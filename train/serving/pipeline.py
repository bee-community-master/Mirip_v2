from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .bundle import ServingBundleManifest, sha256sum, write_manifest
from training.utils import project_relative_path, resolve_model_source, resolve_project_path


@dataclass(slots=True)
class PromotionDecision:
    default_encoder: str
    promote_int8: bool
    reason: str


THREAD_SWEEP = (8, 12, 16)
DIAGNOSIS_HEAD_FILENAME = "diagnosis_head.pt"
ANCHORS_FILENAME = "anchors.pt"


def resolve_int8_tier_agreement(raw_value: float | None) -> float:
    if raw_value is None:
        return 0.0
    if 0.0 <= raw_value <= 1.0:
        return raw_value
    raise ValueError("--int8-tier-agreement must be between 0.0 and 1.0")


def choose_default_encoder(
    quality_report: dict[str, Any],
    benchmarks: dict[str, Any],
) -> PromotionDecision:
    agreement = float(quality_report.get("int8_tier_agreement_vs_fp32", 0.0))
    fp32_p50 = float(benchmarks.get("encoder_fp32", {}).get("latency_ms_p50", 0.0))
    int8_p50 = float(benchmarks.get("encoder_int8", {}).get("latency_ms_p50", 0.0))
    improvement_ratio = 0.0
    if fp32_p50 > 0 and int8_p50 > 0:
        improvement_ratio = max(0.0, (fp32_p50 - int8_p50) / fp32_p50)

    if agreement >= 0.99 and improvement_ratio >= 0.20:
        return PromotionDecision(
            default_encoder="encoder_int8.onnx",
            promote_int8=True,
            reason="INT8 passed quality gate and improved latency by at least 20%",
        )
    return PromotionDecision(
        default_encoder="encoder_fp32.onnx",
        promote_int8=False,
        reason="FP32 retained as baseline because INT8 did not clear the promotion gate",
    )


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def _cpu_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            payload[key] = value.detach().cpu()
        else:
            payload[key] = value
    return payload


def build_diagnosis_head_payload(
    *,
    model_name: str,
    feature_dim: int,
    projector_hidden_dim: int,
    projector_output_dim: int,
    dropout: float,
    projector_state_dict: dict[str, Any],
    score_head_state_dict: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "model_name": model_name,
        "feature_dim": int(feature_dim),
        "projector_hidden_dim": int(projector_hidden_dim),
        "projector_output_dim": int(projector_output_dim),
        "dropout": float(dropout),
        "projector_state_dict": _cpu_state_dict(projector_state_dict),
        "score_head_state_dict": _cpu_state_dict(score_head_state_dict),
    }


def write_diagnosis_artifacts(
    *,
    bundle_dir: str | Path,
    diagnosis_head_payload: dict[str, Any],
    anchors_path: str | Path,
) -> dict[str, str]:
    bundle_path = Path(bundle_dir)
    bundle_path.mkdir(parents=True, exist_ok=True)

    diagnosis_head_target = bundle_path / DIAGNOSIS_HEAD_FILENAME
    torch.save(diagnosis_head_payload, diagnosis_head_target)

    anchors_source = resolve_project_path(anchors_path)
    anchors_target = bundle_path / ANCHORS_FILENAME
    shutil.copy2(anchors_source, anchors_target)

    return {
        "diagnosis_head": diagnosis_head_target.name,
        "anchors": anchors_target.name,
    }


def validate_diagnosis_artifacts(
    *,
    bundle_model_source: str | Path,
    backbone_hidden_size: int,
    checkpoint_path: str | Path,
    checkpoint_config: dict[str, Any],
    diagnosis_head_payload: dict[str, Any],
    anchors_payload: dict[str, Any],
) -> None:
    expected_model_source = Path(resolve_model_source(str(bundle_model_source))).resolve()
    checkpoint_model_source = Path(
        resolve_model_source(str(checkpoint_config.get("model_name", "")))
    ).resolve()
    if checkpoint_model_source != expected_model_source:
        raise ValueError(
            "Checkpoint model_name must point at the same student export directory as --backbone-dir"
        )

    feature_dim = int(diagnosis_head_payload["feature_dim"])
    if feature_dim != int(backbone_hidden_size):
        raise ValueError(
            "Diagnosis head feature_dim does not match the backbone hidden_size exported by the student"
        )

    metadata = dict(anchors_payload.get("metadata", {}))
    anchor_model_source_raw = metadata.get("model_source") or metadata.get("model_name")
    if not anchor_model_source_raw:
        raise ValueError("Anchors metadata is missing model_source/model_name")
    anchor_model_source = Path(resolve_model_source(str(anchor_model_source_raw))).resolve()
    if anchor_model_source != expected_model_source:
        raise ValueError("Anchors were not built from the same student export directory")

    expected_checkpoint_relative = project_relative_path(checkpoint_path)
    anchor_checkpoint_relative = metadata.get("checkpoint_relative")
    if anchor_checkpoint_relative is None:
        raise ValueError("Anchors metadata is missing checkpoint_relative")
    if str(anchor_checkpoint_relative) != expected_checkpoint_relative:
        raise ValueError("Anchors were not built from the provided checkpoint")

    anchor_feature_dim = metadata.get("feature_dim")
    if anchor_feature_dim is None or int(anchor_feature_dim) != feature_dim:
        raise ValueError("Anchors feature_dim does not match the diagnosis head")

    expected_projector_output_dim = int(diagnosis_head_payload["projector_output_dim"])
    anchor_projector_output_dim = metadata.get("projector_output_dim")
    if (
        anchor_projector_output_dim is None
        or int(anchor_projector_output_dim) != expected_projector_output_dim
    ):
        raise ValueError("Anchors projector_output_dim does not match the diagnosis head")

    features = anchors_payload.get("features")
    if not isinstance(features, dict) or not features:
        raise ValueError("Anchors payload is missing tier features")
    for tier, anchor_features in features.items():
        if not torch.is_tensor(anchor_features):
            raise ValueError(f"Anchor tier '{tier}' must be stored as a tensor")
        if anchor_features.ndim != 2 or anchor_features.shape[1] != expected_projector_output_dim:
            raise ValueError(
                f"Anchor tier '{tier}' feature shape must be [N, {expected_projector_output_dim}]"
            )


def copytree_into_bundle(source_dir: str | Path, bundle_dir: str | Path, *, destination_name: str) -> str:
    source_path = Path(source_dir)
    target_dir = Path(bundle_dir) / destination_name
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_path, target_dir)
    return destination_name


def build_serving_bundle(
    *,
    bundle_dir: str | Path,
    model_name: str,
    export_source: str,
    image_size: int,
    quality_report: dict[str, Any],
    benchmarks: dict[str, Any],
    diagnosis_extras: dict[str, str] | None = None,
) -> tuple[ServingBundleManifest, PromotionDecision]:
    bundle_path = Path(bundle_dir)
    bundle_path.mkdir(parents=True, exist_ok=True)

    decision = choose_default_encoder(quality_report, benchmarks)
    benchmarks_payload = dict(benchmarks)
    best_threads = _resolve_best_thread_count(decision.default_encoder, benchmarks_payload)
    if best_threads is not None:
        benchmarks_payload.setdefault("best_intra_op_num_threads", best_threads)
    files = {
        "encoder_fp32.onnx": "encoder_fp32.onnx",
        "preprocessor.json": "preprocessor.json",
        "benchmarks.json": "benchmarks.json",
        "quality_report.json": "quality_report.json",
        "model_sha256.txt": "model_sha256.txt",
    }
    int8_path = bundle_path / "encoder_int8.onnx"
    if int8_path.exists():
        files["encoder_int8.onnx"] = "encoder_int8.onnx"
    elif decision.promote_int8:
        raise FileNotFoundError(
            "INT8 promotion was selected but encoder_int8.onnx is missing from the bundle"
        )

    metadata = {
        "built_at_epoch_seconds": time.time(),
        "promotion_reason": decision.reason,
    }
    manifest = ServingBundleManifest(
        schema_version="1.0",
        model_name=model_name,
        export_source=export_source,
        image_size=image_size,
        default_encoder=decision.default_encoder,
        files=files,
        extras=dict(diagnosis_extras or {}),
        metadata=metadata,
    )
    write_json(bundle_path / "benchmarks.json", benchmarks_payload)
    write_json(bundle_path / "quality_report.json", quality_report)
    write_manifest(bundle_path, manifest)
    encoder_path = bundle_path / decision.default_encoder
    if encoder_path.exists():
        (bundle_path / "model_sha256.txt").write_text(sha256sum(encoder_path), encoding="utf-8")
    return manifest, decision


def _resolve_best_thread_count(
    default_encoder: str,
    benchmarks: dict[str, Any],
) -> int | None:
    benchmark_key = default_encoder.removesuffix(".onnx")
    candidate = benchmarks.get(benchmark_key, {}).get("thread_count")
    if isinstance(candidate, (int, float)) and int(candidate) in THREAD_SWEEP:
        return int(candidate)
    return None
