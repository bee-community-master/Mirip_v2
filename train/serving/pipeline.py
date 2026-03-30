from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bundle import ServingBundleManifest, sha256sum, write_manifest


@dataclass(slots=True)
class PromotionDecision:
    default_encoder: str
    promote_int8: bool
    reason: str


THREAD_SWEEP = (8, 12, 16)


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
