from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .anchors import build_anchor_store, evaluate_anchor_tier_accuracy
from .config import DEFAULT_DINOV3_MODEL_NAME, DinoV3TrainingConfig
from .datasets import DinoPairBatchCollator, DinoPairDataset
from .evaluation import evaluate_pairwise
from .models import DinoV3PairwiseModel
from .postprocess_registry import update_postprocess_registry
from .utils import project_relative_path, resolve_project_path


def _resolve_model_name(config_dict: dict[str, Any]) -> str:
    return str(config_dict.get("model_name", DEFAULT_DINOV3_MODEL_NAME))


def load_checkpoint_model(
    checkpoint_path: str | Path,
    map_location: str,
) -> tuple[dict[str, Any], dict[str, Any], DinoV3PairwiseModel]:
    resolved_checkpoint = resolve_project_path(checkpoint_path)
    checkpoint = torch.load(resolved_checkpoint, map_location=map_location)
    config_dict = checkpoint.get("config", DinoV3TrainingConfig().to_dict())
    model_name = _resolve_model_name(config_dict)
    model = DinoV3PairwiseModel(
        model_name=model_name,
        projector_hidden_dim=int(config_dict.get("projector_hidden_dim", 512)),
        projector_output_dim=int(config_dict.get("projector_output_dim", 256)),
        dropout=float(config_dict.get("dropout", 0.3)),
        margin=float(config_dict.get("margin", 0.3)),
        freeze_backbone=True,
        backbone_dtype=str(config_dict.get("backbone_dtype", "auto")),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint, config_dict, model


def _build_evaluation_loader(
    *,
    pairs_val: str | Path,
    image_root: str | Path,
    model_name: str,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
) -> DataLoader:
    dataset = DinoPairDataset(pairs_csv=pairs_val)
    collator = DinoPairBatchCollator(image_root=image_root, model_name=model_name)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": persistent_workers and num_workers > 0,
        "collate_fn": collator,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def _build_postprocess_payload(
    checkpoint_path: str | Path,
    anchors_output_path: str | Path,
    metrics: dict[str, Any],
    config_dict: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_resolved = resolve_project_path(checkpoint_path)
    return {
        "checkpoint": str(checkpoint_resolved),
        "checkpoint_relative": project_relative_path(checkpoint_resolved),
        "metrics": metrics,
        "config": config_dict,
        "anchors_output": str(anchors_output_path),
        "anchors_output_relative": project_relative_path(anchors_output_path),
    }


def run_postprocess_for_checkpoint(
    *,
    checkpoint_path: str | Path,
    pairs_val: str | Path,
    metadata_train: str | Path,
    metadata_eval: str | Path,
    image_root: str | Path,
    anchors_output: str | Path,
    report_output: str | Path,
    registry_output: str | Path,
    best_checkpoint: str | Path | None = None,
    best_report: str | Path | None = None,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
    device: str,
    precision: str,
    model: torch.nn.Module | None = None,
    config_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime_model = model
    runtime_config = config_dict
    if runtime_model is None:
        map_location = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        _, runtime_config, runtime_model = load_checkpoint_model(checkpoint_path, map_location=map_location)
        runtime_model.to(map_location)
    elif runtime_config is None:
        runtime_config = DinoV3TrainingConfig().to_dict()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    assert runtime_model is not None
    assert runtime_config is not None
    model_name = _resolve_model_name(runtime_config)

    anchors = build_anchor_store(
        model=runtime_model,
        metadata_csv=metadata_train,
        image_root=image_root,
        model_name=model_name,
    )
    anchors_output_path = anchors.save(anchors_output)
    loader = _build_evaluation_loader(
        pairs_val=pairs_val,
        image_root=image_root,
        model_name=model_name,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    metrics = evaluate_pairwise(model=runtime_model, loader=loader, device=device, precision=precision)
    metrics.update(
        evaluate_anchor_tier_accuracy(
            model=runtime_model,
            anchors=anchors,
            metadata_csv=metadata_eval,
            image_root=image_root,
            model_name=model_name,
            precision=precision,
        )
    )

    payload = _build_postprocess_payload(
        checkpoint_path=checkpoint_path,
        anchors_output_path=anchors_output_path,
        metrics=metrics,
        config_dict=runtime_config,
    )
    report_path = resolve_project_path(report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    registry_payload = update_postprocess_registry(
        current_checkpoint=checkpoint_path,
        current_report=report_path,
        output_registry=registry_output,
        best_checkpoint=best_checkpoint,
        best_report=best_report,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "report": payload,
        "registry": registry_payload,
    }
