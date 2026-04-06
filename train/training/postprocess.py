from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .anchors import evaluate_anchor_tier_accuracy_bootstrap
from .config import DEFAULT_DINOV3_MODEL_NAME, DinoV3TrainingConfig
from .datasets import DinoPairBatchCollator, DinoPairDataset
from .evaluation import evaluate_pairwise
from .models import DinoV3PairwiseModel, resolve_pairwise_model_kwargs
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
    model = DinoV3PairwiseModel(**resolve_pairwise_model_kwargs({"model_name": model_name, **config_dict}))
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint, config_dict, model


def _build_evaluation_loader(
    *,
    pairs_val: str | Path,
    image_root: str | Path,
    model_name: str,
    input_size: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
) -> DataLoader:
    dataset = DinoPairDataset(pairs_csv=pairs_val)
    collator = DinoPairBatchCollator(
        image_root=image_root,
        model_name=model_name,
        input_size=input_size,
        is_train=False,
    )
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
    registry_output: str | Path | None = None,
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
    input_size = int(runtime_config.get("input_size", 448))

    anchors, anchor_metrics = evaluate_anchor_tier_accuracy_bootstrap(
        model=runtime_model,
        metadata_train_csv=metadata_train,
        metadata_eval_csv=metadata_eval,
        image_root=image_root,
        model_name=model_name,
        input_size=input_size,
        n_per_tier=int(runtime_config.get("anchor_eval_n_per_tier", 24)),
        seeds=[int(seed) for seed in runtime_config.get("anchor_eval_bootstrap_seeds", [42, 43, 44])],
        precision=precision,
        group_balanced=bool(runtime_config.get("anchor_eval_group_balanced", True)),
        source_checkpoint=checkpoint_path,
    )
    anchors_output_path = anchors.save(anchors_output)
    loader = _build_evaluation_loader(
        pairs_val=pairs_val,
        image_root=image_root,
        model_name=model_name,
        input_size=input_size,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    metrics = evaluate_pairwise(model=runtime_model, loader=loader, device=device, precision=precision)
    metrics.update(anchor_metrics)

    payload = _build_postprocess_payload(
        checkpoint_path=checkpoint_path,
        anchors_output_path=anchors_output_path,
        metrics=metrics,
        config_dict=runtime_config,
    )
    report_path = resolve_project_path(report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    registry_payload = None
    if registry_output is not None:
        registry_payload = update_postprocess_registry(
            current_checkpoint=checkpoint_path,
            current_report=report_path,
            output_registry=registry_output,
            best_checkpoint=best_checkpoint,
            best_report=best_report,
            min_improvement=float(runtime_config.get("anchor_eval_min_improvement", 0.0)),
        )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "report": payload,
        "registry": registry_payload,
    }
