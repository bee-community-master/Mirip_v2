#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from torch.utils.data import DataLoader, IterableDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import DistillationExperimentConfig, apply_runtime_overrides, load_config
from datasets import DistillationBatchCollator, build_stage_datasets
from engine import run_experiment
from utils import (
    DEFAULT_IMAGE_MEAN,
    DEFAULT_IMAGE_STD,
    align_to_patch_multiple,
    save_json,
    set_seed,
    try_enable_tf32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Mirip_v2 DINOv3 ViT-L distillation model.")
    parser.add_argument("--config", required=True, help="Path to a YAML config, relative to train/distillation.")
    parser.add_argument("--smoke", action="store_true", help="Run a 1-epoch smoke configuration on a small subset.")
    parser.add_argument("--resume", help="Resume from a checkpoint path relative to train/ or an absolute path.")
    parser.add_argument(
        "--validate-data",
        action="store_true",
        help="Validate dataset availability and load a sample batch without creating a model.",
    )
    parser.add_argument(
        "--report",
        help="Optional JSON report path relative to train/ or absolute path. Used for --validate-data output.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved config JSON before executing.",
    )
    return parser.parse_args()


def _dataset_length(dataset: object) -> int | None:
    if isinstance(dataset, IterableDataset):
        return None
    if hasattr(dataset, "__len__"):
        return int(len(dataset))
    return None


def _peek_batch(dataset: object, batch_size: int) -> dict[str, object]:
    loader = DataLoader(
        dataset,
        batch_size=max(1, batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=DistillationBatchCollator(),
    )
    return next(iter(loader))


def validate_data(config: DistillationExperimentConfig, report_path: str | None = None) -> dict[str, object]:
    """Builds a minimal sample batch to verify staged data availability and split wiring."""

    stages = config.active_stages()
    if not stages:
        raise RuntimeError("No enabled stages are configured for distillation")
    resolution = align_to_patch_multiple(stages[0].resolution, config.data.patch_size or 16)
    train_dataset, val_dataset = build_stage_datasets(
        config,
        resolution=resolution,
        mean=DEFAULT_IMAGE_MEAN,
        std=DEFAULT_IMAGE_STD,
    )
    train_count = _dataset_length(train_dataset)
    val_count = _dataset_length(val_dataset)
    train_batch = _peek_batch(train_dataset, 2)
    val_batch = _peek_batch(val_dataset, 2)
    payload = {
        "config_name": config.experiment.name,
        "source_type": config.data.source_type,
        "resolution": resolution,
        "prepared_train_csv": config.paths.prepared_train_csv,
        "prepared_val_csv": config.paths.prepared_val_csv,
        "train_count": train_count,
        "val_count": val_count,
        "train_batch_shape": list(train_batch["pixel_values"].shape),
        "val_batch_shape": list(val_batch["pixel_values"].shape),
        "train_sample_ids": [record.sample_id for record in train_batch["records"]],
        "val_sample_ids": [record.sample_id for record in val_batch["records"]],
        "train_image_paths": [record.image_path for record in train_batch["records"]],
        "val_image_paths": [record.image_path for record in val_batch["records"]],
    }
    if report_path:
        save_json(report_path, payload)
    return payload


def main() -> int:
    """CLI entry point for distillation training and lightweight data validation."""

    args = parse_args()
    config = load_config(args.config)
    config = apply_runtime_overrides(config, smoke=args.smoke, resume_from=args.resume)
    set_seed(config.experiment.seed)
    try_enable_tf32()

    if args.print_config:
        print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))

    if args.validate_data:
        summary = validate_data(config, report_path=args.report)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    summary = run_experiment(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
