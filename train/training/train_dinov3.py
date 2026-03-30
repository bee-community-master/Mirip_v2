#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.config import DEFAULT_DINOV3_MODEL_NAME, DinoV3TrainingConfig, default_num_workers
from training.datasets import DinoPairBatchCollator, DinoPairDataset
from training.models import DinoV3PairwiseModel
from training.trainer import DinoV3Trainer
from training.utils import resolve_project_path, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Mirip_v2 DINOv3 pairwise ranker.")
    parser.add_argument("--pairs-train", required=True)
    parser.add_argument("--pairs-val", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default=DEFAULT_DINOV3_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--num-workers", type=int, default=default_num_workers())
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from")
    parser.add_argument("--wandb-project", default="mirip-v2-dinov3")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    config = DinoV3TrainingConfig(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_epochs=args.epochs,
        early_stopping_patience=args.patience,
        checkpoint_dir=args.output_dir,
        num_workers=args.num_workers,
        persistent_workers=not args.no_persistent_workers,
        prefetch_factor=args.prefetch_factor,
        device=args.device,
        precision=args.precision,
        seed=args.seed,
        dropout=args.dropout,
        margin=args.margin,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    train_dataset = DinoPairDataset(
        pairs_csv=args.pairs_train,
    )
    val_dataset = DinoPairDataset(
        pairs_csv=args.pairs_val,
    )
    train_collator = DinoPairBatchCollator(image_root=args.image_root, model_name=args.model_name)
    val_collator = DinoPairBatchCollator(image_root=args.image_root, model_name=args.model_name)
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "persistent_workers": config.persistent_workers and config.num_workers > 0,
        "collate_fn": train_collator,
    }
    if config.num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader_kwargs = dict(loader_kwargs)
    val_loader_kwargs["collate_fn"] = val_collator
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **val_loader_kwargs,
    )

    model = DinoV3PairwiseModel(
        model_name=config.model_name,
        projector_hidden_dim=config.projector_hidden_dim,
        projector_output_dim=config.projector_output_dim,
        dropout=config.dropout,
        margin=config.margin,
        freeze_backbone=True,
    )
    trainer = DinoV3Trainer(model=model, config=config, resume_from=args.resume_from)
    summary = trainer.train(train_loader, val_loader)

    report_path = resolve_project_path(args.report) if args.report else Path(config.checkpoint_dir) / "training_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(config.checkpoint_dir)
    payload = {
        "config": config.to_dict(),
        **summary,
        "paths": {
            "best_checkpoint": str(checkpoint_dir / "best_model.pt"),
            "checkpoint_dir": str(checkpoint_dir),
        },
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
