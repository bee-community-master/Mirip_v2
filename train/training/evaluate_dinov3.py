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

from training.anchors import AnchorStore, evaluate_anchor_tier_accuracy
from training.config import DinoV3TrainingConfig
from training.datasets import DinoPairDataset
from training.evaluation import evaluate_pairwise
from training.models import DinoV3PairwiseModel
from training.utils import resolve_project_path, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Mirip_v2 DINOv3 pairwise ranker.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pairs-val", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--anchors")
    parser.add_argument("--metadata-eval")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    map_location = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    checkpoint_path = resolve_project_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    config_dict = checkpoint.get("config", DinoV3TrainingConfig().to_dict())
    model = DinoV3PairwiseModel(
        model_name=config_dict.get("model_name", "facebook/dinov3-vitl16-pretrain-lvd1689m"),
        projector_hidden_dim=int(config_dict.get("projector_hidden_dim", 512)),
        projector_output_dim=int(config_dict.get("projector_output_dim", 256)),
        dropout=float(config_dict.get("dropout", 0.3)),
        margin=float(config_dict.get("margin", 0.3)),
        freeze_backbone=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    dataset = DinoPairDataset(
        pairs_csv=args.pairs_val,
        image_root=args.image_root,
        model_name=config_dict.get("model_name", "facebook/dinov3-vitl16-pretrain-lvd1689m"),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    results = evaluate_pairwise(model=model, loader=loader, device=args.device, precision=args.precision)

    if args.anchors and args.metadata_eval:
        anchors = AnchorStore.load(args.anchors, map_location=map_location)
        model.to(map_location)
        results.update(
            evaluate_anchor_tier_accuracy(
                model=model,
                anchors=anchors,
                metadata_csv=args.metadata_eval,
                image_root=args.image_root,
                model_name=config_dict.get("model_name", "facebook/dinov3-vitl16-pretrain-lvd1689m"),
                precision=args.precision,
            )
        )

    payload = {
        "checkpoint": str(checkpoint_path),
        "metrics": results,
        "config": config_dict,
    }
    output_path = resolve_project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
