#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.anchors import build_anchor_store
from training.config import DEFAULT_DINOV3_MODEL_NAME, DinoV3TrainingConfig
from training.models import DinoV3PairwiseModel
from training.utils import resolve_project_path, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build tier anchors for Mirip_v2 DINOv3 model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-per-tier", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    map_location = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    checkpoint = torch.load(resolve_project_path(args.checkpoint), map_location=map_location)
    config_dict = checkpoint.get("config", DinoV3TrainingConfig().to_dict())
    model = DinoV3PairwiseModel(
        model_name=config_dict.get("model_name", DEFAULT_DINOV3_MODEL_NAME),
        projector_hidden_dim=int(config_dict.get("projector_hidden_dim", 512)),
        projector_output_dim=int(config_dict.get("projector_output_dim", 256)),
        dropout=float(config_dict.get("dropout", 0.3)),
        margin=float(config_dict.get("margin", 0.3)),
        freeze_backbone=True,
        backbone_dtype=str(config_dict.get("backbone_dtype", "auto")),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(map_location)

    anchors = build_anchor_store(
        model=model,
        metadata_csv=args.metadata,
        image_root=args.image_root,
        model_name=config_dict.get("model_name", DEFAULT_DINOV3_MODEL_NAME),
        n_per_tier=args.n_per_tier,
        seed=args.seed,
        source_checkpoint=args.checkpoint,
    )
    output_path = anchors.save(args.output)
    payload = {
        "output": str(output_path),
        "tiers": sorted(anchors.features.keys()),
        "counts": {tier: int(features.shape[0]) for tier, features in anchors.features.items()},
        "metadata": anchors.metadata,
    }
    if args.report:
        report_path = resolve_project_path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
