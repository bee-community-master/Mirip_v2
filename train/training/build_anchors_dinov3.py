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
from training.models import DinoV3PairwiseModel, resolve_pairwise_model_kwargs
from training.utils import resolve_project_path, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build tier anchors for Mirip_v2 DINOv3 model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-per-tier", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--group-balanced", action=argparse.BooleanOptionalAction, default=None)
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
        **resolve_pairwise_model_kwargs(
            {
                "model_name": config_dict.get("model_name", DEFAULT_DINOV3_MODEL_NAME),
                **config_dict,
            }
        )
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(map_location)

    anchors = build_anchor_store(
        model=model,
        metadata_csv=args.metadata,
        image_root=args.image_root,
        model_name=config_dict.get("model_name", DEFAULT_DINOV3_MODEL_NAME),
        input_size=int(config_dict.get("input_size", 448)),
        n_per_tier=int(args.n_per_tier or config_dict.get("anchor_eval_n_per_tier", 24)),
        seed=int(args.seed or (config_dict.get("anchor_eval_bootstrap_seeds") or [42])[0]),
        group_balanced=bool(
            config_dict.get("anchor_eval_group_balanced", True)
            if args.group_balanced is None
            else args.group_balanced
        ),
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
