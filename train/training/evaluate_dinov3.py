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

from training.config import DEFAULT_DINOV3_MODEL_NAME, default_num_workers
from training.datasets import DinoPairBatchCollator, DinoPairDataset
from training.postprocess import load_checkpoint_model
from training.anchors import AnchorStore, evaluate_anchor_tier_accuracy
from training.evaluation import evaluate_pairwise
from training.utils import project_relative_path, resolve_project_path, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Mirip_v2 DINOv3 pairwise ranker.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pairs-val", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--anchors")
    parser.add_argument("--metadata-eval")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=default_num_workers())
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    map_location = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    checkpoint_path = resolve_project_path(args.checkpoint)
    _, config_dict, model = load_checkpoint_model(checkpoint_path, map_location=map_location)

    dataset = DinoPairDataset(pairs_csv=args.pairs_val)
    collator = DinoPairBatchCollator(
        image_root=args.image_root,
        model_name=config_dict.get("model_name", DEFAULT_DINOV3_MODEL_NAME),
        input_size=int(config_dict.get("input_size", 448)),
        is_train=False,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": not args.no_persistent_workers and args.num_workers > 0,
        "collate_fn": collator,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)
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
                model_name=config_dict.get("model_name", DEFAULT_DINOV3_MODEL_NAME),
                input_size=int(config_dict.get("input_size", 448)),
                precision=args.precision,
            )
        )

    payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_relative": project_relative_path(checkpoint_path),
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
