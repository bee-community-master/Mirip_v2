#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.postprocess import run_postprocess_for_checkpoint
from training.utils import resolve_project_path, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-evaluate a checkpoint with robust anchor bootstrap metrics.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pairs-val", required=True)
    parser.add_argument("--metadata-train", required=True)
    parser.add_argument("--metadata-eval", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--anchors-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    result = run_postprocess_for_checkpoint(
        checkpoint_path=args.checkpoint,
        pairs_val=args.pairs_val,
        metadata_train=args.metadata_train,
        metadata_eval=args.metadata_eval,
        image_root=args.image_root,
        anchors_output=args.anchors_output,
        report_output=args.output,
        registry_output=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=not args.no_persistent_workers,
        device=args.device,
        precision=args.precision,
    )
    output_path = resolve_project_path(args.output)
    payload = {
        "checkpoint": args.checkpoint,
        "output": str(output_path),
        "report": result["report"],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
