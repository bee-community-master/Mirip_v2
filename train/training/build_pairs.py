#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.pairs import PairGenerationError, build_pair_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate train/val/test metadata and pair CSVs.")
    parser.add_argument("--manifest", default="training/data/snapshot_manifest.csv")
    parser.add_argument("--output-dir", default="training/data")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--total-pairs", type=int, default=50_000)
    parser.add_argument("--same-dept-ratio", type=float, default=0.5)
    parser.add_argument("--min-score-gap", type=float, default=5.0)
    parser.add_argument("--max-appearances", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.dry_run:
        temp_dir = tempfile.TemporaryDirectory()
        output_dir = temp_dir.name

    try:
        stats = build_pair_outputs(
            manifest_csv=args.manifest,
            output_dir=output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            total_pairs=args.total_pairs,
            same_dept_ratio=args.same_dept_ratio,
            min_score_gap=args.min_score_gap,
            max_appearances=args.max_appearances,
            seed=args.seed,
            strict=True,
        )
    except PairGenerationError as exc:
        print(json.dumps(exc.stats, indent=2, ensure_ascii=False))
        return 1
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
