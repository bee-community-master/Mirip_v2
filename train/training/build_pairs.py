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


def _tier_pair_minimums(split: str, args: argparse.Namespace) -> dict[str, int]:
    return {
        "A-S": int(getattr(args, f"{split}_tier_pair_min_a_s")),
        "B-C": int(getattr(args, f"{split}_tier_pair_min_b_c")),
        "A-C": int(getattr(args, f"{split}_tier_pair_min_a_c")),
        "C-S": int(getattr(args, f"{split}_tier_pair_min_c_s")),
    }


def _tier_pair_caps(split: str, args: argparse.Namespace) -> dict[str, int]:
    return {
        "A-B": int(getattr(args, f"{split}_tier_pair_cap_a_b")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate train/val/test metadata and pair CSVs.")
    parser.add_argument("--manifest", default="training/data/snapshot_manifest.csv")
    parser.add_argument("--output-dir", default="training/data")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--train-pairs-target", type=int, default=40_000)
    parser.add_argument("--val-pairs-target", type=int, default=5_000)
    parser.add_argument("--same-dept-ratio", type=float, default=0.5)
    parser.add_argument("--min-score-gap", type=float, default=5.0)
    parser.add_argument("--max-appearances", type=int, default=48)
    parser.add_argument("--distance1-ratio", type=float, default=0.6)
    parser.add_argument("--distance2-ratio", type=float, default=0.3)
    parser.add_argument("--distance3-ratio", type=float, default=0.1)
    parser.add_argument("--train-tier-pair-min-a-s", type=int, default=4_000)
    parser.add_argument("--train-tier-pair-min-b-c", type=int, default=4_000)
    parser.add_argument("--train-tier-pair-min-a-c", type=int, default=3_000)
    parser.add_argument("--train-tier-pair-min-c-s", type=int, default=3_000)
    parser.add_argument("--train-tier-pair-cap-a-b", type=int, default=18_000)
    parser.add_argument("--val-tier-pair-min-a-s", type=int, default=400)
    parser.add_argument("--val-tier-pair-min-b-c", type=int, default=400)
    parser.add_argument("--val-tier-pair-min-a-c", type=int, default=300)
    parser.add_argument("--val-tier-pair-min-c-s", type=int, default=300)
    parser.add_argument("--val-tier-pair-cap-a-b", type=int, default=2_250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-shortfall", action="store_true")
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
            train_pairs_target=args.train_pairs_target,
            val_pairs_target=args.val_pairs_target,
            same_dept_ratio=args.same_dept_ratio,
            min_score_gap=args.min_score_gap,
            max_appearances=args.max_appearances,
            distance1_ratio=args.distance1_ratio,
            distance2_ratio=args.distance2_ratio,
            distance3_ratio=args.distance3_ratio,
            train_tier_pair_minimums=_tier_pair_minimums("train", args),
            val_tier_pair_minimums=_tier_pair_minimums("val", args),
            train_tier_pair_caps=_tier_pair_caps("train", args),
            val_tier_pair_caps=_tier_pair_caps("val", args),
            seed=args.seed,
            strict=not args.allow_shortfall,
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
