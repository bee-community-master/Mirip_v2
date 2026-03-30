#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.postprocess_registry import update_postprocess_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update the Mirip_v2 postprocess best-checkpoint registry.")
    parser.add_argument("--current-checkpoint", required=True)
    parser.add_argument("--current-report", required=True)
    parser.add_argument("--output-registry", required=True)
    parser.add_argument("--best-checkpoint")
    parser.add_argument("--best-report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = update_postprocess_registry(
        current_checkpoint=args.current_checkpoint,
        current_report=args.current_report,
        output_registry=args.output_registry,
        best_checkpoint=args.best_checkpoint,
        best_report=args.best_report,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
