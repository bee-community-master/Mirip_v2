#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.snapshot import build_snapshot_manifest
from training.utils import write_json, write_rows_to_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a crawl snapshot manifest for Mirip_v2 training.")
    parser.add_argument("--metadata-dir", default="data/crawled/metadata")
    parser.add_argument("--image-root", default="data/crawled")
    parser.add_argument("--output-manifest", default="train/training/data/snapshot_manifest.csv")
    parser.add_argument("--report", default="train/reports/snapshot_report.json")
    parser.add_argument("--min-group-size", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, report = build_snapshot_manifest(
        metadata_dir=args.metadata_dir,
        image_root=args.image_root,
        min_group_size=args.min_group_size,
    )
    print(f"eligible_items={report['eligible_items']} parse_errors={report['parse_errors']}")
    print(f"tier_distribution={report['tier_distribution']}")
    print(f"skipped={report['skipped']}")

    if args.dry_run:
        return 0

    fieldnames = list(rows[0].keys()) if rows else report["required_manifest_columns"]
    write_rows_to_csv(args.output_manifest, rows, fieldnames)
    write_json(args.report, report)
    print(f"manifest={args.output_manifest}")
    print(f"report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
