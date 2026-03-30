#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.pairs import PairGenerationError, build_pair_outputs
from training.snapshot import build_snapshot_manifest
from training.utils import (
    normalize_staged_image_reference,
    resolve_project_path,
    resolve_staged_image_path,
    write_json,
    write_rows_to_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate staged Mirip_v2 training data readiness.")
    parser.add_argument("--metadata-dir", default="data/metadata")
    parser.add_argument("--image-root", default="data")
    parser.add_argument("--report", default="reports/readiness_report.json")
    parser.add_argument("--min-group-size", type=int, default=15)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--total-pairs", type=int, default=50_000)
    parser.add_argument("--same-dept-ratio", type=float, default=0.5)
    parser.add_argument("--min-score-gap", type=float, default=5.0)
    parser.add_argument("--max-appearances", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def inspect_staged_inputs(metadata_dir: Path, image_root: Path) -> tuple[dict[str, Any], list[str]]:
    raw_images_dir = image_root / "raw_images"
    summary = {
        "metadata_dir_exists": metadata_dir.is_dir(),
        "image_root_exists": image_root.is_dir(),
        "raw_images_dir_exists": raw_images_dir.is_dir(),
        "metadata_files": 0,
        "raw_images_files": 0,
        "parse_errors": 0,
        "metadata_without_images": 0,
        "absolute_image_references": 0,
        "non_jpg_image_references": 0,
        "invalid_image_references": 0,
        "unresolved_image_references": 0,
        "non_jpg_files_in_raw_images": 0,
    }
    failures: list[str] = []

    if not metadata_dir.is_dir():
        failures.append(f"missing_metadata_dir:{metadata_dir}")
    if not image_root.is_dir():
        failures.append(f"missing_image_root:{image_root}")
    if not raw_images_dir.is_dir():
        failures.append(f"missing_raw_images_dir:{raw_images_dir}")
    if failures:
        return summary, failures

    metadata_files = sorted(metadata_dir.glob("*.json"), key=lambda path: path.name)
    raw_image_files = sorted(
        path for path in raw_images_dir.iterdir() if path.is_file() and not path.name.startswith(".")
    )
    summary["metadata_files"] = len(metadata_files)
    summary["raw_images_files"] = len(raw_image_files)

    if not metadata_files:
        failures.append("metadata_dir_empty")
    if not raw_image_files:
        failures.append("raw_images_dir_empty")

    for image_file in raw_image_files:
        if image_file.suffix.lower() != ".jpg":
            summary["non_jpg_files_in_raw_images"] += 1

    for metadata_file in metadata_files:
        try:
            payload = json.loads(metadata_file.read_text(encoding="utf-8"))
        except Exception:
            summary["parse_errors"] += 1
            continue

        images = payload.get("images") or []
        if not images:
            summary["metadata_without_images"] += 1
            continue

        for image_reference in images:
            candidate = Path(str(image_reference).strip())
            if candidate.is_absolute():
                summary["absolute_image_references"] += 1
            if candidate.suffix.lower() != ".jpg":
                summary["non_jpg_image_references"] += 1

            normalized = normalize_staged_image_reference(image_reference)
            if normalized is None:
                summary["invalid_image_references"] += 1
                continue

            if resolve_staged_image_path(image_root, normalized) is None:
                summary["unresolved_image_references"] += 1

    if summary["parse_errors"]:
        failures.append("metadata_parse_errors")
    if summary["metadata_without_images"]:
        failures.append("metadata_missing_images")
    if summary["absolute_image_references"]:
        failures.append("absolute_image_references_present")
    if summary["non_jpg_image_references"]:
        failures.append("non_jpg_image_references_present")
    if summary["invalid_image_references"]:
        failures.append("invalid_image_references_present")
    if summary["unresolved_image_references"]:
        failures.append("unresolved_image_references_present")
    if summary["non_jpg_files_in_raw_images"]:
        failures.append("non_jpg_files_present_in_raw_images")

    return summary, failures


def main() -> int:
    args = parse_args()
    metadata_dir = resolve_project_path(args.metadata_dir)
    image_root = resolve_project_path(args.image_root)

    input_summary, failures = inspect_staged_inputs(metadata_dir=metadata_dir, image_root=image_root)
    snapshot_report: dict[str, Any] | None = None
    pair_stats: dict[str, Any] | None = None

    if not failures:
        rows, snapshot_report = build_snapshot_manifest(
            metadata_dir=args.metadata_dir,
            image_root=args.image_root,
            min_group_size=args.min_group_size,
        )
        if not rows:
            failures.append("snapshot_manifest_empty")
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                manifest_path = Path(temp_dir) / "snapshot_manifest.csv"
                write_rows_to_csv(manifest_path, rows, list(rows[0].keys()))
                try:
                    pair_stats = build_pair_outputs(
                        manifest_csv=manifest_path,
                        output_dir=temp_dir,
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
                    pair_stats = exc.stats
                    failures.append("pair_shortfall")

    payload = {
        "ready": not failures,
        "metadata_dir": str(metadata_dir),
        "image_root": str(image_root),
        "input_summary": input_summary,
        "snapshot_report": snapshot_report,
        "pair_stats": pair_stats,
        "failures": failures,
    }
    write_json(args.report, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
