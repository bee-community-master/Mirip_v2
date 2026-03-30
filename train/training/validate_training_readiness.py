#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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
    load_rows_from_csv,
    read_json,
    normalize_staged_image_reference,
    resolve_project_path,
    resolve_staged_image_path,
    write_json,
    write_rows_to_csv,
)

DEFAULT_RAW_REPORT = "output_models/logs/readiness_report.json"
DEFAULT_PREPARED_REPORT = "output_models/logs/prepared_readiness_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate staged Mirip_v2 training data readiness.")
    parser.add_argument("--mode", choices=["raw", "prepared"], default="raw")
    parser.add_argument("--metadata-dir", default="data/metadata")
    parser.add_argument("--image-root", default="data")
    parser.add_argument("--report")
    parser.add_argument("--manifest", default="training/data/snapshot_manifest.csv")
    parser.add_argument("--prepared-dir", default="training/data")
    parser.add_argument("--baseline-readiness-report", default=DEFAULT_RAW_REPORT)
    parser.add_argument("--baseline-snapshot-report", default="output_models/logs/snapshot_report.json")
    parser.add_argument("--min-group-size", type=int, default=15)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--total-pairs", type=int, default=50_000)
    parser.add_argument("--same-dept-ratio", type=float, default=0.5)
    parser.add_argument("--min-score-gap", type=float, default=5.0)
    parser.add_argument("--max-appearances", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _build_listing_signature(paths: list[Path]) -> tuple[str, int, int]:
    hasher = hashlib.sha256()
    total_bytes = 0
    latest_mtime = 0
    for path in paths:
        stat = path.stat()
        size = int(stat.st_size)
        mtime = int(stat.st_mtime)
        total_bytes += size
        latest_mtime = max(latest_mtime, mtime)
        hasher.update(f"{path.name}:{size}:{mtime}\n".encode("utf-8"))
    return hasher.hexdigest(), total_bytes, latest_mtime


def inspect_staged_inputs(metadata_dir: Path, image_root: Path) -> tuple[dict[str, Any], list[str]]:
    raw_images_dir = image_root / "raw_images"
    summary = {
        "metadata_dir_exists": metadata_dir.is_dir(),
        "image_root_exists": image_root.is_dir(),
        "raw_images_dir_exists": raw_images_dir.is_dir(),
        "metadata_files": 0,
        "raw_images_files": 0,
        "metadata_total_bytes": 0,
        "raw_images_total_bytes": 0,
        "metadata_latest_mtime": 0,
        "raw_images_latest_mtime": 0,
        "metadata_listing_sha256": "",
        "raw_images_listing_sha256": "",
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
    metadata_sig, metadata_bytes, metadata_mtime = _build_listing_signature(metadata_files)
    raw_sig, raw_bytes, raw_mtime = _build_listing_signature(raw_image_files)
    summary["metadata_total_bytes"] = metadata_bytes
    summary["raw_images_total_bytes"] = raw_bytes
    summary["metadata_latest_mtime"] = metadata_mtime
    summary["raw_images_latest_mtime"] = raw_mtime
    summary["metadata_listing_sha256"] = metadata_sig
    summary["raw_images_listing_sha256"] = raw_sig

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


def compare_input_summaries(current: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    keys_to_compare = [
        "metadata_files",
        "raw_images_files",
        "metadata_total_bytes",
        "raw_images_total_bytes",
        "metadata_latest_mtime",
        "raw_images_latest_mtime",
        "metadata_listing_sha256",
        "raw_images_listing_sha256",
        "parse_errors",
        "metadata_without_images",
        "absolute_image_references",
        "non_jpg_image_references",
        "invalid_image_references",
        "unresolved_image_references",
        "non_jpg_files_in_raw_images",
    ]
    mismatched = [key for key in keys_to_compare if current.get(key) != baseline.get(key)]
    if mismatched:
        failures.append("raw_input_changed_since_freeze")
        failures.extend(f"raw_input_mismatch:{key}" for key in mismatched)
    return failures


def inspect_prepared_artifacts(args: argparse.Namespace, input_summary: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    prepared_dir = resolve_project_path(args.prepared_dir)
    manifest_path = resolve_project_path(args.manifest)
    baseline_readiness_path = resolve_project_path(args.baseline_readiness_report)
    baseline_snapshot_path = resolve_project_path(args.baseline_snapshot_report)
    image_root = resolve_project_path(args.image_root)

    summary: dict[str, Any] = {
        "prepared_dir": str(prepared_dir),
        "manifest_path": str(manifest_path),
        "baseline_readiness_report": str(baseline_readiness_path),
        "baseline_snapshot_report": str(baseline_snapshot_path),
        "manifest_rows": 0,
        "metadata_rows": {
            "train": 0,
            "val": 0,
            "test": 0,
        },
        "pair_rows": {
            "train": 0,
            "val": 0,
        },
        "pair_targets": {
            "train": int(args.total_pairs * args.train_ratio),
            "val": max(int(args.total_pairs * args.val_ratio), 1_000),
        },
        "missing_files": [],
    }

    if not baseline_readiness_path.is_file():
        failures.append(f"missing_baseline_readiness_report:{baseline_readiness_path}")
        return summary, failures
    if not baseline_snapshot_path.is_file():
        failures.append(f"missing_baseline_snapshot_report:{baseline_snapshot_path}")
        return summary, failures

    baseline_readiness = read_json(baseline_readiness_path)
    baseline_snapshot = read_json(baseline_snapshot_path)
    baseline_input_summary = baseline_readiness.get("input_summary") or {}
    failures.extend(compare_input_summaries(input_summary, baseline_input_summary))

    required_paths = {
        "manifest": manifest_path,
        "metadata_train": prepared_dir / "metadata_train.csv",
        "metadata_val": prepared_dir / "metadata_val.csv",
        "metadata_test": prepared_dir / "metadata_test.csv",
        "pairs_train": prepared_dir / "pairs_train.csv",
        "pairs_val": prepared_dir / "pairs_val.csv",
    }
    missing_files = [label for label, path in required_paths.items() if not path.is_file()]
    summary["missing_files"] = missing_files
    if missing_files:
        failures.extend(f"missing_prepared_file:{label}" for label in missing_files)
        return summary, failures

    manifest_rows = load_rows_from_csv(manifest_path)
    summary["manifest_rows"] = len(manifest_rows)
    if not manifest_rows:
        failures.append("prepared_manifest_empty")
        return summary, failures

    required_manifest_columns = baseline_snapshot.get("required_manifest_columns") or []
    manifest_columns = set(manifest_rows[0].keys())
    missing_columns = [column for column in required_manifest_columns if column not in manifest_columns]
    if missing_columns:
        failures.extend(f"prepared_manifest_missing_column:{column}" for column in missing_columns)

    expected_manifest_rows = int(baseline_snapshot.get("eligible_items") or 0)
    if expected_manifest_rows and len(manifest_rows) != expected_manifest_rows:
        failures.append("prepared_manifest_row_count_mismatch")

    manifest_image_paths = {row["image_path"] for row in manifest_rows}
    manifest_posts = {row["post_no"] for row in manifest_rows}
    unresolved_manifest_rows = 0
    for row in manifest_rows:
        if resolve_staged_image_path(image_root, row["image_path"]) is None:
            unresolved_manifest_rows += 1
    if unresolved_manifest_rows:
        failures.append("prepared_manifest_has_unresolved_images")
    summary["manifest_unresolved_images"] = unresolved_manifest_rows

    split_rows: dict[str, list[dict[str, str]]] = {}
    split_posts: dict[str, set[str]] = {}
    split_images: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        rows = load_rows_from_csv(required_paths[f"metadata_{split}"])
        split_rows[split] = rows
        split_posts[split] = {row["post_no"] for row in rows}
        split_images[split] = {row["image_path"] for row in rows}
        summary["metadata_rows"][split] = len(rows)
        if len(split_posts[split]) != len(rows):
            failures.append(f"prepared_metadata_duplicate_post_no:{split}")
        if len(split_images[split]) != len(rows):
            failures.append(f"prepared_metadata_duplicate_image_path:{split}")
        missing_from_manifest = split_images[split] - manifest_image_paths
        if missing_from_manifest:
            failures.append(f"prepared_metadata_not_in_manifest:{split}")

    if split_posts["train"] & split_posts["val"]:
        failures.append("prepared_metadata_overlap:train_val")
    if split_posts["train"] & split_posts["test"]:
        failures.append("prepared_metadata_overlap:train_test")
    if split_posts["val"] & split_posts["test"]:
        failures.append("prepared_metadata_overlap:val_test")

    combined_posts = split_posts["train"] | split_posts["val"] | split_posts["test"]
    if combined_posts != manifest_posts:
        failures.append("prepared_metadata_union_mismatch")

    pair_targets = summary["pair_targets"]
    for split in ("train", "val"):
        rows = load_rows_from_csv(required_paths[f"pairs_{split}"])
        summary["pair_rows"][split] = len(rows)
        if len(rows) != pair_targets[split]:
            failures.append(f"prepared_pair_count_mismatch:{split}")
        unresolved_pairs = 0
        for row in rows:
            if row["image_path_1"] not in manifest_image_paths or row["image_path_2"] not in manifest_image_paths:
                unresolved_pairs += 1
        if unresolved_pairs:
            failures.append(f"prepared_pairs_not_in_manifest:{split}")
        summary[f"pairs_missing_manifest_refs_{split}"] = unresolved_pairs

    return summary, failures


def main() -> int:
    args = parse_args()
    metadata_dir = resolve_project_path(args.metadata_dir)
    image_root = resolve_project_path(args.image_root)
    report_path = args.report or (DEFAULT_RAW_REPORT if args.mode == "raw" else DEFAULT_PREPARED_REPORT)

    input_summary, failures = inspect_staged_inputs(metadata_dir=metadata_dir, image_root=image_root)
    snapshot_report: dict[str, Any] | None = None
    pair_stats: dict[str, Any] | None = None
    prepared_report: dict[str, Any] | None = None

    if args.mode == "raw" and not failures:
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
    elif args.mode == "prepared" and not failures:
        prepared_report, prepared_failures = inspect_prepared_artifacts(args=args, input_summary=input_summary)
        failures.extend(prepared_failures)

    payload = {
        "mode": args.mode,
        "ready": not failures,
        "metadata_dir": str(metadata_dir),
        "image_root": str(image_root),
        "input_summary": input_summary,
        "snapshot_report": snapshot_report,
        "pair_stats": pair_stats,
        "prepared_report": prepared_report,
        "failures": failures,
    }
    write_json(report_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
