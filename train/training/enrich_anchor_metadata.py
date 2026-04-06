#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.snapshot import VALID_TIERS, make_anchor_group_key
from training.utils import normalize_staged_image_reference, resolve_project_path, resolve_staged_image_path, write_json

HEADER_PATTERN = re.compile(
    r"^\s*(?:\d{4}학년도\s*)?(?:\([^\n)]*\)\s*)?(?P<university>[^\n]+?)\s+(?P<department>[^\n]+?)\s+합격"
)
INVALID_NORMALIZED_DEPTS = {"", "other", "unknown"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich raw metadata anchor fields from interview headers.")
    parser.add_argument("--metadata-dir", default="data/metadata")
    parser.add_argument("--report", default="output_models/logs/anchor_group_enrichment_report.json")
    parser.add_argument("--min-group-size", type=int, default=15)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def _strip_text(value: Any) -> str:
    return str(value or "").strip()


def _is_valid_normalized_dept(value: Any) -> bool:
    return _strip_text(value) not in INVALID_NORMALIZED_DEPTS


def extract_header_fields(interview_clean: str) -> tuple[str | None, str | None]:
    first_line = (interview_clean or "").splitlines()[0].strip() if interview_clean else ""
    if not first_line:
        return None, None
    match = HEADER_PATTERN.search(first_line)
    if not match:
        return None, None
    university = match.group("university").strip()
    department = match.group("department").strip()
    return university or None, department or None


def _load_metadata_payloads(metadata_dir: str | Path) -> list[dict[str, Any]]:
    metadata_root = resolve_project_path(metadata_dir)
    payloads: list[dict[str, Any]] = []
    for path in sorted(metadata_root.glob("*.json"), key=lambda candidate: int(candidate.stem)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_path"] = str(path)
        payloads.append(payload)
    return payloads


def _freeze_counter_map(counter_map: dict[str, Counter[str]]) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    resolved: dict[str, str] = {}
    ambiguous: dict[str, dict[str, int]] = {}
    for key, counter in sorted(counter_map.items()):
        values = {candidate: count for candidate, count in counter.items() if candidate}
        if len(values) == 1:
            resolved[key] = next(iter(values))
        elif len(values) > 1:
            ambiguous[key] = dict(sorted(values.items()))
    return resolved, ambiguous


def build_exact_mappings(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    header_university_counter: dict[str, Counter[str]] = defaultdict(Counter)
    department_raw_counter: dict[str, Counter[str]] = defaultdict(Counter)
    header_department_counter: dict[str, Counter[str]] = defaultdict(Counter)

    for payload in payloads:
        if not _strip_text(payload.get("anchor_group")):
            continue

        header_university, header_department = extract_header_fields(_strip_text(payload.get("interview_clean")))
        university = _strip_text(payload.get("university"))
        department_raw = _strip_text(payload.get("department_raw"))
        normalized_dept = _strip_text(payload.get("normalized_dept"))
        if normalized_dept in INVALID_NORMALIZED_DEPTS:
            continue
        if header_university and university:
            header_university_counter[header_university][university] += 1
        if department_raw:
            department_raw_counter[department_raw][normalized_dept] += 1
        if header_department:
            header_department_counter[header_department][normalized_dept] += 1

    header_university_map, header_university_ambiguous = _freeze_counter_map(header_university_counter)
    department_raw_map, department_raw_ambiguous = _freeze_counter_map(department_raw_counter)
    header_department_map, header_department_ambiguous = _freeze_counter_map(header_department_counter)

    return {
        "header_university": header_university_map,
        "department_raw": department_raw_map,
        "header_department": header_department_map,
        "ambiguous": {
            "header_university": header_university_ambiguous,
            "department_raw": department_raw_ambiguous,
            "header_department": header_department_ambiguous,
        },
    }


def _count_snapshot_eligible(
    payloads: list[dict[str, Any]],
    *,
    image_root: str | Path,
    min_group_size: int,
) -> dict[str, int]:
    group_counter: Counter[str] = Counter()
    for payload in payloads:
        anchor_key = make_anchor_group_key(payload.get("university", ""), payload.get("normalized_dept", ""))
        if anchor_key:
            group_counter[anchor_key] += 1

    valid_groups = {group for group, count in group_counter.items() if count >= min_group_size}
    eligible_items = 0
    for payload in payloads:
        if _strip_text(payload.get("tier")) not in VALID_TIERS:
            continue
        images = payload.get("images") or []
        if not images:
            continue
        normalized_image_path = normalize_staged_image_reference(images[0])
        if normalized_image_path is None:
            continue
        if resolve_staged_image_path(image_root, normalized_image_path) is None:
            continue
        anchor_key = make_anchor_group_key(payload.get("university", ""), payload.get("normalized_dept", ""))
        if anchor_key and anchor_key in valid_groups:
            eligible_items += 1
    return {
        "eligible_items": eligible_items,
        "valid_anchor_groups": len(valid_groups),
    }


def enrich_payload(
    payload: dict[str, Any],
    *,
    mappings: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if _strip_text(payload.get("anchor_group")):
        return deepcopy(payload), None

    updated = deepcopy(payload)
    header_university, header_department = extract_header_fields(_strip_text(payload.get("interview_clean")))
    current_university = _strip_text(payload.get("university"))
    current_department_raw = _strip_text(payload.get("department_raw"))
    current_normalized_dept = _strip_text(payload.get("normalized_dept"))

    before = {
        "university": current_university or None,
        "department_raw": current_department_raw or None,
        "normalized_dept": current_normalized_dept or None,
        "anchor_group": _strip_text(payload.get("anchor_group")) or None,
    }
    change_summary: dict[str, Any] = {
        "file": Path(_strip_text(payload.get("_path"))).name,
        "header_university": header_university,
        "header_department": header_department,
        "before": before,
        "after": None,
        "fields_changed": [],
        "unmatched_reasons": [],
    }

    if not current_university:
        mapped_university = mappings["header_university"].get(header_university or "")
        if mapped_university:
            updated["university"] = mapped_university
            change_summary["fields_changed"].append("university")
        elif header_university:
            change_summary["unmatched_reasons"].append("unmatched_header_university")
        else:
            change_summary["unmatched_reasons"].append("missing_header_university")

    if not current_department_raw and header_department:
        updated["department_raw"] = header_department
        change_summary["fields_changed"].append("department_raw")
    elif not header_department:
        change_summary["unmatched_reasons"].append("missing_header_department")

    if not _is_valid_normalized_dept(current_normalized_dept):
        department_raw_for_lookup = _strip_text(updated.get("department_raw"))
        mapped_department = None
        if department_raw_for_lookup:
            mapped_department = mappings["department_raw"].get(department_raw_for_lookup)
            if mapped_department is None and current_department_raw:
                change_summary["unmatched_reasons"].append("unmatched_department_raw")
        if mapped_department is None and header_department:
            mapped_department = mappings["header_department"].get(header_department)
            if mapped_department is None:
                change_summary["unmatched_reasons"].append("unmatched_header_department")
        if mapped_department:
            updated["normalized_dept"] = mapped_department
            change_summary["fields_changed"].append("normalized_dept")

    anchor_group = make_anchor_group_key(updated.get("university", ""), updated.get("normalized_dept", ""))
    if anchor_group:
        updated["anchor_group"] = anchor_group
        change_summary["fields_changed"].append("anchor_group")
    else:
        change_summary["unmatched_reasons"].append("unable_to_form_anchor_group")

    change_summary["fields_changed"] = sorted(set(change_summary["fields_changed"]))
    change_summary["unmatched_reasons"] = sorted(set(change_summary["unmatched_reasons"]))
    after = {
        "university": _strip_text(updated.get("university")) or None,
        "department_raw": _strip_text(updated.get("department_raw")) or None,
        "normalized_dept": _strip_text(updated.get("normalized_dept")) or None,
        "anchor_group": _strip_text(updated.get("anchor_group")) or None,
    }
    change_summary["after"] = after
    if not change_summary["fields_changed"]:
        return updated, change_summary
    return updated, change_summary


def enrich_metadata(
    *,
    metadata_dir: str | Path,
    min_group_size: int,
    apply: bool,
    report_path: str | Path,
) -> dict[str, Any]:
    metadata_root = resolve_project_path(metadata_dir)
    image_root = metadata_root.parent
    payloads = _load_metadata_payloads(metadata_root)
    mappings = build_exact_mappings(payloads)

    simulated_payloads: list[dict[str, Any]] = []
    changed_files: list[dict[str, Any]] = []
    unchanged_null_anchor: Counter[str] = Counter()
    field_changes: Counter[str] = Counter()

    for payload in payloads:
        updated, change_summary = enrich_payload(payload, mappings=mappings)
        simulated_payloads.append(updated)
        if change_summary is None:
            continue
        if change_summary["fields_changed"]:
            changed_files.append(change_summary)
            for field_name in change_summary["fields_changed"]:
                field_changes[field_name] += 1
        else:
            for reason in change_summary["unmatched_reasons"]:
                unchanged_null_anchor[reason] += 1

    before_eligibility = _count_snapshot_eligible(payloads, image_root=image_root, min_group_size=min_group_size)
    after_eligibility = _count_snapshot_eligible(simulated_payloads, image_root=image_root, min_group_size=min_group_size)

    if apply:
        for payload, updated in zip(payloads, simulated_payloads, strict=True):
            if payload == updated:
                continue
            target = Path(_strip_text(payload.get("_path")))
            updated_payload = {key: value for key, value in updated.items() if key != "_path"}
            target.write_text(json.dumps(updated_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    report = {
        "metadata_dir": str(metadata_root),
        "image_root": str(image_root),
        "apply": apply,
        "min_group_size": min_group_size,
        "totals": {
            "total_json_files": len(payloads),
            "existing_anchor_groups": sum(1 for payload in payloads if _strip_text(payload.get("anchor_group"))),
            "null_anchor_groups": sum(1 for payload in payloads if not _strip_text(payload.get("anchor_group"))),
            "modified_files": len(changed_files),
        },
        "mapping_counts": {
            "header_university": len(mappings["header_university"]),
            "department_raw": len(mappings["department_raw"]),
            "header_department": len(mappings["header_department"]),
        },
        "ambiguous_counts": {
            key: len(value)
            for key, value in mappings["ambiguous"].items()
        },
        "field_changes": dict(sorted(field_changes.items())),
        "unmatched_counts": dict(sorted(unchanged_null_anchor.items())),
        "eligibility": {
            "before": before_eligibility,
            "after": after_eligibility,
            "expected_eligible_delta": after_eligibility["eligible_items"] - before_eligibility["eligible_items"],
        },
        "changed_files": changed_files,
    }
    write_json(report_path, report)
    return report


def main() -> int:
    args = parse_args()
    report = enrich_metadata(
        metadata_dir=args.metadata_dir,
        min_group_size=args.min_group_size,
        apply=args.apply,
        report_path=args.report,
    )
    print(
        json.dumps(
            {
                "modified_files": report["totals"]["modified_files"],
                "expected_eligible_delta": report["eligibility"]["expected_eligible_delta"],
                "field_changes": report["field_changes"],
            },
            ensure_ascii=False,
        )
    )
    print(f"report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
