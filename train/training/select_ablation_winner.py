#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.postprocess_registry import PostprocessRecord, choose_best_record
from training.utils import read_json, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the best DINOv3 ablation run.")
    parser.add_argument("--candidate", action="append", required=True, help="Format: name=registry_path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-improvement", type=float, default=0.005)
    return parser.parse_args()


def _parse_candidate(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise SystemExit(f"Invalid candidate value: {value!r}. Expected name=registry_path")
    return name, resolve_project_path(raw_path)


def _load_record(name: str, registry_path: Path) -> tuple[str, dict[str, object], PostprocessRecord]:
    payload = read_json(registry_path)
    checkpoint = payload.get("selected_best_checkpoint_after_compare")
    report = payload.get("selected_best_report_after_compare")
    metrics = payload.get("selected_best_metrics_after_compare")
    if not checkpoint or not report or not isinstance(metrics, dict):
        raise SystemExit(f"Incomplete ablation registry: {registry_path}")
    return name, payload, PostprocessRecord(
        checkpoint_relative=str(checkpoint),
        report_relative=str(report),
        metrics=metrics,
    )


def main() -> int:
    args = parse_args()
    records: list[tuple[str, dict[str, object], PostprocessRecord]] = [
        _load_record(*_parse_candidate(value))
        for value in args.candidate
    ]
    winner_name, winner_payload, winner_record = records[0]
    decision_trace: list[dict[str, object]] = []

    for name, payload, record in records[1:]:
        selected, decision = choose_best_record(record, winner_record, min_improvement=args.min_improvement)
        decision_trace.append(
            {
                "candidate": name,
                "incumbent": winner_name,
                "decision": decision,
            }
        )
        if selected is record:
            winner_name = name
            winner_payload = payload
            winner_record = record

    def _candidate_payload(name: str, record: PostprocessRecord) -> dict[str, object]:
        report_payload = read_json(record.report_relative)
        config = report_payload.get("config", {})
        return {
            "name": name,
            "checkpoint": record.checkpoint_relative,
            "report": record.report_relative,
            "metrics": record.metric_snapshot,
            "config": config,
        }

    candidates = [_candidate_payload(name, record) for name, _payload, record in records]
    winner_candidate = next(item for item in candidates if item["name"] == winner_name)
    output_path = resolve_project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "winner_name": winner_name,
        "winner_checkpoint": winner_record.checkpoint_relative,
        "winner_report": winner_record.report_relative,
        "winner_metrics": winner_record.metric_snapshot,
        "winner_config": winner_candidate["config"],
        "winner_registry": winner_payload,
        "candidates": candidates,
        "decision_trace": decision_trace,
    }
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
