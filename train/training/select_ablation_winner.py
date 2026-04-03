#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.postprocess_registry import PostprocessRecord
from training.utils import read_json, resolve_project_path
from training.winner_selection import NamedWinnerRecord, choose_named_winner, parse_named_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the best DINOv3 ablation run.")
    parser.add_argument("--candidate", action="append", required=True, help="Format: name=registry_path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-improvement", type=float, default=0.005)
    return parser.parse_args()

def _load_record(name: str, registry_path) -> NamedWinnerRecord:
    payload = read_json(registry_path)
    checkpoint = payload.get("selected_best_checkpoint_after_compare")
    report = payload.get("selected_best_report_after_compare")
    metrics = payload.get("selected_best_metrics_after_compare")
    if not checkpoint or not report or not isinstance(metrics, dict):
        raise SystemExit(f"Incomplete ablation registry: {registry_path}")
    return NamedWinnerRecord(
        name=name,
        payload=payload,
        record=PostprocessRecord(
            checkpoint_relative=str(checkpoint),
            report_relative=str(report),
            metrics=metrics,
        ),
    )


def main() -> int:
    args = parse_args()
    records: list[NamedWinnerRecord] = [
        _load_record(*parse_named_path(value, value_label="candidate"))
        for value in args.candidate
    ]
    winner, decision_trace = choose_named_winner(records, min_improvement=args.min_improvement)

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

    candidates = [_candidate_payload(candidate.name, candidate.record) for candidate in records]
    winner_candidate = next(item for item in candidates if item["name"] == winner.name)
    output_path = resolve_project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "winner_name": winner.name,
        "winner_checkpoint": winner.record.checkpoint_relative,
        "winner_report": winner.record.report_relative,
        "winner_metrics": winner.record.metric_snapshot,
        "winner_config": winner_candidate["config"],
        "winner_registry": winner.payload,
        "candidates": candidates,
        "decision_trace": decision_trace,
    }
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
