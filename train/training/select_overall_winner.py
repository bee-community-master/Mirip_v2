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
    parser = argparse.ArgumentParser(description="Select the overall best winner across frozen and unfreeze summaries.")
    parser.add_argument("--summary", action="append", required=True, help="Format: name=summary_path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-improvement", type=float, default=0.005)
    return parser.parse_args()


def _parse_summary(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise SystemExit(f"Invalid summary value: {value!r}. Expected name=summary_path")
    return name, resolve_project_path(raw_path)


def _load_summary(name: str, summary_path: Path) -> tuple[str, dict[str, object], PostprocessRecord] | None:
    payload = read_json(summary_path)
    winner_checkpoint = payload.get("winner_checkpoint")
    winner_report = payload.get("winner_report")
    winner_metrics = payload.get("winner_metrics")
    if not winner_checkpoint or not winner_report or not isinstance(winner_metrics, dict):
        return None
    return name, payload, PostprocessRecord(
        checkpoint_relative=str(winner_checkpoint),
        report_relative=str(winner_report),
        metrics=winner_metrics,
    )


def main() -> int:
    args = parse_args()
    summaries = []
    for raw_summary in args.summary:
        parsed = _load_summary(*_parse_summary(raw_summary))
        if parsed is not None:
            summaries.append(parsed)
    if not summaries:
        raise SystemExit("No valid summaries were provided.")

    winner_name, winner_payload, winner_record = summaries[0]
    decision_trace: list[dict[str, object]] = []

    for name, payload, record in summaries[1:]:
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

    output_path = resolve_project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "winner_name": winner_name,
        "winner_checkpoint": winner_record.checkpoint_relative,
        "winner_report": winner_record.report_relative,
        "winner_metrics": winner_record.metric_snapshot,
        "winner_config": winner_payload.get("winner_config", {}),
        "winner_summary": winner_payload,
        "decision_trace": decision_trace,
        "candidates": [
            {
                "name": name,
                "checkpoint": record.checkpoint_relative,
                "report": record.report_relative,
                "metrics": record.metric_snapshot,
                "config": payload.get("winner_config", {}),
            }
            for name, payload, record in summaries
        ],
    }
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
