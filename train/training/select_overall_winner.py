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
    parser = argparse.ArgumentParser(description="Select the overall best winner across frozen and unfreeze summaries.")
    parser.add_argument("--summary", action="append", required=True, help="Format: name=summary_path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-improvement", type=float, default=0.005)
    return parser.parse_args()

def _load_summary(name: str, summary_path) -> NamedWinnerRecord | None:
    payload = read_json(summary_path)
    winner_checkpoint = payload.get("winner_checkpoint")
    winner_report = payload.get("winner_report")
    winner_metrics = payload.get("winner_metrics")
    if not winner_checkpoint or not winner_report or not isinstance(winner_metrics, dict):
        return None
    return NamedWinnerRecord(
        name=name,
        payload=payload,
        record=PostprocessRecord(
            checkpoint_relative=str(winner_checkpoint),
            report_relative=str(winner_report),
            metrics=winner_metrics,
        ),
    )


def main() -> int:
    args = parse_args()
    summaries = []
    for raw_summary in args.summary:
        parsed = _load_summary(*parse_named_path(raw_summary, value_label="summary"))
        if parsed is not None:
            summaries.append(parsed)
    winner, decision_trace = choose_named_winner(summaries, min_improvement=args.min_improvement)

    output_path = resolve_project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "winner_name": winner.name,
        "winner_checkpoint": winner.record.checkpoint_relative,
        "winner_report": winner.record.report_relative,
        "winner_metrics": winner.record.metric_snapshot,
        "winner_config": winner.payload.get("winner_config", {}),
        "winner_summary": winner.payload,
        "decision_trace": decision_trace,
        "candidates": [
            {
                "name": candidate.name,
                "checkpoint": candidate.record.checkpoint_relative,
                "report": candidate.record.report_relative,
                "metrics": candidate.record.metric_snapshot,
                "config": candidate.payload.get("winner_config", {}),
            }
            for candidate in summaries
        ],
    }
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
