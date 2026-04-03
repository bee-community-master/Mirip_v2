from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .postprocess_registry import PostprocessRecord, choose_best_record
from .utils import resolve_project_path


@dataclass(frozen=True)
class NamedWinnerRecord:
    name: str
    payload: dict[str, object]
    record: PostprocessRecord


def parse_named_path(value: str, *, value_label: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise SystemExit(f"Invalid {value_label} value: {value!r}. Expected name=path")
    return name, resolve_project_path(raw_path)


def choose_named_winner(
    candidates: list[NamedWinnerRecord],
    *,
    min_improvement: float,
) -> tuple[NamedWinnerRecord, list[dict[str, object]]]:
    if not candidates:
        raise SystemExit("No valid candidates were provided.")

    winner = candidates[0]
    decision_trace: list[dict[str, object]] = []

    for candidate in candidates[1:]:
        selected, decision = choose_best_record(candidate.record, winner.record, min_improvement=min_improvement)
        decision_trace.append(
            {
                "candidate": candidate.name,
                "incumbent": winner.name,
                "decision": decision,
            }
        )
        if selected is candidate.record:
            winner = candidate

    return winner, decision_trace
