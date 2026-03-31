from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .utils import float_or_none, project_relative_path, read_json, resolve_project_path, write_json

SELECTION_CRITERIA: tuple[tuple[str, bool], ...] = (
    ("anchor_tier_accuracy", True),
    ("val_accuracy", True),
    ("same_dept_accuracy", True),
    ("val_loss", False),
    ("epoch", True),
)

CHECKPOINT_EPOCH_PATTERN = re.compile(r"checkpoint_epoch_(\d+)\.pt$")


def _normalize_train_relative(path: str | Path | None) -> str | None:
    if path in (None, ""):
        return None

    raw = str(path).strip()
    if not raw:
        return None

    candidate = Path(raw)
    if not candidate.is_absolute():
        return raw

    parts = list(candidate.parts)
    for marker in ("train", "workspace"):
        if marker in parts:
            if marker == "train":
                index = parts.index(marker)
                return str(Path(*parts[index + 1 :]))
            if marker == "workspace" and "train" in parts[parts.index(marker) + 1 :]:
                train_index = parts.index("train", parts.index(marker) + 1)
                return str(Path(*parts[train_index + 1 :]))
    return raw


def infer_epoch_from_checkpoint(path: str | Path | None) -> int | None:
    if path in (None, ""):
        return None
    match = CHECKPOINT_EPOCH_PATTERN.search(str(path))
    if not match:
        return None
    return int(match.group(1))


def _metric_snapshot(metrics: dict[str, Any], checkpoint_relative: str) -> dict[str, Any]:
    epoch = metrics.get("epoch")
    if epoch is None:
        epoch = infer_epoch_from_checkpoint(checkpoint_relative)
    snapshot = {
        "anchor_tier_accuracy": float_or_none(metrics.get("anchor_tier_accuracy")),
        "val_accuracy": float_or_none(metrics.get("val_accuracy")),
        "same_dept_accuracy": float_or_none(metrics.get("same_dept_accuracy")),
        "val_loss": float_or_none(metrics.get("val_loss")),
        "epoch": int(epoch) if epoch is not None else None,
    }
    return snapshot


def _compare_metric(candidate_value: Any, incumbent_value: Any, higher_is_better: bool) -> int:
    if candidate_value == incumbent_value:
        return 0
    if candidate_value is None:
        return -1
    if incumbent_value is None:
        return 1
    if higher_is_better:
        return 1 if candidate_value > incumbent_value else -1
    return 1 if candidate_value < incumbent_value else -1


@dataclass(frozen=True)
class PostprocessRecord:
    checkpoint_relative: str
    report_relative: str
    metrics: dict[str, Any]

    @property
    def metric_snapshot(self) -> dict[str, Any]:
        return _metric_snapshot(self.metrics, self.checkpoint_relative)

    def to_payload(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint_relative,
            "report": self.report_relative,
            "metrics": self.metric_snapshot,
        }


def load_report_record(
    checkpoint_path: str | Path | None,
    report_path: str | Path,
) -> PostprocessRecord:
    payload = read_json(report_path)
    checkpoint_relative = _normalize_train_relative(checkpoint_path)
    if checkpoint_relative is None:
        checkpoint_relative = (
            payload.get("checkpoint_relative")
            or _normalize_train_relative(payload.get("checkpoint"))
        )
    if checkpoint_relative is None:
        raise ValueError("Unable to resolve checkpoint path for postprocess report.")

    report_relative = project_relative_path(report_path)
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Postprocess report must contain a metrics object.")
    return PostprocessRecord(
        checkpoint_relative=checkpoint_relative,
        report_relative=report_relative,
        metrics=metrics,
    )


def load_existing_best(
    registry_path: str | Path,
    best_checkpoint: str | Path | None = None,
    best_report: str | Path | None = None,
) -> PostprocessRecord | None:
    registry_file = resolve_project_path(registry_path)
    if registry_file.exists():
        payload = read_json(registry_file)
        checkpoint_relative = payload.get("selected_best_checkpoint_after_compare")
        report_relative = payload.get("selected_best_report_after_compare")
        metrics = payload.get("selected_best_metrics_after_compare")
        if checkpoint_relative and report_relative and isinstance(metrics, dict):
            return PostprocessRecord(
                checkpoint_relative=checkpoint_relative,
                report_relative=report_relative,
                metrics=metrics,
            )

    if best_report:
        return load_report_record(best_checkpoint, best_report)
    return None


def choose_best_record(
    candidate: PostprocessRecord,
    incumbent: PostprocessRecord | None,
) -> tuple[PostprocessRecord, dict[str, Any]]:
    if incumbent is None:
        return candidate, {
            "decision": "candidate_selected_initial",
            "criterion": "no_existing_best",
        }

    candidate_snapshot = candidate.metric_snapshot
    incumbent_snapshot = incumbent.metric_snapshot
    comparison_trace: list[dict[str, Any]] = []

    for key, higher_is_better in SELECTION_CRITERIA:
        candidate_value = candidate_snapshot.get(key)
        incumbent_value = incumbent_snapshot.get(key)
        comparison_trace.append(
            {
                "metric": key,
                "candidate": candidate_value,
                "incumbent": incumbent_value,
                "higher_is_better": higher_is_better,
            }
        )
        result = _compare_metric(candidate_value, incumbent_value, higher_is_better)
        if result > 0:
            return candidate, {
                "decision": "candidate_selected",
                "criterion": key,
                "comparison_trace": comparison_trace,
            }
        if result < 0:
            return incumbent, {
                "decision": "incumbent_retained",
                "criterion": key,
                "comparison_trace": comparison_trace,
            }

    return candidate, {
        "decision": "candidate_selected_tie",
        "criterion": "full_tie",
        "comparison_trace": comparison_trace,
    }


def build_registry_payload(
    candidate: PostprocessRecord,
    incumbent: PostprocessRecord | None,
    selected: PostprocessRecord,
    decision: dict[str, Any],
) -> dict[str, Any]:
    compared_at = datetime.now(timezone.utc).isoformat()
    retained_checkpoints = [selected.checkpoint_relative]
    if candidate.checkpoint_relative != selected.checkpoint_relative:
        retained_checkpoints.append(candidate.checkpoint_relative)
    return {
        "registry_version": 1,
        "selection_policy": [
            {
                "metric": metric,
                "higher_is_better": higher_is_better,
            }
            for metric, higher_is_better in SELECTION_CRITERIA
        ],
        "compared_at": compared_at,
        "current_candidate_checkpoint": candidate.checkpoint_relative,
        "current_candidate_report": candidate.report_relative,
        "current_candidate_metrics": candidate.metric_snapshot,
        "current_best_checkpoint_before_compare": incumbent.checkpoint_relative if incumbent else None,
        "current_best_report_before_compare": incumbent.report_relative if incumbent else None,
        "current_best_metrics_before_compare": incumbent.metric_snapshot if incumbent else None,
        "selected_best_checkpoint_after_compare": selected.checkpoint_relative,
        "selected_best_report_after_compare": selected.report_relative,
        "selected_best_metrics_after_compare": selected.metric_snapshot,
        "retained_checkpoints": retained_checkpoints,
        "decision": decision,
    }


def update_postprocess_registry(
    current_checkpoint: str | Path | None,
    current_report: str | Path,
    output_registry: str | Path,
    best_checkpoint: str | Path | None = None,
    best_report: str | Path | None = None,
) -> dict[str, Any]:
    candidate = load_report_record(current_checkpoint, current_report)
    incumbent = load_existing_best(output_registry, best_checkpoint=best_checkpoint, best_report=best_report)
    selected, decision = choose_best_record(candidate, incumbent)
    payload = build_registry_payload(candidate, incumbent, selected, decision)
    write_json(output_registry, payload)
    return payload
