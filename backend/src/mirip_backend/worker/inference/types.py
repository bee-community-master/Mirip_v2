"""Shared inference output types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class InferenceOutput:
    tier: str
    scores: dict[str, float]
    probabilities: list[dict[str, object]]
    feedback: dict[str, object] | None
    summary: str | None
