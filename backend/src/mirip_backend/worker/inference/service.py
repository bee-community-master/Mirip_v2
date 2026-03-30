"""Inference worker service."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from mirip_backend.domain.diagnosis.entities import DiagnosisJob


@dataclass(slots=True, frozen=True)
class InferenceOutput:
    tier: str
    scores: dict[str, float]
    probabilities: list[dict[str, object]]
    feedback: dict[str, object] | None
    summary: str | None


class InferenceService(Protocol):
    async def evaluate(self, job: DiagnosisJob) -> InferenceOutput: ...


class GpuInferenceService:
    """Stub-orchestrated inference service for the worker process."""

    def __init__(self, *, mode: str, model_uri: str | None) -> None:
        self._mode = mode
        self._model_uri = model_uri
        self._loaded = False

    async def load(self) -> None:
        if self._loaded:
            return
        # The real GPU path will attach the actual PyTorch / Transformers runtime.
        self._loaded = True

    async def evaluate(self, job: DiagnosisJob) -> InferenceOutput:
        await self.load()
        digest = hashlib.sha256(job.id.encode("utf-8")).hexdigest()
        tier = ["S", "A", "B", "C"][int(digest[0], 16) % 4]
        base = 55.0 + (int(digest[1:3], 16) / 255.0) * 35.0
        scores = {
            "composition": round(base, 1),
            "technique": round(min(100.0, base + 3.1), 1),
            "creativity": round(max(0.0, base - 1.8), 1),
            "completeness": round(min(100.0, base + 1.2), 1),
        }
        universities = [
            ("홍익대학교", "시각디자인과"),
            ("국민대학교", "공업디자인학과"),
            ("서울대학교", "디자인학부"),
        ]
        probabilities = [
            {
                "university": university,
                "department": department,
                "probability": round(max(0.05, min(0.95, 0.4 + index * 0.12)), 3),
            }
            for index, (university, department) in enumerate(universities)
        ]
        feedback: dict[str, object] | None = None
        if job.include_feedback:
            feedback = {
                "strengths": ["구도 안정감이 좋습니다.", "의도 전달력이 분명합니다."],
                "improvements": ["표현 밀도를 더 높여보세요."],
                "overall": f"{job.department} 기준으로 {tier} 티어 예측입니다.",
            }
        summary = (
            f"Compared {len(job.upload_ids)} uploads."
            if job.job_type == "compare"
            else "Single diagnosis completed."
        )
        return InferenceOutput(
            tier=tier,
            scores=scores,
            probabilities=probabilities,
            feedback=feedback,
            summary=summary,
        )
