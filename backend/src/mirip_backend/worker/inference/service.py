"""Inference worker service."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from mirip_backend.domain.diagnosis.entities import DiagnosisJob
from mirip_backend.infrastructure.gcs.service import GCSStorageService
from mirip_backend.worker.inference.model_bundle import (
    MaterializedModelBundle,
    materialize_model_bundle,
)


@dataclass(slots=True, frozen=True)
class InferenceOutput:
    tier: str
    scores: dict[str, float]
    probabilities: list[dict[str, object]]
    feedback: dict[str, object] | None
    summary: str | None


class InferenceService(Protocol):
    async def evaluate(self, job: DiagnosisJob) -> InferenceOutput: ...


class WorkerInferenceService:
    """Worker inference service for stub and CPU ONNX runtimes."""

    def __init__(
        self,
        *,
        mode: str,
        model_uri: str | None,
        storage_service: GCSStorageService,
        local_model_cache_dir: str,
    ) -> None:
        self._mode = mode
        self._model_uri = model_uri
        self._storage_service = storage_service
        self._local_model_cache_dir = local_model_cache_dir
        self._loaded = False
        self._bundle: MaterializedModelBundle | None = None
        self._session = None

    async def load(self) -> None:
        if self._loaded:
            return
        if self._mode == "stub":
            self._loaded = True
            return
        if self._mode != "cpu_onnx":
            raise RuntimeError(f"Unsupported worker mode: {self._mode}")
        if not self._model_uri:
            raise RuntimeError("cpu_onnx worker mode requires MIRIP_WORKER__MODEL_URI")
        self._bundle = await materialize_model_bundle(
            model_uri=self._model_uri,
            storage_service=self._storage_service,
            cache_dir=self._local_model_cache_dir,
            require_diagnosis_extras=True,
        )
        self._session = self._build_onnx_session(self._bundle)
        self._loaded = True

    @staticmethod
    def _build_onnx_session(bundle: MaterializedModelBundle):  # type: ignore[no-untyped-def]
        import onnxruntime as ort  # type: ignore[import-untyped]

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = bundle.manifest.best_thread_count(bundle.local_dir)
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.add_session_config_entry("session.intra_op.allow_spinning", "1")
        return ort.InferenceSession(
            str(bundle.manifest.encoder_path(bundle.local_dir)),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

    @staticmethod
    def _build_stub_output(job: DiagnosisJob) -> InferenceOutput:
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

    async def evaluate(self, job: DiagnosisJob) -> InferenceOutput:
        await self.load()
        if self._mode == "stub":
            return self._build_stub_output(job)
        if self._session is None or self._bundle is None:
            raise RuntimeError("CPU ONNX runtime failed to initialize")
        raise RuntimeError(
            "CPU ONNX runtime loaded the serving bundle and initialized the encoder session, "
            "but diagnosis-head execution is not implemented yet. Production cutover should "
            "remain blocked until the diagnosis artifact contract is finalized."
        )


GpuInferenceService = WorkerInferenceService
