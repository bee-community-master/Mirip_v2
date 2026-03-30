"""Inference worker service."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from typing import Protocol

from mirip_backend.domain.diagnosis.entities import DiagnosisJob
from mirip_backend.domain.diagnosis.metadata import INPUT_OBJECT_NAMES_METADATA_KEY
from mirip_backend.infrastructure.gcs.service import GCSStorageService
from mirip_backend.worker.inference.diagnosis_runtime import DiagnosisBundleRuntime
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


class NonRetryableInferenceError(RuntimeError):
    """Raised when worker inference cannot succeed on a retry."""


class WorkerInferenceService:
    """Worker inference service for stub and CPU ONNX runtimes."""

    UNIVERSITY_MAPPING: dict[str, list[dict[str, str]]] = {
        "visual_design": [
            {"university": "홍익대학교", "department": "시각디자인과"},
            {"university": "국민대학교", "department": "시각디자인학과"},
            {"university": "건국대학교", "department": "커뮤니케이션디자인학과"},
            {"university": "이화여자대학교", "department": "디자인학부"},
        ],
        "industrial_design": [
            {"university": "홍익대학교", "department": "산업디자인과"},
            {"university": "국민대학교", "department": "공업디자인학과"},
            {"university": "서울대학교", "department": "디자인학부"},
            {"university": "KAIST", "department": "산업디자인학과"},
        ],
        "fine_art": [
            {"university": "서울대학교", "department": "서양화과"},
            {"university": "홍익대학교", "department": "회화과"},
            {"university": "이화여자대학교", "department": "조형예술학부"},
            {"university": "중앙대학교", "department": "서양화과"},
        ],
        "craft": [
            {"university": "홍익대학교", "department": "도예유리과"},
            {"university": "이화여자대학교", "department": "조형예술학부"},
            {"university": "국민대학교", "department": "공예학과"},
            {"university": "서울대학교", "department": "공예과"},
        ],
    }

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
        self._runtime: DiagnosisBundleRuntime | None = None

    async def load(self) -> None:
        if self._loaded:
            return
        if self._mode in {"stub", "gpu_torch"}:
            self._loaded = True
            return
        if self._mode != "cpu_onnx":
            raise NonRetryableInferenceError(f"Unsupported worker mode: {self._mode}")
        if not self._model_uri:
            raise NonRetryableInferenceError(
                "cpu_onnx worker mode requires MIRIP_WORKER__MODEL_URI"
            )
        try:
            self._bundle = await materialize_model_bundle(
                model_uri=self._model_uri,
                storage_service=self._storage_service,
                cache_dir=self._local_model_cache_dir,
                require_diagnosis_extras=True,
            )
            self._session = self._build_onnx_session(self._bundle)
            self._runtime = DiagnosisBundleRuntime.load(self._bundle)
        except NonRetryableInferenceError:
            raise
        except Exception as exc:
            raise NonRetryableInferenceError("CPU ONNX model bundle initialization failed") from exc
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

    @staticmethod
    def _extract_input_object_names(job: DiagnosisJob) -> list[str]:
        object_names = job.metadata.get(INPUT_OBJECT_NAMES_METADATA_KEY)
        if not isinstance(object_names, list) or not object_names:
            raise NonRetryableInferenceError(
                f"Diagnosis job metadata is missing {INPUT_OBJECT_NAMES_METADATA_KEY}"
            )
        normalized = [str(value) for value in object_names if str(value).strip()]
        if not normalized:
            raise NonRetryableInferenceError(
                "Diagnosis job metadata does not contain any usable "
                f"{INPUT_OBJECT_NAMES_METADATA_KEY}"
            )
        return normalized

    def _calculate_probabilities(
        self,
        *,
        tier: str,
        confidence: float,
        department: str,
    ) -> list[dict[str, object]]:
        universities = self.UNIVERSITY_MAPPING.get(
            department,
            self.UNIVERSITY_MAPPING["visual_design"],
        )
        tier_probs = {"S": 0.85, "A": 0.65, "B": 0.45, "C": 0.25}
        base_prob = tier_probs.get(tier, 0.4)
        probabilities: list[dict[str, object]] = []

        for index, university in enumerate(universities):
            difficulty_weight = 1.0 - (index * 0.08)
            adjusted_prob = base_prob * (0.7 + confidence * 0.3)
            final_prob = max(0.05, min(0.95, adjusted_prob * difficulty_weight))
            probabilities.append(
                {
                    "university": university["university"],
                    "department": university["department"],
                    "probability": round(final_prob, 3),
                }
            )

        probabilities.sort(key=lambda item: float(item["probability"]), reverse=True)
        return probabilities

    @staticmethod
    def _build_feedback(
        *,
        job: DiagnosisJob,
        tier: str,
        scores: dict[str, float],
    ) -> dict[str, object] | None:
        if not job.include_feedback:
            return None

        labels = {
            "composition": "구성",
            "technique": "기술",
            "creativity": "창의성",
            "completeness": "완성도",
        }
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        strengths = [f"{labels[key]}이 상대적으로 강합니다." for key, _ in ranked[:2]]
        improvements = [f"{labels[key]} 보완이 필요합니다." for key, _ in ranked[-2:]]
        return {
            "strengths": strengths,
            "improvements": improvements,
            "overall": f"{job.department} 기준으로 {tier} 티어 예측입니다.",
        }

    async def _evaluate_cpu_onnx(self, job: DiagnosisJob) -> InferenceOutput:
        if job.job_type != "evaluate":
            raise NonRetryableInferenceError(
                "cpu_onnx runtime only supports single-image evaluate jobs"
            )

        object_names = self._extract_input_object_names(job)
        if len(object_names) != 1:
            raise NonRetryableInferenceError(
                "cpu_onnx runtime requires exactly one uploaded image for evaluate jobs"
            )
        if self._session is None or self._runtime is None:
            raise NonRetryableInferenceError("CPU ONNX runtime failed to initialize")

        image_bytes = await self._storage_service.download_bytes(object_name=object_names[0])
        runtime_result = await asyncio.to_thread(
            self._runtime.evaluate_image,
            session=self._session,
            image_bytes=image_bytes,
        )
        probabilities = self._calculate_probabilities(
            tier=runtime_result.tier,
            confidence=runtime_result.confidence,
            department=job.department,
        )
        feedback = self._build_feedback(
            job=job,
            tier=runtime_result.tier,
            scores=runtime_result.scores,
        )
        return InferenceOutput(
            tier=runtime_result.tier,
            scores=runtime_result.scores,
            probabilities=probabilities,
            feedback=feedback,
            summary="Single diagnosis completed.",
        )

    async def evaluate(self, job: DiagnosisJob) -> InferenceOutput:
        await self.load()
        if self._mode in {"stub", "gpu_torch"}:
            return self._build_stub_output(job)
        return await self._evaluate_cpu_onnx(job)


GpuInferenceService = WorkerInferenceService
