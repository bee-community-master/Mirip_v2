"""Tests for CPU ONNX diagnosis runtime execution."""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from mirip_backend.domain.diagnosis.entities import DiagnosisJob
from mirip_backend.infrastructure.config.settings import WorkerSettings
from mirip_backend.shared.enums import JobStatus
from mirip_backend.worker.inference.diagnosis_runtime import ImagePreprocessor
from mirip_backend.worker.inference.service import (
    NonRetryableInferenceError,
    WorkerInferenceService,
)


class FakeOnnxSession:
    def run(self, _output_names, inputs):  # type: ignore[no-untyped-def]
        assert "pixel_values" in inputs
        return [np.zeros((1, 4), dtype=np.float32)]


class StubStorageService:
    def __init__(self, image_bytes: bytes) -> None:
        self._image_bytes = image_bytes
        self.downloaded: list[str] = []

    async def download_bytes(self, *, object_name: str) -> bytes:
        self.downloaded.append(object_name)
        return self._image_bytes


def _make_image_bytes() -> bytes:
    image = Image.new("RGB", (16, 16), color=(128, 96, 64))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _make_gradient_image_bytes(*, width: int, height: int) -> bytes:
    values = np.tile(np.arange(width, dtype=np.uint8), (height, 1))
    image = Image.fromarray(values, mode="L").convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _write_bundle_fixture(bundle_dir: Path) -> Path:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("encoder_fp32.onnx", "model_sha256.txt"):
        (bundle_dir / filename).write_text("x", encoding="utf-8")
    (bundle_dir / "preprocessor.json").write_text(
        json.dumps(
            {
                "image_mean": [0.485, 0.456, 0.406],
                "image_std": [0.229, 0.224, 0.225],
                "size": {"height": 8, "width": 8},
            }
        ),
        encoding="utf-8",
    )
    (bundle_dir / "benchmarks.json").write_text(
        json.dumps({"best_intra_op_num_threads": 8}),
        encoding="utf-8",
    )
    (bundle_dir / "quality_report.json").write_text(
        json.dumps({"int8_tier_agreement_vs_fp32": 0.0}),
        encoding="utf-8",
    )

    dropout = 0.0
    projector = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.LayerNorm(4),
        torch.nn.GELU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(4, 4),
        torch.nn.LayerNorm(4),
        torch.nn.GELU(),
        torch.nn.Dropout(dropout),
    )
    score_head = torch.nn.Sequential(
        torch.nn.Linear(4, 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(64, 1),
    )
    for module in list(projector.modules()) + list(score_head.modules()):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.zeros_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    torch.save(
        {
            "schema_version": "1.0",
            "model_name": "student-export",
            "feature_dim": 4,
            "projector_hidden_dim": 4,
            "projector_output_dim": 4,
            "dropout": 0.0,
            "projector_state_dict": projector.state_dict(),
            "score_head_state_dict": score_head.state_dict(),
        },
        bundle_dir / "diagnosis_head.pt",
    )
    torch.save(
        {
            "features": {
                tier: torch.zeros((2, 4), dtype=torch.float32)
                for tier in ("S", "A", "B", "C")
            },
            "image_paths": {
                tier: [f"{tier.lower()}-1.png", f"{tier.lower()}-2.png"]
                for tier in ("S", "A", "B", "C")
            },
            "metadata": {
                "model_source": "student-export",
                "checkpoint_relative": "checkpoints/demo.pt",
                "feature_dim": 4,
                "projector_output_dim": 4,
            },
        },
        bundle_dir / "anchors.pt",
    )
    (bundle_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "model_name": "student-export",
                "export_source": "student-export",
                "image_size": 8,
                "default_encoder": "encoder_fp32.onnx",
                "files": {
                    "encoder_fp32.onnx": "encoder_fp32.onnx",
                    "preprocessor.json": "preprocessor.json",
                    "benchmarks.json": "benchmarks.json",
                    "quality_report.json": "quality_report.json",
                    "model_sha256.txt": "model_sha256.txt",
                },
                "extras": {
                    "diagnosis_head": "diagnosis_head.pt",
                    "anchors": "anchors.pt",
                },
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )
    return bundle_dir


def _make_job(
    *,
    job_type: str = "evaluate",
    input_object_names: list[str] | None = None,
) -> DiagnosisJob:
    return DiagnosisJob(
        id="job-cpu-onnx",
        user_id="user-1",
        upload_ids=["upl-1"] if job_type == "evaluate" else ["upl-1", "upl-2"],
        job_type=job_type,
        department="visual_design",
        include_feedback=True,
        theme=None,
        language="ko",
        status=JobStatus.QUEUED,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        metadata={
            "input_object_names": (
                input_object_names or ["users/user-1/diagnosis/upl-1/piece.png"]
            )
        },
    )


async def test_cpu_onnx_worker_evaluates_local_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle_dir = _write_bundle_fixture(tmp_path / "bundle")
    storage = StubStorageService(_make_image_bytes())
    monkeypatch.setattr(
        WorkerInferenceService,
        "_build_onnx_session",
        staticmethod(lambda _bundle: FakeOnnxSession()),
    )
    service = WorkerInferenceService(
        mode="cpu_onnx",
        model_uri=str(bundle_dir),
        storage_service=storage,  # type: ignore[arg-type]
        local_model_cache_dir=str(tmp_path / "cache"),
    )

    result = await service.evaluate(_make_job())

    assert result.tier == "C"
    assert set(result.scores) == {"composition", "technique", "creativity", "completeness"}
    assert result.summary == "Single diagnosis completed."
    assert storage.downloaded == ["users/user-1/diagnosis/upl-1/piece.png"]
    assert result.probabilities[0]["probability"] >= result.probabilities[-1]["probability"]


async def test_cpu_onnx_worker_rejects_compare_jobs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle_dir = _write_bundle_fixture(tmp_path / "bundle")
    storage = StubStorageService(_make_image_bytes())
    monkeypatch.setattr(
        WorkerInferenceService,
        "_build_onnx_session",
        staticmethod(lambda _bundle: FakeOnnxSession()),
    )
    service = WorkerInferenceService(
        mode="cpu_onnx",
        model_uri=str(bundle_dir),
        storage_service=storage,  # type: ignore[arg-type]
        local_model_cache_dir=str(tmp_path / "cache"),
    )

    with pytest.raises(NonRetryableInferenceError):
        await service.evaluate(
            _make_job(
                job_type="compare",
                input_object_names=[
                    "users/user-1/diagnosis/upl-1/piece-a.png",
                    "users/user-1/diagnosis/upl-2/piece-b.png",
                ],
            )
        )


def test_image_preprocessor_respects_shortest_edge_resize_and_center_crop(tmp_path: Path) -> None:
    config_path = tmp_path / "preprocessor.json"
    config_path.write_text(
        json.dumps(
            {
                "do_resize": True,
                "size": {"shortest_edge": 8},
                "do_center_crop": True,
                "crop_size": {"height": 8, "width": 8},
                "do_rescale": False,
                "do_normalize": False,
                "resample": Image.Resampling.BICUBIC.value,
            }
        ),
        encoding="utf-8",
    )
    preprocessor = ImagePreprocessor.load(config_path)
    image_bytes = _make_gradient_image_bytes(width=12, height=6)

    pixel_values = preprocessor.preprocess_bytes(image_bytes)

    source_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    expected_image = source_image.resize((16, 8), resample=Image.Resampling.BICUBIC).crop(
        (4, 0, 12, 8)
    )
    expected = np.transpose(np.asarray(expected_image, dtype=np.float32), (2, 0, 1))[None, ...]
    assert np.allclose(pixel_values, expected)


async def test_gpu_torch_mode_keeps_stub_placeholder_behavior(tmp_path: Path) -> None:
    service = WorkerInferenceService(
        mode="gpu_torch",
        model_uri=None,
        storage_service=StubStorageService(_make_image_bytes()),  # type: ignore[arg-type]
        local_model_cache_dir=str(tmp_path / "cache"),
    )

    result = await service.evaluate(_make_job())

    assert result.summary == "Single diagnosis completed."
    assert result.tier in {"S", "A", "B", "C"}


def test_worker_settings_accept_legacy_gpu_alias() -> None:
    assert WorkerSettings(mode="gpu").mode == "gpu_torch"
