"""Tests for local serving bundle validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mirip_backend.infrastructure.config.settings import GCSSettings
from mirip_backend.infrastructure.gcs.service import GCSStorageService
from mirip_backend.worker.inference.model_bundle import materialize_model_bundle


async def test_model_bundle_requires_diagnosis_extras(tmp_path: Path) -> None:
    for filename in (
        "encoder_fp32.onnx",
        "preprocessor.json",
        "benchmarks.json",
        "quality_report.json",
        "model_sha256.txt",
    ):
        (tmp_path / filename).write_text("x", encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "default_encoder": "encoder_fp32.onnx",
        "files": {
            "encoder_fp32.onnx": "encoder_fp32.onnx",
            "preprocessor.json": "preprocessor.json",
            "benchmarks.json": "benchmarks.json",
            "quality_report.json": "quality_report.json",
            "model_sha256.txt": "model_sha256.txt",
        },
        "extras": {},
        "metadata": {},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    storage = GCSStorageService(GCSSettings(), backend="fake")

    with pytest.raises(RuntimeError):
        await materialize_model_bundle(
            model_uri=str(tmp_path),
            storage_service=storage,
            cache_dir="/tmp/mirip-model-cache-test",
            require_diagnosis_extras=True,
        )
