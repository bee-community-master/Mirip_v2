"""Tests for local serving bundle validation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from mirip_backend.infrastructure.config.settings import GCSSettings
from mirip_backend.infrastructure.gcs.service import GCSStorageService
from mirip_backend.worker.inference.model_bundle import (
    CACHE_READY_SENTINEL,
    ModelBundleManifest,
    _resolve_cached_bundle_dir,
    materialize_model_bundle,
)


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


class StubDownloadStorage:
    async def download_tree(self, *, gcs_uri: str, destination_dir: str | Path) -> list[Path]:
        def _write_partial_bundle() -> Path:
            target_dir = Path(destination_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / "encoder_fp32.onnx"
            target.write_text("x", encoding="utf-8")
            return target

        target = await asyncio.to_thread(_write_partial_bundle)
        return [target]


def test_model_bundle_uses_default_encoder_thread_hint_when_top_level_hint_is_missing(
    tmp_path: Path,
) -> None:
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
    benchmarks = {
        "encoder_fp32": {
            "latency_ms_p50": 42.0,
            "latency_ms_p95": 43.0,
            "thread_count": 8,
        }
    }
    for filename in (
        "encoder_fp32.onnx",
        "preprocessor.json",
        "quality_report.json",
        "model_sha256.txt",
    ):
        (tmp_path / filename).write_text("x", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "benchmarks.json").write_text(json.dumps(benchmarks), encoding="utf-8")

    loaded = ModelBundleManifest.load(tmp_path)

    assert loaded.best_thread_count(tmp_path) == 8


async def test_gcs_download_tree_rejects_bucket_root_uri(tmp_path: Path) -> None:
    storage = GCSStorageService(GCSSettings(), backend="fake")

    with pytest.raises(ValueError):
        await storage.download_tree(
            gcs_uri="gs://mirip-v2-assets",
            destination_dir=tmp_path,
        )


async def test_gcs_download_tree_scopes_to_directory_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeBlob:
        def __init__(self, name: str) -> None:
            self.name = name

        def download_to_filename(self, filename: str) -> None:
            Path(filename).write_text(self.name, encoding="utf-8")

    class FakeClient:
        def bucket(self, bucket_name: str) -> str:
            return bucket_name

        def list_blobs(self, bucket: str, prefix: str) -> list[FakeBlob]:
            assert bucket == "mirip-v2-assets"
            assert prefix == "models/vitl/"
            return [
                FakeBlob("models/vitl/manifest.json"),
                FakeBlob("models/vitl-extra/manifest.json"),
            ]

    storage = GCSStorageService(GCSSettings(project_id="mirip-v2"), backend="gcs")
    monkeypatch.setattr(GCSStorageService, "_client", lambda self: FakeClient())

    downloaded = await storage.download_tree(
        gcs_uri="gs://mirip-v2-assets/models/vitl",
        destination_dir=tmp_path,
    )

    assert downloaded == [tmp_path / "manifest.json"]
    assert not (tmp_path / "extra" / "manifest.json").exists()


async def test_failed_remote_bundle_materialization_does_not_mark_cache_ready(
    tmp_path: Path,
) -> None:
    model_uri = "gs://mirip-v2-assets/models/vitl"

    with pytest.raises(FileNotFoundError):
        await materialize_model_bundle(
            model_uri=model_uri,
            storage_service=StubDownloadStorage(),  # type: ignore[arg-type]
            cache_dir=tmp_path,
            require_diagnosis_extras=True,
        )

    local_dir, is_cached = _resolve_cached_bundle_dir(model_uri, tmp_path)
    assert is_cached is False
    assert not (local_dir / CACHE_READY_SENTINEL).exists()


def test_bundle_cache_requires_ready_sentinel(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    local_dir, is_cached = _resolve_cached_bundle_dir("gs://mirip-v2-assets/models/vitl", cache_dir)

    assert local_dir.parent == cache_dir
    assert is_cached is False

    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / CACHE_READY_SENTINEL).write_text("ready\n", encoding="utf-8")

    _, is_cached = _resolve_cached_bundle_dir("gs://mirip-v2-assets/models/vitl", cache_dir)

    assert is_cached is True
