"""Serving bundle materialization and validation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mirip_backend.infrastructure.gcs.service import GCSStorageService

REQUIRED_FILES = (
    "manifest.json",
    "encoder_fp32.onnx",
    "preprocessor.json",
    "benchmarks.json",
    "quality_report.json",
    "model_sha256.txt",
)
REQUIRED_DIAGNOSIS_EXTRAS = ("diagnosis_head", "anchors")
THREAD_SWEEP = (8, 12, 16)
DEFAULT_INTRA_OP_THREADS = 16
CACHE_READY_SENTINEL = ".bundle_ready"


@dataclass(slots=True, frozen=True)
class ModelBundleManifest:
    schema_version: str
    model_name: str | None
    export_source: str | None
    image_size: int | None
    default_encoder: str
    files: dict[str, str]
    extras: dict[str, str]
    metadata: dict[str, Any]

    @classmethod
    def load(cls, bundle_dir: str | Path) -> ModelBundleManifest:
        raw = json.loads((Path(bundle_dir) / "manifest.json").read_text(encoding="utf-8"))
        return cls(
            schema_version=str(raw["schema_version"]),
            model_name=str(raw["model_name"]) if raw.get("model_name") is not None else None,
            export_source=(
                str(raw["export_source"]) if raw.get("export_source") is not None else None
            ),
            image_size=int(raw["image_size"]) if raw.get("image_size") is not None else None,
            default_encoder=str(raw["default_encoder"]),
            files={str(key): str(value) for key, value in dict(raw["files"]).items()},
            extras={str(key): str(value) for key, value in dict(raw.get("extras", {})).items()},
            metadata=dict(raw.get("metadata", {})),
        )

    def validate(self, bundle_dir: str | Path, *, require_diagnosis_extras: bool) -> None:
        bundle_path = Path(bundle_dir)
        for filename in REQUIRED_FILES:
            target = (
                bundle_path / filename
                if filename == "manifest.json"
                else bundle_path / self.files.get(filename, filename)
            )
            if not target.exists():
                raise RuntimeError(f"Model bundle is missing required file: {target}")
        if require_diagnosis_extras:
            missing = [key for key in REQUIRED_DIAGNOSIS_EXTRAS if key not in self.extras]
            if missing:
                raise RuntimeError(
                    "Model bundle is missing diagnosis extras: " + ", ".join(sorted(missing))
                )
        for key, relative in self.extras.items():
            if not (bundle_path / relative).exists():
                raise RuntimeError(
                    f"Model bundle extra '{key}' is missing: {bundle_path / relative}"
                )

    def encoder_path(self, bundle_dir: str | Path) -> Path:
        bundle_path = Path(bundle_dir)
        return bundle_path / self.files.get(self.default_encoder, self.default_encoder)

    def preprocessor_path(self, bundle_dir: str | Path) -> Path:
        bundle_path = Path(bundle_dir)
        return bundle_path / self.files.get("preprocessor.json", "preprocessor.json")

    def extra_path(self, bundle_dir: str | Path, key: str) -> Path:
        bundle_path = Path(bundle_dir)
        relative = self.extras.get(key)
        if relative is None:
            raise RuntimeError(f"Model bundle extra '{key}' is not declared")
        return bundle_path / relative

    def best_thread_count(self, bundle_dir: str | Path) -> int:
        benchmarks = json.loads(
            (Path(bundle_dir) / self.files["benchmarks.json"]).read_text(encoding="utf-8")
        )
        candidate = benchmarks.get("best_intra_op_num_threads")
        if isinstance(candidate, int) and candidate in THREAD_SWEEP:
            return candidate
        candidate = benchmarks.get(
            self.default_encoder.removesuffix(".onnx"),
            {},
        ).get("thread_count")
        if isinstance(candidate, (int, float)) and int(candidate) in THREAD_SWEEP:
            return int(candidate)
        return DEFAULT_INTRA_OP_THREADS


@dataclass(slots=True, frozen=True)
class MaterializedModelBundle:
    model_uri: str
    local_dir: Path
    manifest: ModelBundleManifest


async def materialize_model_bundle(
    *,
    model_uri: str,
    storage_service: GCSStorageService,
    cache_dir: str | Path,
    require_diagnosis_extras: bool,
) -> MaterializedModelBundle:
    local_dir, downloaded_remote = await _resolve_local_bundle_dir(
        model_uri=model_uri,
        storage_service=storage_service,
        cache_dir=cache_dir,
    )
    try:
        manifest = ModelBundleManifest.load(local_dir)
        manifest.validate(local_dir, require_diagnosis_extras=require_diagnosis_extras)
    except Exception:
        if downloaded_remote:
            await asyncio.to_thread(_reset_bundle_cache_dir, local_dir)
        raise
    if downloaded_remote:
        await asyncio.to_thread(_mark_bundle_cache_ready, local_dir)
    return MaterializedModelBundle(model_uri=model_uri, local_dir=local_dir, manifest=manifest)


async def _resolve_local_bundle_dir(
    *,
    model_uri: str,
    storage_service: GCSStorageService,
    cache_dir: str | Path,
) -> tuple[Path, bool]:
    candidate = await asyncio.to_thread(_resolve_existing_local_bundle_dir, model_uri)
    if candidate is not None:
        return candidate, False

    local_dir, is_cached = await asyncio.to_thread(_resolve_cached_bundle_dir, model_uri, cache_dir)
    if is_cached:
        return local_dir, False
    await asyncio.to_thread(_reset_bundle_cache_dir, local_dir)
    await storage_service.download_tree(gcs_uri=model_uri, destination_dir=local_dir)
    return local_dir, True


def _resolve_existing_local_bundle_dir(model_uri: str) -> Path | None:
    candidate = Path(model_uri).expanduser()
    if candidate.exists():
        return candidate.resolve()
    return None


def _resolve_cached_bundle_dir(model_uri: str, cache_dir: str | Path) -> tuple[Path, bool]:
    cache_root = Path(cache_dir)
    digest = hashlib.sha256(model_uri.encode("utf-8")).hexdigest()[:16]
    local_dir = cache_root / digest
    return local_dir, (local_dir / CACHE_READY_SENTINEL).exists()


def _reset_bundle_cache_dir(local_dir: Path) -> None:
    if local_dir.exists():
        shutil.rmtree(local_dir)


def _mark_bundle_cache_ready(local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / CACHE_READY_SENTINEL).write_text("ready\n", encoding="utf-8")
