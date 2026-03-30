from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REQUIRED_BUNDLE_FILES = (
    "manifest.json",
    "encoder_fp32.onnx",
    "preprocessor.json",
    "benchmarks.json",
    "quality_report.json",
    "model_sha256.txt",
)
OPTIONAL_BUNDLE_FILES = ("encoder_int8.onnx",)
REQUIRED_DIAGNOSIS_EXTRA_KEYS = ("diagnosis_head", "anchors")


@dataclass(slots=True)
class ServingBundleManifest:
    schema_version: str
    model_name: str
    export_source: str
    image_size: int
    default_encoder: str
    files: dict[str, str]
    extras: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(
        self,
        bundle_dir: str | Path,
        *,
        require_diagnosis_extras: bool = False,
    ) -> None:
        bundle_path = Path(bundle_dir)
        if self.default_encoder not in {"encoder_fp32.onnx", "encoder_int8.onnx"}:
            raise ValueError("default_encoder must be one of: encoder_fp32.onnx, encoder_int8.onnx")

        for filename in REQUIRED_BUNDLE_FILES:
            if filename == "manifest.json":
                target = bundle_path / filename
            else:
                target = bundle_path / self.files.get(filename, filename)
            if not target.exists():
                raise FileNotFoundError(f"Missing required serving bundle file: {target}")

        for filename in OPTIONAL_BUNDLE_FILES:
            relative = self.files.get(filename)
            if relative is not None and not (bundle_path / relative).exists():
                raise FileNotFoundError(f"Manifest references missing optional file: {bundle_path / relative}")

        if require_diagnosis_extras:
            missing = [key for key in REQUIRED_DIAGNOSIS_EXTRA_KEYS if key not in self.extras]
            if missing:
                raise ValueError(
                    "Serving bundle is missing diagnosis extras: " + ", ".join(sorted(missing))
                )
        for key, relative in self.extras.items():
            if not (bundle_path / relative).exists():
                raise FileNotFoundError(f"Manifest references missing extra '{key}': {bundle_path / relative}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_manifest(bundle_dir: str | Path, manifest: ServingBundleManifest) -> Path:
    bundle_path = Path(bundle_dir)
    bundle_path.mkdir(parents=True, exist_ok=True)
    path = bundle_path / "manifest.json"
    path.write_text(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_manifest(bundle_dir: str | Path) -> ServingBundleManifest:
    path = Path(bundle_dir) / "manifest.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return ServingBundleManifest(
        schema_version=str(raw["schema_version"]),
        model_name=str(raw["model_name"]),
        export_source=str(raw["export_source"]),
        image_size=int(raw["image_size"]),
        default_encoder=str(raw["default_encoder"]),
        files={str(key): str(value) for key, value in dict(raw["files"]).items()},
        extras={str(key): str(value) for key, value in dict(raw.get("extras", {})).items()},
        metadata=dict(raw.get("metadata", {})),
    )


def sha256sum(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
