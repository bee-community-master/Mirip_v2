from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from train.serving.bundle import ServingBundleManifest, load_manifest, write_manifest
from train.serving.pipeline import (
    build_serving_bundle,
    choose_default_encoder,
    resolve_int8_tier_agreement,
)
from train.training.utils import resolve_model_source


class ServingBundleTests(unittest.TestCase):
    def test_manifest_requires_core_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_dir = Path(temp_dir)
            for filename in (
                "encoder_fp32.onnx",
                "preprocessor.json",
                "benchmarks.json",
                "quality_report.json",
                "model_sha256.txt",
            ):
                (bundle_dir / filename).write_text("x", encoding="utf-8")
            manifest = ServingBundleManifest(
                schema_version="1.0",
                model_name="demo-vitl",
                export_source="local",
                image_size=518,
                default_encoder="encoder_fp32.onnx",
                files={
                    "encoder_fp32.onnx": "encoder_fp32.onnx",
                    "preprocessor.json": "preprocessor.json",
                    "benchmarks.json": "benchmarks.json",
                    "quality_report.json": "quality_report.json",
                    "model_sha256.txt": "model_sha256.txt",
                },
            )
            write_manifest(bundle_dir, manifest)
            loaded = load_manifest(bundle_dir)
            loaded.validate(bundle_dir, require_diagnosis_extras=False)

    def test_manifest_rejects_missing_diagnosis_extras_when_required(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_dir = Path(temp_dir)
            for filename in (
                "manifest.json",
                "encoder_fp32.onnx",
                "preprocessor.json",
                "benchmarks.json",
                "quality_report.json",
                "model_sha256.txt",
            ):
                (bundle_dir / filename).write_text("x", encoding="utf-8")
            payload = {
                "schema_version": "1.0",
                "model_name": "demo-vitl",
                "export_source": "local",
                "image_size": 518,
                "default_encoder": "encoder_fp32.onnx",
                "files": {
                    "encoder_fp32.onnx": "encoder_fp32.onnx",
                    "preprocessor.json": "preprocessor.json",
                    "benchmarks.json": "benchmarks.json",
                    "quality_report.json": "quality_report.json",
                    "model_sha256.txt": "model_sha256.txt",
                },
                "extras": {},
            }
            (bundle_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
            manifest = load_manifest(bundle_dir)
            with self.assertRaises(ValueError):
                manifest.validate(bundle_dir, require_diagnosis_extras=True)

    def test_promotion_keeps_fp32_when_int8_gate_is_not_met(self) -> None:
        decision = choose_default_encoder(
            quality_report={"int8_tier_agreement_vs_fp32": 0.985},
            benchmarks={
                "encoder_fp32": {"latency_ms_p50": 100.0},
                "encoder_int8": {"latency_ms_p50": 70.0},
            },
        )
        self.assertEqual(decision.default_encoder, "encoder_fp32.onnx")
        self.assertFalse(decision.promote_int8)

    def test_build_serving_bundle_omits_optional_int8_when_not_materialized(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_dir = Path(temp_dir)
            for filename in ("encoder_fp32.onnx", "preprocessor.json"):
                (bundle_dir / filename).write_text("x", encoding="utf-8")

            manifest, decision = build_serving_bundle(
                bundle_dir=bundle_dir,
                model_name="demo-vitl",
                export_source="local",
                image_size=518,
                quality_report={"int8_tier_agreement_vs_fp32": 0.0},
                benchmarks={"encoder_fp32": {"latency_ms_p50": 100.0, "thread_count": 8}},
            )

            self.assertEqual(decision.default_encoder, "encoder_fp32.onnx")
            self.assertNotIn("encoder_int8.onnx", manifest.files)
            payload = json.loads((bundle_dir / "benchmarks.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["best_intra_op_num_threads"], 8)
            load_manifest(bundle_dir).validate(bundle_dir, require_diagnosis_extras=False)

    def test_resolve_model_source_accepts_local_export_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "student_export"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")

            resolved = resolve_model_source(str(model_dir))
            self.assertEqual(resolved, str(model_dir))

    def test_resolve_model_source_rejects_invalid_local_export_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "broken_export"
            model_dir.mkdir(parents=True, exist_ok=True)

            with self.assertRaises(ValueError):
                resolve_model_source(str(model_dir))

    def test_resolve_int8_tier_agreement_defaults_to_zero_without_validation_input(self) -> None:
        self.assertEqual(resolve_int8_tier_agreement(None), 0.0)

    def test_resolve_int8_tier_agreement_rejects_out_of_range_values(self) -> None:
        with self.assertRaises(ValueError):
            resolve_int8_tier_agreement(1.2)


if __name__ == "__main__":
    unittest.main()
