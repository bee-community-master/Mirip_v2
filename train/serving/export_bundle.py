#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import resource
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serving.bundle import load_manifest
from serving.pipeline import build_serving_bundle, resolve_int8_tier_agreement, write_json
from training.utils import resolve_project_path

THREAD_SWEEP = (8, 12, 16)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a Mirip ViT-L CPU serving bundle.")
    parser.add_argument("--backbone-dir", required=True, help="Local Hugging Face backbone export directory")
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--skip-int8", action="store_true")
    parser.add_argument(
        "--int8-tier-agreement",
        type=float,
        default=None,
        help="Measured INT8 tier agreement versus FP32 in the range [0.0, 1.0]",
    )
    return parser.parse_args()


def _export_fp32_encoder(backbone_dir: Path, output_path: Path, image_size: int) -> None:
    from transformers import AutoModel

    class EncoderWrapper(torch.nn.Module):
        def __init__(self, model: torch.nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            outputs = self.model(pixel_values=pixel_values)
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                raise RuntimeError("AutoModel output missing last_hidden_state")
            return hidden[:, 0, :]

    model = AutoModel.from_pretrained(backbone_dir)
    model.eval()
    wrapper = EncoderWrapper(model)
    dummy = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(output_path),
        input_names=["pixel_values"],
        output_names=["embeddings"],
        dynamic_axes={"pixel_values": {0: "batch"}, "embeddings": {0: "batch"}},
        opset_version=17,
    )


def _quantize_int8(fp32_path: Path, int8_path: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
        per_channel=False,
    )


def _benchmark_encoder(onnx_path: Path, image_size: int, threads: int) -> dict[str, float]:
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = threads
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    startup_begin = time.perf_counter()
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    startup_time_ms = (time.perf_counter() - startup_begin) * 1000.0
    inputs = np.random.rand(1, 3, image_size, image_size).astype("float32")
    session.run(None, {"pixel_values": inputs})
    latencies_ms: list[float] = []
    for _ in range(3):
        start = time.perf_counter()
        session.run(None, {"pixel_values": inputs})
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    return {
        "latency_ms_p50": float(np.percentile(latencies_ms, 50)),
        "latency_ms_p95": float(np.percentile(latencies_ms, 95)),
        "startup_time_ms": float(startup_time_ms),
        "rss_mib": float(_peak_rss_mib()),
        "thread_count": int(threads),
    }


def _peak_rss_mib() -> float:
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return peak_rss / (1024 * 1024)
    return peak_rss / 1024


def _benchmark_thread_sweep(
    onnx_path: Path,
    image_size: int,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    runs = {
        str(threads): _benchmark_encoder(onnx_path, image_size, threads=threads)
        for threads in THREAD_SWEEP
    }
    best = min(runs.values(), key=lambda item: item["latency_ms_p50"])
    return dict(best), runs


def main() -> int:
    args = _parse_args()
    backbone_dir = resolve_project_path(args.backbone_dir)
    bundle_dir = resolve_project_path(args.bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    if not backbone_dir.exists():
        raise SystemExit(f"Backbone export directory does not exist: {backbone_dir}")
    if not (backbone_dir / "config.json").exists():
        raise SystemExit(f"Backbone export directory is missing config.json: {backbone_dir}")
    if args.skip_int8 and args.int8_tier_agreement is not None:
        raise SystemExit("--int8-tier-agreement cannot be used with --skip-int8")

    shutil.copy2(backbone_dir / "preprocessor_config.json", bundle_dir / "preprocessor.json") if (backbone_dir / "preprocessor_config.json").exists() else write_json(bundle_dir / "preprocessor.json", {"image_size": args.image_size})

    fp32_path = bundle_dir / "encoder_fp32.onnx"
    _export_fp32_encoder(backbone_dir, fp32_path, args.image_size)

    fp32_best, fp32_sweep = _benchmark_thread_sweep(fp32_path, args.image_size)
    benchmarks: dict[str, Any] = {
        "encoder_fp32": fp32_best,
        "encoder_fp32_thread_sweep": fp32_sweep,
    }
    quality_report = {
        "int8_tier_agreement_vs_fp32": resolve_int8_tier_agreement(
            args.int8_tier_agreement
        ),
        "export_source": str(backbone_dir),
    }
    if not args.skip_int8:
        int8_path = bundle_dir / "encoder_int8.onnx"
        _quantize_int8(fp32_path, int8_path)
        int8_best, int8_sweep = _benchmark_thread_sweep(int8_path, args.image_size)
        benchmarks["encoder_int8"] = int8_best
        benchmarks["encoder_int8_thread_sweep"] = int8_sweep

    manifest, decision = build_serving_bundle(
        bundle_dir=bundle_dir,
        model_name=args.model_name,
        export_source=str(backbone_dir),
        image_size=args.image_size,
        quality_report=quality_report,
        benchmarks=benchmarks,
    )
    manifest.validate(bundle_dir, require_diagnosis_extras=False)
    print(json.dumps({"manifest": manifest.to_dict(), "promotion": decision.__dict__}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
