#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.config import DEFAULT_DINOV3_MODEL_NAME
from training.datasets import DinoPairBatchCollator, DinoPairDataset
from training.models import DinoV3PairwiseModel
from training.trainer import resolve_precision
from training.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe the largest stable DINOv3 micro-batch size.")
    parser.add_argument("--pairs-train", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--model-name", default=DEFAULT_DINOV3_MODEL_NAME)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--feature-pool", default="cls_mean_patch_concat", choices=["cls", "cls_mean_patch_concat"])
    parser.add_argument("--head-type", default="mlp_small", choices=["linear", "mlp_small"])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--backbone-dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--batch-size-candidates", default="8,6,4,2")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _is_oom_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def _first_batch(dataset: DinoPairDataset, collator: DinoPairBatchCollator, batch_size: int):
    batch = [dataset[index] for index in range(min(batch_size, len(dataset)))]
    if not batch:
        raise SystemExit("pairs-train dataset is empty; cannot probe batch size.")
    return collator(batch)


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    precision = resolve_precision(args.precision, "cuda" if device.type == "cuda" else "cpu")
    candidates = [int(value.strip()) for value in args.batch_size_candidates.split(",") if value.strip()]
    dataset = DinoPairDataset(args.pairs_train)
    collator = DinoPairBatchCollator(
        image_root=args.image_root,
        model_name=args.model_name,
        input_size=args.input_size,
        is_train=True,
    )

    failures: list[dict[str, object]] = []
    for batch_size in candidates:
        model = DinoV3PairwiseModel(
            model_name=args.model_name,
            dropout=args.dropout,
            margin=args.margin,
            freeze_backbone=True,
            backbone_dtype=args.backbone_dtype,
            feature_pool=args.feature_pool,
            head_type=args.head_type,
        ).to(device)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1e-4,
        )
        try:
            img1, img2, labels, _ = _first_batch(dataset, collator, batch_size)
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, precision):
                score1, score2 = model(img1, img2)
                loss = model.compute_loss(score1, score2, labels)
            loss.backward()
            optimizer.step()
            payload = {
                "selected_batch_size": batch_size,
                "candidates": candidates,
                "device": str(device),
                "precision": precision,
            }
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0
        except RuntimeError as exc:
            if not _is_oom_error(exc):
                raise
            failures.append({"batch_size": batch_size, "error": str(exc)})
            if device.type == "cuda":
                torch.cuda.empty_cache()
        finally:
            del model
            del optimizer

    print(json.dumps({"selected_batch_size": None, "failures": failures}, indent=2, ensure_ascii=False))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
