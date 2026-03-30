from __future__ import annotations

import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from .trainer import resolve_precision


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def evaluate_pairwise(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    precision: str = "auto",
) -> dict[str, Any]:
    target_device = torch.device(device if torch.cuda.is_available() else "cpu")
    resolved_precision = resolve_precision(precision, "cuda" if target_device.type == "cuda" else "cpu")
    model = model.to(target_device)
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_pairs = 0
    same_dept_correct = 0
    same_dept_total = 0
    latency_total = 0.0

    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)

    with torch.no_grad():
        for img1, img2, labels, is_same_dept in loader:
            img1 = img1.to(target_device, non_blocking=True)
            img2 = img2.to(target_device, non_blocking=True)
            labels = labels.to(target_device, non_blocking=True)
            is_same_dept = is_same_dept.to(target_device, non_blocking=True)

            start = time.perf_counter()
            with _autocast(target_device, resolved_precision):
                score1, score2 = model(img1, img2)
                loss = model.compute_loss(score1, score2, labels)
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            latency_total += time.perf_counter() - start

            predictions = torch.sign(score1 - score2).squeeze(-1)
            correct_mask = predictions == labels.float()

            total_loss += loss.item()
            total_correct += int(correct_mask.sum().item())
            total_pairs += labels.shape[0]

            same_mask = is_same_dept == 1
            if same_mask.any():
                same_dept_correct += int((correct_mask & same_mask).sum().item())
                same_dept_total += int(same_mask.sum().item())

    peak_vram_gb = 0.0
    if target_device.type == "cuda":
        peak_vram_gb = torch.cuda.max_memory_allocated(target_device) / (1024 ** 3)

    return {
        "val_loss": total_loss / max(len(loader), 1),
        "val_accuracy": total_correct / max(total_pairs, 1),
        "same_dept_accuracy": same_dept_correct / max(same_dept_total, 1) if same_dept_total else 0.0,
        "latency_ms_per_pair": (latency_total / max(total_pairs, 1)) * 1000.0,
        "peak_vram_gb": peak_vram_gb,
        "total_pairs": total_pairs,
        "precision": resolved_precision,
    }
