from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoImageProcessor

from .datasets import load_metadata_rows, resolve_image_path
from .trainer import resolve_precision
from .utils import resolve_project_path


class AnchorStore:
    def __init__(self, features: dict[str, torch.Tensor], image_paths: dict[str, list[str]], metadata: dict[str, Any]) -> None:
        self.features = features
        self.image_paths = image_paths
        self.metadata = metadata

    def save(self, path: str | Path) -> Path:
        target = resolve_project_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "features": self.features,
                "image_paths": self.image_paths,
                "metadata": self.metadata,
            },
            target,
        )
        return target

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "AnchorStore":
        payload = torch.load(resolve_project_path(path), map_location=map_location)
        return cls(
            features=payload["features"],
            image_paths=payload["image_paths"],
            metadata=payload.get("metadata", {}),
        )


def _load_processor(model_name: str):
    return AutoImageProcessor.from_pretrained(model_name)


def build_anchor_store(
    model: torch.nn.Module,
    metadata_csv: str | Path,
    image_root: str | Path,
    model_name: str,
    n_per_tier: int = 10,
    seed: int = 42,
) -> AnchorStore:
    rows = load_metadata_rows(metadata_csv)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["tier"]].append(row)

    processor = _load_processor(model_name)
    rng = random.Random(seed)
    device = next(model.parameters()).device
    features: dict[str, torch.Tensor] = {}
    image_paths: dict[str, list[str]] = {}

    model.eval()
    with torch.no_grad():
        for tier in ("S", "A", "B", "C"):
            tier_rows = grouped.get(tier, [])
            if not tier_rows:
                continue
            selected = tier_rows[:] if len(tier_rows) <= n_per_tier else rng.sample(tier_rows, n_per_tier)
            tier_features = []
            tier_paths = []
            for row in selected:
                path = resolve_image_path(image_root, row["image_path"])
                image = Image.open(path).convert("RGB")
                pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
                projected = model.project_features(model.extract_features(pixel_values))
                tier_features.append(projected.squeeze(0).cpu())
                tier_paths.append(row["image_path"])
            if tier_features:
                features[tier] = torch.stack(tier_features)
                image_paths[tier] = tier_paths

    return AnchorStore(
        features=features,
        image_paths=image_paths,
        metadata={
            "n_per_tier": n_per_tier,
            "model_name": model_name,
            "seed": seed,
            "metadata_csv": str(metadata_csv),
        },
    )


class TierRanker:
    def __init__(self, model: torch.nn.Module, anchors: AnchorStore) -> None:
        self.model = model
        self.anchors = anchors
        self.device = next(model.parameters()).device

    def rank_projected_feature(self, projected_feature: torch.Tensor) -> dict[str, Any]:
        projected_feature = projected_feature.to(self.device)
        input_score = self.model.score_features(projected_feature).squeeze().item()

        win_rates: dict[str, float] = {}
        for tier, anchor_features in self.anchors.features.items():
            anchor_features = anchor_features.to(self.device)
            anchor_scores = self.model.score_features(anchor_features).squeeze(-1)
            wins = (input_score > anchor_scores).sum().item()
            win_rates[tier] = wins / max(anchor_features.shape[0], 1)

        if win_rates.get("S", 0.0) >= 0.5:
            tier = "S"
            confidence = win_rates["S"]
        elif win_rates.get("A", 0.0) >= 0.5:
            tier = "A"
            confidence = win_rates["A"]
        elif win_rates.get("B", 0.0) >= 0.5:
            tier = "B"
            confidence = win_rates["B"]
        else:
            tier = "C"
            confidence = 1.0 - win_rates.get("C", 0.0)

        return {
            "tier": tier,
            "confidence": round(confidence, 4),
            "win_rates": win_rates,
            "score": input_score,
        }


def _evaluation_autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16 if precision == "bf16" else torch.float16,
    )


def evaluate_anchor_tier_accuracy(
    model: torch.nn.Module,
    anchors: AnchorStore,
    metadata_csv: str | Path,
    image_root: str | Path,
    model_name: str,
    precision: str = "auto",
) -> dict[str, Any]:
    rows = load_metadata_rows(metadata_csv)
    processor = _load_processor(model_name)
    device = next(model.parameters()).device
    resolved_precision = resolve_precision(precision, "cuda" if device.type == "cuda" else "cpu")
    ranker = TierRanker(model, anchors)

    total = 0
    correct = 0
    per_tier = defaultdict(lambda: {"total": 0, "correct": 0})

    model.eval()
    with torch.no_grad():
        for row in rows:
            true_tier = row["tier"]
            path = resolve_image_path(image_root, row["image_path"])
            image = Image.open(path).convert("RGB")
            pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
            with _evaluation_autocast(device, resolved_precision):
                projected = model.project_features(model.extract_features(pixel_values))
            result = ranker.rank_projected_feature(projected)
            predicted_tier = result["tier"]
            total += 1
            per_tier[true_tier]["total"] += 1
            if predicted_tier == true_tier:
                correct += 1
                per_tier[true_tier]["correct"] += 1

    return {
        "anchor_tier_accuracy": correct / max(total, 1),
        "anchor_tier_total": total,
        "anchor_tier_correct": correct,
        "anchor_tier_per_tier": {
            tier: {
                "accuracy": values["correct"] / max(values["total"], 1),
                "total": values["total"],
                "correct": values["correct"],
            }
            for tier, values in sorted(per_tier.items())
        },
    }
