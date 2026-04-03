from __future__ import annotations

import random
from collections import defaultdict
from math import sqrt
from pathlib import Path
from typing import Any

import torch

from .datasets import load_metadata_rows, preprocess_rgb_image, resolve_image_path, load_rgb_image
from .trainer import resolve_precision
from .utils import project_relative_path, resolve_model_source, resolve_project_path


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

def build_anchor_store(
    model: torch.nn.Module,
    metadata_csv: str | Path,
    image_root: str | Path,
    model_name: str,
    input_size: int,
    n_per_tier: int = 10,
    seed: int = 42,
    group_balanced: bool = False,
    source_checkpoint: str | Path | None = None,
) -> AnchorStore:
    rows = load_metadata_rows(metadata_csv)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["tier"]].append(row)

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
            selected = _select_anchor_rows(
                tier_rows,
                rng=rng,
                n_per_tier=n_per_tier,
                group_balanced=group_balanced,
            )
            tier_features = []
            tier_paths = []
            for row in selected:
                path = resolve_image_path(image_root, row["image_path"])
                pixel_values = preprocess_rgb_image(
                    load_rgb_image(path),
                    model_name=model_name,
                    input_size=input_size,
                    is_train=False,
                ).unsqueeze(0).to(device)
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
            "group_balanced": group_balanced,
            "model_name": model_name,
            "model_source": resolve_model_source(model_name),
            "seed": seed,
            "metadata_csv": str(metadata_csv),
            "checkpoint": str(resolve_project_path(source_checkpoint)) if source_checkpoint else None,
            "checkpoint_relative": project_relative_path(source_checkpoint) if source_checkpoint else None,
            "feature_dim": getattr(getattr(model, "feature_extractor", None), "output_dim", None),
            "projector_output_dim": getattr(getattr(model, "score_head", [None])[0], "in_features", None),
        },
    )


def _select_anchor_rows(
    tier_rows: list[dict[str, str]],
    *,
    rng: random.Random,
    n_per_tier: int,
    group_balanced: bool,
) -> list[dict[str, str]]:
    if len(tier_rows) <= n_per_tier:
        return list(tier_rows)
    if not group_balanced:
        return rng.sample(tier_rows, n_per_tier)

    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in tier_rows:
        grouped_rows[row.get("anchor_group") or row["image_path"]].append(row)

    group_keys = list(grouped_rows)
    rng.shuffle(group_keys)
    selected: list[dict[str, str]] = []
    seen_paths: set[str] = set()

    for group_key in group_keys:
        row = rng.choice(grouped_rows[group_key])
        selected.append(row)
        seen_paths.add(row["image_path"])
        if len(selected) >= n_per_tier:
            return selected

    remaining_rows = [row for row in tier_rows if row["image_path"] not in seen_paths]
    rng.shuffle(remaining_rows)
    selected.extend(remaining_rows[: max(n_per_tier - len(selected), 0)])
    return selected[:n_per_tier]


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
    input_size: int,
    precision: str = "auto",
) -> dict[str, Any]:
    rows = load_metadata_rows(metadata_csv)
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
            pixel_values = preprocess_rgb_image(
                load_rgb_image(path),
                model_name=model_name,
                input_size=input_size,
                is_train=False,
            ).unsqueeze(0).to(device)
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


def _evaluate_anchor_tier_accuracy_once(
    *,
    model: torch.nn.Module,
    metadata_train_csv: str | Path,
    metadata_eval_csv: str | Path,
    image_root: str | Path,
    model_name: str,
    input_size: int,
    n_per_tier: int,
    seed: int,
    precision: str,
    group_balanced: bool,
    source_checkpoint: str | Path | None = None,
) -> tuple[AnchorStore, dict[str, Any]]:
    anchors = build_anchor_store(
        model=model,
        metadata_csv=metadata_train_csv,
        image_root=image_root,
        model_name=model_name,
        input_size=input_size,
        n_per_tier=n_per_tier,
        seed=seed,
        group_balanced=group_balanced,
        source_checkpoint=source_checkpoint,
    )
    metrics = evaluate_anchor_tier_accuracy(
        model=model,
        anchors=anchors,
        metadata_csv=metadata_eval_csv,
        image_root=image_root,
        model_name=model_name,
        input_size=input_size,
        precision=precision,
    )
    return anchors, metrics


def _aggregate_anchor_bootstrap_results(
    per_seed_results: list[dict[str, Any]],
) -> dict[str, Any]:
    if not per_seed_results:
        raise ValueError("per_seed_results must not be empty")

    accuracies = [float(result["anchor_tier_accuracy"]) for result in per_seed_results]
    mean_accuracy = sum(accuracies) / len(accuracies)
    total = int(per_seed_results[0]["anchor_tier_total"])
    variance = sum((value - mean_accuracy) ** 2 for value in accuracies) / len(accuracies)

    aggregated_per_tier: dict[str, dict[str, Any]] = {}
    tier_names = sorted(
        {
            tier
            for result in per_seed_results
            for tier in result.get("anchor_tier_per_tier", {})
        }
    )
    for tier in tier_names:
        tier_entries = [result.get("anchor_tier_per_tier", {}).get(tier, {}) for result in per_seed_results]
        total_for_tier = int(next((entry.get("total", 0) for entry in tier_entries if entry), 0))
        tier_accuracies = [float(entry.get("accuracy", 0.0)) for entry in tier_entries]
        mean_tier_accuracy = sum(tier_accuracies) / len(tier_accuracies)
        aggregated_per_tier[tier] = {
            "accuracy": mean_tier_accuracy,
            "total": total_for_tier,
            "correct": int(round(mean_tier_accuracy * total_for_tier)),
        }

    return {
        "anchor_tier_accuracy": mean_accuracy,
        "anchor_tier_accuracy_mean": mean_accuracy,
        "anchor_tier_accuracy_std": sqrt(variance),
        "anchor_tier_total": total,
        "anchor_tier_correct": int(round(mean_accuracy * total)),
        "anchor_tier_per_tier": aggregated_per_tier,
        "anchor_tier_per_tier_mean": aggregated_per_tier,
    }


def evaluate_anchor_tier_accuracy_bootstrap(
    *,
    model: torch.nn.Module,
    metadata_train_csv: str | Path,
    metadata_eval_csv: str | Path,
    image_root: str | Path,
    model_name: str,
    input_size: int,
    n_per_tier: int,
    seeds: list[int],
    precision: str = "auto",
    group_balanced: bool = False,
    source_checkpoint: str | Path | None = None,
) -> tuple[AnchorStore, dict[str, Any]]:
    per_seed_payloads: list[dict[str, Any]] = []
    primary_anchors: AnchorStore | None = None

    for index, seed in enumerate(seeds):
        anchors, metrics = _evaluate_anchor_tier_accuracy_once(
            model=model,
            metadata_train_csv=metadata_train_csv,
            metadata_eval_csv=metadata_eval_csv,
            image_root=image_root,
            model_name=model_name,
            input_size=input_size,
            n_per_tier=n_per_tier,
            seed=seed,
            precision=precision,
            group_balanced=group_balanced,
            source_checkpoint=source_checkpoint,
        )
        if primary_anchors is None or index == 0:
            primary_anchors = anchors
        per_seed_payloads.append(
            {
                "seed": seed,
                **metrics,
            }
        )

    assert primary_anchors is not None
    aggregated = _aggregate_anchor_bootstrap_results(per_seed_payloads)
    aggregated["anchor_tier_accuracy_per_seed"] = per_seed_payloads
    aggregated["anchor_eval_config"] = {
        "n_per_tier": n_per_tier,
        "seeds": list(seeds),
        "group_balanced": group_balanced,
    }
    return primary_anchors, aggregated
