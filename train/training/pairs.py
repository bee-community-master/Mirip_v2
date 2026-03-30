from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any

from .utils import load_rows_from_csv, write_json, write_rows_to_csv


class PairGenerationError(RuntimeError):
    def __init__(self, message: str, stats: dict[str, Any]) -> None:
        super().__init__(message)
        self.stats = stats


def _row_to_item(row: dict[str, str]) -> dict[str, Any]:
    return {
        "post_no": int(row["post_no"]),
        "image_path": row["image_path"],
        "tier": row["tier"],
        "tier_score": float(row["tier_score"]),
        "normalized_dept": row["normalized_dept"],
        "anchor_group": row["anchor_group"],
        "university": row["university"],
        "work_type": row.get("work_type", "unknown"),
        "exam_topic": row.get("exam_topic", ""),
    }


def split_items_by_image(
    items: list[dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def _compute_quality_tier(high: dict[str, Any], low: dict[str, Any]) -> int:
    high_work = high.get("work_type", "unknown")
    low_work = low.get("work_type", "unknown")
    both_repro = high_work == "재현작" and low_work == "재현작"
    any_repro = high_work == "재현작" or low_work == "재현작"
    if both_repro:
        return 3
    if any_repro:
        return 2
    return 1


def _generate_same_dept_candidates(items: list[dict[str, Any]], min_score_gap: float) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[item["normalized_dept"]].append(item)

    candidates: list[dict[str, Any]] = []
    for dept, dept_items in grouped.items():
        if len(dept_items) < 2:
            continue
        for i in range(len(dept_items)):
            for j in range(i + 1, len(dept_items)):
                first = dept_items[i]
                second = dept_items[j]
                gap = abs(first["tier_score"] - second["tier_score"])
                if gap < min_score_gap:
                    continue
                if first["tier_score"] >= second["tier_score"]:
                    high, low = first, second
                else:
                    high, low = second, first
                quality = _compute_quality_tier(high, low)
                candidates.append(
                    {
                        "image_path_1": high["image_path"],
                        "image_path_2": low["image_path"],
                        "label": 1,
                        "tier_score_1": high["tier_score"],
                        "tier_score_2": low["tier_score"],
                        "score_gap": round(gap, 4),
                        "pair_type": "same_dept",
                        "dept": dept,
                        "quality_tier": quality,
                        "post_no_1": high["post_no"],
                        "post_no_2": low["post_no"],
                    }
                )
                candidates.append(
                    {
                        "image_path_1": low["image_path"],
                        "image_path_2": high["image_path"],
                        "label": -1,
                        "tier_score_1": low["tier_score"],
                        "tier_score_2": high["tier_score"],
                        "score_gap": round(gap, 4),
                        "pair_type": "same_dept",
                        "dept": dept,
                        "quality_tier": quality,
                        "post_no_1": low["post_no"],
                        "post_no_2": high["post_no"],
                    }
                )
    return candidates


def _generate_cross_dept_candidates(
    items: list[dict[str, Any]],
    min_score_gap: float,
    seed: int,
    max_candidates: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    candidates: list[dict[str, Any]] = []
    if len(items) < 2:
        return candidates

    max_attempts = max_candidates * 3
    attempts = 0
    while len(candidates) < max_candidates and attempts < max_attempts:
        attempts += 1
        first, second = rng.sample(items, 2)
        if first["normalized_dept"] == second["normalized_dept"]:
            continue
        gap = abs(first["tier_score"] - second["tier_score"])
        if gap < min_score_gap:
            continue
        if first["tier_score"] >= second["tier_score"]:
            high, low = first, second
        else:
            high, low = second, first
        candidates.append(
            {
                "image_path_1": high["image_path"],
                "image_path_2": low["image_path"],
                "label": 1,
                "tier_score_1": high["tier_score"],
                "tier_score_2": low["tier_score"],
                "score_gap": round(gap, 4),
                "pair_type": "cross_dept",
                "dept": f"{high['normalized_dept']}_vs_{low['normalized_dept']}",
                "quality_tier": 0,
                "post_no_1": high["post_no"],
                "post_no_2": low["post_no"],
            }
        )
    rng.shuffle(candidates)
    return candidates


def _select_with_appearance_limit(
    candidates: list[dict[str, Any]],
    target: int,
    max_appearances: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for pair in candidates:
        if len(selected) >= target:
            break
        first = pair["image_path_1"]
        second = pair["image_path_2"]
        if counts[first] >= max_appearances or counts[second] >= max_appearances:
            continue
        selected.append(pair)
        counts[first] += 1
        counts[second] += 1
    return selected


def _build_shortfall_reasons(
    pair_label: str,
    target: int,
    selected: int,
    available_candidates: int,
) -> list[str]:
    reasons: list[str] = []
    if selected >= target:
        return reasons
    if available_candidates < target:
        reasons.append(f"insufficient_{pair_label}_candidates")
    if selected < min(target, available_candidates):
        reasons.append(f"{pair_label}_selection_limited_by_max_appearances")
    return reasons


def generate_pairs(
    items: list[dict[str, Any]],
    total_pairs: int,
    same_dept_ratio: float = 0.5,
    min_score_gap: float = 5.0,
    max_appearances: int = 20,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    same_target = int(total_pairs * same_dept_ratio)
    cross_target = total_pairs - same_target

    same_candidates = _generate_same_dept_candidates(items, min_score_gap=min_score_gap)
    same_candidates.sort(key=lambda pair: (-pair["quality_tier"], -pair["score_gap"]))
    same_selected = _select_with_appearance_limit(
        candidates=same_candidates,
        target=same_target,
        max_appearances=max_appearances,
    )

    cross_candidates = _generate_cross_dept_candidates(
        items=items,
        min_score_gap=min_score_gap,
        seed=seed,
        max_candidates=max(total_pairs * 20, 100_000),
    )
    cross_selected = _select_with_appearance_limit(
        candidates=cross_candidates,
        target=cross_target,
        max_appearances=max_appearances,
    )

    pairs = same_selected + cross_selected
    rng.shuffle(pairs)
    diagnostics = {
        "requested_pairs": total_pairs,
        "requested_same_dept_pairs": same_target,
        "requested_cross_dept_pairs": cross_target,
        "available_same_dept_candidates": len(same_candidates),
        "available_cross_dept_candidates": len(cross_candidates),
        "selected_same_dept_pairs": len(same_selected),
        "selected_cross_dept_pairs": len(cross_selected),
        "produced_pairs": len(pairs),
        "shortfall_reasons": _build_shortfall_reasons(
            pair_label="same_dept",
            target=same_target,
            selected=len(same_selected),
            available_candidates=len(same_candidates),
        )
        + _build_shortfall_reasons(
            pair_label="cross_dept",
            target=cross_target,
            selected=len(cross_selected),
            available_candidates=len(cross_candidates),
        ),
    }
    return pairs, diagnostics


def build_pair_outputs(
    manifest_csv: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    total_pairs: int = 50_000,
    same_dept_ratio: float = 0.5,
    min_score_gap: float = 5.0,
    max_appearances: int = 20,
    seed: int = 42,
    strict: bool = True,
) -> dict[str, Any]:
    rows = [_row_to_item(row) for row in load_rows_from_csv(manifest_csv)]
    train_items, val_items, test_items = split_items_by_image(
        rows,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    train_target = int(total_pairs * train_ratio)
    val_target = max(int(total_pairs * val_ratio), 1_000)

    train_pairs, train_pair_diagnostics = generate_pairs(
        items=train_items,
        total_pairs=train_target,
        same_dept_ratio=same_dept_ratio,
        min_score_gap=min_score_gap,
        max_appearances=max_appearances,
        seed=seed,
    )
    val_pairs, val_pair_diagnostics = generate_pairs(
        items=val_items,
        total_pairs=val_target,
        same_dept_ratio=same_dept_ratio,
        min_score_gap=min_score_gap,
        max_appearances=max_appearances,
        seed=seed + 1,
    )

    metadata_fields = [
        "post_no",
        "image_path",
        "tier",
        "tier_score",
        "normalized_dept",
        "anchor_group",
        "university",
        "work_type",
        "exam_topic",
    ]
    pair_fields = [
        "image_path_1",
        "image_path_2",
        "label",
        "tier_score_1",
        "tier_score_2",
        "score_gap",
        "pair_type",
        "dept",
        "quality_tier",
        "post_no_1",
        "post_no_2",
    ]

    write_rows_to_csv(f"{output_dir}/metadata_train.csv", train_items, metadata_fields)
    write_rows_to_csv(f"{output_dir}/metadata_val.csv", val_items, metadata_fields)
    write_rows_to_csv(f"{output_dir}/metadata_test.csv", test_items, metadata_fields)
    write_rows_to_csv(f"{output_dir}/pairs_train.csv", train_pairs, pair_fields)
    write_rows_to_csv(f"{output_dir}/pairs_val.csv", val_pairs, pair_fields)

    stats = {
        "manifest_rows": len(rows),
        "train_items": len(train_items),
        "val_items": len(val_items),
        "test_items": len(test_items),
        "pairs_train": len(train_pairs),
        "pairs_val": len(val_pairs),
        "pair_targets": {
            "train": train_target,
            "val": val_target,
        },
        "pair_generation_train": train_pair_diagnostics,
        "pair_generation_val": val_pair_diagnostics,
        "same_dept_ratio_train": (
            sum(1 for row in train_pairs if row["pair_type"] == "same_dept") / max(len(train_pairs), 1)
        ),
        "same_dept_ratio_val": (
            sum(1 for row in val_pairs if row["pair_type"] == "same_dept") / max(len(val_pairs), 1)
        ),
        "tier_distribution_train": dict(Counter(item["tier"] for item in train_items)),
        "tier_distribution_val": dict(Counter(item["tier"] for item in val_items)),
        "tier_distribution_test": dict(Counter(item["tier"] for item in test_items)),
        "pair_type_train": dict(Counter(pair["pair_type"] for pair in train_pairs)),
        "pair_type_val": dict(Counter(pair["pair_type"] for pair in val_pairs)),
        "quality_tiers_train": dict(Counter(pair["quality_tier"] for pair in train_pairs)),
        "quality_tiers_val": dict(Counter(pair["quality_tier"] for pair in val_pairs)),
        "seed": seed,
        "min_score_gap": min_score_gap,
        "max_appearances": max_appearances,
    }
    stats["pair_shortfall"] = {
        "train": {
            "requested": train_target,
            "produced": len(train_pairs),
            "missing": max(train_target - len(train_pairs), 0),
            "reasons": train_pair_diagnostics["shortfall_reasons"],
        },
        "val": {
            "requested": val_target,
            "produced": len(val_pairs),
            "missing": max(val_target - len(val_pairs), 0),
            "reasons": val_pair_diagnostics["shortfall_reasons"],
        },
    }
    stats["pair_generation_ok"] = all(details["missing"] == 0 for details in stats["pair_shortfall"].values())
    write_json(f"{output_dir}/pair_statistics.json", stats)
    if strict and not stats["pair_generation_ok"]:
        raise PairGenerationError("Requested pair counts could not be satisfied.", stats)
    return stats
