from __future__ import annotations

import random
from collections import Counter, defaultdict
from math import floor
from typing import Any

from .utils import load_rows_from_csv, write_json, write_rows_to_csv

TIER_ORDER = {
    "S": 0,
    "A": 1,
    "B": 2,
    "C": 3,
}


class PairGenerationError(RuntimeError):
    def __init__(self, message: str, stats: dict[str, Any]) -> None:
        super().__init__(message)
        self.stats = stats


PAIR_TYPES: tuple[str, str] = ("same_dept", "cross_dept")
DISTANCE_BUCKETS: tuple[int, int, int] = (1, 2, 3)
DEFAULT_TIER_PAIR_MINIMUMS_TRAIN: dict[str, int] = {
    "A-S": 4_000,
    "B-C": 4_000,
    "A-C": 3_000,
    "C-S": 3_000,
}
DEFAULT_TIER_PAIR_MINIMUMS_VAL: dict[str, int] = {
    "A-S": 400,
    "B-C": 400,
    "A-C": 300,
    "C-S": 300,
}
DEFAULT_TIER_PAIR_CAP_AB_TRAIN = 18_000
DEFAULT_TIER_PAIR_CAP_AB_VAL = 2_250


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


def _unordered_tier_pair(first_tier: str, second_tier: str) -> str:
    return "-".join(sorted((first_tier, second_tier)))


def _tier_distance(first: dict[str, Any], second: dict[str, Any]) -> int:
    first_rank = TIER_ORDER.get(first["tier"])
    second_rank = TIER_ORDER.get(second["tier"])
    if first_rank is None or second_rank is None:
        return 99
    return abs(first_rank - second_rank)


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
                        "tier_1": high["tier"],
                        "tier_2": low["tier"],
                        "label": 1,
                        "tier_score_1": high["tier_score"],
                        "tier_score_2": low["tier_score"],
                        "score_gap": round(gap, 4),
                        "tier_distance": _tier_distance(high, low),
                        "unordered_tier_pair": _unordered_tier_pair(high["tier"], low["tier"]),
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
                        "tier_1": low["tier"],
                        "tier_2": high["tier"],
                        "label": -1,
                        "tier_score_1": low["tier_score"],
                        "tier_score_2": high["tier_score"],
                        "score_gap": round(gap, 4),
                        "tier_distance": _tier_distance(high, low),
                        "unordered_tier_pair": _unordered_tier_pair(high["tier"], low["tier"]),
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
                "tier_1": high["tier"],
                "tier_2": low["tier"],
                "label": 1,
                "tier_score_1": high["tier_score"],
                "tier_score_2": low["tier_score"],
                "score_gap": round(gap, 4),
                "tier_distance": _tier_distance(high, low),
                "unordered_tier_pair": _unordered_tier_pair(high["tier"], low["tier"]),
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
    counts: Counter[str] | None = None,
    selected_keys: set[tuple[int, int, int]] | None = None,
    tier_pair_counts: Counter[str] | None = None,
    tier_pair_caps: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    image_counts: Counter[str] = counts if counts is not None else Counter()
    seen_keys = selected_keys if selected_keys is not None else set()
    pair_counts = tier_pair_counts if tier_pair_counts is not None else Counter()
    for pair in candidates:
        if len(selected) >= target:
            break
        first = pair["image_path_1"]
        second = pair["image_path_2"]
        pair_key = (pair["post_no_1"], pair["post_no_2"], pair["label"])
        unordered_tier_pair = str(pair.get("unordered_tier_pair", ""))
        if pair_key in seen_keys:
            continue
        if image_counts[first] >= max_appearances or image_counts[second] >= max_appearances:
            continue
        if tier_pair_caps and unordered_tier_pair in tier_pair_caps:
            if pair_counts[unordered_tier_pair] >= tier_pair_caps[unordered_tier_pair]:
                continue
        selected.append(pair)
        seen_keys.add(pair_key)
        image_counts[first] += 1
        image_counts[second] += 1
        if unordered_tier_pair:
            pair_counts[unordered_tier_pair] += 1
    return selected


def _sort_same_dept_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda pair: (
            pair["tier_distance"] != 1,
            -pair["quality_tier"],
            pair["tier_distance"],
            pair["score_gap"],
        ),
    )


def _sort_cross_dept_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda pair: (
            pair["tier_distance"] != 1,
            pair["tier_distance"],
            pair["score_gap"],
        ),
    )


def _normalize_distance_ratio_targets(
    distance1_ratio: float,
    distance2_ratio: float,
    distance3_ratio: float,
) -> dict[int, float]:
    raw = {
        1: distance1_ratio,
        2: distance2_ratio,
        3: distance3_ratio,
    }
    if any(value < 0 for value in raw.values()):
        raise ValueError("distance ratios must be non-negative")
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("at least one distance ratio must be positive")
    return {
        distance: value / total
        for distance, value in raw.items()
    }


def _allocate_integer_targets(total: int, ratios: dict[Any, float]) -> dict[Any, int]:
    if total <= 0:
        return {key: 0 for key in ratios}

    raw_targets = {key: total * ratio for key, ratio in ratios.items()}
    allocated = {key: int(floor(value)) for key, value in raw_targets.items()}
    remaining = total - sum(allocated.values())
    for key, _ in sorted(
        raw_targets.items(),
        key=lambda item: (item[1] - floor(item[1]), str(item[0])),
        reverse=True,
    ):
        if remaining <= 0:
            break
        allocated[key] += 1
        remaining -= 1
    return allocated


def _allocate_pair_type_distance_targets(
    total_pairs: int,
    same_dept_ratio: float,
    distance_ratio_targets: dict[int, float],
) -> tuple[dict[str, int], dict[int, int], dict[tuple[str, int], int]]:
    pair_type_targets = {
        "same_dept": int(total_pairs * same_dept_ratio),
        "cross_dept": total_pairs - int(total_pairs * same_dept_ratio),
    }
    distance_targets = _allocate_integer_targets(total_pairs, distance_ratio_targets)
    cell_targets: dict[tuple[str, int], int] = {}
    for distance, distance_target in distance_targets.items():
        same_for_distance = int(distance_target * same_dept_ratio)
        cell_targets[("same_dept", distance)] = same_for_distance
        cell_targets[("cross_dept", distance)] = distance_target - same_for_distance

    current_same = sum(cell_targets[("same_dept", distance)] for distance in DISTANCE_BUCKETS)
    delta = pair_type_targets["same_dept"] - current_same
    if delta != 0:
        preferred_distances = sorted(
            DISTANCE_BUCKETS,
            key=lambda distance: distance_ratio_targets[distance],
            reverse=True,
        )
        while delta != 0:
            changed = False
            for distance in preferred_distances:
                same_key = ("same_dept", distance)
                cross_key = ("cross_dept", distance)
                if delta > 0 and cell_targets[cross_key] > 0:
                    cell_targets[same_key] += 1
                    cell_targets[cross_key] -= 1
                    delta -= 1
                    changed = True
                elif delta < 0 and cell_targets[same_key] > 0:
                    cell_targets[same_key] -= 1
                    cell_targets[cross_key] += 1
                    delta += 1
                    changed = True
                if delta == 0:
                    break
            if not changed:
                break

    return pair_type_targets, distance_targets, cell_targets


def _bucket_candidates_by_cell(
    candidates: list[dict[str, Any]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = {
        (pair_type, distance): []
        for pair_type in PAIR_TYPES
        for distance in DISTANCE_BUCKETS
    }
    for pair in candidates:
        distance = int(pair["tier_distance"])
        if distance not in DISTANCE_BUCKETS:
            continue
        buckets[(pair["pair_type"], distance)].append(pair)
    return buckets


def _bucket_candidates_by_tier_pair(
    candidates: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in candidates:
        buckets[str(pair["unordered_tier_pair"])].append(pair)
    return buckets


def _select_tier_pair_minimums(
    candidates: list[dict[str, Any]],
    *,
    tier_pair_minimums: dict[str, int],
    max_appearances: int,
    tier_pair_caps: dict[str, int],
) -> tuple[list[dict[str, Any]], Counter[str], Counter[str], set[tuple[int, int, int]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    image_counts: Counter[str] = Counter()
    tier_pair_counts: Counter[str] = Counter()
    selected_keys: set[tuple[int, int, int]] = set()
    candidates_by_tier_pair = _bucket_candidates_by_tier_pair(candidates)

    tier_pair_order = sorted(
        tier_pair_minimums,
        key=lambda pair_key: (
            len(candidates_by_tier_pair.get(pair_key, [])) / max(tier_pair_minimums[pair_key], 1)
            if tier_pair_minimums[pair_key] > 0
            else float("inf"),
            pair_key,
        ),
    )

    selected_counts: dict[str, int] = {}
    shortfalls: dict[str, dict[str, Any]] = {}
    for pair_key in tier_pair_order:
        target = tier_pair_minimums[pair_key]
        chosen = _select_with_appearance_limit(
            candidates=candidates_by_tier_pair.get(pair_key, []),
            target=target,
            max_appearances=max_appearances,
            counts=image_counts,
            selected_keys=selected_keys,
            tier_pair_counts=tier_pair_counts,
            tier_pair_caps=tier_pair_caps,
        )
        selected.extend(chosen)
        selected_counts[pair_key] = len(chosen)
        available = len(candidates_by_tier_pair.get(pair_key, []))
        if len(chosen) < target:
            reasons = []
            if available < target:
                reasons.append("insufficient_candidates")
            if len(chosen) < min(target, available):
                reasons.append("selection_limited_by_max_appearances_or_caps")
            shortfalls[pair_key] = {
                "target": target,
                "selected": len(chosen),
                "available": available,
                "reasons": reasons,
            }

    diagnostics = {
        "tier_pair_minimum_targets": dict(tier_pair_minimums),
        "tier_pair_minimum_selected": selected_counts,
        "tier_pair_available": {
            pair_key: len(candidates_by_tier_pair.get(pair_key, []))
            for pair_key in tier_pair_minimums
        },
        "pair_quota_shortfalls": shortfalls,
    }
    return selected, image_counts, tier_pair_counts, selected_keys, diagnostics


def _select_quota_constrained_pairs(
    candidates: list[dict[str, Any]],
    *,
    total_pairs: int,
    same_dept_ratio: float,
    max_appearances: int,
    distance_ratio_targets: dict[int, float],
    initial_selected: list[dict[str, Any]] | None = None,
    counts: Counter[str] | None = None,
    selected_keys: set[tuple[int, int, int]] | None = None,
    tier_pair_counts: Counter[str] | None = None,
    tier_pair_caps: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pair_type_targets, distance_targets, cell_targets = _allocate_pair_type_distance_targets(
        total_pairs=total_pairs,
        same_dept_ratio=same_dept_ratio,
        distance_ratio_targets=distance_ratio_targets,
    )
    candidates_by_cell = _bucket_candidates_by_cell(candidates)
    selected: list[dict[str, Any]] = list(initial_selected or [])
    counts = counts if counts is not None else Counter()
    selected_keys = selected_keys if selected_keys is not None else set()
    tier_pair_counts = tier_pair_counts if tier_pair_counts is not None else Counter()
    selected_by_cell: dict[tuple[str, int], int] = defaultdict(int)
    for pair in selected:
        selected_by_cell[(pair["pair_type"], int(pair["tier_distance"]))] += 1

    prioritized_cells = sorted(
        [
            (pair_type, distance)
            for distance in DISTANCE_BUCKETS
            for pair_type in PAIR_TYPES
        ],
        key=lambda cell: (
            len(candidates_by_cell[cell]) / max(cell_targets[cell], 1) if cell_targets[cell] else float("inf"),
            -cell[1],
            PAIR_TYPES.index(cell[0]),
        ),
    )
    for cell in prioritized_cells:
        chosen = _select_with_appearance_limit(
            candidates=candidates_by_cell[cell],
            target=max(cell_targets[cell] - selected_by_cell[cell], 0),
            max_appearances=max_appearances,
            counts=counts,
            selected_keys=selected_keys,
            tier_pair_counts=tier_pair_counts,
            tier_pair_caps=tier_pair_caps,
        )
        selected.extend(chosen)
        selected_by_cell[cell] += len(chosen)

    pair_type_counts = Counter(pair["pair_type"] for pair in selected)
    distance_counts = Counter(int(pair["tier_distance"]) for pair in selected)
    remaining_candidates = [
        pair
        for pair in candidates
        if (pair["post_no_1"], pair["post_no_2"], pair["label"]) not in selected_keys
    ]
    remaining_candidates.sort(
        key=lambda pair: (
            pair_type_targets[pair["pair_type"]] - pair_type_counts[pair["pair_type"]] <= 0,
            distance_targets[int(pair["tier_distance"])] - distance_counts[int(pair["tier_distance"])] <= 0,
            -(distance_targets[int(pair["tier_distance"])] - distance_counts[int(pair["tier_distance"])]),
            -(pair_type_targets[pair["pair_type"]] - pair_type_counts[pair["pair_type"]]),
            -int(pair["tier_distance"]),
        )
    )
    if len(selected) < total_pairs:
        selected.extend(
            _select_with_appearance_limit(
                candidates=remaining_candidates,
                target=total_pairs - len(selected),
                max_appearances=max_appearances,
                counts=counts,
                selected_keys=selected_keys,
                tier_pair_counts=tier_pair_counts,
                tier_pair_caps=tier_pair_caps,
            )
        )

    final_pair_type_counts = Counter(pair["pair_type"] for pair in selected)
    final_distance_counts = Counter(int(pair["tier_distance"]) for pair in selected)
    final_tier_pair_counts = Counter(str(pair["unordered_tier_pair"]) for pair in selected)
    return selected[:total_pairs], {
        "pair_type_targets": dict(pair_type_targets),
        "distance_targets": {str(distance): distance_targets[distance] for distance in DISTANCE_BUCKETS},
        "cell_targets": {
            pair_type: {str(distance): cell_targets[(pair_type, distance)] for distance in DISTANCE_BUCKETS}
            for pair_type in PAIR_TYPES
        },
        "available_by_cell": {
            pair_type: {str(distance): len(candidates_by_cell[(pair_type, distance)]) for distance in DISTANCE_BUCKETS}
            for pair_type in PAIR_TYPES
        },
        "selected_by_cell": {
            pair_type: {str(distance): selected_by_cell[(pair_type, distance)] for distance in DISTANCE_BUCKETS}
            for pair_type in PAIR_TYPES
        },
        "selected_pair_type_counts": dict(final_pair_type_counts),
        "selected_distance_counts": {str(distance): final_distance_counts[distance] for distance in DISTANCE_BUCKETS},
        "selected_distance_ratios": {
            str(distance): final_distance_counts[distance] / max(len(selected), 1)
            for distance in DISTANCE_BUCKETS
        },
        "distance_quota_attainment": {
            str(distance): final_distance_counts[distance] / max(distance_targets[distance], 1)
            for distance in DISTANCE_BUCKETS
        },
        "pair_type_quota_attainment": {
            pair_type: final_pair_type_counts[pair_type] / max(pair_type_targets[pair_type], 1)
            for pair_type in PAIR_TYPES
        },
        "unordered_tier_pair_counts": dict(final_tier_pair_counts),
        "distance_ratio_targets": {
            str(distance): distance_ratio_targets[distance]
            for distance in DISTANCE_BUCKETS
        },
    }


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
    max_appearances: int = 48,
    seed: int = 42,
    distance1_ratio: float = 0.6,
    distance2_ratio: float = 0.3,
    distance3_ratio: float = 0.1,
    tier_pair_minimums: dict[str, int] | None = None,
    tier_pair_caps: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    same_target = int(total_pairs * same_dept_ratio)
    cross_target = total_pairs - same_target
    distance_ratio_targets = _normalize_distance_ratio_targets(
        distance1_ratio=distance1_ratio,
        distance2_ratio=distance2_ratio,
        distance3_ratio=distance3_ratio,
    )
    tier_pair_minimums = {
        pair_key: target
        for pair_key, target in (tier_pair_minimums or {}).items()
        if target > 0
    }
    tier_pair_caps = {
        pair_key: target
        for pair_key, target in (tier_pair_caps or {}).items()
        if target > 0
    }

    same_candidates = _generate_same_dept_candidates(items, min_score_gap=min_score_gap)
    same_candidates = _sort_same_dept_candidates(same_candidates)
    same_candidates = [
        pair
        for pair in same_candidates
        if int(pair["tier_distance"]) in DISTANCE_BUCKETS
    ]
    cross_candidates = _generate_cross_dept_candidates(
        items=items,
        min_score_gap=min_score_gap,
        seed=seed,
        max_candidates=max(total_pairs * 20, 100_000),
    )
    cross_candidates = _sort_cross_dept_candidates(cross_candidates)
    cross_candidates = [
        pair
        for pair in cross_candidates
        if int(pair["tier_distance"]) in DISTANCE_BUCKETS
    ]
    all_candidates = same_candidates + cross_candidates
    minimum_selected, counts, tier_pair_counts, selected_keys, tier_pair_diagnostics = _select_tier_pair_minimums(
        all_candidates,
        tier_pair_minimums=tier_pair_minimums,
        max_appearances=max_appearances,
        tier_pair_caps=tier_pair_caps,
    )
    pairs, quota_diagnostics = _select_quota_constrained_pairs(
        candidates=all_candidates,
        total_pairs=total_pairs,
        same_dept_ratio=same_dept_ratio,
        max_appearances=max_appearances,
        distance_ratio_targets=distance_ratio_targets,
        initial_selected=minimum_selected,
        counts=counts,
        selected_keys=selected_keys,
        tier_pair_counts=tier_pair_counts,
        tier_pair_caps=tier_pair_caps,
    )
    rng.shuffle(pairs)
    selected_same = sum(1 for pair in pairs if pair["pair_type"] == "same_dept")
    selected_cross = len(pairs) - selected_same
    diagnostics = {
        "requested_pairs": total_pairs,
        "requested_same_dept_pairs": same_target,
        "requested_cross_dept_pairs": cross_target,
        "available_same_dept_candidates": len(same_candidates),
        "available_cross_dept_candidates": len(cross_candidates),
        "selected_same_dept_pairs": selected_same,
        "selected_cross_dept_pairs": selected_cross,
        "produced_pairs": len(pairs),
        "shortfall_reasons": _build_shortfall_reasons(
            pair_label="same_dept",
            target=same_target,
            selected=selected_same,
            available_candidates=len(same_candidates),
        )
        + _build_shortfall_reasons(
            pair_label="cross_dept",
            target=cross_target,
            selected=selected_cross,
            available_candidates=len(cross_candidates),
        ),
        **quota_diagnostics,
        **tier_pair_diagnostics,
        "tier_pair_caps": dict(tier_pair_caps),
        "tier_pair_cap_hits": {
            pair_key: quota_diagnostics["unordered_tier_pair_counts"].get(pair_key, 0) >= cap
            for pair_key, cap in tier_pair_caps.items()
        },
    }
    return pairs, diagnostics


def build_pair_outputs(
    manifest_csv: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    train_pairs_target: int = 40_000,
    val_pairs_target: int = 5_000,
    same_dept_ratio: float = 0.5,
    min_score_gap: float = 5.0,
    max_appearances: int = 48,
    seed: int = 42,
    strict: bool = True,
    distance1_ratio: float = 0.6,
    distance2_ratio: float = 0.3,
    distance3_ratio: float = 0.1,
    train_tier_pair_minimums: dict[str, int] | None = None,
    val_tier_pair_minimums: dict[str, int] | None = None,
    train_tier_pair_caps: dict[str, int] | None = None,
    val_tier_pair_caps: dict[str, int] | None = None,
) -> dict[str, Any]:
    rows = [_row_to_item(row) for row in load_rows_from_csv(manifest_csv)]
    train_items, val_items, test_items = split_items_by_image(
        rows,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    train_target = train_pairs_target
    val_target = val_pairs_target

    train_pairs, train_pair_diagnostics = generate_pairs(
        items=train_items,
        total_pairs=train_target,
        same_dept_ratio=same_dept_ratio,
        min_score_gap=min_score_gap,
        max_appearances=max_appearances,
        seed=seed,
        distance1_ratio=distance1_ratio,
        distance2_ratio=distance2_ratio,
        distance3_ratio=distance3_ratio,
        tier_pair_minimums=train_tier_pair_minimums,
        tier_pair_caps=train_tier_pair_caps,
    )
    val_pairs, val_pair_diagnostics = generate_pairs(
        items=val_items,
        total_pairs=val_target,
        same_dept_ratio=same_dept_ratio,
        min_score_gap=min_score_gap,
        max_appearances=max_appearances,
        seed=seed + 1,
        distance1_ratio=distance1_ratio,
        distance2_ratio=distance2_ratio,
        distance3_ratio=distance3_ratio,
        tier_pair_minimums=val_tier_pair_minimums,
        tier_pair_caps=val_tier_pair_caps,
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
        "tier_1",
        "tier_2",
        "tier_distance",
        "unordered_tier_pair",
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
        "split_ratio_targets": {
            "train": train_ratio,
            "val": val_ratio,
            "test": max(0.0, 1.0 - train_ratio - val_ratio),
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
        "tier_direction_train": dict(Counter(f"{pair['tier_1']}>{pair['tier_2']}" for pair in train_pairs)),
        "tier_direction_val": dict(Counter(f"{pair['tier_1']}>{pair['tier_2']}" for pair in val_pairs)),
        "tier_distance_train": dict(Counter(str(pair["tier_distance"]) for pair in train_pairs)),
        "tier_distance_val": dict(Counter(str(pair["tier_distance"]) for pair in val_pairs)),
        "quality_tiers_train": dict(Counter(pair["quality_tier"] for pair in train_pairs)),
        "quality_tiers_val": dict(Counter(pair["quality_tier"] for pair in val_pairs)),
        "unordered_tier_pair_train": dict(Counter(pair["unordered_tier_pair"] for pair in train_pairs)),
        "unordered_tier_pair_val": dict(Counter(pair["unordered_tier_pair"] for pair in val_pairs)),
        "seed": seed,
        "min_score_gap": min_score_gap,
        "max_appearances": max_appearances,
        "distance_ratio_targets": {
            "1": distance1_ratio,
            "2": distance2_ratio,
            "3": distance3_ratio,
        },
        "tier_pair_quota_targets": {
            "train_minimums": train_tier_pair_minimums or {},
            "val_minimums": val_tier_pair_minimums or {},
            "train_caps": train_tier_pair_caps or {},
            "val_caps": val_tier_pair_caps or {},
        },
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
