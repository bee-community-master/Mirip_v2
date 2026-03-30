from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .utils import normalize_staged_image_reference, resolve_project_path, resolve_staged_image_path

VALID_TIERS = ("S", "A", "B", "C")
TIER_BASE_SCORES = {
    "S": 90.0,
    "A": 72.0,
    "B": 55.0,
    "C": 38.0,
}
TIER_RANGES = {
    "S": (82.0, 98.0),
    "A": (65.0, 84.0),
    "B": (48.0, 68.0),
    "C": (30.0, 50.0),
}


def extract_competition_ratio(text: str) -> float | None:
    patterns = [
        r"경쟁률\s*[:：]?\s*(\d+\.?\d*)\s*[:：]\s*1",
        r"경쟁률\s*(\d+\.?\d*)\s*[:：]\s*1",
        r"경쟁률\s*[:：]?\s*(\d+\.?\d*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        for group in match.groups():
            if group is None:
                continue
            try:
                value = float(group)
            except ValueError:
                continue
            if value > 0:
                return round(value, 2)
    return None


def extract_qa_answer(text: str, question_pattern: str, max_length: int = 500) -> str | None:
    match = re.search(question_pattern, text, re.IGNORECASE)
    if not match:
        return None

    start = match.end()
    remaining_after_match = text[start:]
    q_end = re.search(r"\?", remaining_after_match)
    if q_end and q_end.start() < 200:
        start += q_end.end()

    remaining = text[start:]
    end_patterns = [
        r"\n\s*Q\.",
        r"\n\s*★",
        r"\n\s*●",
        r"\n\s*자료제공",
        r"\n\s*미대입시닷컴 합격생",
        r"\n\s*본문내용",
    ]
    end_pos = len(remaining)
    for pattern in end_patterns:
        end_match = re.search(pattern, remaining)
        if end_match and end_match.start() < end_pos:
            end_pos = end_match.start()

    answer = remaining[:end_pos].strip()
    answer = re.sub(r"미대입시닷컴\s*midaeipsi\.com", "", answer)
    answer = re.sub(r"홍대\s+\S+\s*(프리미엄[^미]*)?(미술학원|학원)?", "", answer)
    answer = re.sub(r"\S+\s*(미술학원|학원)\s*$", "", answer, flags=re.MULTILINE)
    answer = re.sub(r"\s+", " ", answer).strip()

    if len(answer) < 10:
        return None
    if len(answer) > max_length:
        answer = answer[:max_length].rsplit(".", 1)[0] + "."
    return answer or None


def extract_exam_topic(text: str) -> str | None:
    answer = extract_qa_answer(text, r"Q\.\s*고사장.*출제문제.*발상")
    if not answer:
        answer = extract_qa_answer(text, r"Q\.\s*실기고사\s*당일\s*출제문제")
    if not answer:
        return None

    patterns = [
        r"출제\s*문제는?\s*[\"'“”]?(.{10,200}?)[\"'“”]?\s*(?:였|이었)",
        r"주제는?\s*[\"'“”]?(.{10,200}?)[\"'“”]?\s*(?:였|이었)",
        r"문제는?\s*[\"'“”]?(.{10,200}?)[\"'“”]?\s*(?:였|이었)",
    ]
    for pattern in patterns:
        match = re.search(pattern, answer)
        if match:
            topic = re.sub(r"\s+", " ", match.group(1)).strip(" '\"“”")
            if len(topic) >= 5:
                return topic[:200]

    sentences = re.split(r"(?<=[.!?다요])\s+", answer)
    first = sentences[0].strip() if sentences else ""
    return first[:200] if len(first) >= 10 else None


def make_anchor_group_key(university: str, normalized_dept: str) -> str | None:
    university = (university or "").strip()
    normalized_dept = (normalized_dept or "").strip()
    if not university or not normalized_dept:
        return None
    if normalized_dept in {"other", "unknown"}:
        return None
    return f"{university}_{normalized_dept}"


def resolve_image_path(image_root: str | Path, image_path: str) -> Path | None:
    return resolve_staged_image_path(image_root, image_path)


def compute_tier_score(
    tier: str,
    competition_ratio: float | None,
    ratios_by_tier: dict[str, list[float]],
) -> tuple[float, float, float | None]:
    if tier not in TIER_BASE_SCORES:
        raise ValueError(f"Unsupported tier: {tier}")

    score_min, score_max = TIER_RANGES[tier]
    if competition_ratio and len(ratios_by_tier.get(tier, [])) >= 5:
        sorted_ratios = sorted(ratios_by_tier[tier])
        below = sum(1 for ratio in sorted_ratios if ratio <= competition_ratio)
        percentile = below / len(sorted_ratios)
        tier_score = score_min + percentile * (score_max - score_min)
        tier_score = round(max(score_min, min(score_max, tier_score)), 1)
        return tier_score, 0.8, round(percentile, 3)
    return TIER_BASE_SCORES[tier], 0.4, None


def build_snapshot_manifest(
    metadata_dir: str | Path,
    image_root: str | Path,
    min_group_size: int = 15,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata_root = resolve_project_path(metadata_dir)
    files = sorted(metadata_root.glob("*.json"), key=lambda path: int(path.stem))

    raw_items: list[dict[str, Any]] = []
    parse_errors = 0
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            parse_errors += 1
            continue
        payload["_source_path"] = str(path)
        raw_items.append(payload)

    group_counter: Counter[str] = Counter()
    ratios_by_tier: dict[str, list[float]] = defaultdict(list)
    for item in raw_items:
        anchor_key = make_anchor_group_key(item.get("university", ""), item.get("normalized_dept", ""))
        if anchor_key:
            group_counter[anchor_key] += 1
        competition_ratio = extract_competition_ratio(item.get("interview_raw", "") or "")
        tier = item.get("tier")
        if tier in VALID_TIERS and competition_ratio:
            ratios_by_tier[tier].append(competition_ratio)

    valid_groups = {group for group, count in group_counter.items() if count >= min_group_size}
    manifest_rows: list[dict[str, Any]] = []
    skipped = Counter()
    tier_distribution = Counter()
    dept_distribution = Counter()

    for item in raw_items:
        tier = item.get("tier")
        if tier not in VALID_TIERS:
            skipped["invalid_tier"] += 1
            continue

        images = item.get("images") or []
        if not images:
            skipped["missing_images"] += 1
            continue

        normalized_dept = (item.get("normalized_dept") or "").strip()
        if not normalized_dept:
            skipped["missing_normalized_dept"] += 1
            continue

        anchor_group = make_anchor_group_key(item.get("university", ""), normalized_dept)
        if not anchor_group or anchor_group not in valid_groups:
            skipped["missing_anchor_group"] += 1
            continue

        normalized_image_path = normalize_staged_image_reference(images[0])
        if normalized_image_path is None:
            skipped["invalid_image_reference"] += 1
            continue

        resolved_image = resolve_image_path(image_root, normalized_image_path)
        if resolved_image is None:
            skipped["missing_image_file"] += 1
            continue

        competition_ratio = extract_competition_ratio(item.get("interview_raw", "") or "")
        tier_score, tier_confidence, competition_percentile = compute_tier_score(
            tier=tier,
            competition_ratio=competition_ratio,
            ratios_by_tier=ratios_by_tier,
        )
        exam_topic = extract_exam_topic(item.get("interview_raw", "") or "") or ""

        row = {
            "post_no": int(item.get("post_no") or Path(item["_source_path"]).stem),
            "image_path": normalized_image_path,
            "tier": tier,
            "tier_score": tier_score,
            "normalized_dept": normalized_dept,
            "anchor_group": anchor_group,
            "university": (item.get("university") or "").strip(),
            "work_type": (item.get("work_type") or "unknown").strip(),
            "exam_topic": exam_topic,
            "tier_confidence": tier_confidence,
            "competition_ratio": competition_ratio if competition_ratio is not None else "",
            "competition_percentile": competition_percentile if competition_percentile is not None else "",
        }
        manifest_rows.append(row)
        tier_distribution[tier] += 1
        dept_distribution[normalized_dept] += 1

    report = {
        "metadata_dir": str(metadata_root),
        "image_root": str(resolve_project_path(image_root)),
        "total_json_files": len(files),
        "parsed_json_files": len(raw_items),
        "parse_errors": parse_errors,
        "eligible_items": len(manifest_rows),
        "skipped": dict(skipped),
        "tier_distribution": dict(sorted(tier_distribution.items())),
        "normalized_dept_distribution": dict(sorted(dept_distribution.items())),
        "valid_anchor_groups": len(valid_groups),
        "top_anchor_groups": group_counter.most_common(20),
        "competition_ratio_counts": {tier: len(values) for tier, values in ratios_by_tier.items()},
        "min_group_size": min_group_size,
        "required_manifest_columns": [
            "post_no",
            "image_path",
            "tier",
            "tier_score",
            "normalized_dept",
            "anchor_group",
            "university",
            "work_type",
            "exam_topic",
        ],
    }
    return manifest_rows, report
