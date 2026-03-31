from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path
from unittest import mock

TRAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN_ROOT))

from training import build_pairs as build_pairs_module
from training import validate_training_readiness as readiness_module
from training.pairs import generate_pairs, split_items_by_image


def _item(post_no: int, dept: str, tier: str, tier_score: float) -> dict[str, object]:
    return {
        "post_no": post_no,
        "image_path": f"raw_images/{post_no}.jpg",
        "tier": tier,
        "tier_score": tier_score,
        "normalized_dept": dept,
        "anchor_group": f"uni_{dept}",
        "university": "uni",
        "work_type": "실기작",
        "exam_topic": "topic",
    }


class PairGenerationTests(unittest.TestCase):
    def test_generate_pairs_honors_distance_quota_targets(self) -> None:
        items = [
            _item(1, "design_a", "S", 95.0),
            _item(2, "design_a", "A", 88.0),
            _item(3, "design_a", "B", 80.0),
            _item(4, "design_a", "C", 70.0),
            _item(5, "design_b", "S", 94.0),
            _item(6, "design_b", "A", 87.0),
            _item(7, "design_b", "B", 79.0),
            _item(8, "design_b", "C", 69.0),
        ]

        pairs, diagnostics = generate_pairs(
            items=items,
            total_pairs=10,
            same_dept_ratio=0.5,
            max_appearances=12,
            seed=7,
            distance1_ratio=0.6,
            distance2_ratio=0.3,
            distance3_ratio=0.1,
        )

        self.assertEqual(len(pairs), 10)
        self.assertEqual(diagnostics["distance_targets"], {"1": 6, "2": 3, "3": 1})
        self.assertEqual(diagnostics["selected_distance_counts"], {"1": 6, "2": 3, "3": 1})
        self.assertEqual(diagnostics["selected_pair_type_counts"]["same_dept"], 5)
        self.assertEqual(diagnostics["selected_pair_type_counts"]["cross_dept"], 5)

    def test_build_pairs_cli_defaults_match_legacy_aligned_preset(self) -> None:
        with mock.patch.object(sys, "argv", ["build_pairs.py"]):
            args = build_pairs_module.parse_args()

        self.assertEqual(args.train_ratio, 0.8)
        self.assertEqual(args.val_ratio, 0.1)
        self.assertEqual(args.train_pairs_target, 40_000)
        self.assertEqual(args.val_pairs_target, 5_000)
        self.assertEqual(args.max_appearances, 48)
        self.assertEqual(args.distance1_ratio, 0.6)
        self.assertEqual(args.distance2_ratio, 0.3)
        self.assertEqual(args.distance3_ratio, 0.1)

    def test_readiness_cli_defaults_match_prepared_pair_targets(self) -> None:
        with mock.patch.object(sys, "argv", ["validate_training_readiness.py"]):
            args = readiness_module.parse_args()

        self.assertEqual(args.train_ratio, 0.8)
        self.assertEqual(args.val_ratio, 0.1)
        self.assertEqual(args.train_pairs_target, 40_000)
        self.assertEqual(args.val_pairs_target, 5_000)
        self.assertEqual(args.max_appearances, 48)
        self.assertEqual(args.distance1_ratio, 0.6)
        self.assertEqual(args.distance2_ratio, 0.3)
        self.assertEqual(args.distance3_ratio, 0.1)

    def test_split_items_matches_legacy_snapshot_counts(self) -> None:
        base = TRAIN_ROOT / "training" / "data"
        rows: list[dict[str, str]] = []
        for name in ("metadata_train.csv", "metadata_val.csv", "metadata_test.csv"):
            with (base / name).open("r", encoding="utf-8", newline="") as handle:
                rows.extend(csv.DictReader(handle))

        items = [
            {
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
            for row in rows
        ]

        train_items, val_items, test_items = split_items_by_image(items, train_ratio=0.8, val_ratio=0.1, seed=42)

        self.assertEqual((len(train_items), len(val_items), len(test_items)), (2187, 273, 274))
