from __future__ import annotations

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
        self.assertEqual(sum(diagnostics["unordered_tier_pair_counts"].values()), len(pairs))

    def test_generate_pairs_honors_tier_pair_minimums_and_cap(self) -> None:
        items = [
            _item(1, "design_a", "S", 95.0),
            _item(2, "design_a", "A", 89.0),
            _item(3, "design_a", "B", 84.0),
            _item(4, "design_a", "C", 74.0),
            _item(5, "design_b", "S", 94.0),
            _item(6, "design_b", "A", 88.0),
            _item(7, "design_b", "B", 83.0),
            _item(8, "design_b", "C", 73.0),
            _item(9, "design_c", "S", 93.0),
            _item(10, "design_c", "A", 87.0),
            _item(11, "design_c", "B", 82.0),
            _item(12, "design_c", "C", 72.0),
        ]

        pairs, diagnostics = generate_pairs(
            items=items,
            total_pairs=12,
            same_dept_ratio=0.5,
            max_appearances=20,
            seed=11,
            distance1_ratio=0.5,
            distance2_ratio=0.25,
            distance3_ratio=0.25,
            tier_pair_minimums={"A-S": 2, "B-C": 2},
            tier_pair_caps={"A-B": 2},
        )

        self.assertEqual(len(pairs), 12)
        self.assertGreaterEqual(diagnostics["unordered_tier_pair_counts"]["A-S"], 2)
        self.assertGreaterEqual(diagnostics["unordered_tier_pair_counts"]["B-C"], 2)
        self.assertLessEqual(diagnostics["unordered_tier_pair_counts"].get("A-B", 0), 2)
        self.assertEqual(diagnostics["pair_quota_shortfalls"], {})

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
        self.assertEqual(args.train_tier_pair_min_a_s, 4_000)
        self.assertEqual(args.train_tier_pair_cap_a_b, 18_000)
        self.assertEqual(args.val_tier_pair_min_a_s, 400)
        self.assertEqual(args.val_tier_pair_cap_a_b, 2_250)

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
        self.assertEqual(args.train_tier_pair_min_a_s, 4_000)
        self.assertEqual(args.train_tier_pair_cap_a_b, 18_000)
        self.assertEqual(args.val_tier_pair_min_a_s, 400)
        self.assertEqual(args.val_tier_pair_cap_a_b, 2_250)

    def test_readiness_shortfall_is_not_fatal_when_pairs_were_still_generated(self) -> None:
        self.assertFalse(
            readiness_module._pair_shortfall_is_fatal(
                {
                    "pair_shortfall": {
                        "train": {"produced": 29112},
                        "val": {"produced": 4302},
                    }
                }
            )
        )
        self.assertTrue(
            readiness_module._pair_shortfall_is_fatal(
                {
                    "pair_shortfall": {
                        "train": {"produced": 0},
                        "val": {"produced": 4302},
                    }
                }
            )
        )

    def test_prepared_readiness_uses_baseline_produced_pair_count(self) -> None:
        baseline_readiness = {
            "pair_stats": {
                "pair_shortfall": {
                    "train": {"produced": 29112},
                    "val": {"produced": 4302},
                }
            }
        }

        self.assertEqual(
            readiness_module._expected_pair_rows(baseline_readiness, split="train", fallback_target=40_000),
            29112,
        )
        self.assertEqual(
            readiness_module._expected_pair_rows(baseline_readiness, split="val", fallback_target=5_000),
            4302,
        )

    def test_split_items_matches_current_snapshot_split_math(self) -> None:
        items = [_item(index, f"dept_{index % 8}", "A", 70.0) for index in range(1, 2737)]

        train_items, val_items, test_items = split_items_by_image(items, train_ratio=0.8, val_ratio=0.1, seed=42)

        self.assertEqual((len(train_items), len(val_items), len(test_items)), (2188, 273, 275))
