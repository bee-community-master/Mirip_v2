from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

TRAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN_ROOT))

from training import build_pairs as build_pairs_module
from training import validate_training_readiness as readiness_module
from training.pairs import generate_pairs


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
    def test_generate_pairs_prefers_adjacent_tiers_when_requested(self) -> None:
        items = [
            _item(1, "design_a", "S", 95.0),
            _item(2, "design_a", "A", 88.0),
            _item(3, "design_a", "B", 80.0),
            _item(4, "design_b", "S", 94.0),
            _item(5, "design_b", "A", 87.0),
            _item(6, "design_b", "B", 79.0),
        ]

        pairs, diagnostics = generate_pairs(
            items=items,
            total_pairs=8,
            same_dept_ratio=0.5,
            max_appearances=8,
            seed=7,
            adjacent_tier_ratio=1.0,
        )

        self.assertEqual(len(pairs), 8)
        self.assertTrue(all(pair["tier_distance"] == 1 for pair in pairs))
        self.assertEqual(diagnostics["selected_adjacent_tier_pairs"], 8)

    def test_build_pairs_cli_defaults_match_boundary_focused_preset(self) -> None:
        with mock.patch.object(sys, "argv", ["build_pairs.py"]):
            args = build_pairs_module.parse_args()

        self.assertEqual(args.train_ratio, 0.75)
        self.assertEqual(args.val_ratio, 0.15)
        self.assertEqual(args.adjacent_tier_ratio, 0.7)

    def test_readiness_cli_defaults_match_prepared_pair_targets(self) -> None:
        with mock.patch.object(sys, "argv", ["validate_training_readiness.py"]):
            args = readiness_module.parse_args()

        self.assertEqual(args.train_ratio, 0.75)
        self.assertEqual(args.val_ratio, 0.15)
        self.assertEqual(args.adjacent_tier_ratio, 0.7)
