from __future__ import annotations

import sys
import unittest
from pathlib import Path

TRAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN_ROOT))

from training.postprocess_registry import PostprocessRecord
from training.winner_selection import NamedWinnerRecord, choose_named_winner, parse_named_path


class WinnerSelectionTests(unittest.TestCase):
    def test_parse_named_path_rejects_missing_separator(self) -> None:
        with self.assertRaises(SystemExit):
            parse_named_path("frozen_summary.json", value_label="summary")

    def test_choose_named_winner_tracks_decision_trace(self) -> None:
        candidates = [
            NamedWinnerRecord(
                name="A",
                payload={"winner_config": {"head_type": "linear"}},
                record=PostprocessRecord(
                    checkpoint_relative="output_models/checkpoints/dinov3_vit7b16/ablation/A/checkpoint_epoch_0001.pt",
                    report_relative="output_models/logs/a.json",
                    metrics={"anchor_tier_accuracy_mean": 0.51, "val_accuracy": 0.80, "same_dept_accuracy": 0.79, "val_loss": 0.4},
                ),
            ),
            NamedWinnerRecord(
                name="B",
                payload={"winner_config": {"head_type": "mlp_small"}},
                record=PostprocessRecord(
                    checkpoint_relative="output_models/checkpoints/dinov3_vit7b16/ablation/B/checkpoint_epoch_0001.pt",
                    report_relative="output_models/logs/b.json",
                    metrics={"anchor_tier_accuracy_mean": 0.517, "val_accuracy": 0.78, "same_dept_accuracy": 0.77, "val_loss": 0.5},
                ),
            ),
        ]

        winner, decision_trace = choose_named_winner(candidates, min_improvement=0.005)

        self.assertEqual(winner.name, "B")
        self.assertEqual(len(decision_trace), 1)
        self.assertEqual(decision_trace[0]["candidate"], "B")
        self.assertEqual(decision_trace[0]["incumbent"], "A")


if __name__ == "__main__":
    unittest.main()
