from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

TRAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN_ROOT))

from training.postprocess_registry import update_postprocess_registry


class PostprocessRegistryTests(unittest.TestCase):
    def test_initial_candidate_creates_registry_with_single_retained_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            candidate_report = base / "candidate.json"
            registry_path = base / "registry.json"
            candidate_report.write_text(
                json.dumps(
                    {
                        "checkpoint_relative": "checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0007.pt",
                        "metrics": {
                            "anchor_tier_accuracy": 0.72,
                            "val_accuracy": 0.81,
                            "same_dept_accuracy": 0.79,
                            "val_loss": 0.44,
                        },
                    }
                ),
                encoding="utf-8",
            )

            payload = update_postprocess_registry(
                current_checkpoint="checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0007.pt",
                current_report=candidate_report,
                output_registry=registry_path,
            )

            self.assertEqual(
                payload["selected_best_checkpoint_after_compare"],
                "checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0007.pt",
            )
            self.assertEqual(
                payload["retained_checkpoints"],
                ["checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0007.pt"],
            )
            self.assertEqual(payload["decision"]["decision"], "candidate_selected_initial")

    def test_existing_best_is_retained_when_anchor_accuracy_is_higher(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            candidate_report = base / "candidate.json"
            registry_path = base / "registry.json"

            candidate_report.write_text(
                json.dumps(
                    {
                        "checkpoint_relative": "checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0008.pt",
                        "metrics": {
                            "anchor_tier_accuracy": 0.71,
                            "val_accuracy": 0.90,
                            "same_dept_accuracy": 0.82,
                            "val_loss": 0.30,
                        },
                    }
                ),
                encoding="utf-8",
            )
            registry_path.write_text(
                json.dumps(
                    {
                        "selected_best_checkpoint_after_compare": "checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0006.pt",
                        "selected_best_report_after_compare": "reports/best.json",
                        "selected_best_metrics_after_compare": {
                            "anchor_tier_accuracy": 0.75,
                            "val_accuracy": 0.80,
                            "same_dept_accuracy": 0.78,
                            "val_loss": 0.40,
                            "epoch": 6,
                        },
                    }
                ),
                encoding="utf-8",
            )

            payload = update_postprocess_registry(
                current_checkpoint="checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0008.pt",
                current_report=candidate_report,
                output_registry=registry_path,
            )

            self.assertEqual(
                payload["selected_best_checkpoint_after_compare"],
                "checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0006.pt",
            )
            self.assertEqual(payload["decision"]["criterion"], "anchor_tier_accuracy")
            self.assertEqual(payload["decision"]["decision"], "incumbent_retained")


if __name__ == "__main__":
    unittest.main()
