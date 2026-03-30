from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
import sys
import importlib.util

TRAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN_ROOT))

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from training.config import DinoV3TrainingConfig
    from training.trainer import DinoV3Trainer


if TORCH_AVAILABLE:
    class DummyPairwiseModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([[0.1]], dtype=torch.float32))
            self.feature_extractor = None

        def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            score1 = img1.matmul(self.weight)
            score2 = img2.matmul(self.weight)
            return score1, score2

        def compute_loss(self, score1: torch.Tensor, score2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            target = labels.unsqueeze(-1)
            return ((score1 - score2 - target) ** 2).mean()


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for trainer callback test")
class TrainerPostprocessTests(unittest.TestCase):
    @staticmethod
    def _single_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor([[1.0], [2.0]], dtype=torch.float32),
            torch.tensor([[0.5], [1.5]], dtype=torch.float32),
            torch.tensor([1.0, -1.0], dtype=torch.float32),
            torch.tensor([1, 0], dtype=torch.int64),
        )

    def test_post_epoch_callback_receives_each_saved_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = DinoV3TrainingConfig(
                checkpoint_dir=temp_dir,
                max_epochs=2,
                batch_size=2,
                gradient_accumulation_steps=1,
                num_workers=0,
                persistent_workers=False,
                pin_memory=False,
                device="cpu",
                precision="fp32",
            )
            trainer = DinoV3Trainer(model=DummyPairwiseModel(), config=config)
            batch = self._single_batch()
            train_loader = [batch]
            val_loader = [batch]
            callback_paths: list[str] = []

            def callback(path: Path, _metrics: dict[str, float]) -> None:
                callback_paths.append(path.name)

            summary = trainer.train(train_loader, val_loader, post_epoch_callback=callback)

            self.assertEqual(callback_paths, ["checkpoint_epoch_0001.pt", "checkpoint_epoch_0002.pt"])
            self.assertTrue((Path(temp_dir) / "checkpoint_epoch_0001.pt").exists())
            self.assertTrue((Path(temp_dir) / "checkpoint_epoch_0002.pt").exists())
            self.assertEqual(Path(summary["latest_completed_checkpoint"]).name, "checkpoint_epoch_0002.pt")

    def test_resume_next_epoch_skips_repeating_completed_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = DinoV3TrainingConfig(
                checkpoint_dir=temp_dir,
                max_epochs=1,
                batch_size=2,
                gradient_accumulation_steps=1,
                num_workers=0,
                persistent_workers=False,
                pin_memory=False,
                device="cpu",
                precision="fp32",
            )
            trainer = DinoV3Trainer(model=DummyPairwiseModel(), config=config)
            batch = self._single_batch()
            trainer.train([batch], [batch])

            resumed = DinoV3Trainer(
                model=DummyPairwiseModel(),
                config=DinoV3TrainingConfig(
                    checkpoint_dir=temp_dir,
                    max_epochs=5,
                    batch_size=2,
                    gradient_accumulation_steps=1,
                    num_workers=0,
                    persistent_workers=False,
                    pin_memory=False,
                    device="cpu",
                    precision="fp32",
                ),
                resume_from=str(Path(temp_dir) / "checkpoint_epoch_0001.pt"),
                resume_next_epoch=True,
            )

            self.assertEqual(resumed.current_epoch, 1)

    def test_anchor_metric_updates_history_and_resume_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = DinoV3TrainingConfig(
                checkpoint_dir=temp_dir,
                max_epochs=2,
                batch_size=2,
                gradient_accumulation_steps=1,
                num_workers=0,
                persistent_workers=False,
                pin_memory=False,
                device="cpu",
                precision="fp32",
                early_stopping_metric="anchor_tier_accuracy",
                early_stopping_patience=3,
            )
            trainer = DinoV3Trainer(model=DummyPairwiseModel(), config=config)
            callback_metrics = iter([0.61, 0.57])

            def callback(_path: Path, _metrics: dict[str, float]) -> dict[str, object]:
                return {
                    "report": {
                        "metrics": {
                            "anchor_tier_accuracy": next(callback_metrics),
                        }
                    }
                }

            batch = self._single_batch()
            summary = trainer.train([batch], [batch], post_epoch_callback=callback)

            self.assertEqual(summary["history"]["anchor_tier_accuracy"], [0.61, 0.57])
            self.assertEqual(trainer.best_selection_metric, 0.61)

            resumed = DinoV3Trainer(
                model=DummyPairwiseModel(),
                config=DinoV3TrainingConfig(
                    checkpoint_dir=temp_dir,
                    max_epochs=5,
                    batch_size=2,
                    gradient_accumulation_steps=1,
                    num_workers=0,
                    persistent_workers=False,
                    pin_memory=False,
                    device="cpu",
                    precision="fp32",
                    early_stopping_metric="anchor_tier_accuracy",
                ),
                resume_from=str(Path(temp_dir) / "checkpoint_epoch_0002.pt"),
                resume_next_epoch=True,
            )

            self.assertEqual(resumed.current_epoch, 2)
            self.assertEqual(resumed.best_selection_metric, 0.61)
            self.assertEqual(resumed.best_selection_metric_name, "anchor_tier_accuracy")

    def test_val_loss_metric_saves_best_model_on_first_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = DinoV3TrainingConfig(
                checkpoint_dir=temp_dir,
                max_epochs=1,
                batch_size=2,
                gradient_accumulation_steps=1,
                num_workers=0,
                persistent_workers=False,
                pin_memory=False,
                device="cpu",
                precision="fp32",
                early_stopping_metric="val_loss",
            )
            trainer = DinoV3Trainer(model=DummyPairwiseModel(), config=config)
            batch = self._single_batch()

            trainer.train([batch], [batch])

            self.assertTrue((Path(temp_dir) / "best_model.pt").exists())
            self.assertTrue(math.isfinite(trainer.best_val_loss))


if __name__ == "__main__":
    unittest.main()
