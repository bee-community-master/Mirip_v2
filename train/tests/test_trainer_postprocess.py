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

    def test_optimizer_uses_non_foreach_adamw_for_lower_peak_memory(self) -> None:
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
            self.assertIs(trainer.optimizer.defaults.get("foreach"), False)

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
            self.assertEqual(resumed.best_selection_metric_name, "anchor_tier_accuracy_mean")

    def test_reset_training_state_on_resume_keeps_best_metric_but_clears_progress(self) -> None:
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
            trainer.train([batch], [batch], post_epoch_callback=callback)

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
                reset_training_state_on_resume=True,
            )

            self.assertEqual(resumed.current_epoch, 2)
            self.assertEqual(resumed.best_selection_metric, 0.61)
            self.assertEqual(resumed.best_selection_metric_name, "anchor_tier_accuracy_mean")
            self.assertEqual(resumed.patience_counter, 0)
            self.assertEqual(resumed.global_step, 0)
            self.assertEqual(resumed.optimizer.state_dict()["state"], {})

    def test_restart_from_best_after_three_non_improving_anchor_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = DinoV3TrainingConfig(
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
                early_stopping_patience=8,
                restart_from_best_patience=3,
            )
            trainer = DinoV3Trainer(model=DummyPairwiseModel(), config=config)
            callback_metrics = iter([0.61, 0.57, 0.56, 0.55, 0.58])

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

            self.assertEqual(
                summary["history"]["anchor_tier_accuracy"],
                [0.61, 0.57, 0.56, 0.55, 0.58],
            )
            self.assertEqual(len(summary["restart_from_best_events"]), 1)
            self.assertEqual(summary["restart_from_best_events"][0]["trigger_epoch"], 4)
            self.assertEqual(summary["restart_from_best_events"][0]["best_checkpoint_name"], "checkpoint_epoch_0001.pt")
            self.assertEqual(summary["restart_from_best_events"][0]["selection_metric_name"], "anchor_tier_accuracy_mean")
            self.assertEqual(trainer.best_selection_metric, 0.61)
            self.assertEqual(trainer.patience_counter, 1)
            self.assertEqual(Path(temp_dir, "best_model.pt").readlink(), Path("checkpoint_epoch_0001.pt"))

    def test_registry_incumbent_retained_keeps_best_link_and_patience(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            best_checkpoint = str(Path(temp_dir) / "checkpoint_epoch_0001.pt")
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
                early_stopping_patience=4,
            )
            trainer = DinoV3Trainer(model=DummyPairwiseModel(), config=config)
            callback_payloads = iter(
                [
                    {
                        "report": {"metrics": {"anchor_tier_accuracy_mean": 0.5311}},
                        "registry": {
                            "selected_best_checkpoint_after_compare": best_checkpoint,
                            "selected_best_metrics_after_compare": {"anchor_tier_accuracy_mean": 0.5311},
                        },
                    },
                    {
                        "report": {"metrics": {"anchor_tier_accuracy_mean": 0.5320}},
                        "registry": {
                            "selected_best_checkpoint_after_compare": best_checkpoint,
                            "selected_best_metrics_after_compare": {"anchor_tier_accuracy_mean": 0.5311},
                        },
                    },
                ]
            )

            def callback(_path: Path, _metrics: dict[str, float]) -> dict[str, object]:
                return next(callback_payloads)

            batch = self._single_batch()
            trainer.train([batch], [batch], post_epoch_callback=callback)

            self.assertEqual(trainer.best_selection_metric, 0.5311)
            self.assertEqual(trainer.patience_counter, 1)
            self.assertEqual(Path(temp_dir, "best_model.pt").readlink(), Path("checkpoint_epoch_0001.pt"))

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

            best_path = Path(temp_dir) / "best_model.pt"
            self.assertTrue(best_path.is_symlink())
            self.assertEqual(best_path.readlink(), Path("checkpoint_epoch_0001.pt"))
            self.assertTrue(math.isfinite(trainer.best_val_loss))

    def test_best_model_link_replaces_existing_regular_file(self) -> None:
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
            checkpoint_path = Path(temp_dir) / "checkpoint_epoch_0001.pt"
            checkpoint_path.write_text("placeholder", encoding="utf-8")
            best_path = Path(temp_dir) / "best_model.pt"
            best_path.write_text("old-best", encoding="utf-8")

            trainer.update_best_checkpoint_link(checkpoint_path)

            self.assertTrue(best_path.is_symlink())
            self.assertEqual(best_path.readlink(), Path("checkpoint_epoch_0001.pt"))


if __name__ == "__main__":
    unittest.main()
