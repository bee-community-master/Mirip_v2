from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TRAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN_ROOT))

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from training import postprocess as postprocess_module


if TORCH_AVAILABLE:
    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for postprocess tests")
class PostprocessTests(unittest.TestCase):
    def test_run_postprocess_reuses_runtime_model_without_reloading_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            anchors_output = base / "anchors.pt"
            report_output = base / "report.json"
            registry_output = base / "registry.json"
            model = TinyModel()
            anchor_store = mock.Mock()
            anchor_store.save.return_value = anchors_output

            with (
                mock.patch.object(postprocess_module, "load_checkpoint_model") as load_checkpoint_model,
                mock.patch.object(postprocess_module, "build_anchor_store", return_value=anchor_store) as build_anchor_store,
                mock.patch.object(postprocess_module, "_build_evaluation_loader", return_value=object()) as build_loader,
                mock.patch.object(
                    postprocess_module,
                    "evaluate_pairwise",
                    return_value={"val_loss": 0.2, "val_accuracy": 0.7, "same_dept_accuracy": 0.71},
                ) as evaluate_pairwise,
                mock.patch.object(
                    postprocess_module,
                    "evaluate_anchor_tier_accuracy",
                    return_value={"anchor_tier_accuracy": 0.61},
                ) as evaluate_anchor_tier_accuracy,
                mock.patch.object(
                    postprocess_module,
                    "update_postprocess_registry",
                    return_value={"selected_best_checkpoint_after_compare": "checkpoint_epoch_0005.pt"},
                ) as update_postprocess_registry,
            ):
                result = postprocess_module.run_postprocess_for_checkpoint(
                    checkpoint_path="output_models/checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0005.pt",
                    pairs_val="training/data/pairs_val.csv",
                    metadata_train="training/data/metadata_train.csv",
                    metadata_eval="training/data/metadata_val.csv",
                    image_root="data",
                    anchors_output=anchors_output,
                    report_output=report_output,
                    registry_output=registry_output,
                    batch_size=16,
                    num_workers=0,
                    prefetch_factor=2,
                    persistent_workers=False,
                    device="cpu",
                    precision="fp32",
                    model=model,
                    config_dict={"model_name": "dummy-model"},
                )

            load_checkpoint_model.assert_not_called()
            build_anchor_store.assert_called_once_with(
                model=model,
                metadata_csv="training/data/metadata_train.csv",
                image_root="data",
                model_name="dummy-model",
                input_size=448,
            )
            build_loader.assert_called_once()
            evaluate_pairwise.assert_called_once()
            evaluate_anchor_tier_accuracy.assert_called_once()
            update_postprocess_registry.assert_called_once()
            self.assertEqual(result["registry"]["selected_best_checkpoint_after_compare"], "checkpoint_epoch_0005.pt")

            payload = json.loads(report_output.read_text(encoding="utf-8"))
            self.assertEqual(payload["config"]["model_name"], "dummy-model")
            self.assertEqual(payload["metrics"]["anchor_tier_accuracy"], 0.61)
            self.assertEqual(payload["anchors_output"], str(anchors_output))


if __name__ == "__main__":
    unittest.main()
