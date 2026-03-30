from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

MODELS_PATH = Path(__file__).resolve().parents[1] / "training" / "models.py"


if TORCH_AVAILABLE:
    class _FakeEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = torch.nn.ModuleList(
                [
                    torch.nn.Linear(8, 8),
                    torch.nn.Linear(8, 8),
                    torch.nn.Linear(8, 8),
                ]
            )


    class _FakeBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = types.SimpleNamespace(hidden_size=8)
            self.encoder = _FakeEncoder()
            self.layernorm = torch.nn.LayerNorm(8)

        def forward(self, pixel_values: torch.Tensor) -> types.SimpleNamespace:
            return types.SimpleNamespace(last_hidden_state=pixel_values.unsqueeze(1))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for model tests")
class ModelTests(unittest.TestCase):
    def test_selective_unfreeze_supports_encoder_layer_backbones(self) -> None:
        fake_transformers = types.SimpleNamespace(
            AutoModel=types.SimpleNamespace(from_pretrained=mock.Mock(return_value=_FakeBackbone()))
        )
        spec = importlib.util.spec_from_file_location("mirip_training_models_test", MODELS_PATH)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)

        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            assert spec and spec.loader
            spec.loader.exec_module(module)

        extractor = module.DinoV3FeatureExtractor(
            model_name="dummy-model",
            freeze_backbone=True,
            unfreeze_last_n_layers=2,
        )

        trainable_names = {
            name
            for name, param in extractor.model.named_parameters()
            if param.requires_grad
        }
        self.assertFalse(extractor.freeze_backbone)
        self.assertIn("encoder.layer.1.weight", trainable_names)
        self.assertIn("encoder.layer.2.weight", trainable_names)
        self.assertIn("layernorm.weight", trainable_names)
        self.assertNotIn("encoder.layer.0.weight", trainable_names)

        extractor.train()
        self.assertFalse(extractor.model.training)
        self.assertFalse(extractor.model.encoder.layer[0].training)
        self.assertTrue(extractor.model.encoder.layer[1].training)
        self.assertTrue(extractor.model.encoder.layer[2].training)
        self.assertTrue(extractor.model.layernorm.training)
