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
            self.gradient_checkpointing_enabled = False
            self.inputs_require_grads_enabled = False

        def forward(self, pixel_values: torch.Tensor) -> types.SimpleNamespace:
            return types.SimpleNamespace(
                last_hidden_state=torch.stack((pixel_values, pixel_values * 2), dim=1)
            )

        def gradient_checkpointing_enable(self) -> None:
            self.gradient_checkpointing_enabled = True

        def gradient_checkpointing_disable(self) -> None:
            self.gradient_checkpointing_enabled = False

        def enable_input_require_grads(self) -> None:
            self.inputs_require_grads_enabled = True


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

    def test_cls_mean_patch_concat_doubles_feature_dimension(self) -> None:
        fake_transformers = types.SimpleNamespace(
            AutoModel=types.SimpleNamespace(from_pretrained=mock.Mock(return_value=_FakeBackbone()))
        )
        spec = importlib.util.spec_from_file_location("mirip_training_models_pool_test", MODELS_PATH)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)

        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            assert spec and spec.loader
            spec.loader.exec_module(module)

        extractor = module.DinoV3FeatureExtractor(
            model_name="dummy-model",
            freeze_backbone=True,
            feature_pool="cls_mean_patch_concat",
        )
        features = extractor(torch.arange(1, 9, dtype=torch.float32).unsqueeze(0))

        self.assertEqual(tuple(features.shape), (1, 16))
        self.assertTrue(torch.allclose(features[:, :8].norm(dim=1), torch.ones(1), atol=1e-5))
        self.assertTrue(torch.allclose(features[:, 8:].norm(dim=1), torch.ones(1), atol=1e-5))

    def test_pairwise_model_linear_head_emits_single_score(self) -> None:
        fake_transformers = types.SimpleNamespace(
            AutoModel=types.SimpleNamespace(from_pretrained=mock.Mock(return_value=_FakeBackbone()))
        )
        spec = importlib.util.spec_from_file_location("mirip_training_models_head_test", MODELS_PATH)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)

        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            assert spec and spec.loader
            spec.loader.exec_module(module)

        model = module.DinoV3PairwiseModel(
            model_name="dummy-model",
            head_type="linear",
            feature_pool="cls_mean_patch_concat",
            freeze_backbone=True,
        )
        scores = model.predict_score(torch.arange(1, 17, dtype=torch.float32).reshape(2, 8))
        self.assertEqual(tuple(scores.shape), (2, 1))

    def test_unfrozen_backbone_enables_gradient_checkpointing_and_sequential_pair_forward(self) -> None:
        fake_transformers = types.SimpleNamespace(
            AutoModel=types.SimpleNamespace(from_pretrained=mock.Mock(return_value=_FakeBackbone()))
        )
        spec = importlib.util.spec_from_file_location("mirip_training_models_gc_test", MODELS_PATH)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)

        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            assert spec and spec.loader
            spec.loader.exec_module(module)

        model = module.DinoV3PairwiseModel(
            model_name="dummy-model",
            head_type="linear",
            feature_pool="cls_mean_patch_concat",
            freeze_backbone=False,
        )
        self.assertTrue(model.feature_extractor.model.gradient_checkpointing_enabled)
        self.assertTrue(model.feature_extractor.model.inputs_require_grads_enabled)

        first = torch.ones((1, 1))
        second = torch.zeros((1, 1))
        with (
            mock.patch.object(module, "checkpoint", side_effect=lambda fn, *args, **kwargs: fn(*args)) as checkpoint_fn,
            mock.patch.object(model, "predict_score", side_effect=[first, second]) as predict_score,
        ):
            score1, score2 = model(torch.ones((1, 8)), torch.zeros((1, 8)))

        self.assertEqual(checkpoint_fn.call_count, 2)
        self.assertEqual(predict_score.call_count, 2)
        self.assertTrue(torch.equal(score1, first))
        self.assertTrue(torch.equal(score2, second))
