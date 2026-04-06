from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path
from unittest import mock

TRAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN_ROOT))

try:
    import torch
    from training.anchors import _aggregate_anchor_bootstrap_results, _select_anchor_rows, build_anchor_store
except ModuleNotFoundError as exc:  # pragma: no cover - exercised via skip path on CI without torch
    _IMPORT_ERROR = exc
    torch = None
    _aggregate_anchor_bootstrap_results = None
    _select_anchor_rows = None
    build_anchor_store = None
else:
    _IMPORT_ERROR = None


@unittest.skipIf(_IMPORT_ERROR is not None, f"anchor bootstrap tests require optional dependency: {_IMPORT_ERROR}")
class AnchorBootstrapTests(unittest.TestCase):
    def test_build_anchor_store_uses_requested_precision_autocast(self) -> None:
        class DummyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))

            def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
                return pixel_values

            def project_features(self, features: torch.Tensor) -> torch.Tensor:
                return features

        model = DummyModel()
        autocast_calls: list[tuple[str, str]] = []

        @mock.patch("training.anchors._evaluation_autocast")
        @mock.patch("training.anchors.preprocess_rgb_image", return_value=torch.ones(1, dtype=torch.float32))
        @mock.patch("training.anchors.load_rgb_image", return_value=object())
        @mock.patch("training.anchors.resolve_image_path", return_value=Path("data/sample.jpg"))
        @mock.patch("training.anchors.load_metadata_rows")
        def run_case(
            load_metadata_rows: mock.Mock,
            _resolve_image_path: mock.Mock,
            _load_rgb_image: mock.Mock,
            _preprocess_rgb_image: mock.Mock,
            evaluation_autocast: mock.Mock,
        ) -> None:
            load_metadata_rows.return_value = [
                {"tier": "S", "image_path": "sample.jpg", "anchor_group": "g1"},
            ]
            evaluation_autocast.side_effect = lambda device, precision: (
                autocast_calls.append((device.type, precision)) or mock.MagicMock()
            )
            build_anchor_store(
                model=model,
                metadata_csv="training/data/metadata_train.csv",
                image_root="data",
                model_name="dummy-model",
                input_size=144,
                n_per_tier=1,
                precision="bf16",
            )

        run_case()
        self.assertEqual(autocast_calls, [("cpu", "fp32")])

    def test_group_balanced_anchor_selection_prefers_distinct_anchor_groups(self) -> None:
        rng = random.Random(42)
        tier_rows = [
            {"image_path": "1.jpg", "anchor_group": "g1"},
            {"image_path": "2.jpg", "anchor_group": "g1"},
            {"image_path": "3.jpg", "anchor_group": "g2"},
            {"image_path": "4.jpg", "anchor_group": "g2"},
            {"image_path": "5.jpg", "anchor_group": "g3"},
        ]

        selected = _select_anchor_rows(
            tier_rows,
            rng=rng,
            n_per_tier=3,
            group_balanced=True,
        )

        self.assertEqual(len(selected), 3)
        self.assertEqual(len({row["anchor_group"] for row in selected}), 3)

    def test_bootstrap_aggregation_returns_mean_std_and_per_tier_mean(self) -> None:
        aggregated = _aggregate_anchor_bootstrap_results(
            [
                {
                    "anchor_tier_accuracy": 0.50,
                    "anchor_tier_total": 10,
                    "anchor_tier_per_tier": {
                        "A": {"accuracy": 0.4, "total": 5, "correct": 2},
                        "B": {"accuracy": 0.6, "total": 5, "correct": 3},
                    },
                },
                {
                    "anchor_tier_accuracy": 0.70,
                    "anchor_tier_total": 10,
                    "anchor_tier_per_tier": {
                        "A": {"accuracy": 0.6, "total": 5, "correct": 3},
                        "B": {"accuracy": 0.8, "total": 5, "correct": 4},
                    },
                },
            ]
        )

        self.assertAlmostEqual(aggregated["anchor_tier_accuracy_mean"], 0.60)
        self.assertAlmostEqual(aggregated["anchor_tier_accuracy_std"], 0.10)
        self.assertEqual(aggregated["anchor_tier_total"], 10)
        self.assertEqual(aggregated["anchor_tier_correct"], 6)
        self.assertAlmostEqual(aggregated["anchor_tier_per_tier_mean"]["A"]["accuracy"], 0.50)
        self.assertAlmostEqual(aggregated["anchor_tier_per_tier_mean"]["B"]["accuracy"], 0.70)


if __name__ == "__main__":
    unittest.main()
