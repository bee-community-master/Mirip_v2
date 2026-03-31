from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

TRAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN_ROOT))

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None
DATASET_TESTS_AVAILABLE = TORCH_AVAILABLE and PIL_AVAILABLE
if DATASET_TESTS_AVAILABLE:
    from training import datasets as datasets_module
if PIL_AVAILABLE:
    from PIL import Image


@unittest.skipUnless(DATASET_TESTS_AVAILABLE, "torch and Pillow are required for dataset transform tests")
class DatasetTransformTests(unittest.TestCase):
    def test_eval_batch_builder_uses_requested_448_input_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_root = Path(temp_dir)
            raw_images = image_root / "raw_images"
            raw_images.mkdir(parents=True, exist_ok=True)
            image_path = raw_images / "sample.jpg"
            Image.new("RGB", (64, 64), color=(128, 96, 64)).save(image_path)

            with mock.patch.object(
                datasets_module,
                "load_image_processor",
                return_value=types.SimpleNamespace(image_mean=[0.5, 0.5, 0.5], image_std=[0.25, 0.25, 0.25]),
            ):
                batch = datasets_module.build_pixel_batch(
                    image_root=image_root,
                    image_paths=["raw_images/sample.jpg"],
                    model_name="dummy-model",
                    input_size=448,
                    is_train=False,
                )

        self.assertEqual(tuple(batch.shape), (1, 3, 448, 448))

    def test_pair_collator_returns_448_train_crops(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_root = Path(temp_dir)
            raw_images = image_root / "raw_images"
            raw_images.mkdir(parents=True, exist_ok=True)
            for name in ("left.jpg", "right.jpg"):
                Image.new("RGB", (96, 96), color=(140, 110, 90)).save(raw_images / name)

            collator = datasets_module.DinoPairBatchCollator(
                image_root=image_root,
                model_name="dummy-model",
                input_size=448,
                is_train=True,
            )

            with mock.patch.object(
                datasets_module,
                "load_image_processor",
                return_value=types.SimpleNamespace(image_mean=[0.5, 0.5, 0.5], image_std=[0.25, 0.25, 0.25]),
            ):
                img1, img2, labels, is_same_dept = collator(
                    [("raw_images/left.jpg", "raw_images/right.jpg", 1, 1)]
                )

        self.assertEqual(tuple(img1.shape), (1, 3, 448, 448))
        self.assertEqual(tuple(img2.shape), (1, 3, 448, 448))
        self.assertEqual(labels.tolist(), [1.0])
        self.assertEqual(is_same_dept.tolist(), [1])
