from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from transformers import AutoImageProcessor

from .utils import load_rows_from_csv, resolve_staged_image_path

RESIZE_PADDING = 32
_BICUBIC = getattr(getattr(Image, "Resampling", Image), "BICUBIC")


def resolve_image_path(image_root: str | Path, image_path: str) -> Path:
    resolved = resolve_staged_image_path(image_root, image_path)
    if resolved is None:
        raise FileNotFoundError(f"Unable to resolve staged image path: {image_path}")
    return resolved


def load_metadata_rows(path: str | Path) -> list[dict[str, str]]:
    return load_rows_from_csv(path)


@lru_cache(maxsize=8)
def load_image_processor(model_name: str):
    return AutoImageProcessor.from_pretrained(model_name)


def _resolve_image_mean_std(model_name: str) -> tuple[list[float], list[float]]:
    processor = load_image_processor(model_name)
    image_mean = getattr(processor, "image_mean", None) or [0.485, 0.456, 0.406]
    image_std = getattr(processor, "image_std", None) or [0.229, 0.224, 0.225]
    return list(image_mean), list(image_std)


def load_rgb_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _resize_square(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), resample=_BICUBIC)


def _random_crop(image: Image.Image, size: int, rng: np.random.Generator) -> Image.Image:
    if image.width == size and image.height == size:
        return image
    max_left = max(image.width - size, 0)
    max_top = max(image.height - size, 0)
    left = 0 if max_left == 0 else int(rng.integers(0, max_left + 1))
    top = 0 if max_top == 0 else int(rng.integers(0, max_top + 1))
    return image.crop((left, top, left + size, top + size))


def _apply_color_jitter(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    brightness = 1.0 + float(rng.uniform(-0.2, 0.2))
    contrast = 1.0 + float(rng.uniform(-0.2, 0.2))
    saturation = 1.0 + float(rng.uniform(-0.1, 0.1))
    image = ImageEnhance.Brightness(image).enhance(max(brightness, 0.0))
    image = ImageEnhance.Contrast(image).enhance(max(contrast, 0.0))
    image = ImageEnhance.Color(image).enhance(max(saturation, 0.0))
    return image


def preprocess_rgb_image(
    image: Image.Image,
    *,
    model_name: str,
    input_size: int,
    is_train: bool,
) -> torch.Tensor:
    image_mean, image_std = _resolve_image_mean_std(model_name)
    if is_train:
        rng = np.random.default_rng()
        image = _resize_square(image, input_size + RESIZE_PADDING)
        image = _random_crop(image, input_size, rng)
        if float(rng.random()) < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image = _apply_color_jitter(image, rng)
    else:
        image = _resize_square(image, input_size)

    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    mean = torch.tensor(image_mean, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(image_std, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def build_pixel_batch(
    *,
    image_root: str | Path,
    image_paths: Sequence[str],
    model_name: str,
    input_size: int,
    is_train: bool,
) -> torch.Tensor:
    tensors = [
        preprocess_rgb_image(
            load_rgb_image(resolve_image_path(image_root, image_path)),
            model_name=model_name,
            input_size=input_size,
            is_train=is_train,
        )
        for image_path in image_paths
    ]
    return torch.stack(tensors, dim=0)


class DinoPairDataset(Dataset):
    def __init__(
        self,
        pairs_csv: str | Path,
    ) -> None:
        self.rows = load_metadata_rows(pairs_csv)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        return (
            row["image_path_1"],
            row["image_path_2"],
            int(row["label"]),
            1 if row.get("pair_type", "") == "same_dept" else 0,
        )


@dataclass
class DinoPairBatchCollator:
    image_root: str | Path
    model_name: str
    input_size: int
    is_train: bool

    def _load_pixel_values(self, image_paths: Sequence[str]) -> torch.Tensor:
        return build_pixel_batch(
            image_root=self.image_root,
            image_paths=image_paths,
            model_name=self.model_name,
            input_size=self.input_size,
            is_train=self.is_train,
        )

    def __call__(self, batch: Sequence[tuple[str, str, int, int]]):
        image_paths_1, image_paths_2, labels, is_same_dept = zip(*batch)
        merged_pixel_values = self._load_pixel_values((*image_paths_1, *image_paths_2))
        split_index = len(image_paths_1)
        return (
            merged_pixel_values[:split_index],
            merged_pixel_values[split_index:],
            torch.tensor(labels, dtype=torch.float32),
            torch.tensor(is_same_dept, dtype=torch.long),
        )


class DinoMetadataDataset(Dataset):
    def __init__(
        self,
        metadata_csv: str | Path,
    ) -> None:
        self.rows = load_metadata_rows(metadata_csv)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        return {
            "tier": row["tier"],
            "image_path": row["image_path"],
            "post_no": row.get("post_no", ""),
        }


@dataclass
class DinoMetadataBatchCollator:
    image_root: str | Path
    model_name: str
    input_size: int

    def __call__(self, batch: Sequence[dict[str, str]]):
        pixel_values = build_pixel_batch(
            image_root=self.image_root,
            image_paths=[row["image_path"] for row in batch],
            model_name=self.model_name,
            input_size=self.input_size,
            is_train=False,
        )
        return {
            "pixel_values": pixel_values,
            "tier": [row["tier"] for row in batch],
            "image_path": [row["image_path"] for row in batch],
            "post_no": [row["post_no"] for row in batch],
        }
