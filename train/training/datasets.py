from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoImageProcessor

from .utils import load_rows_from_csv, resolve_staged_image_path


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


def load_rgb_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


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

    def _load_pixel_values(self, image_paths: Sequence[str]) -> torch.Tensor:
        processor = load_image_processor(self.model_name)
        images = [load_rgb_image(resolve_image_path(self.image_root, image_path)) for image_path in image_paths]
        return processor(images=images, return_tensors="pt")["pixel_values"]

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

    def __call__(self, batch: Sequence[dict[str, str]]):
        processor = load_image_processor(self.model_name)
        images = [load_rgb_image(resolve_image_path(self.image_root, row["image_path"])) for row in batch]
        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]
        return {
            "pixel_values": pixel_values,
            "tier": [row["tier"] for row in batch],
            "image_path": [row["image_path"] for row in batch],
            "post_no": [row["post_no"] for row in batch],
        }
