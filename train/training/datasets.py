from __future__ import annotations

from pathlib import Path
from typing import Any

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


class DinoPairDataset(Dataset):
    def __init__(
        self,
        pairs_csv: str | Path,
        image_root: str | Path,
        model_name: str,
    ) -> None:
        self.image_root = image_root
        self.model_name = model_name
        self._processor = None
        self.rows = load_metadata_rows(pairs_csv)

    def _get_processor(self):
        if self._processor is None:
            self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        return self._processor

    def _load_pixel_values(self, image_path: str) -> torch.Tensor:
        path = resolve_image_path(self.image_root, image_path)
        image = Image.open(path).convert("RGB")
        processor = self._get_processor()
        pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]
        return pixel_values.squeeze(0)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        img1 = self._load_pixel_values(row["image_path_1"])
        img2 = self._load_pixel_values(row["image_path_2"])
        label = torch.tensor(int(row["label"]), dtype=torch.float32)
        is_same_dept = torch.tensor(
            1 if row.get("pair_type", "") == "same_dept" else 0,
            dtype=torch.long,
        )
        return img1, img2, label, is_same_dept


class DinoMetadataDataset(Dataset):
    def __init__(
        self,
        metadata_csv: str | Path,
        image_root: str | Path,
        model_name: str,
    ) -> None:
        self.rows = load_metadata_rows(metadata_csv)
        self.image_root = image_root
        self.model_name = model_name
        self._processor = None

    def _get_processor(self):
        if self._processor is None:
            self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        return self._processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = resolve_image_path(self.image_root, row["image_path"])
        image = Image.open(path).convert("RGB")
        pixel_values = self._get_processor()(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        return {
            "pixel_values": pixel_values,
            "tier": row["tier"],
            "image_path": row["image_path"],
            "post_no": row.get("post_no", ""),
        }
