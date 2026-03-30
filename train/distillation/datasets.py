from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image
import torch
from torch.utils.data import Dataset, IterableDataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from config import DistillationExperimentConfig
from utils import (
    deterministic_split_bucket,
    normalize_image_reference,
    resolve_image_path,
    resolve_train_path,
)

DEFAULT_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class DistillRecord:
    """Normalized image-level record used by training and evaluation pipelines."""

    sample_id: str
    image_path: str
    split: str
    metadata_path: str
    post_no: str
    tier: str
    department: str
    normalized_dept: str
    anchor_group: str
    university: str
    work_type: str

    @property
    def retrieval_target(self) -> str:
        if self.anchor_group:
            return self.anchor_group
        return self.department or self.normalized_dept


class RecordSource(ABC):
    """Abstract interface for loading distillation samples from different storage layouts."""

    @abstractmethod
    def load_records(self) -> list[DistillRecord]:
        raise NotImplementedError


def _load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with resolve_train_path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _build_prepared_split_lookup(train_csv: str | Path, val_csv: str | Path) -> dict[str, tuple[str, dict[str, str]]]:
    lookup: dict[str, tuple[str, dict[str, str]]] = {}
    for split, csv_path in (("train", train_csv), ("val", val_csv)):
        resolved = resolve_train_path(csv_path)
        if not resolved.exists():
            continue
        for row in _load_csv_rows(resolved):
            image_path = normalize_image_reference(row.get("image_path", ""))
            if image_path:
                lookup[image_path] = (split, row)
    return lookup


def _resolve_hash_split(bucket: float, *, train_ratio: float, val_ratio: float) -> str | None:
    if bucket < train_ratio:
        return "train"
    if bucket < (train_ratio + val_ratio):
        return "val"
    return None


def _resolve_webdataset_url_pattern(url_pattern: str, split: str) -> str:
    if "{split}" not in url_pattern:
        raise ValueError(
            "webdataset_url_pattern must include a '{split}' placeholder to avoid train/val leakage"
        )
    return url_pattern.format(split=split)


def _resolve_metadata_text(
    payload: dict[str, Any],
    prepared_row: dict[str, str] | None,
    *keys: str,
) -> str:
    """Returns the first non-empty metadata value from JSON payload or prepared split row."""

    for key in keys:
        value = payload.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
        if prepared_row is not None:
            prepared_value = prepared_row.get(key)
            if prepared_value is not None and str(prepared_value).strip():
                return str(prepared_value).strip()
    return ""


class MiripStagedSource(RecordSource):
    """Loads image records from Mirip staged metadata JSON files."""

    def __init__(self, config: DistillationExperimentConfig) -> None:
        self.config = config
        self.metadata_dir = resolve_train_path(config.paths.metadata_dir)
        self.prepared_lookup = {}
        if config.data.prepared_split_preferred:
            self.prepared_lookup = _build_prepared_split_lookup(
                config.paths.prepared_train_csv,
                config.paths.prepared_val_csv,
            )

    def _resolve_split(self, image_path: str, payload: dict[str, Any]) -> tuple[str | None, dict[str, str] | None]:
        if image_path in self.prepared_lookup:
            split, row = self.prepared_lookup[image_path]
            return split, row
        split_key = str(payload.get("post_no") or image_path)
        bucket = deterministic_split_bucket(split_key, salt=self.config.data.split_salt)
        return (
            _resolve_hash_split(
                bucket,
                train_ratio=self.config.data.train_ratio,
                val_ratio=self.config.data.val_ratio,
            ),
            None,
        )

    def _build_record(
        self,
        *,
        payload: dict[str, Any],
        metadata_path: Path,
        image_path: str,
        image_index: int,
        split: str,
        prepared_row: dict[str, str] | None,
    ) -> DistillRecord:
        department = _resolve_metadata_text(payload, prepared_row, "department", "normalized_dept")
        post_no = str(payload.get("post_no") or metadata_path.stem)
        return DistillRecord(
            sample_id=f"{post_no}_{image_index}",
            image_path=image_path,
            split=split,
            metadata_path=str(metadata_path.relative_to(self.metadata_dir.parent)),
            post_no=post_no,
            tier=_resolve_metadata_text(payload, prepared_row, "tier"),
            department=department,
            normalized_dept=_resolve_metadata_text(payload, prepared_row, "normalized_dept", "department"),
            anchor_group=_resolve_metadata_text(payload, prepared_row, "anchor_group"),
            university=_resolve_metadata_text(payload, prepared_row, "university"),
            work_type=_resolve_metadata_text(payload, prepared_row, "work_type"),
        )

    def load_records(self) -> list[DistillRecord]:
        records: list[DistillRecord] = []
        for metadata_path in sorted(self.metadata_dir.glob("*.json")):
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            image_values = payload.get("images") or []
            if not isinstance(image_values, list):
                continue
            for index, raw_image_path in enumerate(image_values):
                normalized_path = normalize_image_reference(raw_image_path)
                if normalized_path is None:
                    continue
                split, prepared_row = self._resolve_split(normalized_path, payload)
                if split is None:
                    continue
                records.append(
                    self._build_record(
                        payload=payload,
                        metadata_path=metadata_path,
                        image_path=normalized_path,
                        image_index=index,
                        split=split,
                        prepared_row=prepared_row,
                    )
                )
        return records


class ImageFolderSource(RecordSource):
    """Loads image records from a plain image folder tree."""

    def __init__(self, root: str | Path, *, train_ratio: float, val_ratio: float, split_salt: str) -> None:
        self.root = resolve_train_path(root)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.split_salt = split_salt

    def load_records(self) -> list[DistillRecord]:
        records: list[DistillRecord] = []
        for image_path in sorted(self.root.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in DEFAULT_IMAGE_EXTENSIONS:
                continue
            relative = image_path.relative_to(self.root)
            bucket = deterministic_split_bucket(str(relative), salt=self.split_salt)
            split = _resolve_hash_split(bucket, train_ratio=self.train_ratio, val_ratio=self.val_ratio)
            if split is None:
                continue
            records.append(
                DistillRecord(
                    sample_id=str(relative).replace("/", "_"),
                    image_path=str(relative),
                    split=split,
                    metadata_path="",
                    post_no="",
                    tier="",
                    department="",
                    normalized_dept="",
                    anchor_group="",
                    university="",
                    work_type="",
                )
            )
        return records


def build_transforms(
    *,
    resolution: int,
    mean: Sequence[float],
    std: Sequence[float],
    is_train: bool,
    augmentation,
) -> transforms.Compose:
    """Builds weak augmentation transforms suitable for feature distillation."""

    interpolation = InterpolationMode.BICUBIC
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(resolution, scale=(augmentation.min_scale, 1.0), interpolation=interpolation),
                transforms.RandomHorizontalFlip(p=augmentation.horizontal_flip_prob),
                transforms.ColorJitter(
                    brightness=augmentation.color_jitter_brightness,
                    contrast=augmentation.color_jitter_contrast,
                    saturation=augmentation.color_jitter_saturation,
                    hue=augmentation.color_jitter_hue,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    resize_size = int(round(resolution * 1.1))
    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


class DistillationImageDataset(Dataset):
    """Map-style dataset that resolves image paths and applies shared transforms."""

    def __init__(
        self,
        records: Sequence[DistillRecord],
        *,
        image_root: str | Path,
        transform: transforms.Compose,
    ) -> None:
        self.records = list(records)
        self.image_root = image_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_image_path(self, image_path: str) -> Path:
        image_root = resolve_train_path(self.image_root)
        direct_path = image_root / image_path
        if direct_path.exists():
            return direct_path
        return resolve_image_path(self.image_root, image_path)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = self._resolve_image_path(record.image_path)
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
        pixel_values = self.transform(rgb)
        return {
            "pixel_values": pixel_values,
            "record": record,
        }


class WebDatasetSource(IterableDataset):
    """Iterable WebDataset adapter for split-specific tar shard streams."""

    def __init__(
        self,
        *,
        url_pattern: str,
        split: str,
        transform: transforms.Compose,
        limit: int | None = None,
    ) -> None:
        super().__init__()
        self.url_pattern = url_pattern
        self.split = split
        self.transform = transform
        self.limit = limit

    def __len__(self) -> int:
        if self.limit is None:
            raise TypeError("WebDatasetSource length requires an explicit limit")
        return self.limit

    def __iter__(self):
        try:
            import webdataset as wds
        except ImportError as exc:
            raise ImportError("webdataset is required for source_type=webdataset") from exc

        dataset = wds.WebDataset(self.url_pattern, shardshuffle=self.split == "train").decode("pil")
        if self.split == "train":
            dataset = dataset.shuffle(512)

        count = 0
        for sample in dataset:
            if self.limit is not None and count >= self.limit:
                break
            image = None
            for key in ("jpg", "jpeg", "png", "webp", "image"):
                if key in sample:
                    image = sample[key]
                    break
            if image is None:
                continue
            if not isinstance(image, Image.Image):
                image = Image.open(image).convert("RGB")
            else:
                image = image.convert("RGB")
            count += 1
            sample_id = str(sample.get("__key__", f"{self.split}_{count:06d}"))
            record = DistillRecord(
                sample_id=sample_id,
                image_path=sample_id,
                split=self.split,
                metadata_path="",
                post_no="",
                tier=str(sample.get("tier", "")),
                department=str(sample.get("department", "")),
                normalized_dept=str(sample.get("normalized_dept", "")),
                anchor_group=str(sample.get("anchor_group", "")),
                university=str(sample.get("university", "")),
                work_type=str(sample.get("work_type", "")),
            )
            yield {"pixel_values": self.transform(image), "record": record}


class DistillationBatchCollator:
    """Collates normalized image tensors and keeps record metadata alongside the batch."""

    def __call__(self, batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
        pixel_values = [item["pixel_values"] for item in batch]
        records = [item["record"] for item in batch]
        return {
            "pixel_values": torch.stack(pixel_values),
            "records": records,
        }


def _apply_limits(records: Iterable[DistillRecord], limit: int | None) -> list[DistillRecord]:
    items = list(records)
    if limit is not None:
        return items[:limit]
    return items


def _build_stage_transforms(
    *,
    config: DistillationExperimentConfig,
    resolution: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> tuple[transforms.Compose, transforms.Compose]:
    return (
        build_transforms(
            resolution=resolution,
            mean=mean,
            std=std,
            is_train=True,
            augmentation=config.data.augmentation,
        ),
        build_transforms(
            resolution=resolution,
            mean=mean,
            std=std,
            is_train=False,
            augmentation=config.data.augmentation,
        ),
    )


def _build_webdataset_stage_datasets(
    *,
    config: DistillationExperimentConfig,
    train_transform: transforms.Compose,
    val_transform: transforms.Compose,
) -> tuple[Dataset, Dataset]:
    if not config.data.webdataset_url_pattern:
        raise ValueError("webdataset_url_pattern must be set for source_type=webdataset")
    if config.data.train_limit is None or config.data.val_limit is None:
        raise ValueError("train_limit and val_limit must be set for source_type=webdataset")
    return (
        WebDatasetSource(
            url_pattern=_resolve_webdataset_url_pattern(config.data.webdataset_url_pattern, "train"),
            split="train",
            transform=train_transform,
            limit=config.data.train_limit,
        ),
        WebDatasetSource(
            url_pattern=_resolve_webdataset_url_pattern(config.data.webdataset_url_pattern, "val"),
            split="val",
            transform=val_transform,
            limit=config.data.val_limit,
        ),
    )


def _split_stage_records(
    records: Sequence[DistillRecord],
    *,
    train_limit: int | None,
    val_limit: int | None,
) -> tuple[list[DistillRecord], list[DistillRecord]]:
    train_records = _apply_limits((record for record in records if record.split == "train"), train_limit)
    val_records = _apply_limits((record for record in records if record.split == "val"), val_limit)
    if not train_records:
        raise RuntimeError("No training records available for distillation")
    if not val_records:
        raise RuntimeError("No validation records available for distillation")
    return train_records, val_records


def build_record_source(config: DistillationExperimentConfig) -> RecordSource:
    """Builds a record source for the configured dataset backend."""

    if config.data.source_type == "mirip_staged":
        return MiripStagedSource(config)
    if config.data.source_type == "imagefolder":
        return ImageFolderSource(
            config.paths.image_root,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            split_salt=config.data.split_salt,
        )
    raise ValueError(f"Unsupported record source: {config.data.source_type}")


def build_stage_datasets(
    config: DistillationExperimentConfig,
    *,
    resolution: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> tuple[Dataset, Dataset]:
    """Builds train/val datasets for a specific training stage resolution."""

    train_transform, val_transform = _build_stage_transforms(
        config=config,
        resolution=resolution,
        mean=mean,
        std=std,
    )

    if config.data.source_type == "webdataset":
        return _build_webdataset_stage_datasets(
            config=config,
            train_transform=train_transform,
            val_transform=val_transform,
        )

    source = build_record_source(config)
    records = source.load_records()
    train_records, val_records = _split_stage_records(
        records,
        train_limit=config.data.train_limit,
        val_limit=config.data.val_limit,
    )
    train_dataset = DistillationImageDataset(
        train_records,
        image_root=config.paths.image_root,
        transform=train_transform,
    )
    val_dataset = DistillationImageDataset(
        val_records,
        image_root=config.paths.image_root,
        transform=val_transform,
    )
    return train_dataset, val_dataset


def collate_records(records: Sequence[DistillRecord]) -> dict[str, list[str]]:
    """Converts record metadata into a logging-friendly columnar dict."""

    return {
        "sample_id": [record.sample_id for record in records],
        "image_path": [record.image_path for record in records],
        "post_no": [record.post_no for record in records],
        "tier": [record.tier for record in records],
        "department": [record.department for record in records],
        "normalized_dept": [record.normalized_dept for record in records],
        "anchor_group": [record.anchor_group for record in records],
        "university": [record.university for record in records],
        "work_type": [record.work_type for record in records],
    }
