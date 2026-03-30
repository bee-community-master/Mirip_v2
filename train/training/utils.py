from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_IMAGES_DIRNAME = "raw_images"


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def normalize_staged_image_reference(image_path: str | Path) -> str | None:
    raw_value = str(image_path).strip()
    if not raw_value:
        return None

    candidate = Path(raw_value)
    if candidate.is_absolute():
        return None
    if candidate.suffix.lower() != ".jpg":
        return None
    return f"{RAW_IMAGES_DIRNAME}/{candidate.name}"


def resolve_staged_image_path(image_root: str | Path, image_path: str | Path) -> Path | None:
    normalized = normalize_staged_image_reference(image_path)
    if normalized is None:
        return None

    image_root_path = resolve_project_path(image_root)
    candidate = image_root_path / normalized
    if candidate.exists():
        return candidate.resolve()
    return None


def write_json(path: str | Path, payload: Any) -> Path:
    target = resolve_project_path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def read_json(path: str | Path) -> Any:
    return json.loads(resolve_project_path(path).read_text(encoding="utf-8"))


def write_rows_to_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    target = resolve_project_path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return target


def load_rows_from_csv(path: str | Path) -> list[dict[str, str]]:
    with resolve_project_path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def float_or_none(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    return float(value)


def int_or_none(value: Any) -> int | None:
    if value in (None, "", "null"):
        return None
    return int(value)
