from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

DISTILL_ROOT = Path(__file__).resolve().parent
TRAIN_ROOT = DISTILL_ROOT.parent
REPO_ROOT = TRAIN_ROOT.parent
RAW_IMAGES_DIRNAME = "raw_images"


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_train_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (TRAIN_ROOT / candidate).resolve()


def resolve_distill_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (DISTILL_ROOT / candidate).resolve()


def save_json(path: str | Path, payload: Any) -> Path:
    target = resolve_train_path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> Path:
    target = resolve_train_path(path)
    ensure_dir(target.parent)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return target


def normalize_image_reference(image_path: str | Path) -> str | None:
    raw_value = str(image_path).strip()
    if not raw_value:
        return None
    candidate = Path(raw_value)
    parts = list(candidate.parts)
    if RAW_IMAGES_DIRNAME in parts:
        start = parts.index(RAW_IMAGES_DIRNAME)
        return str(Path(*parts[start:]))
    if candidate.suffix:
        return str(Path(RAW_IMAGES_DIRNAME) / candidate.name)
    return None


def resolve_image_path(image_root: str | Path, image_path: str | Path) -> Path:
    normalized = normalize_image_reference(image_path)
    if normalized is None:
        raise FileNotFoundError(f"Invalid image path reference: {image_path}")
    image_root_path = resolve_train_path(image_root)
    resolved = image_root_path / normalized
    if not resolved.exists():
        raise FileNotFoundError(f"Image file not found: {resolved}")
    return resolved


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def align_to_patch_multiple(value: int, patch_size: int) -> int:
    if value <= 0:
        raise ValueError("resolution must be positive")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    quotient, remainder = divmod(value, patch_size)
    if remainder == 0:
        return value
    lower = max(patch_size, quotient * patch_size)
    upper = (quotient + 1) * patch_size
    if abs(value - lower) <= abs(upper - value):
        return lower
    return upper


def deterministic_split_bucket(key: str, salt: str = "mirip-distill") -> float:
    digest = hashlib.sha1(f"{salt}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("mirip.distillation")
    if logger.handlers:
        logger.setLevel(level.upper())
        return logger
    logger.setLevel(level.upper())
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def maybe_dataclass_to_dict(payload: Any) -> Any:
    if is_dataclass(payload):
        return asdict(payload)
    return payload


def select_device(device: str | None = None) -> torch.device:
    requested = (device or "cuda").lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def resolve_precision(requested: str, device: torch.device) -> str:
    if device.type != "cuda":
        return "fp32"
    if requested == "auto":
        if torch.cuda.is_bf16_supported():
            return "bf16"
        return "fp16"
    return requested


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def format_seconds(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    clipped = tensor.detach().cpu().clamp(0.0, 1.0)
    return (clipped.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def pca_rgb_map(patch_tokens: torch.Tensor, patch_grid_hw: tuple[int, int]) -> np.ndarray:
    h, w = patch_grid_hw
    tokens = patch_tokens.detach().float().cpu().reshape(h * w, -1).numpy()
    tokens = tokens - tokens.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(tokens, full_matrices=False)
    comps = tokens @ vt[:3].T
    comps = comps.reshape(h, w, 3)
    comps -= comps.min(axis=(0, 1), keepdims=True)
    denom = np.maximum(comps.max(axis=(0, 1), keepdims=True), 1e-6)
    comps = comps / denom
    return (comps * 255.0).astype(np.uint8)


def try_enable_tf32() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}
