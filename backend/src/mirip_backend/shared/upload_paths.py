"""Helpers for upload path normalization."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from mirip_backend.shared.exceptions import ValidationError

_SAFE_PATH_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(filename: str) -> str:
    name = PurePosixPath(filename).name.strip()
    safe_name = _SAFE_PATH_CHARS.sub("-", name).strip(".-")
    if not safe_name:
        raise ValidationError("Filename must contain at least one safe character")
    return safe_name


def sanitize_category(category: str) -> str:
    safe_category = _SAFE_PATH_CHARS.sub("-", category.strip().lower()).strip(".-")
    if not safe_category:
        raise ValidationError("Category must contain at least one safe character")
    return safe_category
