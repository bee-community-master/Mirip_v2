"""ID helpers."""

from __future__ import annotations

from uuid import uuid4


def new_id(prefix: str | None = None) -> str:
    """Create a stable UUID-based identifier with an optional prefix."""

    raw = uuid4().hex
    return f"{prefix}_{raw}" if prefix else raw
