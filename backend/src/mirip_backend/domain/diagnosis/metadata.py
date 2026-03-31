"""Diagnosis job metadata helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

INPUT_OBJECT_NAMES_METADATA_KEY = "input_object_names"
REQUESTED_UPLOAD_COUNT_METADATA_KEY = "requested_upload_count"


def build_diagnosis_job_metadata(*, upload_object_names: Sequence[str]) -> dict[str, Any]:
    object_names = [str(object_name) for object_name in upload_object_names]
    return {
        REQUESTED_UPLOAD_COUNT_METADATA_KEY: str(len(object_names)),
        INPUT_OBJECT_NAMES_METADATA_KEY: object_names,
    }
