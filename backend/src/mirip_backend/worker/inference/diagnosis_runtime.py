"""CPU ONNX diagnosis runtime helpers."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from mirip_backend.worker.inference.model_bundle import MaterializedModelBundle

DEFAULT_IMAGE_MEAN = (0.485, 0.456, 0.406)
DEFAULT_IMAGE_STD = (0.229, 0.224, 0.225)
DEFAULT_IMAGE_SIZE = 518
TIER_ORDER = ("S", "A", "B", "C")


def _clamp_score(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 1)


def _resolve_image_size(config: dict[str, Any], fallback: int) -> int:
    for key in ("crop_size", "size"):
        candidate = config.get(key)
        if isinstance(candidate, int):
            return int(candidate)
        if isinstance(candidate, dict):
            if "height" in candidate and "width" in candidate:
                return int(max(candidate["height"], candidate["width"]))
            for nested_key in ("shortest_edge", "longest_edge"):
                if nested_key in candidate:
                    return int(candidate[nested_key])
    return int(fallback)


def _resolve_resize_config(config: dict[str, Any], fallback: int) -> dict[str, int]:
    candidate = config.get("size")
    if isinstance(candidate, int):
        return {"shortest_edge": int(candidate)}
    if isinstance(candidate, dict):
        if "height" in candidate and "width" in candidate:
            return {
                "height": int(candidate["height"]),
                "width": int(candidate["width"]),
            }
        for key in ("shortest_edge", "longest_edge"):
            value = candidate.get(key)
            if value is not None:
                return {key: int(value)}
    return {"height": int(fallback), "width": int(fallback)}


def _resolve_crop_size(candidate: Any) -> tuple[int, int] | None:
    if isinstance(candidate, int):
        size = int(candidate)
        return (size, size)
    if isinstance(candidate, dict) and "height" in candidate and "width" in candidate:
        return (int(candidate["height"]), int(candidate["width"]))
    return None


def _resolve_resample(candidate: Any) -> Image.Resampling:
    try:
        if isinstance(candidate, str):
            return Image.Resampling[candidate.upper()]
        if candidate is not None:
            return Image.Resampling(int(candidate))
    except (KeyError, ValueError):
        pass
    return Image.Resampling.BICUBIC


@dataclass(slots=True, frozen=True)
class ImagePreprocessor:
    image_size: int
    resize_config: dict[str, int]
    crop_size: tuple[int, int] | None
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    do_resize: bool
    do_center_crop: bool
    resample: Image.Resampling
    do_rescale: bool
    rescale_factor: float
    do_normalize: bool

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        fallback_image_size: int = DEFAULT_IMAGE_SIZE,
    ) -> ImagePreprocessor:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        image_mean = tuple(float(value) for value in config.get("image_mean", DEFAULT_IMAGE_MEAN))
        image_std = tuple(float(value) for value in config.get("image_std", DEFAULT_IMAGE_STD))
        return cls(
            image_size=_resolve_image_size(config, fallback_image_size),
            resize_config=_resolve_resize_config(config, fallback_image_size),
            crop_size=_resolve_crop_size(config.get("crop_size")),
            image_mean=(image_mean[0], image_mean[1], image_mean[2]),
            image_std=(image_std[0], image_std[1], image_std[2]),
            do_resize=bool(config.get("do_resize", True)),
            do_center_crop=bool(config.get("do_center_crop", "crop_size" in config)),
            resample=_resolve_resample(config.get("resample")),
            do_rescale=bool(config.get("do_rescale", True)),
            rescale_factor=float(config.get("rescale_factor", 1.0 / 255.0)),
            do_normalize=bool(config.get("do_normalize", True)),
        )

    def preprocess_bytes(self, image_bytes: bytes) -> np.ndarray:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if self.do_resize:
            image = self._resize_image(image)
        if self.do_center_crop and self.crop_size is not None:
            image = self._center_crop_image(image)
        array = np.asarray(image, dtype=np.float32)
        if self.do_rescale:
            array *= self.rescale_factor
        if self.do_normalize:
            mean = np.asarray(self.image_mean, dtype=np.float32)
            std = np.asarray(self.image_std, dtype=np.float32)
            array = (array - mean) / std
        return np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32)

    def _resize_image(self, image: Image.Image) -> Image.Image:
        if "height" in self.resize_config and "width" in self.resize_config:
            return image.resize(
                (self.resize_config["width"], self.resize_config["height"]),
                resample=self.resample,
            )

        width, height = image.size
        if "shortest_edge" in self.resize_config:
            scale = self.resize_config["shortest_edge"] / float(min(width, height))
        else:
            scale = self.resize_config["longest_edge"] / float(max(width, height))
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        return image.resize((resized_width, resized_height), resample=self.resample)

    def _center_crop_image(self, image: Image.Image) -> Image.Image:
        if self.crop_size is None:
            return image
        target_height, target_width = self.crop_size
        width, height = image.size
        left = max((width - target_width) // 2, 0)
        top = max((height - target_height) // 2, 0)
        return image.crop((left, top, left + target_width, top + target_height))


@dataclass(slots=True, frozen=True)
class DiagnosisHeadArtifact:
    schema_version: str
    model_name: str
    feature_dim: int
    projector_hidden_dim: int
    projector_output_dim: int
    dropout: float
    projector_state_dict: dict[str, Any]
    score_head_state_dict: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> DiagnosisHeadArtifact:
        payload = torch.load(Path(path), map_location="cpu")
        return cls(
            schema_version=str(payload["schema_version"]),
            model_name=str(payload["model_name"]),
            feature_dim=int(payload["feature_dim"]),
            projector_hidden_dim=int(payload["projector_hidden_dim"]),
            projector_output_dim=int(payload["projector_output_dim"]),
            dropout=float(payload.get("dropout", 0.0)),
            projector_state_dict=dict(payload["projector_state_dict"]),
            score_head_state_dict=dict(payload["score_head_state_dict"]),
        )


class DiagnosisScoringHead(nn.Module):
    def __init__(self, artifact: DiagnosisHeadArtifact) -> None:
        super().__init__()
        self.feature_dim = artifact.feature_dim
        self.projector_output_dim = artifact.projector_output_dim
        dropout = artifact.dropout
        self.projector = nn.Sequential(
            nn.Linear(artifact.feature_dim, artifact.projector_hidden_dim),
            nn.LayerNorm(artifact.projector_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(artifact.projector_hidden_dim, artifact.projector_output_dim),
            nn.LayerNorm(artifact.projector_output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.score_head = nn.Sequential(
            nn.Linear(artifact.projector_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.projector.load_state_dict(artifact.projector_state_dict)
        self.score_head.load_state_dict(artifact.score_head_state_dict)
        self.eval()

    @torch.inference_mode()
    def project_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.projector(features.float())

    @torch.inference_mode()
    def score_projected(self, projected_features: torch.Tensor) -> torch.Tensor:
        return self.score_head(projected_features.float())


@dataclass(slots=True)
class DiagnosisRuntimeResult:
    tier: str
    confidence: float
    win_rates: dict[str, float]
    raw_score: float
    scores: dict[str, float]


@dataclass(slots=True)
class DiagnosisBundleRuntime:
    preprocessor: ImagePreprocessor
    scoring_head: DiagnosisScoringHead
    anchors: dict[str, torch.Tensor]

    @classmethod
    def load(cls, bundle: MaterializedModelBundle) -> DiagnosisBundleRuntime:
        manifest = bundle.manifest
        preprocessor = ImagePreprocessor.load(
            manifest.preprocessor_path(bundle.local_dir),
            fallback_image_size=manifest.image_size or DEFAULT_IMAGE_SIZE,
        )
        diagnosis_head = DiagnosisHeadArtifact.load(
            manifest.extra_path(bundle.local_dir, "diagnosis_head")
        )
        if manifest.model_name and diagnosis_head.model_name != manifest.model_name:
            raise RuntimeError(
                "Diagnosis head model_name does not match the serving bundle manifest"
            )
        scoring_head = DiagnosisScoringHead(diagnosis_head)
        anchors_payload = torch.load(
            manifest.extra_path(bundle.local_dir, "anchors"),
            map_location="cpu",
        )
        anchors: dict[str, torch.Tensor] = {}
        for tier, features in dict(anchors_payload.get("features", {})).items():
            if not torch.is_tensor(features):
                raise RuntimeError(f"Anchor tier '{tier}' must be stored as a tensor")
            features = features.float()
            if features.ndim != 2 or features.shape[1] != diagnosis_head.projector_output_dim:
                raise RuntimeError(
                    f"Anchor tier '{tier}' shape must be [N, {diagnosis_head.projector_output_dim}]"
                )
            anchors[str(tier)] = features
        if not anchors:
            raise RuntimeError("Diagnosis runtime requires at least one anchor tier")
        return cls(
            preprocessor=preprocessor,
            scoring_head=scoring_head,
            anchors=anchors,
        )

    def evaluate_image(self, *, session: Any, image_bytes: bytes) -> DiagnosisRuntimeResult:
        encoder_features = self._run_encoder(session=session, image_bytes=image_bytes)
        feature_tensor = torch.from_numpy(encoder_features).float()
        if feature_tensor.ndim != 2 or feature_tensor.shape[1] != self.scoring_head.feature_dim:
            raise RuntimeError(
                "Encoder output shape must be "
                f"[N, {self.scoring_head.feature_dim}], got {tuple(feature_tensor.shape)}"
            )
        projected = self.scoring_head.project_features(feature_tensor)
        raw_score = float(self.scoring_head.score_projected(projected).squeeze().item())
        tier, confidence, win_rates = self._rank_projected_feature(projected)
        scores = self._derive_legacy_scores(projected.squeeze(0), win_rates)
        return DiagnosisRuntimeResult(
            tier=tier,
            confidence=round(confidence, 4),
            win_rates=win_rates,
            raw_score=raw_score,
            scores=scores,
        )

    def _run_encoder(self, *, session: Any, image_bytes: bytes) -> np.ndarray:
        pixel_values = self.preprocessor.preprocess_bytes(image_bytes)
        outputs = session.run(None, {"pixel_values": pixel_values})
        if not outputs:
            raise RuntimeError("ONNX encoder returned no outputs")
        return np.asarray(outputs[0], dtype=np.float32)

    def _rank_projected_feature(
        self,
        projected_feature: torch.Tensor,
    ) -> tuple[str, float, dict[str, float]]:
        input_score = float(self.scoring_head.score_projected(projected_feature).squeeze().item())
        win_rates: dict[str, float] = {}

        for tier in TIER_ORDER:
            anchor_features = self.anchors.get(tier)
            if anchor_features is None:
                continue
            anchor_scores = self.scoring_head.score_projected(anchor_features).squeeze(-1)
            wins = int((input_score > anchor_scores).sum().item())
            win_rates[tier] = wins / max(anchor_features.shape[0], 1)

        if win_rates.get("S", 0.0) >= 0.5:
            return "S", win_rates["S"], win_rates
        if win_rates.get("A", 0.0) >= 0.5:
            return "A", win_rates["A"], win_rates
        if win_rates.get("B", 0.0) >= 0.5:
            return "B", win_rates["B"], win_rates
        return "C", 1.0 - win_rates.get("C", 0.0), win_rates

    def _derive_legacy_scores(
        self,
        projected_feature: torch.Tensor,
        win_rates: dict[str, float],
    ) -> dict[str, float]:
        values = projected_feature.detach().cpu().numpy().astype(np.float32).flatten()
        segments = np.array_split(values, 4)
        segment_std = [float(np.std(segment)) * 10.0 for segment in segments]

        avg_winrate = sum(win_rates.values()) / len(win_rates) if win_rates else 0.5
        base_score = 40.0 + avg_winrate * 55.0
        s_wr = win_rates.get("S", 0.0)
        a_wr = win_rates.get("A", 0.0)
        b_wr = win_rates.get("B", 0.0)
        c_wr = win_rates.get("C", 0.0)

        return {
            "composition": _clamp_score(base_score + segment_std[0] + (b_wr - a_wr) * 10.0),
            "technique": _clamp_score(base_score + segment_std[1] + (c_wr - s_wr) * 8.0),
            "creativity": _clamp_score(base_score + segment_std[2] + (a_wr - b_wr) * 6.0),
            "completeness": _clamp_score(base_score + segment_std[3] + (s_wr + c_wr - 1.0) * 5.0),
        }
