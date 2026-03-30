from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from utils import align_to_patch_multiple, resolve_distill_path


@dataclass
class ExperimentConfig:
    name: str = "vitl_distill"
    seed: int = 42
    device: str = "cuda"
    resume_from: str | None = None
    run_name: str | None = None
    smoke_train_limit: int = 256
    smoke_val_limit: int = 128


@dataclass
class PathsConfig:
    metadata_dir: str = "data/metadata"
    image_root: str = "data"
    prepared_train_csv: str = "training/data/metadata_train.csv"
    prepared_val_csv: str = "training/data/metadata_val.csv"
    checkpoint_root: str = "checkpoints/distill_vitl"
    report_root: str = "reports/distillation"
    student_export_dirname: str = "best_student_backbone"


@dataclass
class AugmentationConfig:
    min_scale: float = 0.75
    horizontal_flip_prob: float = 0.5
    color_jitter_brightness: float = 0.08
    color_jitter_contrast: float = 0.08
    color_jitter_saturation: float = 0.05
    color_jitter_hue: float = 0.02


@dataclass
class DataConfig:
    source_type: str = "mirip_staged"
    prepared_split_preferred: bool = True
    train_ratio: float = 0.9
    val_ratio: float = 0.1
    split_salt: str = "mirip-distill"
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    patch_size: int | None = None
    train_limit: int | None = None
    val_limit: int | None = None
    webdataset_url_pattern: str | None = None
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    def __post_init__(self) -> None:
        if not 0 < self.train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")
        if not 0 < self.val_ratio < 1:
            raise ValueError("val_ratio must be between 0 and 1")
        if (self.train_ratio + self.val_ratio) > 1.0:
            raise ValueError("train_ratio + val_ratio must be <= 1")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive")


@dataclass
class ModelsConfig:
    teacher_name: str = "PIA-SPACE-LAB/dinov3-vit7b16-pretrain-lvd1689m"
    student_name: str = "PIA-SPACE-LAB/dinov3-vitl-pretrain-lvd1689m"
    backend_order: list[str] = field(default_factory=lambda: ["huggingface", "timm"])
    teacher_dtype: str = "auto"
    student_dtype: str = "auto"
    trust_remote_code: bool = False
    gradient_checkpointing: bool = False


@dataclass
class LossWeights:
    cls: float = 1.0
    patch: float = 2.0
    rel: float = 1.0
    mid: float = 0.3
    pool: float = 0.2


@dataclass
class DistillationConfig:
    patch_loss_type: str = "cosine"
    normalize_features: bool = True
    rel_patch_sample_size: int = 128
    use_relational_loss: bool = True
    weights: LossWeights = field(default_factory=LossWeights)

    def __post_init__(self) -> None:
        if self.patch_loss_type not in {"cosine", "normalized_l2"}:
            raise ValueError("patch_loss_type must be one of: cosine, normalized_l2")
        if self.rel_patch_sample_size <= 0:
            raise ValueError("rel_patch_sample_size must be positive")


@dataclass
class OptimizerConfig:
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.05
    gradient_clip_norm: float = 1.0
    precision: str = "auto"
    cosine_min_lr: float = 1e-6


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_every_steps: int = 10
    jsonl_name: str = "history.jsonl"
    tensorboard_enabled: bool = False
    wandb_enabled: bool = False
    wandb_project: str = "mirip-v2-dinov3-distill"
    wandb_run_name: str | None = None


@dataclass
class EvaluationConfig:
    batch_size: int = 8
    retrieval_k: list[int] = field(default_factory=lambda: [1, 5])
    visualization_samples: int = 8
    save_visualizations: bool = True
    max_retrieval_samples: int | None = 512


@dataclass
class StageConfig:
    name: str
    enabled: bool = True
    resolution: int = 256
    epochs: int = 1
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float | None = None
    warmup_epochs: int = 1
    loss_weights: LossWeights = field(default_factory=LossWeights)

    def validate(self, patch_size: int) -> None:
        if self.epochs <= 0:
            raise ValueError(f"{self.name}: epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError(f"{self.name}: batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(f"{self.name}: gradient_accumulation_steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError(f"{self.name}: learning_rate must be positive")
        if self.weight_decay is not None and self.weight_decay < 0:
            raise ValueError(f"{self.name}: weight_decay must be non-negative")
        self.resolution = align_to_patch_multiple(self.resolution, patch_size)


@dataclass
class DistillationExperimentConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    stages: list[StageConfig] = field(
        default_factory=lambda: [
            StageConfig(name="stage1_main", resolution=256, epochs=8, batch_size=8, gradient_accumulation_steps=4, learning_rate=1e-4, warmup_epochs=1),
            StageConfig(name="stage2_highres", resolution=384, epochs=4, batch_size=4, gradient_accumulation_steps=8, learning_rate=5e-5, warmup_epochs=1),
            StageConfig(
                name="stage3_refine",
                resolution=384,
                epochs=2,
                batch_size=4,
                gradient_accumulation_steps=8,
                learning_rate=2.5e-5,
                warmup_epochs=0,
                loss_weights=LossWeights(cls=0.7, patch=2.0, rel=1.0, mid=0.3, pool=0.1),
            ),
        ]
    )
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self) -> None:
        patch_size = self.data.patch_size or 16
        for stage in self.stages:
            stage.validate(patch_size)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def clone(self) -> "DistillationExperimentConfig":
        return copy.deepcopy(self)

    def active_stages(self) -> list[StageConfig]:
        return [stage for stage in self.stages if stage.enabled]


def _loss_weights_from_dict(data: dict[str, Any] | None, defaults: LossWeights | None = None) -> LossWeights:
    base = asdict(defaults or LossWeights())
    if data:
        base.update(data)
    return LossWeights(**base)


def _merge_default_stage_weights(stages: list[StageConfig], default_weights: LossWeights) -> list[StageConfig]:
    merged: list[StageConfig] = []
    baseline = LossWeights()
    for stage in stages:
        copied = copy.deepcopy(stage)
        if copied.loss_weights == baseline:
            copied.loss_weights = copy.deepcopy(default_weights)
        merged.append(copied)
    return merged


def _load_stage_configs(
    values: list[dict[str, Any]] | None,
    *,
    default_weights: LossWeights,
) -> list[StageConfig]:
    if not values:
        return _merge_default_stage_weights(DistillationExperimentConfig().stages, default_weights)
    stages: list[StageConfig] = []
    for entry in values:
        payload = dict(entry)
        payload["loss_weights"] = _loss_weights_from_dict(payload.get("loss_weights"), default_weights)
        stages.append(StageConfig(**payload))
    return stages


def load_config(path: str | Path) -> DistillationExperimentConfig:
    config_path = resolve_distill_path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    distillation_config = DistillationConfig(
        weights=_loss_weights_from_dict(raw.get("distillation", {}).get("weights"), LossWeights()),
        **{k: v for k, v in raw.get("distillation", {}).items() if k != "weights"},
    )
    cfg = DistillationExperimentConfig(
        experiment=ExperimentConfig(**raw.get("experiment", {})),
        paths=PathsConfig(**raw.get("paths", {})),
        data=DataConfig(
            augmentation=AugmentationConfig(**raw.get("data", {}).get("augmentation", {})),
            **{k: v for k, v in raw.get("data", {}).items() if k != "augmentation"},
        ),
        models=ModelsConfig(**raw.get("models", {})),
        distillation=distillation_config,
        stages=_load_stage_configs(raw.get("stages"), default_weights=distillation_config.weights),
        optimizer=OptimizerConfig(**raw.get("optimizer", {})),
        logging=LoggingConfig(**raw.get("logging", {})),
        evaluation=EvaluationConfig(**raw.get("evaluation", {})),
    )
    return cfg


def apply_runtime_overrides(
    config: DistillationExperimentConfig,
    *,
    smoke: bool = False,
    resume_from: str | None = None,
) -> DistillationExperimentConfig:
    resolved = config.clone()
    if resume_from:
        resolved.experiment.resume_from = resume_from
    if smoke:
        for index, stage in enumerate(resolved.stages):
            stage.enabled = index == 0
            if index == 0:
                stage.epochs = 1
        if resolved.data.train_limit is None:
            resolved.data.train_limit = resolved.experiment.smoke_train_limit
        if resolved.data.val_limit is None:
            resolved.data.val_limit = resolved.experiment.smoke_val_limit
        if resolved.evaluation.max_retrieval_samples is None or resolved.evaluation.max_retrieval_samples > 128:
            resolved.evaluation.max_retrieval_samples = 128
        resolved.evaluation.visualization_samples = min(resolved.evaluation.visualization_samples, 4)
    return resolved
