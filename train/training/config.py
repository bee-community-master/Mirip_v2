from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .utils import resolve_project_path

DEFAULT_DINOV3_MODEL_NAME = "PIA-SPACE-LAB/dinov3-vit7b16-pretrain-lvd1689m"


def default_num_workers() -> int:
    cpu_count = os.cpu_count() or 4
    return max(1, min(8, cpu_count - 2))


@dataclass
class DinoV3TrainingConfig:
    model_name: str = DEFAULT_DINOV3_MODEL_NAME
    backbone_dtype: str = "auto"
    learning_rate: float = 1e-4
    backbone_learning_rate_scale: float = 0.2
    weight_decay: float = 1e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 8
    max_epochs: int = 30
    warmup_epochs: int = 2
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-4
    early_stopping_metric: str = "anchor_tier_accuracy"
    restart_from_best_patience: int = 0
    gradient_clip_norm: float = 1.0
    scheduler_eta_min: float = 1e-6
    anchor_eval_n_per_tier: int = 24
    anchor_eval_bootstrap_seeds: list[int] = field(default_factory=lambda: [42, 43, 44])
    anchor_eval_min_improvement: float = 0.005
    anchor_eval_group_balanced: bool = True
    checkpoint_dir: str = field(default_factory=lambda: "output_models/checkpoints/dinov3_vit7b16")
    save_every_n_epochs: int = 1
    num_workers: int = field(default_factory=default_num_workers)
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    device: str = "cuda"
    precision: str = "bf16"
    seed: int = 42
    input_size: int = 448
    feature_pool: str = "cls_mean_patch_concat"
    head_type: str = "mlp_small"
    freeze_backbone: bool = True
    projector_hidden_dim: int = 512
    projector_output_dim: int = 256
    unfreeze_last_n_layers: int = 0
    dropout: float = 0.1
    margin: float = 0.3
    wandb_enabled: bool = False
    wandb_project: str | None = "mirip-v2-dinov3"
    wandb_run_name: str | None = None

    def __post_init__(self) -> None:
        self.checkpoint_dir = str(resolve_project_path(self.checkpoint_dir))
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.backbone_learning_rate_scale <= 0:
            raise ValueError("backbone_learning_rate_scale must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        if self.restart_from_best_patience < 0:
            raise ValueError("restart_from_best_patience must be non-negative")
        if self.anchor_eval_n_per_tier <= 0:
            raise ValueError("anchor_eval_n_per_tier must be positive")
        if not self.anchor_eval_bootstrap_seeds:
            raise ValueError("anchor_eval_bootstrap_seeds must not be empty")
        if any(not isinstance(seed, int) for seed in self.anchor_eval_bootstrap_seeds):
            raise ValueError("anchor_eval_bootstrap_seeds must contain integers")
        if self.anchor_eval_min_improvement < 0:
            raise ValueError("anchor_eval_min_improvement must be non-negative")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive")
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")
        if self.unfreeze_last_n_layers < 0:
            raise ValueError("unfreeze_last_n_layers must be non-negative")
        if self.backbone_dtype not in {"auto", "bf16", "fp16", "fp32"}:
            raise ValueError("backbone_dtype must be one of: auto, bf16, fp16, fp32")
        if self.early_stopping_metric not in {"val_loss", "anchor_tier_accuracy"}:
            raise ValueError("early_stopping_metric must be one of: val_loss, anchor_tier_accuracy")
        if self.feature_pool not in {"cls", "cls_mean_patch_concat"}:
            raise ValueError("feature_pool must be one of: cls, cls_mean_patch_concat")
        if self.head_type not in {"linear", "mlp_small"}:
            raise ValueError("head_type must be one of: linear, mlp_small")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
