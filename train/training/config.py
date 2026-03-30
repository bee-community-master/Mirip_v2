from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .utils import resolve_project_path

DEFAULT_DINOV3_MODEL_NAME = "camenduru/dinov3-vitl16-pretrain-lvd1689m"


@dataclass
class DinoV3TrainingConfig:
    model_name: str = DEFAULT_DINOV3_MODEL_NAME
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    batch_size: int = 8
    gradient_accumulation_steps: int = 8
    max_epochs: int = 50
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    gradient_clip_norm: float = 1.0
    scheduler_eta_min: float = 1e-6
    checkpoint_dir: str = field(default_factory=lambda: "checkpoints/dinov3_vitl16")
    save_every_n_epochs: int = 1
    num_workers: int = 4
    pin_memory: bool = True
    device: str = "cuda"
    precision: str = "auto"
    seed: int = 42
    projector_hidden_dim: int = 512
    projector_output_dim: int = 256
    dropout: float = 0.3
    margin: float = 0.3
    wandb_enabled: bool = False
    wandb_project: str | None = "mirip-v2-dinov3"
    wandb_run_name: str | None = None

    def __post_init__(self) -> None:
        self.checkpoint_dir = str(resolve_project_path(self.checkpoint_dir))
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
