"""DINOv3 teacher-student distillation package for Mirip_v2."""

from .config import DistillationExperimentConfig, load_config

__all__ = ["DistillationExperimentConfig", "load_config"]
