from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DistillationConfig, LossWeights, StageConfig


def _normalize(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features.float(), dim=-1)


def _cosine_distance(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    cosine = F.cosine_similarity(student, teacher, dim=-1)
    return (1.0 - cosine).mean()


def _normalized_l2(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(_normalize(student), _normalize(teacher))


class PatchTokenDistillationLoss(nn.Module):
    """Distills local patch token features."""

    def __init__(self, loss_type: str = "cosine") -> None:
        super().__init__()
        if loss_type not in {"cosine", "normalized_l2"}:
            raise ValueError("loss_type must be one of: cosine, normalized_l2")
        self.loss_type = loss_type

    def forward(self, student_patch: torch.Tensor, teacher_patch: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "normalized_l2":
            return _normalized_l2(student_patch, teacher_patch)
        return _cosine_distance(student_patch, teacher_patch)


class RelationalPatchLoss(nn.Module):
    """Matches intra-image patch similarity structure."""

    def __init__(self, sample_size: int = 128) -> None:
        super().__init__()
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
        self.sample_size = sample_size

    def forward(self, student_patch: torch.Tensor, teacher_patch: torch.Tensor) -> torch.Tensor:
        patch_count = min(student_patch.shape[1], teacher_patch.shape[1])
        if patch_count < 2:
            return student_patch.new_zeros(())
        sample_size = min(self.sample_size, patch_count)
        if sample_size < patch_count:
            indices = torch.linspace(
                0,
                patch_count - 1,
                steps=sample_size,
                device=student_patch.device,
            ).round().long().unique()
            student_patch = student_patch.index_select(1, indices)
            teacher_patch = teacher_patch.index_select(1, indices)
        student_norm = _normalize(student_patch)
        teacher_norm = _normalize(teacher_patch)
        student_sim = student_norm @ student_norm.transpose(1, 2)
        teacher_sim = teacher_norm @ teacher_norm.transpose(1, 2)
        return F.mse_loss(student_sim, teacher_sim)


class ClsTokenDistillationLoss(nn.Module):
    """Aligns class token representations."""

    def forward(self, student_cls: torch.Tensor, teacher_cls: torch.Tensor) -> torch.Tensor:
        return _cosine_distance(student_cls, teacher_cls)


class PooledTokenDistillationLoss(nn.Module):
    """Aligns global pooled features."""

    def forward(self, student_pool: torch.Tensor, teacher_pool: torch.Tensor) -> torch.Tensor:
        return _cosine_distance(student_pool, teacher_pool)


class IntermediateFeatureDistillationLoss(nn.Module):
    """Aligns mapped intermediate hidden states."""

    def forward(self, student_hidden: list[torch.Tensor], teacher_hidden: list[torch.Tensor]) -> torch.Tensor:
        if not student_hidden or not teacher_hidden:
            device = student_hidden[0].device if student_hidden else teacher_hidden[0].device
            return torch.zeros((), device=device)
        losses = [_cosine_distance(student, teacher) for student, teacher in zip(student_hidden, teacher_hidden)]
        return torch.stack(losses).mean()


@dataclass
class LossBreakdown:
    total: torch.Tensor
    cls: torch.Tensor
    patch: torch.Tensor
    rel: torch.Tensor
    mid: torch.Tensor
    pool: torch.Tensor

    def as_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().cpu().item()),
            "cls": float(self.cls.detach().cpu().item()),
            "patch": float(self.patch.detach().cpu().item()),
            "rel": float(self.rel.detach().cpu().item()),
            "mid": float(self.mid.detach().cpu().item()),
            "pool": float(self.pool.detach().cpu().item()),
        }


class DistillationLossBundle(nn.Module):
    """Computes the weighted distillation objective."""

    def __init__(
        self,
        weights: LossWeights,
        *,
        patch_loss_type: str = "cosine",
        rel_patch_sample_size: int = 128,
        use_relational_loss: bool = True,
    ) -> None:
        super().__init__()
        self.weights = weights
        self.use_relational_loss = use_relational_loss
        self.patch_loss = PatchTokenDistillationLoss(loss_type=patch_loss_type)
        self.rel_loss = RelationalPatchLoss(sample_size=rel_patch_sample_size)
        self.cls_loss = ClsTokenDistillationLoss()
        self.pool_loss = PooledTokenDistillationLoss()
        self.mid_loss = IntermediateFeatureDistillationLoss()

    def forward(
        self,
        *,
        student_patch: torch.Tensor,
        teacher_patch: torch.Tensor,
        student_cls: torch.Tensor,
        teacher_cls: torch.Tensor,
        student_pool: torch.Tensor,
        teacher_pool: torch.Tensor,
        student_mid: list[torch.Tensor],
        teacher_mid: list[torch.Tensor],
    ) -> LossBreakdown:
        patch = self.patch_loss(student_patch, teacher_patch)
        rel = self.rel_loss(student_patch, teacher_patch) if self.use_relational_loss else student_patch.new_zeros(())
        cls = self.cls_loss(student_cls, teacher_cls)
        pool = self.pool_loss(student_pool, teacher_pool)
        mid = self.mid_loss(student_mid, teacher_mid)
        total = (
            self.weights.cls * cls
            + self.weights.patch * patch
            + self.weights.rel * rel
            + self.weights.mid * mid
            + self.weights.pool * pool
        )
        return LossBreakdown(total=total, cls=cls, patch=patch, rel=rel, mid=mid, pool=pool)

    @staticmethod
    def metric_summary(breakdowns: list[LossBreakdown]) -> dict[str, float]:
        if not breakdowns:
            return {key: 0.0 for key in ("loss_total", "loss_cls", "loss_patch", "loss_rel", "loss_mid", "loss_pool")}
        components: dict[str, list[float]] = {
            "loss_total": [],
            "loss_cls": [],
            "loss_patch": [],
            "loss_rel": [],
            "loss_mid": [],
            "loss_pool": [],
        }
        for item in breakdowns:
            components["loss_total"].append(float(item.total.detach().cpu().item()))
            components["loss_cls"].append(float(item.cls.detach().cpu().item()))
            components["loss_patch"].append(float(item.patch.detach().cpu().item()))
            components["loss_rel"].append(float(item.rel.detach().cpu().item()))
            components["loss_mid"].append(float(item.mid.detach().cpu().item()))
            components["loss_pool"].append(float(item.pool.detach().cpu().item()))
        return {key: sum(values) / len(values) for key, values in components.items()}


def build_loss_bundle(
    *,
    distillation_config: DistillationConfig,
    stage: StageConfig,
) -> DistillationLossBundle:
    """Builds a loss bundle using the resolved distillation config and stage weights."""

    return DistillationLossBundle(
        weights=stage.loss_weights,
        patch_loss_type=distillation_config.patch_loss_type,
        rel_patch_sample_size=distillation_config.rel_patch_sample_size,
        use_relational_loss=distillation_config.use_relational_loss,
    )
