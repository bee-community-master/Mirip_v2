#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import DistillationExperimentConfig, StageConfig, apply_runtime_overrides, load_config
from datasets import DistillRecord, DistillationBatchCollator, build_stage_datasets
from losses import DistillationLossBundle, LossBreakdown
from models import DistillationBatch, TeacherStudentDistillModel
from utils import (
    align_to_patch_multiple,
    autocast_context,
    ensure_dir,
    pca_rgb_map,
    resolve_precision,
    resolve_train_path,
    safe_mean,
    save_json,
    select_device,
    set_seed,
    tensor_to_numpy_image,
    try_enable_tf32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Mirip_v2 DINOv3 distillation checkpoints.")
    parser.add_argument("--config", required=True, help="Path to a YAML config, relative to train/distillation.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path relative to train/ or absolute path.")
    parser.add_argument("--output", help="Optional JSON output path relative to train/ or absolute path.")
    parser.add_argument("--stage-name", help="Override the stage name used for evaluation resolution/loss weights.")
    parser.add_argument(
        "--max-batches",
        type=int,
        help="Optional maximum number of validation batches to evaluate.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Apply smoke overrides before evaluation.",
    )
    return parser.parse_args()


def _build_loss_bundle(config: DistillationExperimentConfig, stage: StageConfig) -> DistillationLossBundle:
    return DistillationLossBundle(
        weights=stage.loss_weights,
        patch_loss_type=config.distillation.patch_loss_type,
        rel_patch_sample_size=config.distillation.rel_patch_sample_size,
        use_relational_loss=config.distillation.use_relational_loss,
    )


def _resolve_stage(
    config: DistillationExperimentConfig,
    checkpoint_payload: dict[str, Any],
    stage_name: str | None,
) -> StageConfig:
    active_stages = config.active_stages()
    if not active_stages:
        raise RuntimeError("No enabled stages are configured for distillation")
    if stage_name:
        for stage in active_stages:
            if stage.name == stage_name:
                return stage
        raise ValueError(f"Unknown stage name: {stage_name}")
    checkpoint_index = int(checkpoint_payload.get("stage_index", len(active_stages) - 1))
    checkpoint_index = max(0, min(checkpoint_index, len(active_stages) - 1))
    return active_stages[checkpoint_index]


def _build_val_loader(
    config: DistillationExperimentConfig,
    stage: StageConfig,
    model: TeacherStudentDistillModel,
) -> tuple[DataLoader, int | None]:
    resolution = align_to_patch_multiple(stage.resolution, model.patch_size)
    _, val_dataset = build_stage_datasets(
        config,
        resolution=resolution,
        mean=model.image_mean,
        std=model.image_std,
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": config.evaluation.batch_size,
        "shuffle": False,
        "num_workers": config.data.num_workers,
        "pin_memory": config.data.pin_memory,
        "persistent_workers": config.data.persistent_workers and config.data.num_workers > 0,
        "collate_fn": DistillationBatchCollator(),
    }
    if config.data.num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.data.prefetch_factor
    val_loader = DataLoader(val_dataset, **loader_kwargs)
    val_count = None if isinstance(val_dataset, IterableDataset) else len(val_dataset)
    return val_loader, val_count


def _cosine_mean(student: torch.Tensor, teacher: torch.Tensor) -> float:
    return float(F.cosine_similarity(student.float(), teacher.float(), dim=-1).mean().item())


def _mid_alignment_score(student_mid: Sequence[torch.Tensor], teacher_mid: Sequence[torch.Tensor]) -> float:
    if not student_mid or not teacher_mid:
        return 0.0
    scores = [
        _cosine_mean(student.mean(dim=1), teacher.mean(dim=1))
        for student, teacher in zip(student_mid, teacher_mid)
    ]
    return safe_mean(scores)


def _patch_matrix_alignment_score(
    student_patch: torch.Tensor,
    teacher_patch: torch.Tensor,
    sample_size: int,
) -> float:
    patch_count = min(student_patch.shape[1], teacher_patch.shape[1])
    if patch_count < 2:
        return 0.0
    actual_sample = min(sample_size, patch_count)
    if actual_sample < patch_count:
        indices = torch.linspace(
            0,
            patch_count - 1,
            steps=actual_sample,
            device=student_patch.device,
        ).round().long().unique()
        student_patch = student_patch.index_select(1, indices)
        teacher_patch = teacher_patch.index_select(1, indices)
    student_norm = F.normalize(student_patch.float(), dim=-1)
    teacher_norm = F.normalize(teacher_patch.float(), dim=-1)
    student_sim = student_norm @ student_norm.transpose(1, 2)
    teacher_sim = teacher_norm @ teacher_norm.transpose(1, 2)
    return float(
        F.cosine_similarity(
            student_sim.flatten(1),
            teacher_sim.flatten(1),
            dim=-1,
        ).mean().item()
    )


def _denormalize_image(tensor: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype).view(-1, 1, 1)
    return (tensor.detach().cpu() * std_tensor) + mean_tensor


def _save_patch_visualization(
    *,
    record: DistillRecord,
    image_tensor: torch.Tensor,
    teacher_patch: torch.Tensor,
    student_patch: torch.Tensor,
    patch_grid_hw: tuple[int, int],
    mean: Sequence[float],
    std: Sequence[float],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    original = tensor_to_numpy_image(_denormalize_image(image_tensor, mean, std).clamp(0.0, 1.0))
    teacher_map = pca_rgb_map(teacher_patch, patch_grid_hw)
    student_map = pca_rgb_map(student_patch, patch_grid_hw)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[1].imshow(teacher_map)
    axes[1].set_title("Teacher patch PCA")
    axes[2].imshow(student_map)
    axes[2].set_title("Student patch PCA")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(record.sample_id)
    figure.tight_layout()
    ensure_dir(output_path.parent)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _compute_retrieval_metrics(
    embeddings: Sequence[torch.Tensor],
    targets: Sequence[str],
    *,
    ks: Sequence[int],
    max_samples: int | None,
) -> dict[str, float]:
    if not embeddings:
        metrics = {"retrieval_queries": 0.0, "mrr": 0.0}
        for k in ks:
            metrics[f"recall@{k}"] = 0.0
        return metrics

    stacked = torch.cat(list(embeddings), dim=0)
    if max_samples is not None and stacked.shape[0] > max_samples:
        stacked = stacked[:max_samples]
        targets = list(targets[:max_samples])
    normalized = F.normalize(stacked.float(), dim=-1)
    similarity = normalized @ normalized.T
    similarity.fill_diagonal_(-float("inf"))

    valid_queries = 0
    reciprocal_ranks: list[float] = []
    recall_hits: dict[int, int] = {k: 0 for k in ks}
    targets_list = list(targets)
    for index, target in enumerate(targets_list):
        if not target:
            continue
        positives = [candidate_index for candidate_index, candidate in enumerate(targets_list) if candidate == target and candidate_index != index]
        if not positives:
            continue
        valid_queries += 1
        ranking = torch.argsort(similarity[index], descending=True)
        ranked_indices = ranking.tolist()
        first_positive_rank = None
        for rank, candidate_index in enumerate(ranked_indices, start=1):
            if candidate_index in positives:
                first_positive_rank = rank
                break
        if first_positive_rank is None:
            continue
        reciprocal_ranks.append(1.0 / first_positive_rank)
        for k in ks:
            if any(candidate_index in positives for candidate_index in ranked_indices[:k]):
                recall_hits[k] += 1

    metrics = {
        "retrieval_queries": float(valid_queries),
        "mrr": safe_mean(reciprocal_ranks),
    }
    for k in ks:
        metrics[f"recall@{k}"] = 0.0 if valid_queries == 0 else float(recall_hits[k] / valid_queries)
    return metrics


def _default_output_path(
    config: DistillationExperimentConfig,
    checkpoint_path: Path,
    checkpoint_payload: dict[str, Any],
) -> Path:
    run_name = str(checkpoint_payload.get("run_name") or checkpoint_path.parent.name)
    report_dir = resolve_train_path(Path(config.paths.report_root) / run_name)
    return report_dir / f"eval_{checkpoint_path.stem}.json"


def _pairwise_handoff_command(config: DistillationExperimentConfig, checkpoint_payload: dict[str, Any]) -> str:
    run_name = str(checkpoint_payload.get("run_name") or "distill_run")
    export_dir = resolve_train_path(Path(config.paths.checkpoint_root) / run_name / config.paths.student_export_dirname)
    try:
        relative = export_dir.relative_to(resolve_train_path(".").parent)
    except ValueError:
        relative = export_dir
    return f"python3 train/training/train_dinov3.py --model-name {relative}"


def _maybe_export_student_backbone(
    model: TeacherStudentDistillModel,
    config: DistillationExperimentConfig,
    checkpoint_payload: dict[str, Any],
) -> str | None:
    if not model.student.can_export_hf:
        return None
    run_name = str(checkpoint_payload.get("run_name") or "distill_run")
    export_dir = resolve_train_path(Path(config.paths.checkpoint_root) / run_name / config.paths.student_export_dirname)
    if not export_dir.exists():
        model.export_student_backbone(export_dir)
    return str(export_dir)


def evaluate_checkpoint(
    config: DistillationExperimentConfig,
    *,
    checkpoint_path: Path,
    output_path: Path | None,
    stage_name: str | None,
    max_batches: int | None,
) -> dict[str, Any]:
    """Runs offline evaluation for one distilled checkpoint and stores a JSON report."""

    device = select_device(config.experiment.device)
    checkpoint_payload = torch.load(checkpoint_path, map_location=device)
    stage = _resolve_stage(config, checkpoint_payload, stage_name)
    model = TeacherStudentDistillModel(
        config.models,
        normalize_features=config.distillation.normalize_features,
    )
    model.load_checkpoint_state(checkpoint_payload["model"])
    model.to(device)
    model.eval()
    val_loader, val_count = _build_val_loader(config, stage, model)
    loss_bundle = _build_loss_bundle(config, stage)
    precision = resolve_precision(config.optimizer.precision, device)

    breakdowns: list[LossBreakdown] = []
    patch_cosines: list[float] = []
    cls_cosines: list[float] = []
    pool_cosines: list[float] = []
    patch_alignment_scores: list[float] = []
    mid_alignment_scores: list[float] = []
    student_embeddings: list[torch.Tensor] = []
    retrieval_targets: list[str] = []
    visual_outputs: list[str] = []
    batch_counter = 0

    run_name = str(checkpoint_payload.get("run_name") or checkpoint_path.parent.name)
    report_dir = resolve_train_path(Path(config.paths.report_root) / run_name)
    visual_dir = report_dir / "visuals"

    with torch.no_grad():
        for batch in val_loader:
            batch_counter += 1
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            with autocast_context(device, precision):
                distill_batch = model(pixel_values)
            loss_breakdown = loss_bundle(
                student_patch=distill_batch.student_patch,
                teacher_patch=distill_batch.teacher_patch,
                student_cls=distill_batch.student_cls,
                teacher_cls=distill_batch.teacher_cls,
                student_pool=distill_batch.student_pool,
                teacher_pool=distill_batch.teacher_pool,
                student_mid=distill_batch.student_mid,
                teacher_mid=distill_batch.teacher_mid,
            )
            breakdowns.append(loss_breakdown)
            patch_cosines.append(_cosine_mean(distill_batch.student_patch, distill_batch.teacher_patch))
            cls_cosines.append(_cosine_mean(distill_batch.student_cls, distill_batch.teacher_cls))
            pool_cosines.append(_cosine_mean(distill_batch.student_pool, distill_batch.teacher_pool))
            patch_alignment_scores.append(
                _patch_matrix_alignment_score(
                    distill_batch.student_patch,
                    distill_batch.teacher_patch,
                    config.distillation.rel_patch_sample_size,
                )
            )
            mid_alignment_scores.append(_mid_alignment_score(distill_batch.student_mid, distill_batch.teacher_mid))

            student_embeddings.append(distill_batch.student_pool.detach().cpu())
            retrieval_targets.extend(record.retrieval_target for record in batch["records"])

            remaining_visuals = config.evaluation.visualization_samples - len(visual_outputs)
            if config.evaluation.save_visualizations and remaining_visuals > 0:
                for item_index, record in enumerate(batch["records"][:remaining_visuals]):
                    output_file = visual_dir / f"{len(visual_outputs):03d}_{record.sample_id}.png"
                    _save_patch_visualization(
                        record=record,
                        image_tensor=batch["pixel_values"][item_index],
                        teacher_patch=distill_batch.teacher_patch[item_index],
                        student_patch=distill_batch.student_patch[item_index],
                        patch_grid_hw=distill_batch.teacher.patch_grid_hw,
                        mean=model.image_mean,
                        std=model.image_std,
                        output_path=output_file,
                    )
                    visual_outputs.append(str(output_file))

            if max_batches is not None and batch_counter >= max_batches:
                break

    export_dir = _maybe_export_student_backbone(model, config, checkpoint_payload)
    loss_metrics = DistillationLossBundle.metric_summary(breakdowns)
    retrieval_metrics = _compute_retrieval_metrics(
        student_embeddings,
        retrieval_targets,
        ks=config.evaluation.retrieval_k,
        max_samples=config.evaluation.max_retrieval_samples,
    )
    summary = {
        "run_name": run_name,
        "checkpoint": str(checkpoint_path),
        "stage": stage.name,
        "resolution": align_to_patch_multiple(stage.resolution, model.patch_size),
        "val_count": val_count,
        "evaluated_batches": batch_counter,
        "loss": loss_metrics,
        "feature_alignment": {
            "patch_cosine": safe_mean(patch_cosines),
            "cls_cosine": safe_mean(cls_cosines),
            "pool_cosine": safe_mean(pool_cosines),
            "patch_similarity_alignment": safe_mean(patch_alignment_scores),
            "intermediate_alignment": safe_mean(mid_alignment_scores),
        },
        "retrieval": retrieval_metrics,
        "visualizations": visual_outputs,
        "student_export_dir": export_dir,
        "pairwise_handoff_command": _pairwise_handoff_command(config, checkpoint_payload),
    }
    target_output = output_path or _default_output_path(config, checkpoint_path, checkpoint_payload)
    save_json(target_output, summary)
    return summary


def main() -> int:
    """CLI entry point for checkpoint evaluation."""

    args = parse_args()
    config = load_config(args.config)
    config = apply_runtime_overrides(config, smoke=args.smoke)
    set_seed(config.experiment.seed)
    try_enable_tf32()
    checkpoint_path = resolve_train_path(args.ckpt)
    output_path = resolve_train_path(args.output) if args.output else None
    summary = evaluate_checkpoint(
        config,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        stage_name=args.stage_name,
        max_batches=args.max_batches,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
