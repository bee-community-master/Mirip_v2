from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, IterableDataset

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - optional dependency
    SummaryWriter = None

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

from config import DistillationExperimentConfig, StageConfig
from datasets import DistillationBatchCollator, build_stage_datasets
from losses import DistillationLossBundle, LossBreakdown, build_loss_bundle
from models import DistillationBatch, TeacherStudentDistillModel
from utils import (
    append_jsonl,
    align_to_patch_multiple,
    autocast_context,
    build_pairwise_handoff_command,
    ensure_dir,
    format_seconds,
    resolve_precision,
    resolve_train_path,
    save_json,
    safe_mean,
    select_device,
    setup_logging,
)


@dataclass
class RunDirectories:
    """Resolved output directories for one distillation run."""

    run_name: str
    checkpoint_dir: Path
    report_dir: Path
    export_dir: Path


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _build_run_name(config: DistillationExperimentConfig) -> str:
    if config.experiment.run_name:
        return config.experiment.run_name
    return f"{config.experiment.name}_{_timestamp_slug()}"


def prepare_run_directories(
    config: DistillationExperimentConfig,
    *,
    run_name_override: str | None = None,
) -> RunDirectories:
    """Creates run-specific checkpoint/report directories and returns their paths."""

    run_name = run_name_override or _build_run_name(config)
    checkpoint_dir = ensure_dir(resolve_train_path(Path(config.paths.checkpoint_root) / run_name))
    report_dir = ensure_dir(resolve_train_path(Path(config.paths.report_root) / run_name))
    export_dir = checkpoint_dir / config.paths.student_export_dirname
    return RunDirectories(run_name=run_name, checkpoint_dir=checkpoint_dir, report_dir=report_dir, export_dir=export_dir)


def _build_param_groups(model: TeacherStudentDistillModel, weight_decay: float) -> list[dict[str, Any]]:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _build_scheduler(
    optimizer: AdamW,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr: float,
    base_lr: float,
) -> LambdaLR:
    min_ratio = min_lr / base_lr if base_lr > 0 else 0.0

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(1e-6, float(step + 1) / max(warmup_steps, 1))
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, _lr_lambda)


def _dataloader_length(loader: DataLoader) -> int:
    if hasattr(loader, "__len__"):
        return len(loader)
    raise RuntimeError(
        "DataLoader length is required to build a cosine schedule. "
        "For IterableDataset/WebDataset, provide finite train/val limits."
    )


def _feature_cosine(student: torch.Tensor, teacher: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(student.float(), teacher.float(), dim=-1).mean().item())


def _mid_feature_cosine(student_mid: list[torch.Tensor], teacher_mid: list[torch.Tensor]) -> float:
    if not student_mid or not teacher_mid:
        return 0.0
    values = [_feature_cosine(student, teacher) for student, teacher in zip(student_mid, teacher_mid)]
    return safe_mean(values)


def _peak_vram_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024 ** 3))


def _build_loaders(
    config: DistillationExperimentConfig,
    stage: StageConfig,
    model: TeacherStudentDistillModel,
) -> tuple[DataLoader, DataLoader]:
    resolution = align_to_patch_multiple(stage.resolution, model.patch_size)
    train_dataset, val_dataset = build_stage_datasets(
        config,
        resolution=resolution,
        mean=model.image_mean,
        std=model.image_std,
    )
    collator = DistillationBatchCollator()
    loader_kwargs = {
        "num_workers": config.data.num_workers,
        "pin_memory": config.data.pin_memory,
        "persistent_workers": config.data.persistent_workers and config.data.num_workers > 0,
        "collate_fn": collator,
    }
    if config.data.num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.data.prefetch_factor
    train_loader = DataLoader(
        train_dataset,
        batch_size=stage.batch_size,
        shuffle=not isinstance(train_dataset, IterableDataset),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.evaluation.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader


def _run_forward(
    model: TeacherStudentDistillModel,
    pixel_values: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
) -> DistillationBatch:
    pixel_values = pixel_values.to(device, non_blocking=True)
    with autocast_context(device, precision):
        return model(pixel_values)


def train_one_epoch(
    *,
    model: TeacherStudentDistillModel,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    loss_bundle: DistillationLossBundle,
    device: torch.device,
    precision: str,
    grad_accumulation_steps: int,
    grad_clip_norm: float,
    logger,
    log_every_steps: int,
    global_step: int,
) -> tuple[dict[str, float], int]:
    """Runs one training epoch and returns aggregated metrics plus the updated global step."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    breakdowns: list[LossBreakdown] = []
    patch_cosines: list[float] = []
    pool_cosines: list[float] = []
    mid_cosines: list[float] = []

    for batch_index, batch in enumerate(loader, start=1):
        distill_batch = _run_forward(model, batch["pixel_values"], device=device, precision=precision)
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
        scaled_loss = loss_breakdown.total / grad_accumulation_steps
        if scaler.is_enabled():
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if batch_index % grad_accumulation_steps == 0 or batch_index == len(loader):
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip_norm)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        breakdowns.append(loss_breakdown)
        patch_cosines.append(_feature_cosine(distill_batch.student_patch, distill_batch.teacher_patch))
        pool_cosines.append(_feature_cosine(distill_batch.student_pool, distill_batch.teacher_pool))
        mid_cosines.append(_mid_feature_cosine(distill_batch.student_mid, distill_batch.teacher_mid))
        global_step += 1

        if batch_index % max(log_every_steps, 1) == 0:
            metrics = loss_breakdown.as_dict()
            logger.info(
                "train step=%s loss=%.4f patch=%.4f rel=%.4f cls=%.4f pool=%.4f",
                global_step,
                metrics["total"],
                metrics["patch"],
                metrics["rel"],
                metrics["cls"],
                metrics["pool"],
            )

    metrics = DistillationLossBundle.metric_summary(breakdowns)
    metrics.update(
        {
            "patch_cosine": safe_mean(patch_cosines),
            "pool_cosine": safe_mean(pool_cosines),
            "mid_cosine": safe_mean(mid_cosines),
        }
    )
    return metrics, global_step


@torch.no_grad()
def validate(
    *,
    model: TeacherStudentDistillModel,
    loader: DataLoader,
    loss_bundle: DistillationLossBundle,
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    """Evaluates the current model on the validation split and aggregates metrics."""

    model.eval()
    breakdowns: list[LossBreakdown] = []
    patch_cosines: list[float] = []
    pool_cosines: list[float] = []
    mid_cosines: list[float] = []

    for batch in loader:
        distill_batch = _run_forward(model, batch["pixel_values"], device=device, precision=precision)
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
        patch_cosines.append(_feature_cosine(distill_batch.student_patch, distill_batch.teacher_patch))
        pool_cosines.append(_feature_cosine(distill_batch.student_pool, distill_batch.teacher_pool))
        mid_cosines.append(_mid_feature_cosine(distill_batch.student_mid, distill_batch.teacher_mid))

    metrics = DistillationLossBundle.metric_summary(breakdowns)
    metrics.update(
        {
            "patch_cosine": safe_mean(patch_cosines),
            "pool_cosine": safe_mean(pool_cosines),
            "mid_cosine": safe_mean(mid_cosines),
        }
    )
    return metrics


def _write_checkpoint(
    *,
    path: Path,
    model: TeacherStudentDistillModel,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    config: DistillationExperimentConfig,
    run_dirs: RunDirectories,
    stage_index: int,
    epoch_in_stage: int,
    global_step: int,
    best_val_loss: float,
    precision: str,
) -> Path:
    payload = {
        "model": model.checkpoint_state(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "config": config.to_dict(),
        "run_name": run_dirs.run_name,
        "stage_index": stage_index,
        "epoch_in_stage": epoch_in_stage,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "precision": precision,
    }
    torch.save(payload, path)
    return path


def _load_checkpoint(
    path: str | Path,
    *,
    model: TeacherStudentDistillModel,
    optimizer: AdamW | None,
    scheduler: LambdaLR | None,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(resolve_train_path(path), map_location=device)
    model.load_checkpoint_state(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and scaler.is_enabled() and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    return payload


def _maybe_init_logging(config: DistillationExperimentConfig, run_dirs: RunDirectories):
    logger = setup_logging(config.logging.level)
    tb_writer = None
    if config.logging.tensorboard_enabled and SummaryWriter is not None:
        tb_writer = SummaryWriter(log_dir=str(run_dirs.report_dir / "tensorboard"))
    if config.logging.wandb_enabled and wandb is not None:
        wandb.init(
            project=config.logging.wandb_project,
            name=config.logging.wandb_run_name or run_dirs.run_name,
            config=config.to_dict(),
        )
    return logger, tb_writer


def _log_epoch_metrics(
    *,
    stage_name: str,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    elapsed_seconds: float,
    peak_vram_gb: float,
    report_path: Path,
    tb_writer,
    epoch_index: int,
) -> None:
    payload = {
        "stage": stage_name,
        "epoch": epoch,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_hms": format_seconds(elapsed_seconds),
        "peak_vram_gb": peak_vram_gb,
        "train": train_metrics,
        "val": val_metrics,
    }
    append_jsonl(report_path, payload)
    if tb_writer is None:
        return
    for split_name, metrics in (("train", train_metrics), ("val", val_metrics)):
        for key, value in metrics.items():
            tb_writer.add_scalar(f"{stage_name}/{split_name}/{key}", value, epoch_index)
    tb_writer.add_scalar(f"{stage_name}/system/peak_vram_gb", peak_vram_gb, epoch_index)


def _pairwise_handoff_command(run_dirs: RunDirectories) -> str:
    return build_pairwise_handoff_command(run_dirs.export_dir)


def run_stage(
    *,
    config: DistillationExperimentConfig,
    model: TeacherStudentDistillModel,
    stage_index: int,
    stage: StageConfig,
    run_dirs: RunDirectories,
    logger,
    tb_writer,
    resume_payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Executes one configured stage, including resume-aware optimizer restoration."""

    device = select_device(config.experiment.device)
    precision = resolve_precision(config.optimizer.precision, device)
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_loader, val_loader = _build_loaders(config, stage, model)
    steps_per_epoch = math.ceil(_dataloader_length(train_loader) / stage.gradient_accumulation_steps)
    total_steps = max(steps_per_epoch * stage.epochs, 1)
    warmup_steps = min(total_steps, stage.warmup_epochs * steps_per_epoch)
    weight_decay = stage.weight_decay if stage.weight_decay is not None else config.optimizer.weight_decay
    optimizer = AdamW(
        _build_param_groups(model, weight_decay),
        lr=stage.learning_rate,
        betas=config.optimizer.betas,
        eps=config.optimizer.eps,
    )
    scheduler = _build_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr=config.optimizer.cosine_min_lr,
        base_lr=stage.learning_rate,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=precision == "fp16" and device.type == "cuda")
    best_val_loss = math.inf
    global_step = 0
    start_epoch = 0

    if resume_payload is not None:
        _load_checkpoint(
            resolve_train_path(config.experiment.resume_from or ""),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        best_val_loss = float(resume_payload.get("best_val_loss", math.inf))
        global_step = int(resume_payload.get("global_step", 0))
        if int(resume_payload.get("stage_index", 0)) == stage_index:
            start_epoch = int(resume_payload.get("epoch_in_stage", -1)) + 1

    loss_bundle = build_loss_bundle(distillation_config=config.distillation, stage=stage)
    report_path = run_dirs.report_dir / config.logging.jsonl_name
    best_checkpoint_path = run_dirs.checkpoint_dir / "best.pt"
    last_checkpoint_path = run_dirs.checkpoint_dir / "last.pt"
    last_metrics: dict[str, Any] | None = None
    epoch_counter = 0

    logger.info("stage=%s resolution=%s epochs=%s batch_size=%s", stage.name, stage.resolution, stage.epochs, stage.batch_size)
    for epoch in range(start_epoch, stage.epochs):
        start_time = time.perf_counter()
        train_metrics, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            loss_bundle=loss_bundle,
            device=device,
            precision=precision,
            grad_accumulation_steps=stage.gradient_accumulation_steps,
            grad_clip_norm=config.optimizer.gradient_clip_norm,
            logger=logger,
            log_every_steps=config.logging.log_every_steps,
            global_step=global_step,
        )
        val_metrics = validate(
            model=model,
            loader=val_loader,
            loss_bundle=loss_bundle,
            device=device,
            precision=precision,
        )
        elapsed = time.perf_counter() - start_time
        peak_vram_gb = _peak_vram_gb(device)
        epoch_counter += 1

        _log_epoch_metrics(
            stage_name=stage.name,
            epoch=epoch + 1,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            elapsed_seconds=elapsed,
            peak_vram_gb=peak_vram_gb,
            report_path=report_path,
            tb_writer=tb_writer,
            epoch_index=epoch_counter,
        )

        logger.info(
            "stage=%s epoch=%s/%s train_loss=%.4f val_loss=%.4f patch_cos=%.4f pool_cos=%.4f",
            stage.name,
            epoch + 1,
            stage.epochs,
            train_metrics["loss_total"],
            val_metrics["loss_total"],
            val_metrics["patch_cosine"],
            val_metrics["pool_cosine"],
        )

        _write_checkpoint(
            path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            run_dirs=run_dirs,
            stage_index=stage_index,
            epoch_in_stage=epoch,
            global_step=global_step,
            best_val_loss=best_val_loss,
            precision=precision,
        )

        if val_metrics["loss_total"] < best_val_loss:
            best_val_loss = val_metrics["loss_total"]
            _write_checkpoint(
                path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                run_dirs=run_dirs,
                stage_index=stage_index,
                epoch_in_stage=epoch,
                global_step=global_step,
                best_val_loss=best_val_loss,
                precision=precision,
            )
            if model.student.can_export_hf:
                ensure_dir(run_dirs.export_dir)
                model.export_student_backbone(run_dirs.export_dir)

        last_metrics = {
            "stage": stage.name,
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
            "elapsed_seconds": elapsed,
            "peak_vram_gb": peak_vram_gb,
        }

    return (
        {
            "stage": stage.name,
            "best_val_loss": best_val_loss,
            "last_metrics": last_metrics,
            "precision": precision,
            "global_step": global_step,
        },
        None,
    )


def run_experiment(config: DistillationExperimentConfig) -> dict[str, Any]:
    """Runs all enabled distillation stages and writes a final experiment summary."""

    device = select_device(config.experiment.device)
    resume_payload = None
    start_stage_index = 0
    run_name_override = None
    if config.experiment.resume_from:
        resume_payload = torch.load(resolve_train_path(config.experiment.resume_from), map_location=device)
        run_name_override = str(resume_payload.get("run_name") or "")
        start_stage_index = int(resume_payload.get("stage_index", 0))

    run_dirs = prepare_run_directories(config, run_name_override=run_name_override or None)
    logger, tb_writer = _maybe_init_logging(config, run_dirs)
    save_json(run_dirs.report_dir / "config_resolved.json", config.to_dict())
    model = TeacherStudentDistillModel(
        config.models,
        normalize_features=config.distillation.normalize_features,
    )
    model.to(device)

    if resume_payload is not None:
        model.load_checkpoint_state(resume_payload["model"])

    stage_summaries: list[dict[str, Any]] = []
    active_stages = config.active_stages()
    for stage_index, stage in enumerate(active_stages[start_stage_index:], start=start_stage_index):
        stage_summary, _ = run_stage(
            config=config,
            model=model,
            stage_index=stage_index,
            stage=stage,
            run_dirs=run_dirs,
            logger=logger,
            tb_writer=tb_writer,
            resume_payload=resume_payload if stage_index == start_stage_index else None,
        )
        stage_summaries.append(stage_summary)
        resume_payload = None

    if tb_writer is not None:
        tb_writer.close()
    if config.logging.wandb_enabled and wandb is not None:
        wandb.finish()

    summary = {
        "run_name": run_dirs.run_name,
        "checkpoint_dir": str(run_dirs.checkpoint_dir),
        "report_dir": str(run_dirs.report_dir),
        "student_export_dir": str(run_dirs.export_dir),
        "pairwise_handoff_command": _pairwise_handoff_command(run_dirs),
        "stages": stage_summaries,
    }
    save_json(run_dirs.report_dir / "summary.json", summary)
    return summary
