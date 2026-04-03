from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    wandb = None

from .config import DinoV3TrainingConfig
from .utils import resolve_project_path


def resolve_precision(requested: str, device: str) -> str:
    if device != "cuda":
        return "fp32"
    if requested == "auto":
        if torch.cuda.is_bf16_supported():
            return "bf16"
        return "fp16"
    return requested


class DinoV3Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        config: DinoV3TrainingConfig,
        resume_from: str | None = None,
        resume_next_epoch: bool = False,
        reset_training_state_on_resume: bool = False,
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.precision = resolve_precision(config.precision, "cuda" if self.device.type == "cuda" else "cpu")
        self.model.to(self.device)

        optimizer_groups = self._build_optimizer_groups()
        self.optimizer = AdamW(optimizer_groups, weight_decay=config.weight_decay)
        self.scheduler = self._build_scheduler()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.precision == "fp16" and self.device.type == "cuda")
        self.best_val_loss = math.inf
        self.best_selection_metric: float | None = None
        self.best_selection_metric_name: str | None = None
        self.best_checkpoint_path: Path | None = None
        self.current_epoch = 0
        self.global_step = 0
        self.patience_counter = 0
        self.peak_vram_gb = 0.0

        if self.config.wandb_enabled and wandb is not None:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=self.config.to_dict(),
            )

        if resume_from:
            self.load_checkpoint(
                resume_from,
                restore_optimizer_state=not reset_training_state_on_resume,
                restore_scheduler_state=not reset_training_state_on_resume,
                restore_global_step=not reset_training_state_on_resume,
                restore_patience=not reset_training_state_on_resume,
            )
            if reset_training_state_on_resume:
                self.peak_vram_gb = 0.0
            if resume_next_epoch:
                self.current_epoch += 1

    def _build_optimizer_groups(self) -> list[dict[str, Any]]:
        backbone_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("feature_extractor.model."):
                backbone_params.append(param)
            else:
                head_params.append(param)

        optimizer_groups: list[dict[str, Any]] = []
        if head_params:
            optimizer_groups.append({"params": head_params, "lr": self.config.learning_rate})
        if backbone_params:
            optimizer_groups.append(
                {
                    "params": backbone_params,
                    "lr": self.config.learning_rate * self.config.backbone_learning_rate_scale,
                }
            )
        if not optimizer_groups:
            raise RuntimeError("No trainable parameters were found for optimization.")
        return optimizer_groups

    def _build_scheduler(self):
        if self.config.warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=max(1.0 / max(self.config.warmup_epochs, 1), 1e-3),
                end_factor=1.0,
                total_iters=self.config.warmup_epochs,
            )
            cosine_epochs = max(self.config.max_epochs - self.config.warmup_epochs, 1)
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=cosine_epochs,
                eta_min=self.config.scheduler_eta_min,
            )
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.config.warmup_epochs],
            )
        return CosineAnnealingLR(
            self.optimizer,
            T_max=max(self.config.max_epochs, 1),
            eta_min=self.config.scheduler_eta_min,
        )

    def _reset_optimization_state(self) -> None:
        optimizer_groups = self._build_optimizer_groups()
        self.optimizer = AdamW(optimizer_groups, weight_decay=self.config.weight_decay)
        self.scheduler = self._build_scheduler()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.precision == "fp16" and self.device.type == "cuda")
        self.global_step = 0
        self.patience_counter = 0
        self.peak_vram_gb = 0.0

    def _autocast(self):
        if self.device.type != "cuda" or self.precision == "fp32":
            return torch.autocast(device_type="cpu", enabled=False)
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _maybe_update_peak_vram(self) -> None:
        if self.device.type != "cuda":
            return
        peak = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
        self.peak_vram_gb = max(self.peak_vram_gb, peak)

    def train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        if getattr(self.model, "feature_extractor", None) is not None:
            if getattr(self.model.feature_extractor, "freeze_backbone", False):
                self.model.feature_extractor.eval()

        total_loss = 0.0
        total_batches = 0
        self.optimizer.zero_grad(set_to_none=True)

        for batch_index, batch in enumerate(loader, start=1):
            img1, img2, labels, _ = batch
            img1 = img1.to(self.device, non_blocking=True)
            img2 = img2.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with self._autocast():
                score1, score2 = self.model(img1, img2)
                loss = self.model.compute_loss(score1, score2, labels)
                scaled_loss = loss / self.config.gradient_accumulation_steps

            if self.scaler.is_enabled():
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if batch_index % self.config.gradient_accumulation_steps == 0 or batch_index == len(loader):
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [param for param in self.model.parameters() if param.requires_grad],
                    self.config.gradient_clip_norm,
                )
                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            total_batches += 1
            self.global_step += 1
            self._maybe_update_peak_vram()

        return total_loss / max(total_batches, 1)

    def validate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        same_dept_correct = 0
        same_dept_total = 0

        with torch.no_grad():
            for img1, img2, labels, is_same_dept in loader:
                img1 = img1.to(self.device, non_blocking=True)
                img2 = img2.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                is_same_dept = is_same_dept.to(self.device, non_blocking=True)

                with self._autocast():
                    score1, score2 = self.model(img1, img2)
                    loss = self.model.compute_loss(score1, score2, labels)

                predictions = torch.sign(score1 - score2).squeeze(-1)
                labels_long = labels.float()
                correct_mask = predictions == labels_long

                total_loss += loss.item()
                total_correct += int(correct_mask.sum().item())
                total_samples += labels.shape[0]

                same_mask = is_same_dept == 1
                if same_mask.any():
                    same_dept_correct += int((correct_mask & same_mask).sum().item())
                    same_dept_total += int(same_mask.sum().item())

                self._maybe_update_peak_vram()

        return {
            "val_loss": total_loss / max(len(loader), 1),
            "val_accuracy": total_correct / max(total_samples, 1),
            "same_dept_accuracy": same_dept_correct / max(same_dept_total, 1) if same_dept_total else 0.0,
            "total_pairs": total_samples,
        }

    def save_checkpoint(self, name: str, extra: dict[str, Any] | None = None) -> Path:
        checkpoint_path = Path(self.config.checkpoint_dir) / name
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_selection_metric": self.best_selection_metric,
            "best_selection_metric_name": self.best_selection_metric_name,
            "patience_counter": self.patience_counter,
            "config": self.config.to_dict(),
            "peak_vram_gb": self.peak_vram_gb,
        }
        if extra:
            checkpoint.update(extra)
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def update_best_checkpoint_link(self, checkpoint_path: Path) -> Path:
        best_link = Path(self.config.checkpoint_dir) / "best_model.pt"
        best_link.unlink(missing_ok=True)
        best_link.symlink_to(checkpoint_path.name)
        self.best_checkpoint_path = checkpoint_path.resolve()
        return best_link

    def _resolve_best_checkpoint_path(self) -> Path | None:
        best_link = Path(self.config.checkpoint_dir) / "best_model.pt"
        if best_link.is_symlink():
            resolved = best_link.resolve()
            if resolved.exists():
                return resolved
        if self.best_checkpoint_path and self.best_checkpoint_path.exists():
            return self.best_checkpoint_path
        return None

    def _restart_from_best_checkpoint(self) -> dict[str, Any] | None:
        best_checkpoint_path = self._resolve_best_checkpoint_path()
        if best_checkpoint_path is None:
            return None
        current_epoch = self.current_epoch
        patience_counter_before_restart = self.patience_counter
        self._reset_optimization_state()
        self.load_checkpoint(
            str(best_checkpoint_path),
            restore_optimizer_state=False,
            restore_scheduler_state=False,
            restore_global_step=False,
            restore_patience=False,
        )
        self.current_epoch = current_epoch
        self.best_checkpoint_path = best_checkpoint_path
        return {
            "trigger_epoch": current_epoch + 1,
            "best_checkpoint": str(best_checkpoint_path),
            "best_checkpoint_name": best_checkpoint_path.name,
            "patience_counter_before_restart": patience_counter_before_restart,
        }

    def load_checkpoint(
        self,
        checkpoint_path: str,
        *,
        restore_optimizer_state: bool = True,
        restore_scheduler_state: bool = True,
        restore_global_step: bool = True,
        restore_patience: bool = True,
    ) -> None:
        checkpoint = torch.load(resolve_project_path(checkpoint_path), map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if restore_optimizer_state:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if restore_scheduler_state:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = int(checkpoint.get("epoch", 0))
        if restore_global_step:
            self.global_step = int(checkpoint.get("global_step", 0))
        else:
            self.global_step = 0
        self.best_val_loss = float(checkpoint.get("best_val_loss", math.inf))
        best_selection_metric = checkpoint.get("best_selection_metric")
        self.best_selection_metric = float(best_selection_metric) if best_selection_metric is not None else None
        best_selection_metric_name = checkpoint.get("best_selection_metric_name")
        self.best_selection_metric_name = str(best_selection_metric_name) if best_selection_metric_name else None
        if restore_patience:
            self.patience_counter = int(checkpoint.get("patience_counter", 0))
        else:
            self.patience_counter = 0
        self.peak_vram_gb = float(checkpoint.get("peak_vram_gb", 0.0))
        best_link = Path(self.config.checkpoint_dir) / "best_model.pt"
        if best_link.is_symlink():
            resolved = best_link.resolve()
            if resolved.exists():
                self.best_checkpoint_path = resolved

    def _resolve_selection_metric(
        self,
        validation_metrics: dict[str, float],
        postprocess_result: dict[str, Any] | None,
    ) -> tuple[str, float]:
        if self.config.early_stopping_metric == "anchor_tier_accuracy" and postprocess_result:
            report = postprocess_result.get("report")
            report_metrics = report.get("metrics") if isinstance(report, dict) else None
            metric_value = (
                report_metrics.get("anchor_tier_accuracy_mean")
                if isinstance(report_metrics, dict)
                else None
            )
            if metric_value is None and isinstance(report_metrics, dict):
                metric_value = report_metrics.get("anchor_tier_accuracy")
            if metric_value is not None:
                return "anchor_tier_accuracy_mean", float(metric_value)
        return "val_loss", float(validation_metrics["val_loss"])

    def _is_metric_improved(self, metric_name: str, candidate_value: float) -> bool:
        if metric_name == "val_loss":
            return candidate_value < self.best_val_loss - self.config.early_stopping_min_delta
        if self.best_selection_metric is None:
            return True
        return candidate_value > self.best_selection_metric + self.config.early_stopping_min_delta

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        post_epoch_callback: Callable[[Path, dict[str, float]], dict[str, Any] | None] | None = None,
    ) -> dict[str, Any]:
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "same_dept_accuracy": [],
            "anchor_tier_accuracy": [],
        }
        latest_completed_checkpoint: Path | None = None
        latest_completed_metrics: dict[str, float] | None = None
        restart_from_best_events: list[dict[str, Any]] = []

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch
            train_loss = self.train_one_epoch(train_loader)
            metrics = self.validate(val_loader)
            self.scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(metrics["val_loss"])
            history["val_accuracy"].append(metrics["val_accuracy"])
            history["same_dept_accuracy"].append(metrics["same_dept_accuracy"])

            checkpoint_metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                **metrics,
            }

            if self.config.wandb_enabled and wandb is not None:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": train_loss,
                        "val/loss": metrics["val_loss"],
                        "val/accuracy": metrics["val_accuracy"],
                        "val/same_dept_accuracy": metrics["same_dept_accuracy"],
                        "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                        "system/peak_vram_gb": self.peak_vram_gb,
                    }
                )

            latest_completed_checkpoint = self.save_checkpoint(
                f"checkpoint_epoch_{epoch + 1:04d}.pt",
                extra={"epoch_metrics": checkpoint_metrics},
            )
            latest_completed_metrics = checkpoint_metrics
            postprocess_result: dict[str, Any] | None = None
            best_checkpoint_target: Path | None = None
            if post_epoch_callback is not None:
                postprocess_result = post_epoch_callback(latest_completed_checkpoint, checkpoint_metrics)
            metric_name, selection_metric_value = self._resolve_selection_metric(metrics, postprocess_result)
            if metric_name == "anchor_tier_accuracy_mean":
                history["anchor_tier_accuracy"].append(selection_metric_value)

            current_best_val_loss = min(self.best_val_loss, metrics["val_loss"])
            if self._is_metric_improved(metric_name, selection_metric_value):
                if metric_name == "val_loss":
                    self.best_val_loss = selection_metric_value
                    self.best_selection_metric_name = "val_loss"
                else:
                    self.best_selection_metric = selection_metric_value
                    self.best_selection_metric_name = metric_name
                    self.best_val_loss = current_best_val_loss
                self.patience_counter = 0
                best_checkpoint_target = latest_completed_checkpoint
            else:
                self.patience_counter += 1
                self.best_val_loss = current_best_val_loss

            if latest_completed_checkpoint is not None:
                self.save_checkpoint(
                    latest_completed_checkpoint.name,
                    extra={"epoch_metrics": checkpoint_metrics},
                )
            if best_checkpoint_target is not None:
                self.update_best_checkpoint_link(best_checkpoint_target)

            if (
                self.config.restart_from_best_patience > 0
                and self.patience_counter >= self.config.restart_from_best_patience
            ):
                restart_event = self._restart_from_best_checkpoint()
                if restart_event is not None:
                    restart_event["selection_metric_name"] = metric_name
                    restart_event["selection_metric_value"] = selection_metric_value
                    restart_from_best_events.append(restart_event)
                    print(
                        json.dumps(
                            {
                                "event": "restart_from_best",
                                **restart_event,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

            if self.patience_counter >= self.config.early_stopping_patience:
                break

        if self.config.wandb_enabled and wandb is not None:
            wandb.finish()

        return {
            "history": history,
            "peak_vram_gb": self.peak_vram_gb,
            "best_val_loss": self.best_val_loss,
            "epochs_completed": self.current_epoch + 1,
            "precision": self.precision,
            "latest_completed_checkpoint": str(latest_completed_checkpoint) if latest_completed_checkpoint else None,
            "latest_completed_metrics": latest_completed_metrics,
            "restart_from_best_events": restart_from_best_events,
        }
