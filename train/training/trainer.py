from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.precision = resolve_precision(config.precision, "cuda" if self.device.type == "cuda" else "cpu")
        self.model.to(self.device)

        trainable_params = [param for param in self.model.parameters() if param.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_epochs,
            eta_min=config.scheduler_eta_min,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.precision == "fp16" and self.device.type == "cuda")
        self.best_val_loss = math.inf
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
            self.load_checkpoint(resume_from)

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
            "patience_counter": self.patience_counter,
            "config": self.config.to_dict(),
            "peak_vram_gb": self.peak_vram_gb,
        }
        if extra:
            checkpoint.update(extra)
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(resolve_project_path(checkpoint_path), map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = int(checkpoint.get("epoch", 0))
        self.global_step = int(checkpoint.get("global_step", 0))
        self.best_val_loss = float(checkpoint.get("best_val_loss", math.inf))
        self.patience_counter = int(checkpoint.get("patience_counter", 0))
        self.peak_vram_gb = float(checkpoint.get("peak_vram_gb", 0.0))

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> dict[str, Any]:
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "same_dept_accuracy": [],
        }
        latest_completed_checkpoint: Path | None = None
        latest_completed_metrics: dict[str, float] | None = None

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
            if metrics["val_loss"] < self.best_val_loss - self.config.early_stopping_min_delta:
                self.best_val_loss = metrics["val_loss"]
                self.patience_counter = 0
                self.save_checkpoint(
                    "best_model.pt",
                    extra={
                        "best_metrics": metrics,
                        "source_checkpoint": latest_completed_checkpoint.name if latest_completed_checkpoint else None,
                    },
                )
            else:
                self.patience_counter += 1

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
        }
