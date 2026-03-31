from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

from config import ModelsConfig
from utils import DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD


def resolve_backbone_dtype(backbone_dtype: str) -> torch.dtype | None:
    """Resolves the requested backbone dtype with CUDA capability-aware fallback."""

    if backbone_dtype == "bf16":
        return torch.bfloat16
    if backbone_dtype == "fp16":
        return torch.float16
    if backbone_dtype == "fp32":
        return torch.float32
    if not torch.cuda.is_available():
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def map_teacher_layers(student_depth: int, teacher_depth: int) -> list[int]:
    """Maps student layers to evenly spaced teacher layers for mid-level distillation."""

    if student_depth <= 0 or teacher_depth <= 0:
        return []
    positions = torch.linspace(0, teacher_depth - 1, steps=student_depth)
    return [int(round(position.item())) for position in positions]


@dataclass
class BackboneOutputs:
    """Canonicalized backbone feature outputs consumed by distillation losses."""

    patch_tokens: torch.Tensor
    cls_token: torch.Tensor
    pooled_output: torch.Tensor
    hidden_states: list[torch.Tensor]
    patch_grid_hw: tuple[int, int]
    backend: str


@dataclass
class DistillationBatch:
    """Teacher/student feature bundle after projection, alignment, and optional normalization."""

    teacher: BackboneOutputs
    student: BackboneOutputs
    teacher_patch: torch.Tensor
    student_patch: torch.Tensor
    teacher_cls: torch.Tensor
    student_cls: torch.Tensor
    teacher_pool: torch.Tensor
    student_pool: torch.Tensor
    teacher_mid: list[torch.Tensor]
    student_mid: list[torch.Tensor]


class FeatureProjector(nn.Module):
    """Projects student features into the teacher hidden size before loss computation."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(features))


def _unwrap_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    if hasattr(output, "last_hidden_state") and isinstance(output.last_hidden_state, torch.Tensor):
        return output.last_hidden_state
    raise TypeError(f"Unable to unwrap tensor from output type: {type(output)!r}")


def _infer_patch_size(config: Any, model: nn.Module) -> int:
    for candidate in (
        getattr(config, "patch_size", None),
        getattr(getattr(config, "vision_config", None), "patch_size", None),
        getattr(getattr(model, "patch_embed", None), "patch_size", None),
    ):
        if candidate is None:
            continue
        if isinstance(candidate, (tuple, list)):
            return int(candidate[0])
        return int(candidate)
    return 16


def _infer_hidden_size(config: Any, model: nn.Module) -> int:
    for candidate in (
        getattr(config, "hidden_size", None),
        getattr(getattr(config, "vision_config", None), "hidden_size", None),
        getattr(model, "num_features", None),
        getattr(model, "embed_dim", None),
    ):
        if candidate is not None:
            return int(candidate)
    raise RuntimeError("Unable to infer hidden size for backbone")


def _resolve_transformer_blocks(model: nn.Module) -> list[nn.Module]:
    candidates: list[Any] = [
        getattr(model, "blocks", None),
        getattr(getattr(model, "encoder", None), "layer", None),
        getattr(getattr(model, "vision_model", None), "encoder", None),
        getattr(getattr(getattr(model, "vision_model", None), "encoder", None), "layer", None),
        getattr(getattr(model, "model", None), "blocks", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if hasattr(candidate, "layer"):
            candidate = candidate.layer
        if isinstance(candidate, (list, nn.ModuleList, tuple)):
            return list(candidate)
    return []


def _infer_patch_grid(pixel_values: torch.Tensor, patch_size: int) -> tuple[int, int]:
    height, width = pixel_values.shape[-2:]
    return max(1, height // patch_size), max(1, width // patch_size)


def _split_tokens(sequence: torch.Tensor, patch_grid_hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = patch_grid_hw
    expected_patches = h * w
    if sequence.dim() != 3:
        raise ValueError(f"Expected token sequence with shape [B, N, C], got {tuple(sequence.shape)}")
    if sequence.shape[1] < expected_patches:
        raise ValueError("Sequence length is smaller than the inferred patch grid")
    patch_tokens = sequence[:, -expected_patches:, :]
    extra_tokens = sequence[:, : sequence.shape[1] - expected_patches, :]
    if extra_tokens.shape[1] > 0:
        cls_token = extra_tokens[:, 0, :]
    else:
        cls_token = patch_tokens.mean(dim=1)
    return patch_tokens, cls_token


def _feature_sequence(feature_tensor: torch.Tensor, patch_grid_hw: tuple[int, int]) -> torch.Tensor:
    if feature_tensor.dim() == 4:
        batch_size, channels, height, width = feature_tensor.shape
        return feature_tensor.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
    if feature_tensor.dim() == 3:
        return feature_tensor
    if feature_tensor.dim() == 2:
        return feature_tensor.unsqueeze(1)
    raise ValueError(f"Unsupported feature tensor shape: {tuple(feature_tensor.shape)}")


def align_patch_tokens(
    patch_tokens: torch.Tensor,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
) -> torch.Tensor:
    if src_hw == dst_hw:
        return patch_tokens
    batch_size, _, channels = patch_tokens.shape
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    feature_map = patch_tokens.reshape(batch_size, src_h, src_w, channels).permute(0, 3, 1, 2)
    resized = F.interpolate(feature_map.float(), size=(dst_h, dst_w), mode="bilinear", align_corners=False)
    return resized.permute(0, 2, 3, 1).reshape(batch_size, dst_h * dst_w, channels)


def _resolve_requested_indices(hidden_state_indices: Sequence[int] | None, depth: int) -> tuple[list[int], list[int]]:
    requested = list(hidden_state_indices or range(depth))
    unique_requested = sorted(set(requested))
    return requested, unique_requested


class BackboneAdapter(nn.Module):
    """Backend-agnostic wrapper around a vision backbone used for distillation."""

    def __init__(self, *, model_name: str, backend: str, freeze: bool) -> None:
        super().__init__()
        self.model_name = model_name
        self.backend = backend
        self.freeze = freeze
        self.patch_size = 16
        self.hidden_size = 0
        self.depth = 0
        self.image_mean = list(DEFAULT_IMAGE_MEAN)
        self.image_std = list(DEFAULT_IMAGE_STD)
        self.can_export_hf = False

    def extract_features(
        self,
        pixel_values: torch.Tensor,
        *,
        output_hidden_states: bool = True,
        hidden_state_indices: Sequence[int] | None = None,
    ) -> BackboneOutputs:
        raise NotImplementedError

    def export_backbone(self, path: str | Path) -> None:
        raise RuntimeError(f"Backbone export is not supported for backend={self.backend}")

    def trainable_state_dict(self) -> dict[str, Any]:
        return self.state_dict()

    def load_trainable_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.load_state_dict(state_dict)


class HuggingFaceBackboneAdapter(BackboneAdapter):
    """Loads a Hugging Face vision backbone and exposes a uniform feature extraction API."""

    def __init__(
        self,
        *,
        model_name: str,
        freeze: bool,
        backbone_dtype: str = "auto",
        trust_remote_code: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__(model_name=model_name, backend="huggingface", freeze=freeze)
        self.processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.image_mean = list(getattr(self.processor, "image_mean", self.image_mean))
        self.image_std = list(getattr(self.processor, "image_std", self.image_std))
        model_kwargs: dict[str, Any] = {
            "low_cpu_mem_usage": True,
            "trust_remote_code": trust_remote_code,
        }
        resolved_dtype = resolve_backbone_dtype(backbone_dtype)
        if resolved_dtype is not None:
            model_kwargs["torch_dtype"] = resolved_dtype
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.patch_size = _infer_patch_size(self.model.config, self.model)
        self.hidden_size = _infer_hidden_size(self.model.config, self.model)
        self.blocks = _resolve_transformer_blocks(self.model)
        self.depth = len(self.blocks) or int(getattr(self.model.config, "num_hidden_layers", 0))
        self.can_export_hf = True
        if gradient_checkpointing and not freeze and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def export_backbone(self, path: str | Path) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(target)
        self.processor.save_pretrained(target)

    def extract_features(
        self,
        pixel_values: torch.Tensor,
        *,
        output_hidden_states: bool = True,
        hidden_state_indices: Sequence[int] | None = None,
    ) -> BackboneOutputs:
        grid_hw = _infer_patch_grid(pixel_values, self.patch_size)
        if self.freeze:
            self.model.eval()
        requested_indices, capture_indices = _resolve_requested_indices(hidden_state_indices, self.depth)
        captured: list[tuple[int, torch.Tensor]] = []
        hooks: list[Any] = []

        def _make_hook(index: int):
            def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
                captured.append((index, _unwrap_tensor(output)))

            return _hook

        use_fallback_hidden_states = output_hidden_states and not self.blocks
        if output_hidden_states and self.blocks:
            for index, block in enumerate(self.blocks):
                if index in capture_indices:
                    hooks.append(block.register_forward_hook(_make_hook(index)))

        context = torch.no_grad() if self.freeze else nullcontext()
        with context:
            outputs = self.model(
                pixel_values=pixel_values,
                return_dict=True,
                output_hidden_states=use_fallback_hidden_states,
            )

        for hook in hooks:
            hook.remove()

        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if last_hidden_state is None:
            last_hidden_state = _unwrap_tensor(outputs)
        patch_tokens, cls_token = _split_tokens(last_hidden_state, grid_hw)
        pooled_output = getattr(outputs, "pooler_output", None)
        if pooled_output is None:
            pooled_output = patch_tokens.mean(dim=1)

        hidden_states: list[torch.Tensor] = []
        if output_hidden_states:
            if use_fallback_hidden_states:
                raw_hidden_states = list(getattr(outputs, "hidden_states", []) or [])
                layer_by_index: dict[int, torch.Tensor] = {}
                for index in capture_indices:
                    if 0 <= (index + 1) < len(raw_hidden_states):
                        layer_sequence = _feature_sequence(raw_hidden_states[index + 1], grid_hw)
                        layer_patch, _ = _split_tokens(layer_sequence, grid_hw)
                        layer_by_index[index] = layer_patch
                hidden_states = [layer_by_index[index] for index in requested_indices if index in layer_by_index]
            else:
                layer_by_index: dict[int, torch.Tensor] = {}
                for index, tensor in sorted(captured, key=lambda item: item[0]):
                    layer_sequence = _feature_sequence(tensor, grid_hw)
                    layer_patch, _ = _split_tokens(layer_sequence, grid_hw)
                    layer_by_index[index] = layer_patch
                hidden_states = [layer_by_index[index] for index in requested_indices if index in layer_by_index]
        return BackboneOutputs(
            patch_tokens=patch_tokens,
            cls_token=cls_token,
            pooled_output=pooled_output,
            hidden_states=hidden_states,
            patch_grid_hw=grid_hw,
            backend=self.backend,
        )


class TimmBackboneAdapter(BackboneAdapter):
    """Fallback adapter for timm vision backbones."""

    def __init__(
        self,
        *,
        model_name: str,
        freeze: bool,
        backbone_dtype: str = "auto",
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__(model_name=model_name, backend="timm", freeze=freeze)
        try:
            import timm
        except ImportError as exc:
            raise ImportError("timm is required for timm backbone fallback") from exc
        self.timm = timm
        create_kwargs = {"pretrained": True, "num_classes": 0, "global_pool": ""}
        try:
            self.model = timm.create_model(model_name, **create_kwargs)
        except Exception as exc:
            raise RuntimeError(f"Failed to create timm model '{model_name}'") from exc
        self.patch_size = _infer_patch_size(getattr(self.model, "default_cfg", {}), self.model)
        self.hidden_size = _infer_hidden_size(getattr(self.model, "default_cfg", {}), self.model)
        self.blocks = _resolve_transformer_blocks(self.model)
        self.depth = len(self.blocks)
        if gradient_checkpointing and not freeze and hasattr(self.model, "set_grad_checkpointing"):
            self.model.set_grad_checkpointing(True)
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def extract_features(
        self,
        pixel_values: torch.Tensor,
        *,
        output_hidden_states: bool = True,
        hidden_state_indices: Sequence[int] | None = None,
    ) -> BackboneOutputs:
        grid_hw = _infer_patch_grid(pixel_values, self.patch_size)
        requested_indices, capture_indices = _resolve_requested_indices(hidden_state_indices, self.depth)
        captured: list[tuple[int, torch.Tensor]] = []
        hooks: list[Any] = []

        def _make_hook(index: int):
            def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
                captured.append((index, _unwrap_tensor(output)))

            return _hook

        if output_hidden_states:
            for index, block in enumerate(self.blocks):
                if index in capture_indices:
                    hooks.append(block.register_forward_hook(_make_hook(index)))

        context = torch.no_grad() if self.freeze else nullcontext()
        with context:
            if hasattr(self.model, "forward_features"):
                features = self.model.forward_features(pixel_values)
            else:
                features = self.model(pixel_values)

        for hook in hooks:
            hook.remove()

        sequence = _feature_sequence(_unwrap_tensor(features), grid_hw)
        patch_tokens, cls_token = _split_tokens(sequence, grid_hw)
        pooled_output = patch_tokens.mean(dim=1)
        hidden_states: list[torch.Tensor] = []
        if output_hidden_states:
            layer_by_index: dict[int, torch.Tensor] = {}
            for index, tensor in sorted(captured, key=lambda item: item[0]):
                layer_sequence = _feature_sequence(tensor, grid_hw)
                layer_patch, _ = _split_tokens(layer_sequence, grid_hw)
                layer_by_index[index] = layer_patch
            hidden_states = [layer_by_index[index] for index in requested_indices if index in layer_by_index]
        return BackboneOutputs(
            patch_tokens=patch_tokens,
            cls_token=cls_token,
            pooled_output=pooled_output,
            hidden_states=hidden_states,
            patch_grid_hw=grid_hw,
            backend=self.backend,
        )


def build_backbone_adapter(
    *,
    model_name: str,
    freeze: bool,
    backbone_dtype: str,
    backend_order: Sequence[str],
    trust_remote_code: bool,
    gradient_checkpointing: bool,
) -> BackboneAdapter:
    """Builds the first working backbone adapter from the configured backend order."""

    last_error: Exception | None = None
    for backend in backend_order:
        try:
            if backend == "huggingface":
                return HuggingFaceBackboneAdapter(
                    model_name=model_name,
                    freeze=freeze,
                    backbone_dtype=backbone_dtype,
                    trust_remote_code=trust_remote_code,
                    gradient_checkpointing=gradient_checkpointing,
                )
            if backend == "timm":
                return TimmBackboneAdapter(
                    model_name=model_name,
                    freeze=freeze,
                    backbone_dtype=backbone_dtype,
                    gradient_checkpointing=gradient_checkpointing,
                )
            raise ValueError(f"Unsupported backend: {backend}")
        except Exception as exc:  # pragma: no cover - fallback path depends on environment
            last_error = exc
    if last_error is None:
        raise RuntimeError("No backbone backend candidates were provided")
    raise RuntimeError(f"Unable to initialize backbone '{model_name}' from backends {list(backend_order)}") from last_error


class TeacherStudentDistillModel(nn.Module):
    """Teacher-student wrapper that aligns features for dense representation distillation."""

    def __init__(self, config: ModelsConfig, *, normalize_features: bool = True) -> None:
        super().__init__()
        self.normalize_features = normalize_features
        self.teacher = build_backbone_adapter(
            model_name=config.teacher_name,
            freeze=True,
            backbone_dtype=config.teacher_dtype,
            backend_order=config.backend_order,
            trust_remote_code=config.trust_remote_code,
            gradient_checkpointing=False,
        )
        self.student = build_backbone_adapter(
            model_name=config.student_name,
            freeze=False,
            backbone_dtype=config.student_dtype,
            backend_order=config.backend_order,
            trust_remote_code=config.trust_remote_code,
            gradient_checkpointing=config.gradient_checkpointing,
        )
        self.teacher_layer_indices = map_teacher_layers(self.student.depth, self.teacher.depth)
        self.student_layer_indices = list(range(self.student.depth))
        self.patch_projector = FeatureProjector(self.student.hidden_size, self.teacher.hidden_size)
        self.cls_projector = FeatureProjector(self.student.hidden_size, self.teacher.hidden_size)
        self.pool_projector = FeatureProjector(self.student.hidden_size, self.teacher.hidden_size)
        self.mid_projectors = nn.ModuleList(
            [FeatureProjector(self.student.hidden_size, self.teacher.hidden_size) for _ in self.student_layer_indices]
        )
        self.image_mean = list(self.teacher.image_mean or self.student.image_mean)
        self.image_std = list(self.teacher.image_std or self.student.image_std)
        self.patch_size = self.student.patch_size or self.teacher.patch_size

    def _maybe_normalize(self, features: torch.Tensor) -> torch.Tensor:
        if not self.normalize_features:
            return features
        return F.normalize(features, dim=-1)

    def forward(self, pixel_values: torch.Tensor) -> DistillationBatch:
        with torch.no_grad():
            teacher_outputs = self.teacher.extract_features(
                pixel_values,
                output_hidden_states=True,
                hidden_state_indices=self.teacher_layer_indices,
            )
        student_outputs = self.student.extract_features(
            pixel_values,
            output_hidden_states=True,
            hidden_state_indices=self.student_layer_indices,
        )

        aligned_student_patch = align_patch_tokens(
            student_outputs.patch_tokens,
            student_outputs.patch_grid_hw,
            teacher_outputs.patch_grid_hw,
        )
        student_patch = self._maybe_normalize(self.patch_projector(aligned_student_patch))
        teacher_patch = self._maybe_normalize(teacher_outputs.patch_tokens)

        student_cls = self._maybe_normalize(self.cls_projector(student_outputs.cls_token))
        teacher_cls = self._maybe_normalize(teacher_outputs.cls_token)

        student_pool = self._maybe_normalize(self.pool_projector(student_outputs.pooled_output))
        teacher_pool = self._maybe_normalize(teacher_outputs.pooled_output)

        aligned_student_mid: list[torch.Tensor] = []
        aligned_teacher_mid: list[torch.Tensor] = []
        for projector, student_hidden, teacher_hidden in zip(
            self.mid_projectors,
            student_outputs.hidden_states,
            teacher_outputs.hidden_states,
        ):
            aligned_student = align_patch_tokens(
                student_hidden,
                student_outputs.patch_grid_hw,
                teacher_outputs.patch_grid_hw,
            )
            aligned_student_mid.append(self._maybe_normalize(projector(aligned_student)))
            aligned_teacher_mid.append(self._maybe_normalize(teacher_hidden))

        return DistillationBatch(
            teacher=teacher_outputs,
            student=student_outputs,
            teacher_patch=teacher_patch,
            student_patch=student_patch,
            teacher_cls=teacher_cls,
            student_cls=student_cls,
            teacher_pool=teacher_pool,
            student_pool=student_pool,
            teacher_mid=aligned_teacher_mid,
            student_mid=aligned_student_mid,
        )

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Returns only parameters that should be optimized."""

        return [param for param in self.parameters() if param.requires_grad]

    def checkpoint_state(self) -> dict[str, Any]:
        """Serializes the trainable student state and projector weights."""

        return {
            "student_adapter": self.student.trainable_state_dict(),
            "patch_projector": self.patch_projector.state_dict(),
            "cls_projector": self.cls_projector.state_dict(),
            "pool_projector": self.pool_projector.state_dict(),
            "mid_projectors": self.mid_projectors.state_dict(),
            "teacher_layer_indices": self.teacher_layer_indices,
            "student_layer_indices": self.student_layer_indices,
        }

    def load_checkpoint_state(self, payload: dict[str, Any]) -> None:
        """Restores the serialized student backbone and projector weights."""

        self.student.load_trainable_state_dict(payload["student_adapter"])
        self.patch_projector.load_state_dict(payload["patch_projector"])
        self.cls_projector.load_state_dict(payload["cls_projector"])
        self.pool_projector.load_state_dict(payload["pool_projector"])
        self.mid_projectors.load_state_dict(payload["mid_projectors"])

    def export_student_backbone(self, path: str | Path) -> None:
        """Exports the distilled student backbone in Hugging Face format when supported."""

        self.student.export_backbone(path)
