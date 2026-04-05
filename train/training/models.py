from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


def resolve_backbone_dtype(backbone_dtype: str) -> torch.dtype | None:
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


class DinoV3FeatureExtractor(nn.Module):
    def __init__(
        self,
        model_name: str,
        normalize: bool = True,
        freeze_backbone: bool = True,
        backbone_dtype: str = "auto",
        unfreeze_last_n_layers: int = 0,
        feature_pool: str = "cls_mean_patch_concat",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.normalize = normalize
        self.freeze_backbone = freeze_backbone
        self.backbone_dtype = backbone_dtype
        self.unfreeze_last_n_layers = unfreeze_last_n_layers
        self.feature_pool = feature_pool
        resolved_dtype = resolve_backbone_dtype(backbone_dtype)
        model_kwargs: dict[str, object] = {
            "low_cpu_mem_usage": True,
        }
        if resolved_dtype is not None:
            model_kwargs["torch_dtype"] = resolved_dtype
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
        hidden_size = int(self.model.config.hidden_size)
        self.output_dim = hidden_size * 2 if feature_pool == "cls_mean_patch_concat" else hidden_size
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            if unfreeze_last_n_layers > 0:
                self._unfreeze_last_layers(unfreeze_last_n_layers)
                self.freeze_backbone = False
            else:
                self.model.eval()
        self._configure_gradient_checkpointing()

    def _configure_gradient_checkpointing(self) -> None:
        enabled = not self.freeze_backbone
        if enabled and hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        toggle_name = "gradient_checkpointing_enable" if enabled else "gradient_checkpointing_disable"
        toggle = getattr(self.model, toggle_name, None)
        if callable(toggle):
            toggle()

    def _resolve_backbone_layers(self) -> list[nn.Module]:
        layer_candidates = (
            getattr(self.model, "layer", None),
            getattr(getattr(self.model, "encoder", None), "layer", None),
            getattr(getattr(getattr(self.model, "vision_model", None), "encoder", None), "layer", None),
        )
        for candidate in layer_candidates:
            if candidate is None:
                continue
            try:
                return list(candidate)
            except TypeError:
                continue
        raise RuntimeError(
            "AutoModel backbone does not expose a supported transformer layer stack for selective unfreezing."
        )

    def _resolve_final_norms(self) -> list[nn.Module]:
        norm_candidates = (
            getattr(self.model, "norm", None),
            getattr(self.model, "layernorm", None),
            getattr(self.model, "post_layernorm", None),
            getattr(getattr(self.model, "encoder", None), "layernorm", None),
            getattr(getattr(self.model, "vision_model", None), "layernorm", None),
        )
        return [module for module in norm_candidates if module is not None]

    def _unfreeze_final_norms(self) -> None:
        norm_candidates = self._resolve_final_norms()
        for module in norm_candidates:
            for param in module.parameters():
                param.requires_grad = True

    def _unfreeze_last_layers(self, layer_count: int) -> None:
        backbone_layers = self._resolve_backbone_layers()
        total_layers = len(backbone_layers)
        if total_layers == 0:
            raise RuntimeError("AutoModel backbone contains no layers to unfreeze.")
        start_index = max(total_layers - layer_count, 0)
        for index in range(start_index, total_layers):
            for param in backbone_layers[index].parameters():
                param.requires_grad = True
        self._unfreeze_final_norms()

    def train(self, mode: bool = True) -> DinoV3FeatureExtractor:
        super().train(mode)
        if self.freeze_backbone:
            self.model.eval()
            return self
        if self.unfreeze_last_n_layers > 0:
            self.model.eval()
            for layer in self._resolve_backbone_layers()[-self.unfreeze_last_n_layers :]:
                layer.train(mode)
            for norm in self._resolve_final_norms():
                norm.train(mode)
        return self

    def _pool_last_hidden_state(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        cls_features = last_hidden_state[:, 0, :]
        if self.feature_pool == "cls":
            return F.normalize(cls_features, p=2, dim=1) if self.normalize else cls_features
        if self.feature_pool == "cls_mean_patch_concat":
            if last_hidden_state.shape[1] > 1:
                patch_features = last_hidden_state[:, 1:, :].mean(dim=1)
            else:
                patch_features = cls_features
            if self.normalize:
                cls_features = F.normalize(cls_features, p=2, dim=1)
                patch_features = F.normalize(patch_features, p=2, dim=1)
            return torch.cat((cls_features, patch_features), dim=1)
        raise ValueError(f"Unsupported feature_pool: {self.feature_pool}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values)
        else:
            outputs = self.model(pixel_values=pixel_values)

        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if last_hidden_state is None:
            raise RuntimeError("AutoModel output missing last_hidden_state")
        return self._pool_last_hidden_state(last_hidden_state)


def resolve_pairwise_model_kwargs(config_dict: dict[str, object]) -> dict[str, object]:
    return {
        "model_name": str(config_dict["model_name"]),
        "dropout": float(config_dict.get("dropout", 0.1)),
        "margin": float(config_dict.get("margin", 0.3)),
        "freeze_backbone": bool(config_dict.get("freeze_backbone", True)),
        "backbone_dtype": str(config_dict.get("backbone_dtype", "auto")),
        "unfreeze_last_n_layers": int(config_dict.get("unfreeze_last_n_layers", 0)),
        "feature_pool": str(config_dict.get("feature_pool", "cls_mean_patch_concat")),
        "head_type": str(config_dict.get("head_type", "mlp_small")),
    }


class DinoV3PairwiseModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        projector_hidden_dim: int = 512,
        projector_output_dim: int = 256,
        dropout: float = 0.1,
        margin: float = 0.3,
        freeze_backbone: bool = True,
        backbone_dtype: str = "auto",
        unfreeze_last_n_layers: int = 0,
        feature_pool: str = "cls_mean_patch_concat",
        head_type: str = "mlp_small",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.margin = margin
        self.head_type = head_type
        self.feature_extractor = DinoV3FeatureExtractor(
            model_name=model_name,
            normalize=True,
            freeze_backbone=freeze_backbone,
            backbone_dtype=backbone_dtype,
            unfreeze_last_n_layers=unfreeze_last_n_layers,
            feature_pool=feature_pool,
        )
        feature_dim = self.feature_extractor.output_dim
        self.projector = nn.Identity()
        if head_type == "linear":
            self.score_head = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, 1),
            )
        elif head_type == "mlp_small":
            self.score_head = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, 1024),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 1),
            )
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")
        self.loss_fn = nn.MarginRankingLoss(margin=margin)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in list(self.score_head.modules()):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(pixel_values)

    @staticmethod
    def _align_tensor_for_module(tensor: torch.Tensor, module: nn.Module) -> torch.Tensor:
        parameter = next(module.parameters(), None)
        if parameter is None:
            return tensor
        if tensor.device != parameter.device:
            tensor = tensor.to(parameter.device)
        if not torch.is_autocast_enabled() and tensor.dtype != parameter.dtype:
            tensor = tensor.to(dtype=parameter.dtype)
        return tensor

    def project_features(self, features: torch.Tensor) -> torch.Tensor:
        features = self._align_tensor_for_module(features, self.projector)
        return self.projector(features)

    def score_features(self, projected_features: torch.Tensor) -> torch.Tensor:
        projected_features = self._align_tensor_for_module(projected_features, self.score_head)
        return self.score_head(projected_features)

    def predict_score(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(pixel_values)
        projected = self.project_features(features)
        return self.score_features(projected)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.feature_extractor.freeze_backbone:
            return self.predict_score(img1), self.predict_score(img2)
        merged = torch.cat((img1, img2), dim=0)
        merged_scores = self.predict_score(merged)
        score1, score2 = torch.chunk(merged_scores, chunks=2, dim=0)
        return score1, score2

    def compute_loss(
        self,
        score1: torch.Tensor,
        score2: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return self.loss_fn(score1.squeeze(-1), score2.squeeze(-1), labels.float())
