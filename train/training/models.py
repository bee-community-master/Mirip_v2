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
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.normalize = normalize
        self.freeze_backbone = freeze_backbone
        self.backbone_dtype = backbone_dtype
        self.unfreeze_last_n_layers = unfreeze_last_n_layers
        resolved_dtype = resolve_backbone_dtype(backbone_dtype)
        model_kwargs: dict[str, object] = {
            "low_cpu_mem_usage": True,
        }
        if resolved_dtype is not None:
            model_kwargs["torch_dtype"] = resolved_dtype
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.output_dim = int(self.model.config.hidden_size)
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            if unfreeze_last_n_layers > 0:
                self._unfreeze_last_layers(unfreeze_last_n_layers)
                self.freeze_backbone = False
            else:
                self.model.eval()

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
        features = last_hidden_state[:, 0, :]
        if self.normalize:
            features = F.normalize(features, p=2, dim=1)
        return features


class DinoV3PairwiseModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        projector_hidden_dim: int = 512,
        projector_output_dim: int = 256,
        dropout: float = 0.3,
        margin: float = 0.3,
        freeze_backbone: bool = True,
        backbone_dtype: str = "auto",
        unfreeze_last_n_layers: int = 0,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.margin = margin
        self.feature_extractor = DinoV3FeatureExtractor(
            model_name=model_name,
            normalize=True,
            freeze_backbone=freeze_backbone,
            backbone_dtype=backbone_dtype,
            unfreeze_last_n_layers=unfreeze_last_n_layers,
        )
        feature_dim = self.feature_extractor.output_dim
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, projector_hidden_dim),
            nn.LayerNorm(projector_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projector_hidden_dim, projector_output_dim),
            nn.LayerNorm(projector_output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.score_head = nn.Sequential(
            nn.Linear(projector_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.loss_fn = nn.MarginRankingLoss(margin=margin)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in list(self.projector.modules()) + list(self.score_head.modules()):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(pixel_values)

    @staticmethod
    def _align_tensor_for_module(tensor: torch.Tensor, module: nn.Module) -> torch.Tensor:
        parameter = next(module.parameters())
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
