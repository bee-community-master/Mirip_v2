from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class DinoV3FeatureExtractor(nn.Module):
    def __init__(self, model_name: str, normalize: bool = True, freeze_backbone: bool = True) -> None:
        super().__init__()
        self.model_name = model_name
        self.normalize = normalize
        self.freeze_backbone = freeze_backbone
        self.model = AutoModel.from_pretrained(model_name)
        self.output_dim = int(self.model.config.hidden_size)
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

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
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.margin = margin
        self.feature_extractor = DinoV3FeatureExtractor(
            model_name=model_name,
            normalize=True,
            freeze_backbone=freeze_backbone,
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

    def project_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.projector(features)

    def score_features(self, projected_features: torch.Tensor) -> torch.Tensor:
        return self.score_head(projected_features)

    def predict_score(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(pixel_values)
        projected = self.project_features(features)
        return self.score_features(projected)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.predict_score(img1), self.predict_score(img2)

    def compute_loss(
        self,
        score1: torch.Tensor,
        score2: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return self.loss_fn(score1.squeeze(-1), score2.squeeze(-1), labels.float())
