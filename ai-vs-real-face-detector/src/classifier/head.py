"""Stage 3 — Classification head and hybrid model."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.deep_branch.feature_extractor import DeepFeatureExtractor
from src.fusion.fuse import FeatureFusion
from src.physics_branch.feature_vector import PHYSICS_FEATURE_DIM


class ClassificationHead(nn.Module):
    """MLP on fused feature vector -> binary logits."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 128),
        dropout: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        layers = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, hidden),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev = hidden
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridClassifier(nn.Module):
    """
    Joint model: deep backbone (trainable) + physics features (fixed, no grad)
    fused via concatenation -> MLP classifier.
    """

    LABELS = ("real", "ai")

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        freeze_blocks: int = 5,
        physics_dim: int = PHYSICS_FEATURE_DIM,
        hidden_dims: Tuple[int, ...] = (256, 128),
        dropout: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.feature_extractor = DeepFeatureExtractor(
            model_name=model_name,
            pretrained=pretrained,
            freeze_blocks=freeze_blocks,
        )
        self.deep_dim = self.feature_extractor.embedding_dim
        self.physics_dim = physics_dim
        self.fusion = FeatureFusion()
        fused_dim = self.fusion.output_dim(self.deep_dim, physics_dim)
        self.classifier = ClassificationHead(
            input_dim=fused_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            num_classes=num_classes,
        )

    def forward(
        self,
        images: torch.Tensor,
        physics_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        deep_features = self.feature_extractor(images)
        fused = self.fusion.fuse(deep_features, physics_features)
        logits = self.classifier(fused)
        return logits, deep_features

    def predict_proba(
        self,
        images: torch.Tensor,
        physics_features: torch.Tensor,
    ) -> torch.Tensor:
        logits, _ = self.forward(images, physics_features)
        return torch.softmax(logits, dim=1)

    @staticmethod
    def decode_label(class_idx: int) -> str:
        return HybridClassifier.LABELS[class_idx]
