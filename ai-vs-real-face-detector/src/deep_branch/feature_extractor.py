"""EfficientNet-B0 feature extractor with optional binary classification head."""

from __future__ import annotations

from typing import Optional, Tuple

import timm
import torch
import torch.nn as nn


class DeepFeatureExtractor(nn.Module):
    """
    Pretrained EfficientNet-B0 backbone.
    Early blocks are frozen; forward returns a feature embedding vector.
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        freeze_blocks: int = 5,
        embedding_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.embedding_dim = embedding_dim or self.backbone.num_features
        self._freeze_early_layers(freeze_blocks)

    def _freeze_early_layers(self, freeze_blocks: int) -> None:
        if freeze_blocks <= 0:
            return
        for name, param in self.backbone.named_parameters():
            # Freeze conv_stem, bn1, and blocks.0 .. blocks.(freeze_blocks-1)
            if name.startswith("conv_stem") or name.startswith("bn1"):
                param.requires_grad = False
            elif name.startswith("blocks."):
                block_idx = int(name.split(".")[1])
                if block_idx < freeze_blocks:
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DeepClassifier(nn.Module):
    """Stage 1 standalone model: backbone + binary classification head."""

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        freeze_blocks: int = 5,
        dropout: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.feature_extractor = DeepFeatureExtractor(
            model_name=model_name,
            pretrained=pretrained,
            freeze_blocks=freeze_blocks,
        )
        dim = self.feature_extractor.embedding_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits, features

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x)
        return torch.softmax(logits, dim=1)
