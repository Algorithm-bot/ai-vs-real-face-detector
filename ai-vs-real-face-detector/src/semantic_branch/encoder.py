"""Pretrained ViT / CLIP semantic encoder with configurable layers."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

DEFAULT_VIT_MODEL = "vit_small_patch16_224"
DEFAULT_CLIP_MODEL = "vit_large_patch14_clip_224.openai"


@dataclass
class SemanticAttentionMap:
    """ViT attention-based anomaly map."""

    heatmap: np.ndarray
    base64_png: Optional[str] = None
    grid_size: int = 14


@dataclass
class SemanticResult:
    """Semantic branch output."""

    features: np.ndarray
    embedding_dim: int
    model_name: str
    attention_map: Optional[SemanticAttentionMap] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "embedding_dim": self.embedding_dim,
            "model_name": self.model_name,
            "feature_norm": float(np.linalg.norm(self.features)),
        }


def parse_layers_list(layers: Optional[Union[str, Sequence[int]]]) -> List[int]:
    if not layers:
        return []
    if isinstance(layers, str):
        return [int(p.strip()) for p in layers.split(",") if p.strip()]
    return [int(p) for p in layers]


class SemanticEncoder(nn.Module):
    """
    Extract semantic features from pretrained ViT or CLIP.
    
    Supports:
    - vit_small_patch16_224 (384-d)
    - vit_large_patch14_clip_224.openai (768-d final embedding or concatenated intermediate CLS tokens)
    - Configurable intermediate CLS tokens from specified layer indices
    """

    def __init__(
        self,
        model_name: str = DEFAULT_VIT_MODEL,
        pretrained: bool = True,
        device: Optional[torch.device] = None,
        intermediate_layers: Optional[Union[str, Sequence[int]]] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.intermediate_layers = parse_layers_list(intermediate_layers)
        self.device = device or torch.device("cpu")

        # Resolve model name: support shorthand "vit_large_patch14_clip_224"
        lookup_name = model_name
        if "vit_large_patch14_clip" in model_name and not model_name.endswith(".openai"):
            lookup_name = f"{model_name}.openai"

        # For CLIP models, num_classes is the projection dimension (e.g. 768).
        # We preserve the projection head when intermediate_layers is not set.
        is_clip = "clip" in model_name.lower()
        if is_clip and not self.intermediate_layers:
            self.backbone = timm.create_model(
                lookup_name,
                pretrained=pretrained,
            )
        else:
            self.backbone = timm.create_model(
                lookup_name,
                pretrained=pretrained,
                num_classes=0,
            )

        # Derive normalization stats from model config if available
        if hasattr(self.backbone, "default_cfg") and "mean" in self.backbone.default_cfg:
            self.mean = tuple(float(x) for x in self.backbone.default_cfg["mean"])
            self.std = tuple(float(x) for x in self.backbone.default_cfg["std"])
        elif "clip" in model_name.lower():
            self.mean = OPENAI_CLIP_MEAN
            self.std = OPENAI_CLIP_STD
        else:
            self.mean = IMAGENET_MEAN
            self.std = IMAGENET_STD

        # Calculate embedding dimension
        if self.intermediate_layers:
            # Each block CLS token has dimension backbone.num_features (1024 for ViT-L, 384 for ViT-S)
            feat_dim = self.backbone.num_features
            self.embedding_dim = feat_dim * len(self.intermediate_layers)
        else:
            # Default final embedding: 768 for CLIP ViT-L/14 with projection head, or backbone.num_features
            if hasattr(self.backbone, "head") and hasattr(self.backbone.head, "out_features"):
                self.embedding_dim = self.backbone.head.out_features
            else:
                self.embedding_dim = self.backbone.num_features


        self.backbone.to(self.device)
        self.backbone.eval()

        for param in self.backbone.parameters():
            param.requires_grad = False

    def _extract_intermediate_cls(self, x: torch.Tensor) -> torch.Tensor:
        """Run through transformer blocks and extract CLS tokens from requested layers."""
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        x = self.backbone.norm_pre(x)
        
        cls_tokens = []
        for i, block in enumerate(self.backbone.blocks, 1):
            x = block(x)
            if i in self.intermediate_layers:
                # CLS token is at index 0: (batch_size, feature_dim)
                cls_tokens.append(x[:, 0])
        
        return torch.cat(cls_tokens, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.intermediate_layers:
            return self._extract_intermediate_cls(x)
        return self.backbone(x)

    def extract(
        self,
        image: np.ndarray,
        return_attention: bool = False,
    ) -> SemanticResult:
        """Extract semantic embedding from a preprocessed 224x224 RGB image."""
        tensor = torch.from_numpy(image).float() / 255.0
        tensor = tensor.permute(2, 0, 1)
        mean_t = torch.tensor(self.mean).view(3, 1, 1)
        std_t = torch.tensor(self.std).view(3, 1, 1)
        tensor = ((tensor - mean_t) / std_t).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.forward(tensor).cpu().numpy().flatten()

        return SemanticResult(
            features=features.astype(np.float32),
            embedding_dim=self.embedding_dim,
            model_name=self.model_name,
            attention_map=None,
        )
