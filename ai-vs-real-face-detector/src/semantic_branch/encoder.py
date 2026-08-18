"""Pretrained ViT semantic encoder and attention anomaly maps."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_VIT_MODEL = "vit_small_patch16_224"


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


class SemanticEncoder(nn.Module):
    """Extract semantic features from a pretrained ViT."""

    def __init__(
        self,
        model_name: str = DEFAULT_VIT_MODEL,
        pretrained: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.embedding_dim = self.backbone.num_features
        self.device = device or torch.device("cpu")
        self.backbone.to(self.device)
        self.backbone.eval()

        for param in self.backbone.parameters():
            param.requires_grad = False

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Resize to 224, normalize, return batch tensor."""
        import cv2

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).float() / 255.0
        tensor = tensor.permute(2, 0, 1)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor.unsqueeze(0).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def extract(
        self,
        image: np.ndarray,
        return_attention: bool = True,
    ) -> SemanticResult:
        """Extract semantic embedding and optional attention anomaly map."""
        tensor = self._preprocess(image)

        with torch.no_grad():
            features = self.backbone(tensor).cpu().numpy().flatten()

        attention_map = None
        if return_attention:
            attention_map = self._compute_attention_map(tensor, image)

        return SemanticResult(
            features=features.astype(np.float32),
            embedding_dim=self.embedding_dim,
            model_name=self.model_name,
            attention_map=attention_map,
        )

    def _compute_attention_map(
        self,
        tensor: torch.Tensor,
        original_image: np.ndarray,
    ) -> SemanticAttentionMap:
        """
        Compute attention-based anomaly map from ViT last-layer attention.
        Uses mean attention from CLS token to patch tokens.
        """
        grid_size = 14  # 224/16 for patch16
        heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)

        try:
            # Access attention weights via forward hooks on last block
            attentions = []
            last_block = self._get_last_attention_block()

            def hook_fn(module, input, output):
                # timm ViT blocks return (x, attn) or just x depending on version
                if isinstance(output, tuple) and len(output) >= 2:
                    attentions.append(output[1])

            handle = last_block.attn.register_forward_hook(hook_fn)
            with torch.no_grad():
                self.backbone(tensor)
            handle.remove()

            if attentions:
                attn = attentions[-1].cpu().numpy()
                # attn shape: (batch, heads, tokens, tokens)
                if attn.ndim == 4:
                    cls_attn = attn[0, :, 0, 1:].mean(axis=0)  # CLS -> patches
                    if len(cls_attn) == grid_size * grid_size:
                        heatmap = cls_attn.reshape(grid_size, grid_size)
        except Exception:
            # Fallback: gradient-free patch occlusion anomaly map
            heatmap = self._occlusion_anomaly_map(tensor, grid_size)

        # Normalize and resize to image size
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        import cv2
        h, w = original_image.shape[:2]
        heatmap_full = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

        base64_png = self._heatmap_to_base64(heatmap_full, original_image)

        return SemanticAttentionMap(
            heatmap=heatmap_full,
            base64_png=base64_png,
            grid_size=grid_size,
        )

    def _get_last_attention_block(self) -> nn.Module:
        if hasattr(self.backbone, "blocks"):
            return self.backbone.blocks[-1]
        raise AttributeError("Cannot find ViT attention blocks")

    def _occlusion_anomaly_map(
        self,
        tensor: torch.Tensor,
        grid_size: int,
    ) -> np.ndarray:
        """Fallback: measure embedding change under patch occlusion."""
        with torch.no_grad():
            base_feat = self.backbone(tensor).cpu().numpy().flatten()
        base_norm = np.linalg.norm(base_feat)

        patch_size = 224 // grid_size
        heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)

        for gy in range(grid_size):
            for gx in range(grid_size):
                occluded = tensor.clone()
                y0, x0 = gy * patch_size, gx * patch_size
                occluded[:, :, y0:y0 + patch_size, x0:x0 + patch_size] = 0.0
                with torch.no_grad():
                    feat = self.backbone(occluded).cpu().numpy().flatten()
                diff = np.linalg.norm(feat - base_feat) / (base_norm + 1e-8)
                heatmap[gy, gx] = diff

        return heatmap

    def _heatmap_to_base64(
        self,
        heatmap: np.ndarray,
        image: np.ndarray,
    ) -> str:
        import cv2

        colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8),
            cv2.COLORMAP_JET,
        )
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        overlay = (0.5 * image.astype(np.float32) + 0.5 * colored).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(overlay)
        buffer = io.BytesIO()
        pil.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")
