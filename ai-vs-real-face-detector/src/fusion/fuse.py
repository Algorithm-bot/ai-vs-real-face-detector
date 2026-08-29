"""Multi-modal feature fusion: concatenation, gated, and attention fusion."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn


class FusionMode(str, Enum):
    CONCAT = "concat"
    GATED = "gated"
    ATTENTION = "attention"


def concatenate_features(
    *feature_tensors: Union[torch.Tensor, np.ndarray],
) -> torch.Tensor:
    """Concatenate feature vectors along the feature dimension."""
    tensors = []
    for feat in feature_tensors:
        if isinstance(feat, np.ndarray):
            feat = torch.from_numpy(feat)
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)
        tensors.append(feat.float())
    return torch.cat(tensors, dim=1)


class GatedFusion(nn.Module):
    """Learned gating fusion across modality vectors."""

    def __init__(self, modality_dims: Dict[str, int], hidden_dim: int = 64) -> None:
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        total = sum(modality_dims.values())
        self.gate = nn.Sequential(
            nn.Linear(total, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, len(self.modality_names)),
            nn.Sigmoid(),
        )
        self.modality_dims = modality_dims

    def forward(self, modalities: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        for name in self.modality_names:
            feat = modalities[name].float()
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
            parts.append(feat)
        concat = torch.cat(parts, dim=1)
        gates = self.gate(concat)
        gated_parts = []
        offset = 0
        for i, name in enumerate(self.modality_names):
            dim = self.modality_dims[name]
            g = gates[:, i].unsqueeze(1)
            gated_parts.append(parts[i] * g)
            offset += dim
        return torch.cat(gated_parts, dim=1)

    def get_gate_weights(self, modalities: Dict[str, torch.Tensor]) -> Dict[str, float]:
        parts = []
        for name in self.modality_names:
            feat = modalities[name].float()
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
            parts.append(feat)
        concat = torch.cat(parts, dim=1)
        gates = self.gate(concat)[0].cpu().numpy()
        return {name: float(gates[i]) for i, name in enumerate(self.modality_names)}


class AttentionFusion(nn.Module):
    """Multi-head attention fusion across modality embeddings."""

    def __init__(
        self,
        modality_dims: Dict[str, int],
        embed_dim: int = 128,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, embed_dim)
            for name, dim in modality_dims.items()
        })
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.output_dim = embed_dim * len(self.modality_names)

    def forward(self, modalities: Dict[str, torch.Tensor]) -> torch.Tensor:
        projected = []
        for name in self.modality_names:
            feat = modalities[name].float()
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
            projected.append(self.projections[name](feat))
        # Stack modalities as sequence: (batch, num_modalities, embed_dim)
        seq = torch.stack(projected, dim=1)
        attn_out, attn_weights = self.attention(seq, seq, seq)
        self._last_attn_weights = attn_weights
        return attn_out.flatten(start_dim=1)

    def get_attention_weights(self) -> Optional[Dict[str, Dict[str, float]]]:
        if not hasattr(self, "_last_attn_weights") or self._last_attn_weights is None:
            return None
        w = self._last_attn_weights[0].cpu().numpy()
        result = {}
        for i, src in enumerate(self.modality_names):
            result[src] = {
                dst: float(w[i, j])
                for j, dst in enumerate(self.modality_names)
            }
        return result


class FeatureFusion(nn.Module):
    """Configurable fusion supporting concat, gated, and attention modes."""

    def __init__(
        self,
        mode: FusionMode = FusionMode.CONCAT,
        modality_dims: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.modality_dims = modality_dims or {}
        self._gated: Optional[GatedFusion] = None
        self._attention: Optional[AttentionFusion] = None

        if mode == FusionMode.GATED and modality_dims:
            self._gated = GatedFusion(modality_dims)
        elif mode == FusionMode.ATTENTION and modality_dims:
            self._attention = AttentionFusion(modality_dims)

    def fuse(
        self,
        deep_features: torch.Tensor,
        physics_features: torch.Tensor,
        prnu_features: Optional[torch.Tensor] = None,
        semantic_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Backward-compatible fuse for deep + physics; extended for all modalities."""
        modalities: Dict[str, torch.Tensor] = {
            "deep": deep_features,
            "physics": physics_features,
        }
        if prnu_features is not None:
            modalities["prnu"] = prnu_features
        if semantic_features is not None:
            modalities["semantic"] = semantic_features

        if self.mode == FusionMode.CONCAT:
            return concatenate_features(*modalities.values())

        if self.mode == FusionMode.GATED and self._gated is not None:
            return self._gated(modalities)

        if self.mode == FusionMode.ATTENTION and self._attention is not None:
            return self._attention(modalities)

        return concatenate_features(*modalities.values())

    def get_fusion_weights(
        self,
        deep_features: torch.Tensor,
        physics_features: torch.Tensor,
        prnu_features: Optional[torch.Tensor] = None,
        semantic_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        modalities: Dict[str, torch.Tensor] = {
            "deep": deep_features,
            "physics": physics_features,
        }
        if prnu_features is not None:
            modalities["prnu"] = prnu_features
        if semantic_features is not None:
            modalities["semantic"] = semantic_features

        if self.mode == FusionMode.GATED and self._gated is not None:
            return self._gated.get_gate_weights(modalities)

        if self.mode == FusionMode.ATTENTION and self._attention is not None:
            self._attention(modalities)
            attn = self._attention.get_attention_weights()
            if attn:
                return {k: float(np.mean(list(v.values()))) for k, v in attn.items()}

        # Concat mode: equal weighting
        n = len(modalities)
        return {k: 1.0 / n for k in modalities}

    @staticmethod
    def output_dim(
        deep_dim: int,
        physics_dim: int,
        prnu_dim: int = 0,
        semantic_dim: int = 0,
        mode: FusionMode = FusionMode.CONCAT,
        embed_dim: int = 128,
        num_modalities: int = 2,
    ) -> int:
        total = deep_dim + physics_dim + prnu_dim + semantic_dim
        if mode == FusionMode.CONCAT:
            return total
        if mode == FusionMode.GATED:
            return total
        if mode == FusionMode.ATTENTION:
            return embed_dim * num_modalities
        return total
