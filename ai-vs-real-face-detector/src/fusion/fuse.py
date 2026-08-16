"""Stage 3 — Feature fusion (simple concatenation for v1)."""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch


def concatenate_features(
    deep_features: Union[torch.Tensor, np.ndarray],
    physics_features: Union[torch.Tensor, np.ndarray],
) -> torch.Tensor:
    """Concatenate deep and physics vectors along the feature dimension."""
    if isinstance(deep_features, np.ndarray):
        deep_features = torch.from_numpy(deep_features)
    if isinstance(physics_features, np.ndarray):
        physics_features = torch.from_numpy(physics_features)

    if deep_features.dim() == 1:
        deep_features = deep_features.unsqueeze(0)
    if physics_features.dim() == 1:
        physics_features = physics_features.unsqueeze(0)

    deep_features = deep_features.float()
    physics_features = physics_features.float()
    return torch.cat([deep_features, physics_features], dim=1)


class FeatureFusion:
    """v1 fusion: concatenation only. Attention/gating reserved for stretch goal."""

    def __init__(self) -> None:
        pass

    def fuse(
        self,
        deep_features: torch.Tensor,
        physics_features: torch.Tensor,
    ) -> torch.Tensor:
        return concatenate_features(deep_features, physics_features)

    @staticmethod
    def output_dim(deep_dim: int, physics_dim: int) -> int:
        return deep_dim + physics_dim
