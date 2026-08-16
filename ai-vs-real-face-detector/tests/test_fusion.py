"""Tests for feature fusion."""

import numpy as np
import torch

from src.fusion.fuse import FeatureFusion, concatenate_features


def test_concatenate_features_numpy():
    deep = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    physics = np.array([4.0, 5.0], dtype=np.float32)
    fused = concatenate_features(deep, physics)
    assert fused.shape == (1, 5)
    assert torch.allclose(fused[0], torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32))


def test_feature_fusion_output_dim():
    fusion = FeatureFusion()
    assert fusion.output_dim(1280, 20) == 1300

    deep = torch.randn(2, 1280)
    physics = torch.randn(2, 20)
    out = fusion.fuse(deep, physics)
    assert out.shape == (2, 1300)
