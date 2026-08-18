"""Tests for extended fusion modes."""

import torch

from src.fusion.fuse import FeatureFusion, FusionMode, concatenate_features


def test_concatenate_multiple():
    a = torch.randn(1, 1280)
    b = torch.randn(1, 20)
    c = torch.randn(1, 9)
    fused = concatenate_features(a, b, c)
    assert fused.shape == (1, 1309)


def test_gated_fusion_output_dim():
    dims = {"deep": 128, "physics": 20}
    fusion = FeatureFusion(mode=FusionMode.GATED, modality_dims=dims)
    deep = torch.randn(2, 128)
    physics = torch.randn(2, 20)
    out = fusion.fuse(deep, physics)
    assert out.shape == (2, 148)


def test_attention_fusion():
    dims = {"deep": 64, "physics": 20}
    fusion = FeatureFusion(mode=FusionMode.ATTENTION, modality_dims=dims)
    deep = torch.randn(1, 64)
    physics = torch.randn(1, 20)
    out = fusion.fuse(deep, physics)
    assert out.shape[0] == 1
