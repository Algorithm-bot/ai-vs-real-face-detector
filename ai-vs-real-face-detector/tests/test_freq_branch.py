"""
Unit tests for frequency/upsampling branch and branch diagnostics.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import torch

from src.classifier.head import FullHybridClassifier
from src.diagnose_branches import run_diagnostics_from_cache, run_diagnostics_from_predictions
from src.freq_branch.extractor import FREQ_FEATURE_DIM, FreqExtractor
from src.fusion.fuse import FusionMode


def test_freq_extractor_dimensions_and_subcomponents():
    extractor = FreqExtractor()

    # Test various resolutions
    for h, w in [(224, 224), (256, 256), (512, 384), (1024, 1024)]:
        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        res = extractor.extract(img)
        vec = res.vector

        assert len(vec) == FREQ_FEATURE_DIM == 64
        assert np.all(np.isfinite(vec)), f"Found NaN/inf at {h}x{w}"

        # 1. 32 radial bins sum to ~1.0
        rad_bins = vec[:32]
        assert np.isclose(rad_bins.sum(), 1.0, atol=1e-3)

        # 2. 20 periodic peak ratios >= 0
        peaks = vec[32:52]
        assert len(peaks) == 20
        assert np.all(peaks >= 0.0)

        # 3. 12 NPR statistics
        npr = vec[52:64]
        assert len(npr) == 12
        assert np.all(np.isfinite(npr))


def test_freq_extractor_identical_for_same_center_crop_across_resolutions():
    """Verify that images of different sizes with the same center 224x224 crop yield identical freq vectors."""
    extractor = FreqExtractor()
    rng = np.random.RandomState(42)

    # 1. Base 224x224 patch
    patch_224 = rng.randint(0, 256, (224, 224, 3), dtype=np.uint8)

    # 2. Embed into 256x256
    img_256 = rng.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    sy_256, sx_256 = (256 - 224) // 2, (256 - 224) // 2
    img_256[sy_256 : sy_256 + 224, sx_256 : sx_256 + 224] = patch_224

    # 3. Embed into 512x384
    img_512x384 = rng.randint(0, 256, (512, 384, 3), dtype=np.uint8)
    sy_rect, sx_rect = (512 - 224) // 2, (384 - 224) // 2
    img_512x384[sy_rect : sy_rect + 224, sx_rect : sx_rect + 224] = patch_224

    # 4. Embed into 1024x1024
    img_1024 = rng.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
    sy_1024, sx_1024 = (1024 - 224) // 2, (1024 - 224) // 2
    img_1024[sy_1024 : sy_1024 + 224, sx_1024 : sx_1024 + 224] = patch_224

    v_224 = extractor.extract(patch_224).vector
    v_256 = extractor.extract(img_256).vector
    v_512x384 = extractor.extract(img_512x384).vector
    v_1024 = extractor.extract(img_1024).vector

    assert np.allclose(v_224, v_256), "Freq vector mismatch between 224x224 and 256x256 center crop"
    assert np.allclose(v_224, v_512x384), "Freq vector mismatch between 224x224 and 512x384 center crop"
    assert np.allclose(v_224, v_1024), "Freq vector mismatch between 224x224 and 1024x1024 center crop"
    assert np.allclose(v_256, v_1024), "Freq vector mismatch between 256x256 and 1024x1024 center crop"


def test_freq_extractor_edge_cases_nan_inf_safety():
    extractor = FreqExtractor()

    # All zeros
    zeros = np.zeros((256, 256, 3), dtype=np.uint8)
    res_zeros = extractor.extract(zeros)
    assert np.all(np.isfinite(res_zeros.vector))

    # All constant (255)
    ones = np.full((256, 256, 3), 255, dtype=np.uint8)
    res_ones = extractor.extract(ones)
    assert np.all(np.isfinite(res_ones.vector))

    # Grayscale 2D
    gray = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    res_gray = extractor.extract(gray)
    assert len(res_gray.vector) == 64
    assert np.all(np.isfinite(res_gray.vector))

    # RGBA 4-channel
    rgba = np.random.randint(0, 256, (256, 256, 4), dtype=np.uint8)
    res_rgba = extractor.extract(rgba)
    assert len(res_rgba.vector) == 64
    assert np.all(np.isfinite(res_rgba.vector))


def test_freq_extractor_speed():
    extractor = FreqExtractor()
    img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

    # Warmup
    _ = extractor.extract(img)

    t0 = time.perf_counter()
    n_iters = 10
    for _ in range(n_iters):
        _ = extractor.extract(img)
    elapsed = (time.perf_counter() - t0) / n_iters

    # Should run in < 150ms per image on CPU
    assert elapsed < 0.15, f"Extraction too slow: {elapsed * 1000:.1f}ms per image"


def test_full_hybrid_classifier_configurable_branches():
    # Test default 4 branches (deep, physics, semantic, freq) without PRNU
    model = FullHybridClassifier(
        pretrained=False,
        semantic_pretrained=False,
        branches=["deep", "physics", "semantic", "freq"],
        no_down=True,
    )

    bs = 2
    x_img = torch.randn(bs, 3, 224, 224)
    x_phys = torch.randn(bs, 22)
    x_sem = torch.randn(bs, 384)
    x_freq = torch.randn(bs, 64)

    logits, _ = model(x_img, physics=x_phys, semantic=x_sem, freq=x_freq)
    assert logits.shape == (bs, 2)

    # Test all 5 branches (including prnu)
    model5 = FullHybridClassifier(
        pretrained=False,
        semantic_pretrained=False,
        branches=["deep", "physics", "prnu", "semantic", "freq"],
    )
    x_prnu = torch.randn(bs, 9)
    logits5, _ = model5(x_img, physics=x_phys, prnu=x_prnu, semantic=x_sem, freq=x_freq)
    assert logits5.shape == (bs, 2)

    # Test minimal subset (deep + freq)
    model_df = FullHybridClassifier(
        pretrained=False,
        semantic_pretrained=False,
        branches=["deep", "freq"],
    )
    logits_df, _ = model_df(x_img, freq=x_freq)
    assert logits_df.shape == (bs, 2)


def test_diagnose_branches_mock(tmp_path: Path):
    # Create mock test.npz
    n_samples = 40
    # Half FOLD_A (DEV), half FOLD_B (REPORT)
    sources = ["stylegan/car"] * 20 + ["stylegan2/car"] * 20
    labels = np.array([0, 1] * 20, dtype=np.int64)

    physics = np.random.randn(n_samples, 22).astype(np.float32)
    prnu = np.random.randn(n_samples, 9).astype(np.float32)
    freq = np.random.randn(n_samples, 64).astype(np.float32)
    semantic = np.random.randn(n_samples, 384).astype(np.float32)

    test_npz = tmp_path / "test.npz"
    np.savez(
        test_npz,
        paths=np.array([f"img_{i}.jpg" for i in range(n_samples)]),
        sources=np.array(sources),
        labels=labels,
        physics=physics,
        prnu=prnu,
        freq=freq,
        semantic=semantic,
    )

    df = run_diagnostics_from_cache(tmp_path)
    assert "generator" in df.columns
    assert "freq_auc" in df.columns
    assert "prnu_auc" in df.columns
    assert "OVERALL_REPORT" in df["generator"].values
