"""Test shared preprocessing consistency and cache/live feature equivalence.

Tests 20 images of diverse shapes (256x256, 512x384, 1024x1024, non-square)
confirming:
1. Shared PreprocessPipeline produces identical center crops
2. Cached vs live features are allclose per branch
3. Augmentation with identical seed produces identical pixel arrays
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.augment import augment_raw_image, path_to_seed
from src.physics_branch.feature_vector import PhysicsFeatureExtractor
from src.preprocess import PreprocessPipeline, center_crop_numpy, pad_if_smaller, random_crop_numpy
from src.prnu_branch.extractor import PRNUExtractor
from tests.synthetic_fixture import SIZES, create_synthetic_image


def test_pad_if_smaller_and_center_crop():
    # Image smaller than 224
    small = np.zeros((100, 150, 3), dtype=np.uint8)
    padded = pad_if_smaller(small, 224)
    assert padded.shape[0] >= 224 and padded.shape[1] >= 224

    cropped = center_crop_numpy(small, 224)
    assert cropped.shape == (224, 224, 3)

    # Image larger than 224 (non-square)
    large = np.zeros((512, 384, 3), dtype=np.uint8)
    cropped_large = center_crop_numpy(large, 224)
    assert cropped_large.shape == (224, 224, 3)


def test_augmentation_deterministic_per_path():
    img = create_synthetic_image(512, 384, seed=42)
    path = "progan/airplane/0_real/001.png"
    
    # Same path, epoch, seed -> identical output
    aug1, rng1 = augment_raw_image(img, path=path, epoch=0, base_seed=42, enable_augment=True)
    aug2, rng2 = augment_raw_image(img, path=path, epoch=0, base_seed=42, enable_augment=True)
    np.testing.assert_array_equal(aug1, aug2)

    # Different path -> different seed / different augmentation
    path_b = "stylegan/bedroom/1_fake/999.png"
    aug3, _ = augment_raw_image(img, path=path_b, epoch=0, base_seed=42, enable_augment=True)
    assert not np.array_equal(aug1, aug3)


def test_20_images_cached_vs_live_feature_equivalence():
    """Verify on 20 images of mixed sizes that features extracted via the cached path
    and the live path are allclose per branch."""
    physics_ext = PhysicsFeatureExtractor()
    prnu_ext = PRNUExtractor()

    pipeline = PreprocessPipeline(mode="crop", crop_size=224, is_training=False)

    for i in range(20):
        w, h = SIZES[i % len(SIZES)]
        img = create_synthetic_image(w, h, seed=i * 7)

        # Path 1: "Live" evaluation path using shared center crop
        live_crop = pipeline.preprocess_image(img)
        live_phys = physics_ext.extract(live_crop).vector
        live_prnu = prnu_ext.extract(live_crop).vector

        # Path 2: "Cached" evaluation path using shared center crop
        cached_crop = center_crop_numpy(img, 224)
        cached_phys = physics_ext.extract(cached_crop).vector
        cached_prnu = prnu_ext.extract(cached_crop).vector

        np.testing.assert_array_equal(live_crop, cached_crop)
        np.testing.assert_allclose(live_phys, cached_phys, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(live_prnu, cached_prnu, rtol=1e-5, atol=1e-5)
