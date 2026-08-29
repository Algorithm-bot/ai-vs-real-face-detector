"""Tests for PRNU forensic branch."""

import numpy as np

from src.prnu_branch.extractor import PRNUExtractor, PRNU_FEATURE_DIM, extract_noise_residual


def test_prnu_feature_dim():
    assert PRNU_FEATURE_DIM == 9


def test_prnu_extract_limited_reliability():
    image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    extractor = PRNUExtractor()
    result = extractor.extract(image)
    assert result.dim == PRNU_FEATURE_DIM
    assert result.reliability == "limited"
    assert not result.has_reference


def test_noise_residual_shape():
    image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    residual = extract_noise_residual(image)
    assert residual.shape == (64, 64)


def test_prnu_with_reference():
    image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    ref = extract_noise_residual(image)
    extractor = PRNUExtractor(reference_residual=ref)
    result = extractor.extract(image)
    assert result.has_reference
    assert result.reliability == "reference_available"
