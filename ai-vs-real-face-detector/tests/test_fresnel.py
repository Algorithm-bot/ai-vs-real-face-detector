"""Tests for Fresnel reflection calculations."""

import numpy as np

from src.physics_branch.fresnel import (
    analyze_fresnel_reflection,
    fresnel_reflectance_unpolarized,
)
from src.physics_branch.corneal_reflection import HighlightDetection
from src.physics_branch.region_detection import EyeRegion


def test_fresnel_normal_incidence():
    r = fresnel_reflectance_unpolarized(1.0, 1.376, 1.0)
    assert 0.0 < r < 0.1


def test_fresnel_grazing():
    r = fresnel_reflectance_unpolarized(1.0, 1.376, 0.0)
    assert r == 1.0


def test_analyze_fresnel_no_highlight():
    eye = EyeRegion(
        side="left",
        bbox=(10, 10, 40, 30),
        center=(30.0, 25.0),
        contour_points=np.zeros((6, 2)),
        iris_points=np.zeros((5, 2)),
        crop=np.zeros((30, 40, 3), dtype=np.uint8),
    )
    highlight = HighlightDetection(detected=False)
    result = analyze_fresnel_reflection(eye, highlight)
    assert not result.detected
