"""Unit tests for physics branch (deterministic CV — safe to run locally)."""

from __future__ import annotations

import numpy as np
import pytest

from src.physics_branch.corneal_reflection import (
    compute_highlight_iou,
    detect_specular_highlight,
)
from src.physics_branch.feature_vector import (
    PHYSICS_FEATURE_DIM,
    PHYSICS_FEATURE_NAMES,
    PhysicsFeatureExtractor,
)
from src.physics_branch.region_detection import FaceLandmarks, FaceRegionDetector
from src.physics_branch.shadow_geometry import LightDirectionEstimate, analyze_shadow_geometry
from src.physics_branch.iris_pupil import fit_pupil_ellipse


def _synthetic_eye_with_highlight(size: int = 64) -> np.ndarray:
    """Create a synthetic eye crop with a bright specular spot."""
    eye = np.zeros((size, size, 3), dtype=np.uint8)
    eye[:] = (40, 30, 25)
    cv2 = pytest.importorskip("cv2")
    cv2.circle(eye, (size // 2, size // 2), size // 4, (80, 60, 50), -1)
    cv2.circle(eye, (size // 2 + 8, size // 2 - 6), 4, (255, 255, 255), -1)
    return eye


def test_highlight_iou_identical_masks():
    mask = np.ones((32, 32), dtype=np.uint8)
    assert compute_highlight_iou(mask, mask) == pytest.approx(1.0)


def test_highlight_iou_disjoint_masks():
    a = np.zeros((32, 32), dtype=np.uint8)
    b = np.zeros((32, 32), dtype=np.uint8)
    a[:16, :] = 1
    b[16:, :] = 1
    assert compute_highlight_iou(a, b) == pytest.approx(0.0)


def test_detect_specular_highlight_on_synthetic_eye():
    eye = _synthetic_eye_with_highlight()
    result = detect_specular_highlight(eye)
    assert result.detected is True
    assert result.center is not None
    assert result.mask is not None
    assert result.mask.shape == eye.shape[:2]


def test_fit_pupil_ellipse_on_circle():
    t = np.linspace(0, 2 * np.pi, 64)
    contour = np.stack([20 + 10 * np.cos(t), 20 + 10 * np.sin(t)], axis=1).astype(np.float32)
    ellipse = fit_pupil_ellipse(contour)
    assert ellipse.detected is True
    assert ellipse.eccentricity < 0.1
    assert ellipse.regularity > 0.9


def test_physics_feature_vector_dimension():
    """Feature vector has fixed length and named fields."""
    pytest.importorskip("mediapipe")
    assert len(PHYSICS_FEATURE_NAMES) == PHYSICS_FEATURE_DIM
    landmarks = FaceLandmarks(detected=False, image_shape=(128, 128))
    with PhysicsFeatureExtractor() as ext:
        fv = ext.extract_from_landmarks(landmarks)
    assert fv.vector.shape == (PHYSICS_FEATURE_DIM,)
    assert fv.dim == PHYSICS_FEATURE_DIM
    assert set(fv.to_dict().keys()) == set(PHYSICS_FEATURE_NAMES)


def test_shadow_geometry_no_face():
    landmarks = FaceLandmarks(detected=False, image_shape=(100, 100))
    from src.physics_branch.corneal_reflection import CornealReflectionResult, HighlightDetection

    corneal = CornealReflectionResult(
        left=HighlightDetection(detected=False),
        right=HighlightDetection(detected=False),
        highlight_iou=0.0,
        alignment_offset=(0.0, 0.0),
        consistent=False,
        debug={},
    )
    result = analyze_shadow_geometry(landmarks, corneal)
    assert result.consistent is False
    assert result.angle_difference_deg == 180.0


def test_light_direction_estimate():
    from src.physics_branch.corneal_reflection import HighlightDetection
    from src.physics_branch.region_detection import EyeRegion
    from src.physics_branch.shadow_geometry import estimate_light_from_highlight

    eye = EyeRegion(
        side="left",
        bbox=(10, 10, 40, 40),
        center=(30.0, 30.0),
        contour_points=np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.float32),
        iris_points=np.zeros((5, 2)),
        crop=np.zeros((40, 40, 3), dtype=np.uint8),
    )
    highlight = HighlightDetection(detected=True, center=(35.0, 25.0), radius=3.0)
    light = estimate_light_from_highlight(eye, highlight)
    assert light.detected is True
    assert -1.0 <= light.vector[0] <= 1.0
    assert -1.0 <= light.vector[1] <= 1.0
