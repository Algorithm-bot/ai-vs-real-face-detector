"""B4 — Shadow and geometry consistency (heuristic light-direction check)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .corneal_reflection import CornealReflectionResult, HighlightDetection
from .region_detection import EyeRegion, FaceLandmarks


@dataclass
class LightDirectionEstimate:
    detected: bool
    vector: Tuple[float, float] = (0.0, 0.0)  # normalized offset from eye center to highlight
    angle_deg: float = 0.0


@dataclass
class ShadowGeometryResult:
    left_light: LightDirectionEstimate
    right_light: LightDirectionEstimate
    angle_difference_deg: float
    vector_cosine_similarity: float
    consistent: bool


def estimate_light_from_highlight(
    eye: EyeRegion,
    highlight: HighlightDetection,
) -> LightDirectionEstimate:
    if not highlight.detected or highlight.center is None:
        return LightDirectionEstimate(detected=False)

    ex, ey = eye.center
    x, y, w, h = eye.bbox
    # Map highlight center from crop coords to image coords
    hx = x + highlight.center[0]
    hy = y + highlight.center[1]

    dx = hx - ex
    dy = hy - ey
    norm = np.hypot(dx, dy)
    if norm < 1e-6:
        return LightDirectionEstimate(detected=False)

    vec = (float(dx / norm), float(dy / norm))
    angle = float(np.degrees(np.arctan2(dy, dx)))
    return LightDirectionEstimate(detected=True, vector=vec, angle_deg=angle)


def analyze_shadow_geometry(
    landmarks: FaceLandmarks,
    corneal: CornealReflectionResult,
    angle_threshold_deg: float = 25.0,
    cosine_threshold: float = 0.85,
) -> ShadowGeometryResult:
    empty = LightDirectionEstimate(detected=False)

    if not landmarks.detected or landmarks.left_eye is None or landmarks.right_eye is None:
        return ShadowGeometryResult(
            left_light=empty,
            right_light=empty,
            angle_difference_deg=180.0,
            vector_cosine_similarity=-1.0,
            consistent=False,
        )

    left_light = estimate_light_from_highlight(landmarks.left_eye, corneal.left)
    right_light = estimate_light_from_highlight(landmarks.right_eye, corneal.right)

    if not left_light.detected or not right_light.detected:
        return ShadowGeometryResult(
            left_light=left_light,
            right_light=right_light,
            angle_difference_deg=180.0,
            vector_cosine_similarity=-1.0,
            consistent=False,
        )

    # Mirror right vector horizontally (x flip) for bilateral comparison
    lv = np.array(left_light.vector)
    rv = np.array([ -right_light.vector[0], right_light.vector[1] ])

    cosine_sim = float(np.dot(lv, rv) / (np.linalg.norm(lv) * np.linalg.norm(rv) + 1e-8))
    angle_diff = float(abs(left_light.angle_deg - (-right_light.angle_deg)) % 360.0)
    angle_diff = min(angle_diff, 360.0 - angle_diff)

    consistent = angle_diff <= angle_threshold_deg and cosine_sim >= cosine_threshold

    return ShadowGeometryResult(
        left_light=left_light,
        right_light=right_light,
        angle_difference_deg=angle_diff,
        vector_cosine_similarity=cosine_sim,
        consistent=consistent,
    )
