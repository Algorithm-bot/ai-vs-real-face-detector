"""Shadow and illumination geometry analysis with actual shadow maps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .corneal_reflection import CornealReflectionResult, HighlightDetection
from .region_detection import EyeRegion, FaceLandmarks


@dataclass
class LightDirectionEstimate:
    detected: bool
    vector: Tuple[float, float] = (0.0, 0.0)
    angle_deg: float = 0.0


@dataclass
class ShadowMapResult:
    """Shadow region analysis on face."""

    shadow_coverage: float = 0.0
    shadow_asymmetry: float = 0.0
    illumination_uniformity: float = 0.0
    left_brightness: float = 0.0
    right_brightness: float = 0.0
    shadow_mask: Optional[np.ndarray] = None


@dataclass
class ShadowGeometryResult:
    left_light: LightDirectionEstimate
    right_light: LightDirectionEstimate
    angle_difference_deg: float
    vector_cosine_similarity: float
    consistent: bool
    shadow_map: Optional[ShadowMapResult] = None
    metadata: Dict = field(default_factory=dict)


def estimate_light_from_highlight(
    eye: EyeRegion,
    highlight: HighlightDetection,
) -> LightDirectionEstimate:
    if not highlight.detected or highlight.center is None:
        return LightDirectionEstimate(detected=False)

    ex, ey = eye.center
    x, y, w, h = eye.bbox
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


def compute_shadow_map(
    image: np.ndarray,
    landmarks: FaceLandmarks,
    shadow_threshold: float = 0.35,
) -> ShadowMapResult:
    """
    Build shadow mask from luminance and analyze bilateral illumination geometry.
    """
    if not landmarks.detected or landmarks.landmarks_pixel is None:
        return ShadowMapResult()

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    gray_f = gray.astype(np.float32) / 255.0
    h, w = gray.shape

    # Face mask from convex hull of landmarks
    pts = landmarks.landmarks_pixel.astype(np.int32)
    hull = cv2.convexHull(pts)
    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(face_mask, hull, 255)

    face_pixels = gray_f[face_mask > 0]
    if len(face_pixels) == 0:
        return ShadowMapResult()

    median_lum = float(np.median(face_pixels))
    shadow_mask = ((gray_f < median_lum * shadow_threshold) & (face_mask > 0)).astype(np.uint8)
    shadow_coverage = float(shadow_mask.sum() / max(1, face_mask.sum()))

    # Bilateral brightness: left vs right half of face
    mid_x = w // 2
    left_mask = face_mask[:, :mid_x]
    right_mask = face_mask[:, mid_x:]
    left_brightness = float(gray_f[:, :mid_x][left_mask > 0].mean()) if left_mask.sum() > 0 else 0.0
    right_brightness = float(gray_f[:, mid_x:][right_mask > 0].mean()) if right_mask.sum() > 0 else 0.0
    asymmetry = abs(left_brightness - right_brightness)

    # Illumination uniformity via coefficient of variation
    cv_lum = float(face_pixels.std() / (face_pixels.mean() + 1e-8))
    uniformity = float(1.0 - min(1.0, cv_lum))

    return ShadowMapResult(
        shadow_coverage=shadow_coverage,
        shadow_asymmetry=asymmetry,
        illumination_uniformity=uniformity,
        left_brightness=left_brightness,
        right_brightness=right_brightness,
        shadow_mask=shadow_mask,
    )


def analyze_shadow_geometry(
    landmarks: FaceLandmarks,
    corneal: CornealReflectionResult,
    image: Optional[np.ndarray] = None,
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

    shadow_map = None
    if image is not None:
        shadow_map = compute_shadow_map(image, landmarks)

    if not left_light.detected or not right_light.detected:
        return ShadowGeometryResult(
            left_light=left_light,
            right_light=right_light,
            angle_difference_deg=180.0,
            vector_cosine_similarity=-1.0,
            consistent=False,
            shadow_map=shadow_map,
        )

    lv = np.array(left_light.vector)
    rv = np.array([-right_light.vector[0], right_light.vector[1]])

    cosine_sim = float(np.dot(lv, rv) / (np.linalg.norm(lv) * np.linalg.norm(rv) + 1e-8))
    angle_diff = float(abs(left_light.angle_deg - (-right_light.angle_deg)) % 360.0)
    angle_diff = min(angle_diff, 360.0 - angle_diff)

    consistent = angle_diff <= angle_threshold_deg and cosine_sim >= cosine_threshold

    # Shadow consistency: low asymmetry supports consistent lighting
    if shadow_map is not None:
        shadow_consistent = shadow_map.shadow_asymmetry < 0.15
        consistent = consistent and shadow_consistent

    return ShadowGeometryResult(
        left_light=left_light,
        right_light=right_light,
        angle_difference_deg=angle_diff,
        vector_cosine_similarity=cosine_sim,
        consistent=consistent,
        shadow_map=shadow_map,
        metadata={
            "shadow_coverage": shadow_map.shadow_coverage if shadow_map else None,
            "illumination_uniformity": shadow_map.illumination_uniformity if shadow_map else None,
        },
    )
