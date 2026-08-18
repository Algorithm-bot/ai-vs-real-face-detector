"""Fresnel reflection calculations for corneal specular highlights."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .corneal_reflection import HighlightDetection
from .region_detection import EyeRegion


# Refractive index of cornea (~1.376)
CORNEA_IOR = 1.376
# Air refractive index
AIR_IOR = 1.0


@dataclass
class FresnelResult:
    """Fresnel reflectance analysis for a corneal highlight."""

    detected: bool
    fresnel_reflectance: float = 0.0
    expected_reflectance: float = 0.0
    reflectance_deviation: float = 0.0
    incidence_angle_deg: float = 0.0
    highlight_intensity: float = 0.0
    physically_plausible: bool = False


def fresnel_reflectance_unpolarized(
    n1: float,
    n2: float,
    cos_theta_i: float,
) -> float:
    """
    Compute unpolarized Fresnel reflectance for dielectric interface.

    cos_theta_i: cosine of incidence angle (clamped to [0, 1]).
    """
    cos_i = float(np.clip(cos_theta_i, 0.0, 1.0))
    sin_i_sq = 1.0 - cos_i ** 2
    sin_t_sq = (n1 / n2) ** 2 * sin_i_sq

    if sin_t_sq >= 1.0:
        return 1.0  # total internal reflection

    cos_t = math.sqrt(max(0.0, 1.0 - sin_t_sq))
    rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    rp = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)
    return float((rs ** 2 + rp ** 2) / 2.0)


def estimate_incidence_angle(
    eye: EyeRegion,
    highlight: HighlightDetection,
    cornea_radius_ratio: float = 0.45,
) -> float:
    """
    Estimate incidence angle from highlight position on approximated corneal sphere.
    Returns angle in degrees.
    """
    if not highlight.detected or highlight.center is None:
        return 90.0

    x, y, w, h = eye.bbox
    hx = x + highlight.center[0]
    hy = y + highlight.center[1]
    ex, ey = eye.center

    dx = hx - ex
    dy = hy - ey
    dist = np.hypot(dx, dy)
    cornea_r = max(w, h) * cornea_radius_ratio
    if cornea_r < 1e-6:
        return 90.0

    # Arc on sphere: sin(theta) ≈ dist / cornea_r
    sin_theta = min(1.0, dist / cornea_r)
    return float(np.degrees(np.arcsin(sin_theta)))


def compute_highlight_intensity(
    eye_crop: np.ndarray,
    highlight: HighlightDetection,
) -> float:
    """Mean intensity of detected highlight region normalized to [0, 1]."""
    if not highlight.detected or highlight.mask is None:
        return 0.0
    if eye_crop.ndim == 3:
        gray = np.mean(eye_crop, axis=2)
    else:
        gray = eye_crop
    mask = highlight.mask > 0
    if mask.sum() == 0:
        return 0.0
    return float(gray[mask].mean() / 255.0)


def analyze_fresnel_reflection(
    eye: EyeRegion,
    highlight: HighlightDetection,
    intensity_tolerance: float = 0.35,
    angle_max_deg: float = 60.0,
) -> FresnelResult:
    """Analyze whether corneal highlight matches Fresnel physics."""
    if not highlight.detected:
        return FresnelResult(detected=False)

    angle_deg = estimate_incidence_angle(eye, highlight)
    cos_theta = math.cos(math.radians(angle_deg))
    expected_r = fresnel_reflectance_unpolarized(AIR_IOR, CORNEA_IOR, cos_theta)
    intensity = compute_highlight_intensity(eye.crop, highlight)

    # Map Fresnel reflectance to expected highlight intensity range
    # At normal incidence R≈0.02; at grazing approaches 1.0
    expected_intensity = expected_r * 3.0  # empirical scaling for display
    deviation = abs(intensity - expected_intensity)

    plausible = (
        angle_deg <= angle_max_deg
        and deviation <= intensity_tolerance
        and intensity > 0.05
    )

    return FresnelResult(
        detected=True,
        fresnel_reflectance=expected_r,
        expected_reflectance=expected_r,
        reflectance_deviation=deviation,
        incidence_angle_deg=angle_deg,
        highlight_intensity=intensity,
        physically_plausible=plausible,
    )


def analyze_bilateral_fresnel(
    left_eye: EyeRegion,
    left_highlight: HighlightDetection,
    right_eye: EyeRegion,
    right_highlight: HighlightDetection,
) -> Tuple[FresnelResult, FresnelResult, float]:
    """Analyze both eyes and return bilateral Fresnel consistency score."""
    left = analyze_fresnel_reflection(left_eye, left_highlight)
    right = analyze_fresnel_reflection(right_eye, right_highlight)

    if left.detected and right.detected:
        consistency = 1.0 - min(
            1.0,
            abs(left.fresnel_reflectance - right.fresnel_reflectance)
            + abs(left.incidence_angle_deg - right.incidence_angle_deg) / 90.0,
        )
    else:
        consistency = 0.0

    return left, right, float(consistency)
