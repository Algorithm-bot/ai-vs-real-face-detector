"""B3 — Iris/pupil physics: ellipse fitting, eccentricity, iris texture entropy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from skimage.measure import EllipseModel

from .region_detection import EyeRegion, FaceLandmarks


@dataclass
class PupilEllipse:
    detected: bool
    center: Tuple[float, float] = (0.0, 0.0)
    axes: Tuple[float, float] = (0.0, 0.0)  # semi-major, semi-minor
    angle_deg: float = 0.0
    eccentricity: float = 0.0
    regularity: float = 0.0  # 1 - eccentricity, clipped to [0,1]


@dataclass
class IrisPupilResult:
    left_pupil: PupilEllipse
    right_pupil: PupilEllipse
    left_iris_entropy: float
    right_iris_entropy: float
    pupil_eccentricity_diff: float
    pupil_regularity_mean: float


def _iris_mask_from_points(eye_crop: np.ndarray, iris_points: np.ndarray, bbox) -> np.ndarray:
    x, y, _, _ = bbox
    local = iris_points.copy()
    local[:, 0] -= x
    local[:, 1] -= y
    mask = np.zeros(eye_crop.shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(local.astype(np.float32))
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
    return mask


def _estimate_pupil_boundary(eye_crop: np.ndarray, iris_points: np.ndarray, bbox) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(eye_crop, cv2.COLOR_RGB2GRAY) if eye_crop.ndim == 3 else eye_crop
    mask = _iris_mask_from_points(eye_crop, iris_points, bbox)
    masked = cv2.bitwise_and(gray, gray, mask=mask)

    blurred = cv2.GaussianBlur(masked, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.bitwise_and(binary, mask)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        return None
    return contour.reshape(-1, 2).astype(np.float32)


def fit_pupil_ellipse(contour: np.ndarray) -> PupilEllipse:
    if contour is None or len(contour) < 5:
        return PupilEllipse(detected=False)

    try:
        model = EllipseModel.from_estimate(contour)
    except (ValueError, TypeError):
        model = None
    if model is None:
        return PupilEllipse(detected=False)

    cx, cy = model.center
    axis_lengths = model.axis_lengths
    semi_major = float(max(axis_lengths))
    semi_minor = float(min(axis_lengths))
    theta = float(model.theta)
    if semi_major <= 0:
        return PupilEllipse(detected=False)

    ecc = float(np.sqrt(max(0.0, 1.0 - (semi_minor / semi_major) ** 2)))
    regularity = float(np.clip(1.0 - ecc, 0.0, 1.0))

    return PupilEllipse(
        detected=True,
        center=(float(cx), float(cy)),
        axes=(float(semi_major), float(semi_minor)),
        angle_deg=float(np.degrees(theta)),
        eccentricity=ecc,
        regularity=regularity,
    )


def compute_iris_entropy(eye_crop: np.ndarray, iris_points: np.ndarray, bbox) -> float:
    gray = cv2.cvtColor(eye_crop, cv2.COLOR_RGB2GRAY) if eye_crop.ndim == 3 else eye_crop
    mask = _iris_mask_from_points(eye_crop, iris_points, bbox)
    pixels = gray[mask > 0]
    if pixels.size == 0:
        return 0.0
    hist, _ = np.histogram(pixels, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return float(entropy)


def analyze_iris_pupil(landmarks: FaceLandmarks) -> IrisPupilResult:
    empty = PupilEllipse(detected=False)

    if not landmarks.detected or landmarks.left_eye is None or landmarks.right_eye is None:
        return IrisPupilResult(
            left_pupil=empty,
            right_pupil=empty,
            left_iris_entropy=0.0,
            right_iris_entropy=0.0,
            pupil_eccentricity_diff=0.0,
            pupil_regularity_mean=0.0,
        )

    left_eye = landmarks.left_eye
    right_eye = landmarks.right_eye

    left_contour = _estimate_pupil_boundary(left_eye.crop, left_eye.iris_points, left_eye.bbox)
    right_contour = _estimate_pupil_boundary(right_eye.crop, right_eye.iris_points, right_eye.bbox)

    left_pupil = fit_pupil_ellipse(left_contour) if left_contour is not None else empty
    right_pupil = fit_pupil_ellipse(right_contour) if right_contour is not None else empty

    left_entropy = compute_iris_entropy(left_eye.crop, left_eye.iris_points, left_eye.bbox)
    right_entropy = compute_iris_entropy(right_eye.crop, right_eye.iris_points, right_eye.bbox)

    ecc_diff = abs(left_pupil.eccentricity - right_pupil.eccentricity)
    regularities = [p.regularity for p in (left_pupil, right_pupil) if p.detected]
    reg_mean = float(np.mean(regularities)) if regularities else 0.0

    return IrisPupilResult(
        left_pupil=left_pupil,
        right_pupil=right_pupil,
        left_iris_entropy=left_entropy,
        right_iris_entropy=right_entropy,
        pupil_eccentricity_diff=ecc_diff,
        pupil_regularity_mean=reg_mean,
    )
