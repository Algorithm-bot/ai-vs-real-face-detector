"""B2 — Corneal specular highlight analysis (Fresnel reflection consistency)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .region_detection import EyeRegion, FaceLandmarks


@dataclass
class HighlightDetection:
    detected: bool
    center: Optional[Tuple[float, float]] = None  # relative to eye crop
    radius: Optional[float] = None
    mask: Optional[np.ndarray] = None
    edges: Optional[np.ndarray] = None


@dataclass
class CornealReflectionResult:
    left: HighlightDetection
    right: HighlightDetection
    highlight_iou: float
    alignment_offset: Tuple[float, float]
    consistent: bool
    debug: dict


def _to_grayscale(eye_crop: np.ndarray) -> np.ndarray:
    if eye_crop.ndim == 2:
        return eye_crop
    return cv2.cvtColor(eye_crop, cv2.COLOR_RGB2GRAY)


def detect_specular_highlight(
    eye_crop: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    min_radius_ratio: float = 0.03,
    max_radius_ratio: float = 0.18,
    intensity_percentile: float = 97.0,
) -> HighlightDetection:
    """
    Detect bright corneal specular highlight using cv2.HoughCircles
    (replaces skimage's hough_circle_peaks, whose internal regionprops
    call hangs/errors on certain accumulator shapes -- see project notes).
    """
    gray = _to_grayscale(eye_crop)
    h, w = gray.shape
    if h < 8 or w < 8:
        return HighlightDetection(detected=False)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)

    min_r = max(2, int(min(h, w) * min_radius_ratio))
    max_r = max(min_r + 1, int(min(h, w) * max_radius_ratio))

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(min_r, 4),
        param1=canny_high,
        param2=15,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is not None and len(circles[0]) > 0:
        cx_f, cy_f, r_f = (float(v) for v in circles[0][0])
    else:
        # Fallback: brightest point, same as before
        _, _, _, max_loc = cv2.minMaxLoc(blurred)
        cx_f, cy_f = float(max_loc[0]), float(max_loc[1])
        r_f = float(min_r)

    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(mask, (int(cx_f), int(cy_f)), max(1, int(r_f)), 255, -1)

    return HighlightDetection(
        detected=True,
        center=(cx_f, cy_f),
        radius=r_f,
        mask=mask,
        edges=edges,
    )


def _normalize_highlight_mask(
    detection: HighlightDetection,
    eye: EyeRegion,
    target_size: Tuple[int, int] = (64, 64),
) -> Optional[np.ndarray]:
    if not detection.detected or detection.mask is None:
        return None
    mask = detection.mask
    x, y, w, h = eye.bbox
    canvas = np.zeros(eye.contour_points.max(axis=0).astype(int)[::-1] * 0 + 1, dtype=np.uint8)
    full_h = int(eye.contour_points[:, 1].max()) + h
    full_w = int(eye.contour_points[:, 0].max()) + w
    full_h = max(full_h, y + mask.shape[0])
    full_w = max(full_w, x + mask.shape[1])
    canvas = np.zeros((full_h, full_w), dtype=np.uint8)
    canvas[y : y + mask.shape[0], x : x + mask.shape[1]] = mask
    resized = cv2.resize(canvas, target_size, interpolation=cv2.INTER_NEAREST)
    return (resized > 0).astype(np.uint8)


def compute_highlight_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def analyze_corneal_reflections(
    landmarks: FaceLandmarks,
    iou_threshold: float = 0.35,
    target_size: Tuple[int, int] = (64, 64),
) -> CornealReflectionResult:
    empty = HighlightDetection(detected=False)
    if not landmarks.detected or landmarks.left_eye is None or landmarks.right_eye is None:
        return CornealReflectionResult(
            left=empty,
            right=empty,
            highlight_iou=0.0,
            alignment_offset=(0.0, 0.0),
            consistent=False,
            debug={"reason": "no_face_or_eyes"},
        )

    left_det = detect_specular_highlight(landmarks.left_eye.crop)
    right_det = detect_specular_highlight(landmarks.right_eye.crop)

    left_mask = _align_eye_highlight_to_canonical(left_det, landmarks.left_eye, target_size)
    right_mask = _align_eye_highlight_to_canonical(right_det, landmarks.right_eye, target_size)

    if left_mask is None or right_mask is None:
        iou = 0.0
        offset = (0.0, 0.0)
    else:
        right_flipped = np.fliplr(right_mask)
        iou = compute_highlight_iou(left_mask, right_flipped)
        offset = _highlight_offset(left_det, right_det)

    return CornealReflectionResult(
        left=left_det,
        right=right_det,
        highlight_iou=iou,
        alignment_offset=offset,
        consistent=iou >= iou_threshold,
        debug={
            "left_center": left_det.center,
            "right_center": right_det.center,
            "target_size": target_size,
        },
    )


def _align_eye_highlight_to_canonical(
    detection: HighlightDetection,
    eye: EyeRegion,
    target_size: Tuple[int, int],
) -> Optional[np.ndarray]:
    if not detection.detected or detection.mask is None:
        return None
    mask = detection.mask.astype(np.uint8)
    return cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)


def _highlight_offset(
    left: HighlightDetection,
    right: HighlightDetection,
) -> Tuple[float, float]:
    if not left.detected or not right.detected:
        return (0.0, 0.0)
    assert left.center is not None and right.center is not None
    lx, ly = left.center
    rx, ry = right.center
    return (float(lx - rx), float(ly - ry))