"""B1 — Face region and landmark detection via MediaPipe Face Mesh."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Lazy import mediapipe to avoid pulling tensorflow at module import time
mp = None

# MediaPipe Face Mesh landmark indices (refine_landmarks=True enables iris points).
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_OUTER = 263
RIGHT_EYE_INNER = 362
LEFT_IRIS = list(range(468, 473))
RIGHT_IRIS = list(range(473, 478))
LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


@dataclass
class EyeRegion:
    side: str  # "left" or "right"
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    center: Tuple[float, float]
    contour_points: np.ndarray  # (N, 2) pixel coords
    iris_points: np.ndarray  # (5, 2) pixel coords
    crop: np.ndarray  # RGB crop of eye region


@dataclass
class FaceLandmarks:
    detected: bool
    image_shape: Tuple[int, int]
    landmarks_normalized: Optional[np.ndarray] = None  # (478, 2) in [0,1]
    landmarks_pixel: Optional[np.ndarray] = None  # (478, 2)
    left_eye: Optional[EyeRegion] = None
    right_eye: Optional[EyeRegion] = None
    debug: Dict = field(default_factory=dict)


class FaceRegionDetector:
    """Detect face mesh landmarks and extract per-eye regions."""

    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        eye_padding_ratio: float = 0.35,
    ) -> None:
        global mp
        if mp is None:
            import mediapipe as _mp

            mp = _mp
        self.eye_padding_ratio = eye_padding_ratio
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
        )

    def _to_rgb(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        # Assume BGR from cv2.imread
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _landmarks_to_pixel(
        self, landmarks, width: int, height: int
    ) -> np.ndarray:
        points = np.array(
            [(lm.x * width, lm.y * height) for lm in landmarks.landmark],
            dtype=np.float32,
        )
        return points

    def _bbox_from_points(
        self, points: np.ndarray, width: int, height: int
    ) -> Tuple[int, int, int, int]:
        xs = points[:, 0]
        ys = points[:, 1]
        pad_x = (xs.max() - xs.min()) * self.eye_padding_ratio
        pad_y = (ys.max() - ys.min()) * self.eye_padding_ratio
        x_min = max(0, int(xs.min() - pad_x))
        y_min = max(0, int(ys.min() - pad_y))
        x_max = min(width, int(xs.max() + pad_x))
        y_max = min(height, int(ys.max() + pad_y))
        return x_min, y_min, x_max - x_min, y_max - y_min

    def _extract_eye(
        self,
        rgb: np.ndarray,
        side: str,
        contour_indices: List[int],
        iris_indices: List[int],
        all_points: np.ndarray,
    ) -> EyeRegion:
        h, w = rgb.shape[:2]
        contour = all_points[contour_indices]
        iris = all_points[iris_indices]
        bbox = self._bbox_from_points(contour, w, h)
        x, y, bw, bh = bbox
        crop = rgb[y : y + bh, x : x + bw].copy()
        center = (float(contour[:, 0].mean()), float(contour[:, 1].mean()))
        return EyeRegion(
            side=side,
            bbox=bbox,
            center=center,
            contour_points=contour,
            iris_points=iris,
            crop=crop,
        )

    def detect(self, image: np.ndarray) -> FaceLandmarks:
        rgb = self._to_rgb(image)
        h, w = rgb.shape[:2]
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return FaceLandmarks(detected=False, image_shape=(h, w))

        face = results.multi_face_landmarks[0]
        pixel_points = self._landmarks_to_pixel(face, w, h)
        normalized = pixel_points.copy()
        normalized[:, 0] /= w
        normalized[:, 1] /= h

        left_eye = self._extract_eye(
            rgb, "left", LEFT_EYE_CONTOUR, LEFT_IRIS, pixel_points
        )
        right_eye = self._extract_eye(
            rgb, "right", RIGHT_EYE_CONTOUR, RIGHT_IRIS, pixel_points
        )

        return FaceLandmarks(
            detected=True,
            image_shape=(h, w),
            landmarks_normalized=normalized,
            landmarks_pixel=pixel_points,
            left_eye=left_eye,
            right_eye=right_eye,
        )

    def close(self) -> None:
        self._face_mesh.close()

    def __enter__(self) -> "FaceRegionDetector":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def draw_landmarks_debug(image: np.ndarray, landmarks: FaceLandmarks) -> np.ndarray:
    """Visualize eye contours and iris points for debugging."""
    if not landmarks.detected:
        return image.copy()

    vis = image.copy()
    if vis.shape[2] == 3 and vis.dtype == np.uint8:
        # Keep as-is; caller may pass BGR or RGB.
        pass

    for eye in (landmarks.left_eye, landmarks.right_eye):
        if eye is None:
            continue
        x, y, w, h = eye.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
        for px, py in eye.contour_points.astype(int):
            cv2.circle(vis, (int(px), int(py)), 1, (255, 255, 0), -1)
        for px, py in eye.iris_points.astype(int):
            cv2.circle(vis, (int(px), int(py)), 2, (0, 0, 255), -1)
    return vis
