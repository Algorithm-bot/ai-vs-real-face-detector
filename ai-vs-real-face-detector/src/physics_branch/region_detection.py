"""B1 — Face region and landmark detection via dlib (68-point).

Replaces the MediaPipe-based detector to eliminate the TFLite/XNNPACK
threading deadlock encountered during hybrid training. dlib has no
TF/XNNPACK backend, so this class of deadlock isn't possible here.
"""

from __future__ import annotations

import os
import bz2
import shutil
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

dlib = None  # lazy import, mirrors the old lazy mediapipe import

# dlib 68-point (ibug 300-W) landmark scheme.
# Anatomical right eye (appears on the LEFT side of a front-facing image).
DLIB_RIGHT_EYE = [36, 37, 38, 39, 40, 41]
# Anatomical left eye (appears on the RIGHT side of a front-facing image).
DLIB_LEFT_EYE = [42, 43, 44, 45, 46, 47]

_MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
_MODEL_DIR = os.path.expanduser("~/.cache/dlib_models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "shape_predictor_68_face_landmarks.dat")


def _ensure_model() -> str:
    """Download dlib's pretrained 68-point landmark model if not already cached."""
    if os.path.exists(_MODEL_PATH):
        return _MODEL_PATH
    os.makedirs(_MODEL_DIR, exist_ok=True)
    compressed_path = _MODEL_PATH + ".bz2"
    print("Downloading dlib 68-point landmark model (one-time, ~95MB)...")
    urllib.request.urlretrieve(_MODEL_URL, compressed_path)
    with bz2.BZ2File(compressed_path) as f_in, open(_MODEL_PATH, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(compressed_path)
    print("Model ready at", _MODEL_PATH)
    return _MODEL_PATH


@dataclass
class EyeRegion:
    side: str  # "left" or "right"
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    center: Tuple[float, float]
    contour_points: np.ndarray  # (6, 2) pixel coords
    iris_points: np.ndarray  # (5, 2) pixel coords -- approximated, see class docstring
    crop: np.ndarray  # RGB crop of eye region


@dataclass
class FaceLandmarks:
    detected: bool
    image_shape: Tuple[int, int]
    landmarks_normalized: Optional[np.ndarray] = None  # (68, 2) in [0,1]
    landmarks_pixel: Optional[np.ndarray] = None  # (68, 2)
    left_eye: Optional[EyeRegion] = None
    right_eye: Optional[EyeRegion] = None
    debug: Dict = field(default_factory=dict)


class FaceRegionDetector:
    """Detect facial landmarks (dlib 68-point) and extract per-eye regions.

    Drop-in replacement for the old MediaPipe detector: same public
    interface (detect(), close(), context manager) and same EyeRegion /
    FaceLandmarks shapes.

    IMPORTANT DIFFERENCE FROM THE MEDIAPIPE VERSION:
    dlib's 68-point model gives eye *contour* points only -- it has no
    iris/pupil landmarks the way MediaPipe's refine_landmarks=True did.
    iris_points here are therefore APPROXIMATED as a small 5-point ring
    (center + 4 cardinal points) around the eye contour's centroid,
    sized relative to eye width (iris_radius_ratio). This feeds into
    iris_pupil.py's existing intensity-based pupil-boundary search the
    same way MediaPipe's iris points did -- it's a reasonable proxy,
    not true iris localization. If pupil/iris features look degraded
    in debug_physics.py output compared to your earlier MediaPipe runs,
    this approximation (and iris_radius_ratio) is the first thing to
    tune.
    """

    def __init__(
        self,
        max_num_faces: int = 1,
        eye_padding_ratio: float = 0.35,
        iris_radius_ratio: float = 0.22,
    ) -> None:
        global dlib
        if dlib is None:
            import dlib as _dlib

            dlib = _dlib

        self.eye_padding_ratio = eye_padding_ratio
        self.iris_radius_ratio = iris_radius_ratio
        self.max_num_faces = max_num_faces

        model_path = _ensure_model()
        self._face_detector = dlib.get_frontal_face_detector()
        self._shape_predictor = dlib.shape_predictor(model_path)

    def _to_rgb(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

    def _approximate_iris_points(self, contour: np.ndarray) -> np.ndarray:
        cx = float(contour[:, 0].mean())
        cy = float(contour[:, 1].mean())
        eye_width = float(contour[:, 0].max() - contour[:, 0].min())
        r = max(1.0, eye_width * self.iris_radius_ratio)
        return np.array(
            [
                [cx, cy],
                [cx + r, cy],
                [cx - r, cy],
                [cx, cy + r],
                [cx, cy - r],
            ],
            dtype=np.float32,
        )

    def _extract_eye(self, rgb: np.ndarray, side: str, contour: np.ndarray) -> EyeRegion:
        h, w = rgb.shape[:2]
        bbox = self._bbox_from_points(contour, w, h)
        x, y, bw, bh = bbox
        crop = rgb[y : y + bh, x : x + bw].copy()
        center = (float(contour[:, 0].mean()), float(contour[:, 1].mean()))
        iris = self._approximate_iris_points(contour)
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
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        faces = self._face_detector(gray, 1)
        if len(faces) == 0:
            return FaceLandmarks(detected=False, image_shape=(h, w))

        face_rect = faces[0]
        shape = self._shape_predictor(gray, face_rect)
        pixel_points = np.array([(p.x, p.y) for p in shape.parts()], dtype=np.float32)  # (68, 2)

        normalized = pixel_points.copy()
        normalized[:, 0] /= w
        normalized[:, 1] /= h

        right_contour = pixel_points[DLIB_RIGHT_EYE]
        left_contour = pixel_points[DLIB_LEFT_EYE]

        left_eye = self._extract_eye(rgb, "left", left_contour)
        right_eye = self._extract_eye(rgb, "right", right_contour)

        return FaceLandmarks(
            detected=True,
            image_shape=(h, w),
            landmarks_normalized=normalized,
            landmarks_pixel=pixel_points,
            left_eye=left_eye,
            right_eye=right_eye,
        )

    def close(self) -> None:
        # dlib holds no persistent graph/session; kept for interface
        # compatibility with callers using the close()/context-manager pattern.
        pass

    def __enter__(self) -> "FaceRegionDetector":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def draw_landmarks_debug(image: np.ndarray, landmarks: FaceLandmarks) -> np.ndarray:
    """Visualize eye contours and approximated iris points for debugging."""
    if not landmarks.detected:
        return image.copy()

    vis = image.copy()
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