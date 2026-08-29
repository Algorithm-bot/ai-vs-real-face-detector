"""Face detection, alignment, and cropping for the deep learning branch."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.physics_branch.region_detection import FaceLandmarks, FaceRegionDetector


class FaceDetectionStatus(str, Enum):
    OK = "ok"
    NO_FACE = "no_face"
    MULTIPLE_FACES = "multiple_faces"


@dataclass
class FaceAlignResult:
    """Result of face detection and alignment."""

    image: np.ndarray
    status: FaceDetectionStatus
    face_count: int
    bbox: Optional[Tuple[int, int, int, int]] = None
    landmarks: Optional[FaceLandmarks] = None


# Standard 5-point reference for similarity transform (224 scale).
_REF_POINTS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _landmarks_to_5pt(landmarks: FaceLandmarks) -> Optional[np.ndarray]:
    """Extract 5 alignment points from dlib 68-pt landmarks."""
    pts = landmarks.landmarks_pixel
    if not landmarks.detected or pts is None:
        return None
    if len(pts) < 48:
        return None
    left_eye = pts[36:42].mean(axis=0)
    right_eye = pts[42:48].mean(axis=0)
    nose = pts[30]
    left_mouth = pts[48]
    right_mouth = pts[54]
    return np.array([left_eye, right_eye, nose, left_mouth, right_mouth], dtype=np.float32)


def _similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Compute 2x3 affine similarity transform matrix."""
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    a = (src_demean * dst_demean).sum()
    b = (src_demean[:, 0] * dst_demean[:, 1] - src_demean[:, 1] * dst_demean[:, 0]).sum()
    s = np.sqrt((src_demean ** 2).sum())
    if s < 1e-6:
        return np.eye(2, 3, dtype=np.float32)
    scale = np.sqrt(a ** 2 + b ** 2) / s
    angle = np.arctan2(b, a)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    m = np.array(
        [
            [scale * cos_a, -scale * sin_a],
            [scale * sin_a, scale * cos_a],
        ],
        dtype=np.float32,
    )
    t = dst_mean - m @ src_mean
    transform = np.zeros((2, 3), dtype=np.float32)
    transform[:2, :2] = m
    transform[:, 2] = t
    return transform


def align_face(
    image: np.ndarray,
    landmarks: FaceLandmarks,
    output_size: int = 224,
) -> np.ndarray:
    """Align face using 5-point similarity transform."""
    five_pt = _landmarks_to_5pt(landmarks)
    if five_pt is None:
        return _crop_face_bbox(image, landmarks, output_size)

    scale = output_size / 112.0
    dst = _REF_POINTS * scale
    transform = _similarity_transform(five_pt, dst)
    aligned = cv2.warpAffine(
        image,
        transform,
        (output_size, output_size),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return aligned


def _crop_face_bbox(
    image: np.ndarray,
    landmarks: FaceLandmarks,
    output_size: int,
    padding_ratio: float = 0.25,
) -> np.ndarray:
    """Fallback: expand face bounding box and crop."""
    pts = landmarks.landmarks_pixel
    if pts is not None and len(pts) > 0:
        xs = pts[:, 0]
        ys = pts[:, 1]
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
    else:
        return cv2.resize(image, (output_size, output_size), interpolation=cv2.INTER_AREA)

    w, h = x2 - x1, y2 - y1
    pad_w = int(w * padding_ratio)
    pad_h = int(h * padding_ratio)
    ih, iw = image.shape[:2]
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(iw, x2 + pad_w)
    y2 = min(ih, y2 + pad_h)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return cv2.resize(image, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)


class FaceAligner:
    """Detect, align, and crop faces before EfficientNet preprocessing."""

    def __init__(
        self,
        detector: Optional[FaceRegionDetector] = None,
        output_size: int = 224,
        align: bool = True,
    ) -> None:
        self._detector = detector
        self._owns_detector = detector is None
        self.output_size = output_size
        self.align = align

    def _get_detector(self) -> FaceRegionDetector:
        if self._detector is None:
            self._detector = FaceRegionDetector()
        return self._detector

    def process(self, image: np.ndarray) -> FaceAlignResult:
        """Detect face(s), align/crop, return processed image and status."""
        detector = self._get_detector()
        rgb = _ensure_rgb(image)
        landmarks = detector.detect(rgb)

        face_count = int(landmarks.debug.get("face_count", 1 if landmarks.detected else 0))

        if not landmarks.detected or face_count == 0:
            resized = cv2.resize(rgb, (self.output_size, self.output_size), interpolation=cv2.INTER_AREA)
            return FaceAlignResult(
                image=resized,
                status=FaceDetectionStatus.NO_FACE,
                face_count=0,
            )

        if face_count > 1:
            status = FaceDetectionStatus.MULTIPLE_FACES
        else:
            status = FaceDetectionStatus.OK

        if self.align:
            aligned = align_face(rgb, landmarks, self.output_size)
        else:
            aligned = _crop_face_bbox(rgb, landmarks, self.output_size)

        bbox = None
        if landmarks.landmarks_pixel is not None:
            xs = landmarks.landmarks_pixel[:, 0]
            ys = landmarks.landmarks_pixel[:, 1]
            bbox = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min()))
        return FaceAlignResult(
            image=aligned,
            status=status,
            face_count=face_count,
            bbox=bbox,
            landmarks=landmarks,
        )

    def close(self) -> None:
        if self._owns_detector and self._detector is not None:
            self._detector.close()
            self._detector = None

    def __enter__(self) -> "FaceAligner":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Convert image to RGB uint8, handling BGR, grayscale, and RGBA."""
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    if image.shape[2] == 3:
        # Heuristic: if loaded via cv2.imread it's BGR; if max in R channel
        # is suspiciously low compared to B, assume BGR.
        if image[:, :, 0].mean() > image[:, :, 2].mean() + 10:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    return image
