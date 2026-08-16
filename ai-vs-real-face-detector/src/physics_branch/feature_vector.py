"""B5 — Aggregate physics features into a single vector per image."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .corneal_reflection import CornealReflectionResult, analyze_corneal_reflections
from .iris_pupil import IrisPupilResult, analyze_iris_pupil
from .region_detection import FaceLandmarks, FaceRegionDetector
from .shadow_geometry import ShadowGeometryResult, analyze_shadow_geometry


# Fixed feature order for reproducibility
PHYSICS_FEATURE_NAMES: List[str] = [
    "face_detected",
    "highlight_iou",
    "highlight_consistent",
    "highlight_offset_x",
    "highlight_offset_y",
    "left_pupil_eccentricity",
    "right_pupil_eccentricity",
    "pupil_eccentricity_diff",
    "pupil_regularity_mean",
    "left_iris_entropy",
    "right_iris_entropy",
    "iris_entropy_mean",
    "iris_entropy_diff",
    "light_angle_diff_deg",
    "light_cosine_similarity",
    "light_consistent",
    "left_highlight_detected",
    "right_highlight_detected",
    "left_pupil_detected",
    "right_pupil_detected",
]

PHYSICS_FEATURE_DIM = len(PHYSICS_FEATURE_NAMES)


@dataclass
class PhysicsFeatureVector:
    vector: np.ndarray
    names: List[str] = field(default_factory=lambda: list(PHYSICS_FEATURE_NAMES))
    landmarks: Optional[FaceLandmarks] = None
    corneal: Optional[CornealReflectionResult] = None
    iris_pupil: Optional[IrisPupilResult] = None
    shadow: Optional[ShadowGeometryResult] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return len(self.vector)

    def to_dict(self) -> Dict[str, float]:
        return {name: float(val) for name, val in zip(self.names, self.vector)}


class PhysicsFeatureExtractor:
    """Run B1–B4 pipeline and return concatenated feature vector."""

    def __init__(self, detector: Optional[FaceRegionDetector] = None) -> None:
        self._detector = detector
        self._owns_detector = detector is None

    def _get_detector(self) -> FaceRegionDetector:
        if self._detector is None:
            self._detector = FaceRegionDetector()
        return self._detector

    def extract_from_landmarks(
        self,
        landmarks: FaceLandmarks,
        corneal: Optional[CornealReflectionResult] = None,
        iris_pupil: Optional[IrisPupilResult] = None,
        shadow: Optional[ShadowGeometryResult] = None,
    ) -> PhysicsFeatureVector:
        corneal = corneal or analyze_corneal_reflections(landmarks)
        iris_pupil = iris_pupil or analyze_iris_pupil(landmarks)
        shadow = shadow or analyze_shadow_geometry(landmarks, corneal)

        left_pupil = iris_pupil.left_pupil
        right_pupil = iris_pupil.right_pupil
        entropy_mean = (iris_pupil.left_iris_entropy + iris_pupil.right_iris_entropy) / 2.0
        entropy_diff = abs(iris_pupil.left_iris_entropy - iris_pupil.right_iris_entropy)

        values = [
            float(landmarks.detected),
            corneal.highlight_iou,
            float(corneal.consistent),
            corneal.alignment_offset[0],
            corneal.alignment_offset[1],
            left_pupil.eccentricity,
            right_pupil.eccentricity,
            iris_pupil.pupil_eccentricity_diff,
            iris_pupil.pupil_regularity_mean,
            iris_pupil.left_iris_entropy,
            iris_pupil.right_iris_entropy,
            entropy_mean,
            entropy_diff,
            shadow.angle_difference_deg,
            shadow.vector_cosine_similarity,
            float(shadow.consistent),
            float(corneal.left.detected),
            float(corneal.right.detected),
            float(left_pupil.detected),
            float(right_pupil.detected),
        ]

        vector = np.array(values, dtype=np.float32)
        return PhysicsFeatureVector(
            vector=vector,
            landmarks=landmarks,
            corneal=corneal,
            iris_pupil=iris_pupil,
            shadow=shadow,
        )

    def extract(self, image: np.ndarray) -> PhysicsFeatureVector:
        detector = self._get_detector()
        landmarks = detector.detect(image)
        return self.extract_from_landmarks(landmarks)

    def close(self) -> None:
        if self._owns_detector and self._detector is not None:
            self._detector.close()
            self._detector = None

    def __enter__(self) -> "PhysicsFeatureExtractor":
        return self

    def __exit__(self, *args) -> None:
        self.close()
