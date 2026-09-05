"""Aggregate physics features into a single vector per image.

The classifier vector is scene-level and works on faces, objects, and
indoor/outdoor photos. Face-specific optics stay optional metadata when a
face happens to be present; they are not required for classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .scene_physics import SCENE_PHYSICS_FEATURE_NAMES, extract_scene_physics
from .corneal_reflection import CornealReflectionResult, analyze_corneal_reflections
from .fresnel import analyze_bilateral_fresnel
from .iris_pupil import IrisPupilResult, analyze_iris_pupil
from .region_detection import FaceLandmarks, FaceRegionDetector
from .shadow_geometry import ShadowGeometryResult, analyze_shadow_geometry


PHYSICS_FEATURE_NAMES: List[str] = list(SCENE_PHYSICS_FEATURE_NAMES)
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
    """Scene-level physics pipeline, with optional face optics in metadata."""

    def __init__(
        self,
        detector: Optional[FaceRegionDetector] = None,
        include_face_cues: bool = False,
    ) -> None:
        self._detector = detector
        self._owns_detector = detector is None
        self.include_face_cues = include_face_cues

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
        image = None
        if landmarks.debug:
            image = landmarks.debug.get("source_image")
        if image is not None:
            vector = extract_scene_physics(image)
        else:
            vector = np.zeros(PHYSICS_FEATURE_DIM, dtype=np.float32)

        if landmarks.detected:
            corneal = corneal or analyze_corneal_reflections(landmarks)
            iris_pupil = iris_pupil or analyze_iris_pupil(landmarks)
            shadow = shadow or analyze_shadow_geometry(
                landmarks, corneal, image=image
            )

        return PhysicsFeatureVector(
            vector=vector,
            landmarks=landmarks,
            corneal=corneal,
            iris_pupil=iris_pupil,
            shadow=shadow,
            metadata={"face_detected": bool(landmarks.detected)},
        )

    def extract(self, image: np.ndarray) -> PhysicsFeatureVector:
        vector = extract_scene_physics(image)
        result = PhysicsFeatureVector(vector=vector)
        if not self.include_face_cues:
            return result

        try:
            detector = self._get_detector()
            landmarks = detector.detect(image)
            landmarks.debug["source_image"] = image
            result.landmarks = landmarks
            result.metadata["face_detected"] = bool(landmarks.detected)
            if landmarks.detected:
                result.corneal = analyze_corneal_reflections(landmarks)
                result.iris_pupil = analyze_iris_pupil(landmarks)
                result.shadow = analyze_shadow_geometry(
                    landmarks, result.corneal, image=image
                )
                if landmarks.left_eye and landmarks.right_eye and result.corneal:
                    left_f, right_f, fresnel_consistency = analyze_bilateral_fresnel(
                        landmarks.left_eye,
                        result.corneal.left,
                        landmarks.right_eye,
                        result.corneal.right,
                    )
                    result.metadata["fresnel"] = {
                        "left_plausible": left_f.physically_plausible,
                        "right_plausible": right_f.physically_plausible,
                        "bilateral_consistency": fresnel_consistency,
                        "left_incidence_deg": left_f.incidence_angle_deg,
                        "right_incidence_deg": right_f.incidence_angle_deg,
                    }
        except Exception as exc:
            result.metadata["face_cues_error"] = str(exc)
        return result

    def close(self) -> None:
        if self._owns_detector and self._detector is not None:
            self._detector.close()
            self._detector = None

    def __enter__(self) -> "PhysicsFeatureExtractor":
        return self

    def __exit__(self, *args) -> None:
        self.close()
