"""B1 — Face region and landmark detection via dlib (68-point).

Replaces the MediaPipe-based detector to eliminate the TFLite/XNNPACK
threading deadlock encountered during hybrid training.

dlib has no TensorFlow/TFLite/XNNPACK backend.

IMPORTANT:
    dlib's 68-point landmark model provides eye contour landmarks but
    does NOT provide true iris/pupil landmarks. Therefore iris_points
    are approximated from the eye contour centroid and eye width.

This class preserves the public interface expected by the existing
physics branch:
    - FaceRegionDetector.detect()
    - FaceRegionDetector.close()
    - context-manager support
    - EyeRegion
    - FaceLandmarks
"""

from __future__ import annotations

import os
import bz2
import shutil
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------
# Lazy dlib import
# ---------------------------------------------------------------------

dlib = None


# ---------------------------------------------------------------------
# dlib 68-point landmark indices
# ---------------------------------------------------------------------
#
# Anatomical right eye appears on the LEFT side of a front-facing
# image.
#
# Anatomical left eye appears on the RIGHT side of a front-facing
# image.
# ---------------------------------------------------------------------

DLIB_RIGHT_EYE = [36, 37, 38, 39, 40, 41]
DLIB_LEFT_EYE = [42, 43, 44, 45, 46, 47]


# ---------------------------------------------------------------------
# Pretrained dlib landmark model
# ---------------------------------------------------------------------

_MODEL_URL = (
    "http://dlib.net/files/"
    "shape_predictor_68_face_landmarks.dat.bz2"
)

_MODEL_DIR = os.path.expanduser(
    "~/.cache/dlib_models"
)

_MODEL_PATH = os.path.join(
    _MODEL_DIR,
    "shape_predictor_68_face_landmarks.dat"
)


# ---------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------

def _ensure_model() -> str:
    """Download dlib's pretrained 68-point landmark model if necessary."""

    if os.path.exists(_MODEL_PATH):
        return _MODEL_PATH

    os.makedirs(
        _MODEL_DIR,
        exist_ok=True
    )

    compressed_path = _MODEL_PATH + ".bz2"

    print(
        "Downloading dlib 68-point landmark model "
        "(one-time download, approximately 95 MB)..."
    )

    try:
        urllib.request.urlretrieve(
            _MODEL_URL,
            compressed_path
        )

        print("Extracting dlib landmark model...")

        with bz2.BZ2File(compressed_path) as f_in:
            with open(_MODEL_PATH, "wb") as f_out:
                shutil.copyfileobj(
                    f_in,
                    f_out
                )

    finally:
        # Remove compressed file even if extraction fails.
        if os.path.exists(compressed_path):
            os.remove(compressed_path)

    print(
        "Model ready at:",
        _MODEL_PATH
    )

    return _MODEL_PATH


# ---------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------

@dataclass
class EyeRegion:
    """Information about one eye."""

    side: str
    # "left" or "right"

    bbox: Tuple[int, int, int, int]
    # x, y, width, height

    center: Tuple[float, float]

    contour_points: np.ndarray
    # Shape: (6, 2)

    iris_points: np.ndarray
    # Shape: (5, 2)
    #
    # These are approximated because dlib does not provide iris
    # landmarks.

    crop: np.ndarray
    # RGB crop of the eye region


@dataclass
class FaceLandmarks:
    """Complete face and eye landmark information."""

    detected: bool

    image_shape: Tuple[int, int]

    landmarks_normalized: Optional[np.ndarray] = None
    # Shape: (68, 2)
    # Coordinates normalized to [0, 1].

    landmarks_pixel: Optional[np.ndarray] = None
    # Shape: (68, 2)

    left_eye: Optional[EyeRegion] = None

    right_eye: Optional[EyeRegion] = None

    debug: Dict = field(
        default_factory=dict
    )


# ---------------------------------------------------------------------
# Face detector
# ---------------------------------------------------------------------

class FaceRegionDetector:
    """Detect facial landmarks using dlib's 68-point model.

    This is a drop-in replacement for the previous MediaPipe-based
    detector.

    Public interface:
        detector.detect(image)
        detector.close()

    Context manager:
        with FaceRegionDetector() as detector:
            landmarks = detector.detect(image)

    IMPORTANT DIFFERENCE FROM MEDIAPIPE
    -----------------------------------
    MediaPipe Face Mesh with refine_landmarks=True provides detailed
    iris landmarks.

    dlib's 68-point model only provides eye contour landmarks.

    Therefore iris_points are approximated as:

        center
        + right
        + left
        + top
        + bottom

    around the eye contour centroid.

    The existing iris/pupil analysis can still use these points as a
    geometric reference, but they are NOT true iris landmarks.
    """

    def __init__(
        self,
        max_num_faces: int = 1,
        eye_padding_ratio: float = 0.35,
        iris_radius_ratio: float = 0.22,
    ) -> None:

        global dlib

        # Lazy import.
        if dlib is None:
            try:
                import dlib as _dlib
            except ImportError as exc:
                raise ImportError(
                    "dlib is required for FaceRegionDetector. "
                    "Install it with: pip install dlib"
                ) from exc

            dlib = _dlib

        self.eye_padding_ratio = eye_padding_ratio

        self.iris_radius_ratio = iris_radius_ratio

        self.max_num_faces = max_num_faces

        # Load the pretrained landmark model.
        model_path = _ensure_model()

        # dlib frontal face detector.
        self._face_detector = (
            dlib.get_frontal_face_detector()
        )

        # 68-point landmark predictor.
        self._shape_predictor = (
            dlib.shape_predictor(model_path)
        )

    # -----------------------------------------------------------------
    # Image conversion
    # -----------------------------------------------------------------

    def _to_rgb(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """Convert an image to RGB.

        The existing project primarily passes RGB NumPy arrays.

        To avoid accidentally swapping channels, this method assumes
        3-channel NumPy arrays are already RGB.

        Grayscale and RGBA images are converted appropriately.
        """

        if image is None:
            raise ValueError(
                "Image cannot be None."
            )

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "Image must be a NumPy array."
            )

        if image.ndim == 2:
            return cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2RGB
            )

        if image.ndim != 3:
            raise ValueError(
                f"Unsupported image shape: {image.shape}"
            )

        channels = image.shape[2]

        if channels == 4:
            return cv2.cvtColor(
                image,
                cv2.COLOR_RGBA2RGB
            )

        if channels == 3:
            # Project images are already RGB.
            return image

        raise ValueError(
            f"Unsupported number of image channels: {channels}"
        )

    # -----------------------------------------------------------------
    # Eye bounding box
    # -----------------------------------------------------------------

    def _bbox_from_points(
        self,
        points: np.ndarray,
        width: int,
        height: int,
    ) -> Tuple[int, int, int, int]:
        """Create a padded bounding box around eye landmarks."""

        xs = points[:, 0]
        ys = points[:, 1]

        eye_width = xs.max() - xs.min()
        eye_height = ys.max() - ys.min()

        pad_x = eye_width * self.eye_padding_ratio
        pad_y = eye_height * self.eye_padding_ratio

        x_min = max(
            0,
            int(xs.min() - pad_x)
        )

        y_min = max(
            0,
            int(ys.min() - pad_y)
        )

        x_max = min(
            width,
            int(xs.max() + pad_x)
        )

        y_max = min(
            height,
            int(ys.max() + pad_y)
        )

        bbox_width = max(
            1,
            x_max - x_min
        )

        bbox_height = max(
            1,
            y_max - y_min
        )

        return (
            x_min,
            y_min,
            bbox_width,
            bbox_height,
        )

    # -----------------------------------------------------------------
    # Approximate iris points
    # -----------------------------------------------------------------

    def _approximate_iris_points(
        self,
        contour: np.ndarray
    ) -> np.ndarray:
        """Approximate iris points from the eye contour.

        dlib has no iris landmarks, so we estimate a small circular
        region centered on the eye contour centroid.
        """

        cx = float(
            contour[:, 0].mean()
        )

        cy = float(
            contour[:, 1].mean()
        )

        eye_width = float(
            contour[:, 0].max()
            - contour[:, 0].min()
        )

        radius = max(
            1.0,
            eye_width * self.iris_radius_ratio
        )

        return np.array(
            [
                [cx, cy],
                [cx + radius, cy],
                [cx - radius, cy],
                [cx, cy + radius],
                [cx, cy - radius],
            ],
            dtype=np.float32,
        )

    # -----------------------------------------------------------------
    # Eye extraction
    # -----------------------------------------------------------------

    def _extract_eye(
        self,
        rgb: np.ndarray,
        side: str,
        contour: np.ndarray,
    ) -> EyeRegion:
        """Extract an eye region from the full RGB image."""

        height, width = rgb.shape[:2]

        bbox = self._bbox_from_points(
            contour,
            width,
            height,
        )

        x, y, bbox_width, bbox_height = bbox

        crop = rgb[
            y : y + bbox_height,
            x : x + bbox_width,
        ].copy()

        center = (
            float(contour[:, 0].mean()),
            float(contour[:, 1].mean()),
        )

        iris_points = (
            self._approximate_iris_points(
                contour
            )
        )

        return EyeRegion(
            side=side,
            bbox=bbox,
            center=center,
            contour_points=contour.copy(),
            iris_points=iris_points,
            crop=crop,
        )

    # -----------------------------------------------------------------
    # Face detection
    # -----------------------------------------------------------------

    def detect(
        self,
        image: np.ndarray
    ) -> FaceLandmarks:
        """Detect a face and extract 68-point landmarks.

        Parameters
        ----------
        image:
            RGB, RGBA, or grayscale NumPy image.

        Returns
        -------
        FaceLandmarks
            Face and eye information.
        """

        # -------------------------------------------------------------
        # Convert image to RGB
        # -------------------------------------------------------------

        rgb = self._to_rgb(image)

        height, width = rgb.shape[:2]

        # -------------------------------------------------------------
        # Convert to grayscale for dlib detector
        # -------------------------------------------------------------

        gray = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2GRAY
        )

        # -------------------------------------------------------------
        # Downscale large images before face detection.
        #
        # This makes dlib's HOG detector considerably faster.
        # -------------------------------------------------------------

        max_dim = 512

        scale = 1.0

        detect_gray = gray

        if max(height, width) > max_dim:

            scale = (
                max_dim
                / max(height, width)
            )

            detect_width = max(
                1,
                int(width * scale)
            )

            detect_height = max(
                1,
                int(height * scale)
            )

            detect_gray = cv2.resize(
                gray,
                (
                    detect_width,
                    detect_height,
                ),
                interpolation=cv2.INTER_AREA,
            )

        # -------------------------------------------------------------
        # Detect faces.
        #
        # upsample_num_times=0 is intentional for speed.
        # -------------------------------------------------------------

        faces = self._face_detector(
            detect_gray,
            0,
        )

        # -------------------------------------------------------------
        # No face found
        # -------------------------------------------------------------

        if len(faces) == 0:

            return FaceLandmarks(
                detected=False,
                image_shape=(
                    height,
                    width,
                ),
            )

        # -------------------------------------------------------------
        # Select the largest face.
        #
        # This is more robust than simply assuming faces[0] is the
        # desired face.
        # -------------------------------------------------------------

        face_rect = max(
            faces,
            key=lambda rect: (
                rect.right() - rect.left()
            )
            * (
                rect.bottom() - rect.top()
            ),
        )

        # -------------------------------------------------------------
        # Convert detection coordinates back to original resolution.
        # -------------------------------------------------------------

        if scale != 1.0:

            face_rect = dlib.rectangle(
                left=int(
                    face_rect.left() / scale
                ),
                top=int(
                    face_rect.top() / scale
                ),
                right=int(
                    face_rect.right() / scale
                ),
                bottom=int(
                    face_rect.bottom() / scale
                ),
            )

        # -------------------------------------------------------------
        # Run the 68-point landmark predictor on original resolution.
        # -------------------------------------------------------------

        shape = self._shape_predictor(
            gray,
            face_rect,
        )

        # -------------------------------------------------------------
        # Convert dlib points → NumPy array
        # -------------------------------------------------------------

        pixel_points = np.array(
            [
                (point.x, point.y)
                for point in shape.parts()
            ],
            dtype=np.float32,
        )

        # Safety check.
        if pixel_points.shape != (68, 2):

            return FaceLandmarks(
                detected=False,
                image_shape=(
                    height,
                    width,
                ),
                debug={
                    "reason": "unexpected_landmark_shape",
                    "shape": pixel_points.shape,
                },
            )

        # -------------------------------------------------------------
        # Normalize landmark coordinates to [0, 1]
        # -------------------------------------------------------------

        normalized = pixel_points.copy()

        normalized[:, 0] /= max(
            1,
            width
        )

        normalized[:, 1] /= max(
            1,
            height
        )

        # Clamp to [0, 1].
        normalized = np.clip(
            normalized,
            0.0,
            1.0,
        )

        # -------------------------------------------------------------
        # Extract eye contours
        # -------------------------------------------------------------

        right_contour = pixel_points[
            DLIB_RIGHT_EYE
        ]

        left_contour = pixel_points[
            DLIB_LEFT_EYE
        ]

        # -------------------------------------------------------------
        # Extract eye regions
        # -------------------------------------------------------------

        left_eye = self._extract_eye(
            rgb,
            "left",
            left_contour,
        )

        right_eye = self._extract_eye(
            rgb,
            "right",
            right_contour,
        )

        # -------------------------------------------------------------
        # Return complete landmark object
        # -------------------------------------------------------------

        return FaceLandmarks(
            detected=True,
            image_shape=(
                height,
                width,
            ),
            landmarks_normalized=normalized,
            landmarks_pixel=pixel_points,
            left_eye=left_eye,
            right_eye=right_eye,
            debug={
                "detector": "dlib_68",
                "iris_points": "approximated",
                "detection_scale": scale,
                "face_count": len(faces),
            },
        )

    # -----------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------

    def close(self) -> None:
        """Release detector resources.

        dlib does not maintain a persistent TensorFlow/TFLite graph,
        so there is no explicit session to close.

        Kept for compatibility with the previous MediaPipe detector.
        """

        pass

    # -----------------------------------------------------------------
    # Context manager
    # -----------------------------------------------------------------

    def __enter__(
        self
    ) -> "FaceRegionDetector":

        return self

    def __exit__(
        self,
        *args
    ) -> None:

        self.close()


# ---------------------------------------------------------------------
# Debug visualization
# ---------------------------------------------------------------------

def draw_landmarks_debug(
    image: np.ndarray,
    landmarks: FaceLandmarks,
) -> np.ndarray:
    """Visualize eye contours and approximated iris points."""

    if not landmarks.detected:
        return image.copy()

    vis = image.copy()

    for eye in (
        landmarks.left_eye,
        landmarks.right_eye,
    ):

        if eye is None:
            continue

        x, y, width, height = eye.bbox

        # Eye bounding box.
        cv2.rectangle(
            vis,
            (x, y),
            (
                x + width,
                y + height,
            ),
            (0, 255, 0),
            1,
        )

        # dlib eye contour points.
        for px, py in eye.contour_points.astype(int):

            cv2.circle(
                vis,
                (
                    int(px),
                    int(py),
                ),
                1,
                (255, 255, 0),
                -1,
            )

        # Approximated iris points.
        for px, py in eye.iris_points.astype(int):

            cv2.circle(
                vis,
                (
                    int(px),
                    int(py),
                ),
                2,
                (0, 0, 255),
                -1,
            )

    return vis