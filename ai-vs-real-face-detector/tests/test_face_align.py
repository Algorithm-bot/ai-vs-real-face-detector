"""Tests for face alignment preprocessing."""

import numpy as np

from src.deep_branch.face_align import FaceAligner, FaceDetectionStatus, _ensure_rgb
from src.deep_branch.preprocessing import FacePreprocessor, PreprocessConfig


def test_ensure_rgb_from_bgr():
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    bgr[:, :, 0] = 200  # high blue channel in BGR
    rgb = _ensure_rgb(bgr)
    assert rgb.shape == (100, 100, 3)
    assert rgb[0, 0, 2] > rgb[0, 0, 0]  # red channel higher after conversion


def test_preprocessor_without_align():
    pre = FacePreprocessor(PreprocessConfig(align_face=False))
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    out, meta = pre.preprocess_numpy(image)
    assert out.shape == (224, 224, 3)
    assert meta is None or meta.status == FaceDetectionStatus.OK


def test_face_aligner_no_face():
    gray = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    # Random noise unlikely to have a detectable face
    with FaceAligner() as aligner:
        result = aligner.process(gray)
    assert result.face_count >= 0
    assert result.image.shape[0] == 224
