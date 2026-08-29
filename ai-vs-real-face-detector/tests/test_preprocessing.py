"""Tests for deep branch preprocessing."""

import numpy as np
from PIL import Image

from src.deep_branch.preprocessing import FacePreprocessor, preprocess_for_model


def test_preprocessor_output_shape():
    pre = FacePreprocessor()
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    out, _ = pre.preprocess_numpy(image)
    assert out.shape == (224, 224, 3)
    assert out.dtype == np.uint8


def test_preprocess_for_model_tensor_shape():
    pil = Image.new("RGB", (256, 256), color=(128, 64, 32))
    tensor = preprocess_for_model(pil)
    assert tensor.shape == (1, 3, 224, 224)
