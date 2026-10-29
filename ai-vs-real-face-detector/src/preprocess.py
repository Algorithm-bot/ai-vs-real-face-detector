"""
Single source of truth for image preprocessing and augmentations.

Used consistently by training, cache_features.py, evaluate.py, and inference.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_CROP_SIZE = 224


class PreprocessMode(str, Enum):
    """Preprocessing modes for spatial alignment across branches."""
    RESIZE = "resize"           # Legacy: force resize to crop_size x crop_size
    CROP = "crop"               # Random crop (train) / center crop (eval)
    CENTER_CROP = "center_crop" # Always center crop, no resize if large enough
    NO_RESIZE = "no_resize"     # Preserve original resolution (for pixel-level branches)
    ASPECT_CROP = "aspect_crop" # Aspect-preserving center crop, resize only if needed

# Frequency branch rule: Always extract from the unresized center 224x224 crop
# from src/preprocess.py to maintain consistency across resolutions.
def preprocess_for_freq(
    image: np.ndarray,
    crop_size: int = DEFAULT_CROP_SIZE,
    mode: str = "center_crop"
) -> np.ndarray:
    """
    Preprocess image for frequency branch.

    Args:
        image: Input RGB image
        crop_size: Target crop size
        mode: Preprocessing mode (uses PreprocessMode values)

    Returns:
        Preprocessed image for frequency feature extraction
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = image[:, :, :3]

    # Use PreprocessPipeline for consistency with deep branch
    pipeline = PreprocessPipeline(mode=mode, crop_size=crop_size, is_training=False)
    return pipeline.preprocess_image(image)


def pad_if_smaller(image: np.ndarray, target_size: int = DEFAULT_CROP_SIZE) -> np.ndarray:
    """Pad image with edge reflection if width or height is less than target_size."""
    h, w = image.shape[:2]
    pad_h = max(0, target_size - h)
    pad_w = max(0, target_size - w)
    if pad_h > 0 or pad_w > 0:
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT)
    return image


def center_crop_numpy(image: np.ndarray, crop_size: int = DEFAULT_CROP_SIZE) -> np.ndarray:
    """Extract a center crop of size crop_size x crop_size without resizing."""
    image = pad_if_smaller(image, crop_size)
    h, w = image.shape[:2]
    start_y = (h - crop_size) // 2
    start_x = (w - crop_size) // 2
    return image[start_y : start_y + crop_size, start_x : start_x + crop_size]


def random_crop_numpy(
    image: np.ndarray,
    crop_size: int = DEFAULT_CROP_SIZE,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Extract a random crop of size crop_size x crop_size without resizing."""
    image = pad_if_smaller(image, crop_size)
    h, w = image.shape[:2]
    max_y = h - crop_size
    max_x = w - crop_size
    if rng is None:
        start_y = np.random.randint(0, max_y + 1) if max_y > 0 else 0
        start_x = np.random.randint(0, max_x + 1) if max_x > 0 else 0
    else:
        start_y = rng.randint(0, max_y + 1) if max_y > 0 else 0
        start_x = rng.randint(0, max_x + 1) if max_x > 0 else 0
    return image[start_y : start_y + crop_size, start_x : start_x + crop_size]


def resize_numpy(image: np.ndarray, size: int = DEFAULT_CROP_SIZE) -> np.ndarray:
    """Legacy resize with INTER_AREA."""
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def aspect_preserving_center_crop(
    image: np.ndarray,
    crop_size: int = DEFAULT_CROP_SIZE
) -> np.ndarray:
    """
    Aspect-preserving center crop:
    - If image >= crop_size in both dimensions: extract center crop without resizing
    - If smaller in either dimension: pad with edge reflection then center crop
    - If larger but not square: center crop to min dimension, then resize to target
    """
    h, w = image.shape[:2]

    if h >= crop_size and w >= crop_size:
        # Image is large enough - extract center crop without resizing
        # If not square, crop to square first (min dimension), then resize if needed
        if h != w:
            min_dim = min(h, w)
            start_y = (h - min_dim) // 2
            start_x = (w - min_dim) // 2
            image = image[start_y:start_y + min_dim, start_x:start_x + min_dim]
            if min_dim != crop_size:
                image = cv2.resize(image, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
        else:
            # Square and large enough - direct center crop
            start_y = (h - crop_size) // 2
            start_x = (w - crop_size) // 2
            image = image[start_y:start_y + crop_size, start_x:start_x + crop_size]
    else:
        # Image is smaller than crop_size in at least one dimension
        # Pad with edge reflection then center crop
        image = pad_if_smaller(image, crop_size)
        h, w = image.shape[:2]
        start_y = (h - crop_size) // 2
        start_x = (w - crop_size) // 2
        image = image[start_y:start_y + crop_size, start_x:start_x + crop_size]

    return image


class PreprocessPipeline:
    """Shared preprocessing pipeline."""

    def __init__(
        self,
        mode: str = "crop",  # "crop", "resize", "center_crop", "no_resize", "aspect_crop"
        crop_size: int = DEFAULT_CROP_SIZE,
        is_training: bool = False,
        seed: Optional[int] = None,
    ):
        self.mode = mode.lower()
        self.crop_size = crop_size
        self.is_training = is_training
        self.rng = np.random.RandomState(seed) if seed is not None else None

    def preprocess_image(
        self,
        image: np.ndarray,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """Preprocess an RGB uint8 image according to mode."""
        mode = self.mode

        if mode == PreprocessMode.RESIZE.value:
            return resize_numpy(image, self.crop_size)

        elif mode == PreprocessMode.NO_RESIZE.value:
            # Return original image unchanged (caller must handle downstream)
            return image

        elif mode == PreprocessMode.CENTER_CROP.value:
            # Always center crop, no random component
            return center_crop_numpy(image, self.crop_size)

        elif mode == PreprocessMode.ASPECT_CROP.value:
            # Aspect-preserving center crop
            return aspect_preserving_center_crop(image, self.crop_size)

        elif mode == PreprocessMode.CROP.value:
            # Original behavior: random crop for training, center crop for eval
            if self.is_training:
                use_rng = rng or self.rng
                return random_crop_numpy(image, self.crop_size, rng=use_rng)
            else:
                return center_crop_numpy(image, self.crop_size)

        else:
            # Default to center crop for unknown modes
            return center_crop_numpy(image, self.crop_size)


def get_image_tensor_transforms(
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
    is_training: bool = False,
) -> transforms.Compose:
    """
    Torchvision tensor transforms applied after spatial crop.
    For deep branch only during training: horizontal flip + mild color jitter.
    """
    tfms = []
    if is_training:
        tfms.append(transforms.RandomHorizontalFlip(p=0.5))
        tfms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05))
    tfms.append(transforms.ToTensor())
    tfms.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(tfms)
