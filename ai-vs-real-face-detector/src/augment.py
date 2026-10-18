"""
Data augmentation for training: Blur and JPEG compression applied at RAW resolution.

Provides deterministic seeding per image path so cached branches and live
branches see identical augmented-then-cropped pixels for a given path.
"""

from __future__ import annotations

import hashlib
import io
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def path_to_seed(path: str, epoch: int = 0, base_seed: int = 42) -> int:
    """Generate a deterministic 32-bit integer seed from path, epoch, and base_seed."""
    h = hashlib.md5(f"{path}:{epoch}:{base_seed}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def apply_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur with given sigma."""
    if sigma <= 0.0:
        return image
    ksize = int(2 * np.ceil(2 * sigma) + 1)
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def apply_jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    """Apply JPEG compression with given quality (1-100)."""
    quality = max(1, min(100, int(quality)))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result, encimg = cv2.imencode(".jpg", bgr, encode_param)
    if not result:
        return image
    decimg = cv2.imdecode(encimg, 1)
    return cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)


def apply_blur_jpeg_augmentation(
    image: np.ndarray,
    rng: Optional[np.random.RandomState] = None,
    p_blur: float = 0.5,
    p_jpeg: float = 0.5,
    max_sigma: float = 3.0,
    min_quality: int = 30,
) -> np.ndarray:
    """
    Apply Blur + JPEG compression at raw resolution.
    Follows Wang et al. / ForenSynths training augmentation protocol.
    """
    if rng is None:
        rng = np.random.RandomState()

    out = image.copy()
    if rng.rand() < p_blur:
        sigma = float(rng.uniform(0.1, max_sigma))
        out = apply_blur(out, sigma)

    if rng.rand() < p_jpeg:
        quality = int(rng.randint(min_quality, 101))
        out = apply_jpeg(out, quality)

    return out


def augment_raw_image(
    image: np.ndarray,
    path: str = "",
    epoch: int = 0,
    base_seed: int = 42,
    enable_augment: bool = True,
    deterministic: bool = True,
) -> Tuple[np.ndarray, np.random.RandomState]:
    """
    Augment raw image before cropping.
    If deterministic=True: uses path_to_seed(path, epoch, base_seed) so cached and live match.
    If deterministic=False: draws fresh random blur/JPEG per epoch (fresh RNG).
    Returns (augmented_image, crop_rng).
    """
    if not enable_augment:
        rng = np.random.RandomState(path_to_seed(path, epoch, base_seed)) if (deterministic and path) else np.random.RandomState()
        return image, rng

    if deterministic and path:
        seed = path_to_seed(path, epoch, base_seed)
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    augmented = apply_blur_jpeg_augmentation(image, rng=rng)
    return augmented, rng
