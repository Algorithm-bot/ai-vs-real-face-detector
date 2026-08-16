"""Image preprocessing for the deep learning branch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_IMAGE_SIZE = 224


@dataclass
class PreprocessConfig:
    image_size: int = DEFAULT_IMAGE_SIZE
    apply_noise_reduction: bool = True
    apply_clahe: bool = True
    convert_to_lab: bool = False
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)


class FacePreprocessor:
    """OpenCV-based preprocessing before torchvision normalization."""

    def __init__(self, config: Optional[PreprocessConfig] = None) -> None:
        self.config = config or PreprocessConfig()

    def _ensure_rgb_uint8(self, image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 3:
            # Assume BGR from OpenCV file reads; convert to RGB.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def reduce_noise(self, image: np.ndarray) -> np.ndarray:
        cfg = self.config
        return cv2.bilateralFilter(
            image,
            d=cfg.bilateral_d,
            sigmaColor=cfg.bilateral_sigma_color,
            sigmaSpace=cfg.bilateral_sigma_space,
        )

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        cfg = self.config
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip_limit,
            tileGridSize=cfg.clahe_tile_grid_size,
        )
        l_channel = clahe.apply(l_channel)
        enhanced = cv2.merge((l_channel, a_channel, b_channel))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

    def resize(self, image: np.ndarray) -> np.ndarray:
        size = self.config.image_size
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

    def preprocess_numpy(self, image: np.ndarray) -> np.ndarray:
        """Return RGB uint8 array ready for PIL / torchvision."""
        image = self._ensure_rgb_uint8(image)
        if self.config.apply_noise_reduction:
            image = self.reduce_noise(image)
        if self.config.convert_to_lab:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        if self.config.apply_clahe:
            image = self.enhance_contrast(image)
        return self.resize(image)

    def preprocess_pil(self, image: Image.Image) -> Image.Image:
        array = np.array(image.convert("RGB"))
        return Image.fromarray(self.preprocess_numpy(array))

    def preprocess_path(self, path: str) -> Image.Image:
        bgr = cv2.imread(path)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(self.preprocess_numpy(rgb))


def get_train_transforms(image_size: int = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    """Torch transforms applied after OpenCV preprocessing."""
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_val_transforms(image_size: int = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def preprocess_for_model(
    image: Image.Image | np.ndarray | str,
    preprocessor: Optional[FacePreprocessor] = None,
    tensor_transform: Optional[Callable] = None,
) -> torch.Tensor:
    """End-to-end: raw input -> normalized tensor [1, C, H, W]."""
    pre = preprocessor or FacePreprocessor()
    tfm = tensor_transform or get_val_transforms()

    if isinstance(image, str):
        pil = pre.preprocess_path(image)
    elif isinstance(image, np.ndarray):
        pil = pre.preprocess_pil(Image.fromarray(image))
    else:
        pil = pre.preprocess_pil(image)

    tensor = tfm(pil)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    return tensor
