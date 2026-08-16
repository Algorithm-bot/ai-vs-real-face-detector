"""Fast image preprocessing for the deep learning branch."""

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

    # Disabled for training because they are expensive CPU operations.
    # If required for a separate physics/debugging pipeline,
    # they can still be enabled manually.
    apply_noise_reduction: bool = False
    apply_clahe: bool = False

    convert_to_lab: bool = False

    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0

    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)


class FacePreprocessor:
    """
    Lightweight preprocessing for the deep learning branch.

    Default training path:

        image
          ↓
        RGB
          ↓
        resize 224x224
          ↓
        torchvision transforms
          ↓
        ImageNet normalization
    """

    def __init__(
        self,
        config: Optional[PreprocessConfig] = None,
    ) -> None:

        self.config = config or PreprocessConfig()

    # --------------------------------------------------------
    # Convert image to RGB uint8
    # --------------------------------------------------------

    def _ensure_rgb_uint8(
        self,
        image: np.ndarray,
    ) -> np.ndarray:

        if image.dtype != np.uint8:

            image = np.clip(
                image,
                0,
                255,
            ).astype(np.uint8)

        if image.ndim == 2:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2RGB,
            )

        elif image.shape[2] == 4:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGBA2RGB,
            )

        elif image.shape[2] == 3:

            # Input from cv2.imread is BGR.
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

        return image

    # --------------------------------------------------------
    # Optional noise reduction
    # --------------------------------------------------------

    def reduce_noise(
        self,
        image: np.ndarray,
    ) -> np.ndarray:

        cfg = self.config

        return cv2.bilateralFilter(
            image,
            d=cfg.bilateral_d,
            sigmaColor=cfg.bilateral_sigma_color,
            sigmaSpace=cfg.bilateral_sigma_space,
        )

    # --------------------------------------------------------
    # Optional CLAHE
    # --------------------------------------------------------

    def enhance_contrast(
        self,
        image: np.ndarray,
    ) -> np.ndarray:

        cfg = self.config

        lab = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2LAB,
        )

        l_channel, a_channel, b_channel = cv2.split(
            lab
        )

        clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip_limit,
            tileGridSize=cfg.clahe_tile_grid_size,
        )

        l_channel = clahe.apply(
            l_channel
        )

        enhanced = cv2.merge(
            (
                l_channel,
                a_channel,
                b_channel,
            )
        )

        return cv2.cvtColor(
            enhanced,
            cv2.COLOR_LAB2RGB,
        )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    def resize(
        self,
        image: np.ndarray,
    ) -> np.ndarray:

        size = self.config.image_size

        return cv2.resize(
            image,
            (size, size),
            interpolation=cv2.INTER_AREA,
        )

    # --------------------------------------------------------
    # Main preprocessing
    # --------------------------------------------------------

    def preprocess_numpy(
        self,
        image: np.ndarray,
    ) -> np.ndarray:

        image = self._ensure_rgb_uint8(
            image
        )

        cfg = self.config

        # Optional expensive preprocessing.
        # Disabled by default during training.

        if cfg.apply_noise_reduction:

            image = self.reduce_noise(
                image
            )

        if cfg.convert_to_lab:

            lab = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2LAB,
            )

            image = cv2.cvtColor(
                lab,
                cv2.COLOR_LAB2RGB,
            )

        if cfg.apply_clahe:

            image = self.enhance_contrast(
                image
            )

        # Resize is cheap compared with
        # bilateral filtering and CLAHE.
        image = self.resize(
            image
        )

        return image

    # --------------------------------------------------------
    # PIL preprocessing
    # --------------------------------------------------------

    def preprocess_pil(
        self,
        image: Image.Image,
    ) -> Image.Image:

        array = np.array(
            image.convert("RGB")
        )

        processed = self.preprocess_numpy(
            array
        )

        return Image.fromarray(
            processed
        )

    # --------------------------------------------------------
    # Path preprocessing
    # --------------------------------------------------------

    def preprocess_path(
        self,
        path: str,
    ) -> Image.Image:

        bgr = cv2.imread(path)

        if bgr is None:

            raise FileNotFoundError(
                f"Could not read image: {path}"
            )

        rgb = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2RGB,
        )

        return Image.fromarray(
            self.preprocess_numpy(rgb)
        )


# ============================================================
# TRAINING TRANSFORMS
# ============================================================

def get_train_transforms(
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> transforms.Compose:

    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(
                p=0.5
            ),

            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.05,
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


# ============================================================
# VALIDATION TRANSFORMS
# ============================================================

def get_val_transforms(
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> transforms.Compose:

    return transforms.Compose(
        [
            transforms.ToTensor(),

            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


# ============================================================
# SINGLE IMAGE INFERENCE PREPROCESSING
# ============================================================

def preprocess_for_model(
    image: Image.Image | np.ndarray | str,
    preprocessor: Optional[FacePreprocessor] = None,
    tensor_transform: Optional[Callable] = None,
) -> torch.Tensor:

    pre = (
        preprocessor
        if preprocessor is not None
        else FacePreprocessor()
    )

    tfm = (
        tensor_transform
        if tensor_transform is not None
        else get_val_transforms()
    )

    if isinstance(image, str):

        pil = pre.preprocess_path(
            image
        )

    elif isinstance(image, np.ndarray):

        pil = pre.preprocess_pil(
            Image.fromarray(image)
        )

    else:

        pil = pre.preprocess_pil(
            image
        )

    tensor = tfm(
        pil
    )

    if tensor.dim() == 3:

        tensor = tensor.unsqueeze(0)

    return tensor