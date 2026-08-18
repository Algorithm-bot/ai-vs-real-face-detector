"""Fast image preprocessing for the deep learning branch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.deep_branch.face_align import FaceAligner, FaceAlignResult, FaceDetectionStatus, _ensure_rgb as ensure_rgb_uint8


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_IMAGE_SIZE = 224


@dataclass
class PreprocessConfig:
    image_size: int = DEFAULT_IMAGE_SIZE

    # Face detection and alignment before resize/normalize.
    align_face: bool = False
    is_bgr: bool = False

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
        self._aligner: Optional[FaceAligner] = None

    def _get_aligner(self) -> FaceAligner:
        if self._aligner is None:
            self._aligner = FaceAligner(output_size=self.config.image_size, align=self.config.align_face)
        return self._aligner

    def align_and_crop(self, image: np.ndarray) -> FaceAlignResult:
        """Detect face, align, and crop. Returns status for no-face/multiple-face."""
        rgb = self._ensure_rgb_uint8(image)
        if self.config.align_face:
            return self._get_aligner().process(rgb)
        resized = self.resize(rgb)
        return FaceAlignResult(image=resized, status=FaceDetectionStatus.OK, face_count=1)

    # --------------------------------------------------------
    # Convert image to RGB uint8
    # --------------------------------------------------------

    def _ensure_rgb_uint8(
        self,
        image: np.ndarray,
        is_bgr: Optional[bool] = None,
    ) -> np.ndarray:

        use_bgr = is_bgr if is_bgr is not None else self.config.is_bgr
        if use_bgr and image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return ensure_rgb_uint8(image)

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

        if self.config.align_face:
            align_result = self.align_and_crop(image)
            image = align_result.image
        else:
            align_result = None

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

        return image, align_result

    def preprocess_numpy_with_meta(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[FaceAlignResult]]:
        """Like preprocess_numpy but also returns face alignment metadata."""
        return self.preprocess_numpy(image)

    def preprocess_numpy_legacy(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """Backward-compatible: returns only the image array."""
        out, _ = self.preprocess_numpy(image)
        return out

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

        processed, _ = self.preprocess_numpy(
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

        processed, _ = self.preprocess_numpy(rgb)

        return Image.fromarray(
            processed
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
    return_meta: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Optional[FaceAlignResult]]:

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

    align_meta: Optional[FaceAlignResult] = None

    if isinstance(image, str):
        bgr = cv2.imread(image)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {image}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        array, align_meta = pre.preprocess_numpy(rgb)
        pil = Image.fromarray(array)
    elif isinstance(image, np.ndarray):
        array, align_meta = pre.preprocess_numpy(image)
        pil = Image.fromarray(array)
    else:
        array = np.array(image.convert("RGB"))
        array, align_meta = pre.preprocess_numpy(array)
        pil = Image.fromarray(array)

    tensor = tfm(pil)

    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    if return_meta:
        return tensor, align_meta
    return tensor