"""Grad-CAM explainability for the deep branch."""

from __future__ import annotations

import base64
import io
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.classifier.head import HybridClassifier
from src.deep_branch.feature_extractor import DeepClassifier, DeepFeatureExtractor


class GradCAMExplainer:
    """Generate Grad-CAM heatmaps for deep-branch predictions."""

    def __init__(
        self,
        model: nn.Module,
        target_layers: Optional[List[nn.Module]] = None,
        use_cuda: bool = False,
    ) -> None:
        self.model = model
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        if target_layers is None:
            target_layers = self._default_target_layers(model)
        self.target_layers = target_layers
        self.cam = GradCAM(model=self._cam_model_wrapper(), target_layers=target_layers)

    def _default_target_layers(self, model: nn.Module) -> List[nn.Module]:
        if isinstance(model, HybridClassifier):
            backbone = model.feature_extractor.backbone
        elif isinstance(model, DeepClassifier):
            backbone = model.feature_extractor.backbone
        elif isinstance(model, DeepFeatureExtractor):
            backbone = model.backbone
        else:
            raise TypeError(f"Unsupported model type: {type(model)}")

        # EfficientNet last conv block
        if hasattr(backbone, "blocks"):
            return [backbone.blocks[-1]]
        raise ValueError("Could not infer Grad-CAM target layer")

    def _cam_model_wrapper(self) -> nn.Module:
        """Grad-CAM needs a model that outputs classification logits from images."""

        class CamWrapper(nn.Module):
            def __init__(self, outer: "GradCAMExplainer") -> None:
                super().__init__()
                self.outer = outer

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                model = self.outer.model
                if isinstance(model, HybridClassifier):
                    # Use zero physics features for CAM visualization focus on image
                    batch = x.shape[0]
                    physics = torch.zeros(
                        batch, model.physics_dim, device=x.device, dtype=x.dtype
                    )
                    logits, _ = model(x, physics)
                    return logits
                if isinstance(model, DeepClassifier):
                    logits, _ = model(x)
                    return logits
                if isinstance(model, DeepFeatureExtractor):
                    return model(x)
                raise TypeError("Unsupported model for Grad-CAM")

        return CamWrapper(self)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Return RGB heatmap overlay as float32 array in [0, 1]."""
        input_tensor = input_tensor.to(self.device)
        targets = None
        if target_class is not None:
            targets = [ClassifierOutputTarget(target_class)]

        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        cam = grayscale_cam[0]

        # Denormalize tensor for overlay
        img = input_tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        visualization = show_cam_on_image(img.astype(np.float32), cam, use_rgb=True)
        return visualization.astype(np.float32) / 255.0

    def generate_base64_png(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> str:
        overlay = self.generate(input_tensor, target_class=target_class)
        pil = Image.fromarray((overlay * 255).astype(np.uint8))
        buffer = io.BytesIO()
        pil.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def generate_heatmap(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    as_base64: bool = True,
) -> str | np.ndarray:
    explainer = GradCAMExplainer(model)
    if as_base64:
        return explainer.generate_base64_png(input_tensor, target_class)
    return explainer.generate(input_tensor, target_class)
