"""Grad-CAM explainability for the deep branch and Hybrid classifier."""

from __future__ import annotations

import base64
import io
from typing import List, Optional

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
    """Generate Grad-CAM heatmaps for deep or Hybrid models.

    For HybridClassifier, Grad-CAM explains the image/deep branch while
    keeping the physics feature vector fixed. This is necessary because
    Grad-CAM operates on image tensors and cannot directly visualize
    non-image physics features.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layers: Optional[List[nn.Module]] = None,
        use_cuda: bool = False,
        physics_features: Optional[torch.Tensor] = None,
    ) -> None:
        self.model = model
        self.device = torch.device(
            "cuda"
            if use_cuda and torch.cuda.is_available()
            else "cpu"
        )

        self.model.to(self.device)
        self.model.eval()

        if target_layers is None:
            target_layers = self._default_target_layers(model)

        self.target_layers = target_layers

        # IMPORTANT:
        # Grad-CAM must receive a module that actually owns/registers
        # the model parameters. The previous implementation wrapped
        # the model in a module that only stored a reference to the
        # outer explainer, causing:
        #
        #     StopIteration
        #
        # from:
        #
        #     next(model.parameters())
        #
        self.cam_model = self._create_cam_model()

        self.physics_features = None

        if physics_features is not None:
            self.physics_features = physics_features.to(self.device)

        self.cam = GradCAM(
            model=self.cam_model,
            target_layers=self.target_layers,
        )

    # ------------------------------------------------------------
    # TARGET LAYER
    # ------------------------------------------------------------

    def _default_target_layers(
        self,
        model: nn.Module,
    ) -> List[nn.Module]:

        if isinstance(model, HybridClassifier):
            backbone = model.feature_extractor.backbone

        elif isinstance(model, DeepClassifier):
            backbone = model.feature_extractor.backbone

        elif isinstance(model, DeepFeatureExtractor):
            backbone = model.backbone

        else:
            raise TypeError(
                f"Unsupported model type for Grad-CAM: {type(model)}"
            )

        # EfficientNet / timm style backbone.
        if hasattr(backbone, "blocks"):
            return [backbone.blocks[-1]]

        raise ValueError(
            "Could not automatically determine Grad-CAM target layer."
        )

    # ------------------------------------------------------------
    # CAM MODEL
    # ------------------------------------------------------------

    def _create_cam_model(self) -> nn.Module:
        """Create a Grad-CAM-compatible model.

        The returned module owns the original model as a registered
        PyTorch submodule, so model.parameters() works correctly.
        """

        outer = self

        class CamModel(nn.Module):

            def __init__(self) -> None:
                super().__init__()

                # IMPORTANT:
                # Register the actual model as a child module.
                #
                # This makes:
                #
                #     self.parameters()
                #
                # return the underlying model parameters.
                self.model = outer.model

            def forward(
                self,
                x: torch.Tensor,
            ) -> torch.Tensor:

                model = self.model

                if isinstance(model, HybridClassifier):

                    batch_size = x.shape[0]

                    if outer.physics_features is None:

                        physics = torch.zeros(
                            batch_size,
                            model.physics_dim,
                            device=x.device,
                            dtype=x.dtype,
                        )

                    else:

                        physics = outer.physics_features

                        # Support a single physics vector being used
                        # for a single image.
                        if physics.ndim == 1:
                            physics = physics.unsqueeze(0)

                        # Expand the same physics vector across the
                        # batch if necessary.
                        if physics.shape[0] == 1 and batch_size > 1:
                            physics = physics.expand(
                                batch_size,
                                -1,
                            )

                        physics = physics.to(
                            device=x.device,
                            dtype=x.dtype,
                        )

                    logits, _ = model(x, physics)

                    return logits

                if isinstance(model, DeepClassifier):

                    logits, _ = model(x)

                    return logits

                if isinstance(model, DeepFeatureExtractor):

                    return model(x)

                raise TypeError(
                    f"Unsupported model for Grad-CAM: {type(model)}"
                )

        return CamModel()

    # ------------------------------------------------------------
    # GENERATE HEATMAP
    # ------------------------------------------------------------

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generate an RGB Grad-CAM visualization.

        Returns:
            Float32 NumPy array in [0, 1].
        """

        input_tensor = input_tensor.to(self.device)

        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)

        # Make sure gradients are enabled.
        input_tensor = input_tensor.requires_grad_(True)

        targets = None

        if target_class is not None:

            targets = [
                ClassifierOutputTarget(target_class)
            ]

        # Generate CAM.
        grayscale_cam = self.cam(
            input_tensor=input_tensor,
            targets=targets,
        )

        cam = grayscale_cam[0]

        # --------------------------------------------------------
        # DENORMALIZE IMAGE
        # --------------------------------------------------------

        img = (
            input_tensor[0]
            .detach()
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )

        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32,
        )

        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        )

        img = std * img + mean

        img = np.clip(
            img,
            0.0,
            1.0,
        )

        # --------------------------------------------------------
        # OVERLAY
        # --------------------------------------------------------

        visualization = show_cam_on_image(
            img.astype(np.float32),
            cam,
            use_rgb=True,
        )

        return (
            visualization.astype(np.float32)
            / 255.0
        )

    # ------------------------------------------------------------
    # BASE64 PNG
    # ------------------------------------------------------------

    def generate_base64_png(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> str:

        overlay = self.generate(
            input_tensor,
            target_class=target_class,
        )

        pil = Image.fromarray(
            (overlay * 255).astype(np.uint8)
        )

        buffer = io.BytesIO()

        pil.save(
            buffer,
            format="PNG",
        )

        return base64.b64encode(
            buffer.getvalue()
        ).decode("ascii")

    # ------------------------------------------------------------
    # CLEANUP
    # ------------------------------------------------------------

    def close(self) -> None:
        """Release Grad-CAM resources."""

        try:
            if hasattr(self, "cam") and self.cam is not None:
                self.cam.activations_and_grads.release()
        except Exception:
            pass

        self.cam = None


# ================================================================
# CONVENIENCE FUNCTION
# ================================================================

def generate_heatmap(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    as_base64: bool = True,
) -> str | np.ndarray:

    explainer = GradCAMExplainer(
        model=model,
        use_cuda=(
            input_tensor.device.type == "cuda"
        ),
    )

    try:

        if as_base64:

            return explainer.generate_base64_png(
                input_tensor,
                target_class,
            )

        return explainer.generate(
            input_tensor,
            target_class,
        )

    finally:

        explainer.close()