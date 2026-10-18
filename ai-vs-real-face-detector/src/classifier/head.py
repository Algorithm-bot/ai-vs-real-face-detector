"""Stage 3 — Classification head and hybrid models."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.deep_branch.feature_extractor import DeepFeatureExtractor
from src.fusion.fuse import FeatureFusion, FusionMode
from src.physics_branch.feature_vector import PHYSICS_FEATURE_DIM
from src.physics_branch.normalization import PhysicsNormalizer
from src.prnu_branch.extractor import PRNU_FEATURE_DIM
from src.semantic_branch.encoder import DEFAULT_VIT_MODEL


class ClassificationHead(nn.Module):
    """MLP on fused feature vector -> binary logits."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 128),
        dropout: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        layers = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, hidden),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev = hidden
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridClassifier(nn.Module):
    """
    Joint model: deep backbone + physics features fused via concatenation -> MLP.
    Backward-compatible with existing hybrid_best.pt checkpoints.
    """

    LABELS = ("real", "ai")

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        freeze_blocks: int = 5,
        physics_dim: int = PHYSICS_FEATURE_DIM,
        hidden_dims: Tuple[int, ...] = (256, 128),
        dropout: float = 0.3,
        num_classes: int = 2,
        normalize_physics: bool = False,
    ) -> None:
        super().__init__()
        self.feature_extractor = DeepFeatureExtractor(
            model_name=model_name,
            pretrained=pretrained,
            freeze_blocks=freeze_blocks,
        )
        self.deep_dim = self.feature_extractor.embedding_dim
        self.physics_dim = physics_dim
        self.normalize_physics = normalize_physics
        self.physics_normalizer = PhysicsNormalizer() if normalize_physics else None
        self.fusion = FeatureFusion(mode=FusionMode.CONCAT)
        fused_dim = self.fusion.output_dim(self.deep_dim, physics_dim)
        self.classifier = ClassificationHead(
            input_dim=fused_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            num_classes=num_classes,
        )

    def _prepare_physics(self, physics_features: torch.Tensor) -> torch.Tensor:
        if self.normalize_physics and self.physics_normalizer is not None:
            normed = self.physics_normalizer.normalize_batch(physics_features.cpu().numpy())
            return torch.from_numpy(normed).to(physics_features.device)
        return physics_features

    def forward(
        self,
        images: torch.Tensor,
        physics_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        deep_features = self.feature_extractor(images)
        physics = self._prepare_physics(physics_features)
        fused = self.fusion.fuse(deep_features, physics)
        logits = self.classifier(fused)
        return logits, deep_features

    def predict_proba(
        self,
        images: torch.Tensor,
        physics_features: torch.Tensor,
    ) -> torch.Tensor:
        logits, _ = self.forward(images, physics_features)
        return torch.softmax(logits, dim=1)

    @staticmethod
    def decode_label(class_idx: int) -> str:
        return HybridClassifier.LABELS[class_idx]


class FullHybridClassifier(nn.Module):
    """
    Full hybrid model: EfficientNet + Physics + PRNU + ViT with configurable fusion.
    Requires training with mode='full_hybrid'.
    """

    LABELS = ("real", "ai")

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        semantic_model: str = DEFAULT_VIT_MODEL,
        semantic_pretrained: bool = True,
        pretrained: bool = True,
        freeze_blocks: int = 5,
        physics_dim: int = PHYSICS_FEATURE_DIM,
        prnu_dim: int = PRNU_FEATURE_DIM,
        semantic_dim: int = 384,
        freq_dim: int = 64,
        branches: Optional[Sequence[str]] = None,
        fusion_mode: FusionMode = FusionMode.CONCAT,
        branch_dropout: float = 0.0,
        hidden_dims: Tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.3,
        num_classes: int = 2,
        normalize_physics: bool = True,
        normalize_prnu: bool = True,
        no_down: bool = False,
    ) -> None:
        super().__init__()
        from src.features.spec import parse_branches, DEFAULT_BRANCHES
        self.branch_names = parse_branches(branches) if branches is not None else list(DEFAULT_BRANCHES)
        self.semantic_model = semantic_model
        self.semantic_pretrained = semantic_pretrained
        self.feature_extractor = DeepFeatureExtractor(
            model_name=model_name,
            pretrained=pretrained,
            freeze_blocks=freeze_blocks,
            no_down=no_down,
        )
        self.deep_dim = self.feature_extractor.embedding_dim
        self.physics_dim = physics_dim
        self.prnu_dim = prnu_dim
        self.semantic_dim = semantic_dim
        self.freq_dim = freq_dim
        self.fusion_mode = fusion_mode
        self.normalize_physics = normalize_physics
        self.normalize_prnu = normalize_prnu
        self.physics_normalizer = PhysicsNormalizer() if normalize_physics else None
        self.prnu_normalizer = (
            PhysicsNormalizer(
                mean=np.zeros(prnu_dim, dtype=np.float32),
                std=np.ones(prnu_dim, dtype=np.float32),
                feature_names=[f"prnu_{i}" for i in range(prnu_dim)],
            )
            if normalize_prnu
            else None
        )

        all_dims = {
            "deep": self.deep_dim,
            "physics": physics_dim,
            "prnu": prnu_dim,
            "semantic": semantic_dim,
            "freq": freq_dim,
        }
        modality_dims = {name: all_dims[name] for name in self.branch_names}

        self.fusion = FeatureFusion(
            mode=fusion_mode,
            modality_dims=modality_dims,
            branch_dropout=branch_dropout,
        )

        fused_dim = sum(modality_dims.values())
        self.classifier = ClassificationHead(
            input_dim=fused_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            num_classes=num_classes,
        )

    def _prepare_physics(self, physics_features: torch.Tensor) -> torch.Tensor:
        if self.normalize_physics and self.physics_normalizer is not None and self.physics_normalizer.mean.shape[0] == physics_features.shape[-1]:
            normed = self.physics_normalizer.normalize_batch(physics_features.cpu().numpy())
            return torch.from_numpy(normed).to(physics_features.device)
        return physics_features

    def _prepare_prnu(self, prnu_features: torch.Tensor) -> torch.Tensor:
        if self.normalize_prnu and self.prnu_normalizer is not None and self.prnu_normalizer.mean.shape[0] == prnu_features.shape[-1]:
            normed = self.prnu_normalizer.normalize_batch(prnu_features.cpu().numpy())
            return torch.from_numpy(normed).to(prnu_features.device)
        return prnu_features

    def set_feature_scalers(
        self,
        physics_normalizer: Optional[PhysicsNormalizer],
        prnu_normalizer: Optional[PhysicsNormalizer],
    ) -> None:
        if physics_normalizer is not None:
            self.physics_normalizer = physics_normalizer
        if prnu_normalizer is not None:
            self.prnu_normalizer = prnu_normalizer

    def forward(
        self,
        images: torch.Tensor,
        physics_features: Optional[torch.Tensor] = None,
        prnu_features: Optional[torch.Tensor] = None,
        semantic_features: Optional[torch.Tensor] = None,
        freq_features: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        phys_in = physics_features if physics_features is not None else kwargs.get("physics")
        prnu_in = prnu_features if prnu_features is not None else kwargs.get("prnu")
        sem_in = semantic_features if semantic_features is not None else kwargs.get("semantic")
        freq_in = freq_features if freq_features is not None else kwargs.get("freq")

        deep_features = self.feature_extractor(images)
        mods = {"deep": deep_features}
        if "physics" in self.branch_names and phys_in is not None:
            mods["physics"] = self._prepare_physics(phys_in)
        if "prnu" in self.branch_names and prnu_in is not None:
            mods["prnu"] = self._prepare_prnu(prnu_in)
        if "semantic" in self.branch_names and sem_in is not None:
            mods["semantic"] = sem_in
        if "freq" in self.branch_names and freq_in is not None:
            mods["freq"] = freq_in

        fused = self.fusion.fuse(**mods)
        logits = self.classifier(fused)
        return logits, deep_features

    def predict_proba(
        self,
        images: torch.Tensor,
        physics_features: Optional[torch.Tensor] = None,
        prnu_features: Optional[torch.Tensor] = None,
        semantic_features: Optional[torch.Tensor] = None,
        freq_features: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        logits, _ = self.forward(
            images, physics_features, prnu_features, semantic_features, freq_features, **kwargs
        )
        return torch.softmax(logits, dim=1)

    def get_fusion_weights(
        self,
        images: torch.Tensor,
        physics_features: Optional[torch.Tensor] = None,
        prnu_features: Optional[torch.Tensor] = None,
        semantic_features: Optional[torch.Tensor] = None,
        freq_features: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        deep_features = self.feature_extractor(images)
        mods = {"deep": deep_features}
        if "physics" in self.branch_names and physics_features is not None:
            mods["physics"] = self._prepare_physics(physics_features)
        if "prnu" in self.branch_names and prnu_features is not None:
            mods["prnu"] = self._prepare_prnu(prnu_features)
        if "semantic" in self.branch_names and semantic_features is not None:
            mods["semantic"] = semantic_features
        if "freq" in self.branch_names:
            freq_in = freq_features if freq_features is not None else kwargs.get("freq")
            if freq_in is not None:
                mods["freq"] = freq_in
        return self.fusion.get_fusion_weights(**mods)


    @staticmethod
    def decode_label(class_idx: int) -> str:
        return FullHybridClassifier.LABELS[class_idx]


class BranchOnlyClassifier(nn.Module):
    """MLP or Linear Probe on a single branch vector (physics, PRNU, semantic, or freq)."""

    LABELS = ("real", "ai")

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 128),
        dropout: float = 0.3,
        num_classes: int = 2,
        normalize: bool = False,
        linear_probe: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.normalize = normalize
        self.linear_probe = linear_probe
        self.normalizer = PhysicsNormalizer() if normalize else None
        if linear_probe:
            # Item 3d: Linear probe evaluation mode (single linear layer with no hidden layers)
            self.classifier = nn.Linear(input_dim, num_classes)
        else:
            self.classifier = ClassificationHead(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
                num_classes=num_classes,
            )


    def set_normalizer(self, normalizer: Optional[PhysicsNormalizer]) -> None:
        self.normalizer = normalizer

    def _prepare(self, features: torch.Tensor) -> torch.Tensor:
        if self.normalize and self.normalizer is not None:
            normed = self.normalizer.normalize_batch(features.detach().cpu().numpy())
            return torch.from_numpy(normed).to(features.device)
        return features

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        prepared = self._prepare(features)
        return self.classifier(prepared), prepared

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(features)
        return torch.softmax(logits, dim=1)
