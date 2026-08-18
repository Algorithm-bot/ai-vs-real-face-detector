"""Ablation study configurations comparing model variants."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List


class AblationVariant(str, Enum):
    EFFICIENTNET_ONLY = "efficientnet_only"
    PHYSICS_ONLY = "physics_only"
    EFFICIENTNET_PHYSICS = "efficientnet_physics"
    EFFICIENTNET_PHYSICS_PRNU = "efficientnet_physics_prnu"
    EFFICIENTNET_PHYSICS_VIT = "efficientnet_physics_vit"
    FULL = "full"


@dataclass
class AblationConfig:
    """Configuration for one ablation variant."""

    variant: AblationVariant
    use_deep: bool
    use_physics: bool
    use_prnu: bool
    use_semantic: bool
    description: str


ABLATION_CONFIGS: List[AblationConfig] = [
    AblationConfig(
        variant=AblationVariant.EFFICIENTNET_ONLY,
        use_deep=True,
        use_physics=False,
        use_prnu=False,
        use_semantic=False,
        description="EfficientNet-B0 only (stage1 baseline)",
    ),
    AblationConfig(
        variant=AblationVariant.PHYSICS_ONLY,
        use_deep=False,
        use_physics=True,
        use_prnu=False,
        use_semantic=False,
        description="Physics features only (heuristic classifier)",
    ),
    AblationConfig(
        variant=AblationVariant.EFFICIENTNET_PHYSICS,
        use_deep=True,
        use_physics=True,
        use_prnu=False,
        use_semantic=False,
        description="EfficientNet + Physics (original hybrid)",
    ),
    AblationConfig(
        variant=AblationVariant.EFFICIENTNET_PHYSICS_PRNU,
        use_deep=True,
        use_physics=True,
        use_prnu=True,
        use_semantic=False,
        description="EfficientNet + Physics + PRNU",
    ),
    AblationConfig(
        variant=AblationVariant.EFFICIENTNET_PHYSICS_VIT,
        use_deep=True,
        use_physics=True,
        use_prnu=False,
        use_semantic=True,
        description="EfficientNet + Physics + ViT",
    ),
    AblationConfig(
        variant=AblationVariant.FULL,
        use_deep=True,
        use_physics=True,
        use_prnu=True,
        use_semantic=True,
        description="Full model: EfficientNet + Physics + PRNU + ViT",
    ),
]


def get_ablation_config(variant: AblationVariant) -> AblationConfig:
    for cfg in ABLATION_CONFIGS:
        if cfg.variant == variant:
            return cfg
    raise ValueError(f"Unknown ablation variant: {variant}")
