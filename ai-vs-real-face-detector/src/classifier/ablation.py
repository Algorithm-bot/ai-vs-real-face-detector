"""Ablation study configurations comparing model variants."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List


class AblationVariant(str, Enum):
    DEEP_ONLY = "deep_only"
    PHYSICS_ONLY = "physics_only"
    PRNU_ONLY = "prnu_only"
    SEMANTIC_ONLY = "semantic_only"
    FULL = "full"


@dataclass
class AblationConfig:
    """Configuration for one ablation variant."""

    variant: AblationVariant
    train_mode: str
    use_deep: bool
    use_physics: bool
    use_prnu: bool
    use_semantic: bool
    description: str
    checkpoint_name: str


ABLATION_CONFIGS: List[AblationConfig] = [
    AblationConfig(
        variant=AblationVariant.DEEP_ONLY,
        train_mode="stage1",
        use_deep=True,
        use_physics=False,
        use_prnu=False,
        use_semantic=False,
        description="Deep branch only (EfficientNet-B0)",
        checkpoint_name="stage1_best.pt",
    ),
    AblationConfig(
        variant=AblationVariant.PHYSICS_ONLY,
        train_mode="physics_only",
        use_deep=False,
        use_physics=True,
        use_prnu=False,
        use_semantic=False,
        description="Physics branch only (scene optics/forensics)",
        checkpoint_name="physics_only_best.pt",
    ),
    AblationConfig(
        variant=AblationVariant.PRNU_ONLY,
        train_mode="prnu_only",
        use_deep=False,
        use_physics=False,
        use_prnu=True,
        use_semantic=False,
        description="PRNU branch only (sensor-noise residual)",
        checkpoint_name="prnu_only_best.pt",
    ),
    AblationConfig(
        variant=AblationVariant.SEMANTIC_ONLY,
        train_mode="semantic_only",
        use_deep=False,
        use_physics=False,
        use_prnu=False,
        use_semantic=True,
        description="Semantic branch only (ViT embedding)",
        checkpoint_name="semantic_only_best.pt",
    ),
    AblationConfig(
        variant=AblationVariant.FULL,
        train_mode="full_hybrid",
        use_deep=True,
        use_physics=True,
        use_prnu=True,
        use_semantic=True,
        description="All four branches combined",
        checkpoint_name="full_hybrid_best.pt",
    ),
]


def get_ablation_config(variant: AblationVariant) -> AblationConfig:
    for cfg in ABLATION_CONFIGS:
        if cfg.variant == variant:
            return cfg
    raise ValueError(f"Unknown ablation variant: {variant}")
