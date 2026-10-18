"""Single source of truth for branch names, dims, and fusion width."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.physics_branch.feature_vector import PHYSICS_FEATURE_DIM
from src.prnu_branch.extractor import PRNU_FEATURE_DIM

KNOWN_BRANCHES: Tuple[str, ...] = ("deep", "physics", "prnu", "semantic", "freq")
DEFAULT_BRANCHES: Tuple[str, ...] = ("deep", "physics", "semantic", "freq")
LEGACY_BRANCHES: Tuple[str, ...] = ("deep", "physics", "prnu", "semantic")

DEEP_DIM = 1280  # EfficientNet-B0 global pool; verified against timm num_features


def parse_branches(raw: Optional[str | Sequence[str]], legacy: bool = False) -> List[str]:
    if raw is None or raw == "" or raw == ():
        names = list(LEGACY_BRANCHES if legacy else DEFAULT_BRANCHES)
    elif isinstance(raw, str):
        names = [part.strip().lower() for part in raw.split(",") if part.strip()]
    else:
        names = [str(part).strip().lower() for part in raw if str(part).strip()]
    unknown = [name for name in names if name not in KNOWN_BRANCHES]
    if unknown:
        raise ValueError(f"Unknown branches {unknown}; known={list(KNOWN_BRANCHES)}")
    if not names:
        raise ValueError("At least one branch is required.")
    seen = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return seen


def branches_csv(names: Iterable[str]) -> str:
    return ",".join(names)


def semantic_dim_from_config(cfg: Mapping[str, Any]) -> int:
    """Prefer an explicit encoder dim; fall back to known backbones."""
    if cfg.get("semantic_dim"):
        return int(cfg["semantic_dim"])
    layers = cfg.get("semantic_layers") or ""
    model = str(cfg.get("semantic_model") or "vit_small_patch16_224")
    if "vit_large_patch14_clip" in model or "ViT-L-14" in model:
        per_layer = 768
    elif "vit_small" in model:
        per_layer = 384
    else:
        per_layer = int(cfg.get("semantic_dim") or 384)
    if layers:
        n_layers = len([p for p in str(layers).split(",") if p.strip()])
        if n_layers:
            return per_layer * n_layers
    return per_layer


def freq_dim_from_config(cfg: Mapping[str, Any] | None = None) -> int:
    cfg = cfg or {}
    if cfg.get("freq_dim"):
        return int(cfg["freq_dim"])
    try:
        from src.freq_branch.extractor import FREQ_FEATURE_DIM
        return int(FREQ_FEATURE_DIM)
    except Exception:
        return 0


def branch_dims(cfg: Mapping[str, Any]) -> Dict[str, int]:
    names = parse_branches(cfg.get("branches"), legacy=bool(cfg.get("legacy")))
    dims: Dict[str, int] = {}
    for name in names:
        if name == "deep":
            dims[name] = int(cfg.get("deep_dim") or DEEP_DIM)
        elif name == "physics":
            dims[name] = int(cfg.get("physics_dim") or PHYSICS_FEATURE_DIM)
        elif name == "prnu":
            dims[name] = int(cfg.get("prnu_dim") or PRNU_FEATURE_DIM)
        elif name == "semantic":
            dims[name] = semantic_dim_from_config(cfg)
        elif name == "freq":
            dims[name] = freq_dim_from_config(cfg)
    return dims


def fused_dim(cfg: Mapping[str, Any]) -> int:
    return int(sum(branch_dims(cfg).values()))


def uses(names: Sequence[str], branch: str) -> bool:
    return branch in names
