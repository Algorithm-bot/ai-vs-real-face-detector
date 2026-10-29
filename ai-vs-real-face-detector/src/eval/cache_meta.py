"""Feature-cache meta.json contract. Trainers refuse a mismatched cache."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


# Keys that change the numeric content of cached vectors. Extended by later items.
FEATURE_AFFECTING_KEYS = (
    "semantic_model",
    "semantic_pretrained",
    "semantic_dim",
    "semantic_layers",
    "physics_dim",
    "prnu_dim",
    "freq_dim",
    "preprocess",
    "augment",
    "branches",
    "crop_size",
    "max_train_images",
    "subset_seed",
    "p_blur",
    "p_jpeg",
    "p_downscale",
    "max_sigma",
    "min_quality",
    "downscale_min",
    "downscale_max",
)


def normalise_branches(value: Any) -> list[str]:
    """Return the cache branch contract as a stable list, accepting legacy strings."""
    if value is None:
        return []
    values = value.split(",") if isinstance(value, str) else value
    return sorted({str(item).strip() for item in values if str(item).strip()})


def meta_values_equal(key: str, cached: Any, requested: Any) -> bool:
    """Compare cache metadata while remaining compatible with old branch strings."""
    if key == "branches":
        return normalise_branches(cached) == normalise_branches(requested)
    return cached == requested


def load_cache_meta(cache_dir: str | Path) -> Dict[str, Any]:
    path = Path(cache_dir) / "meta.json"
    if not path.exists():
        raise RuntimeError(f"Feature cache is missing meta.json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def requested_meta_from_args(args: Any) -> Dict[str, Any]:
    """Feature-affecting fields present on a training/cache argparse namespace."""
    requested: Dict[str, Any] = {}
    if getattr(args, "semantic_model", None) is not None:
        requested["semantic_model"] = args.semantic_model
    if hasattr(args, "no_semantic_pretrained"):
        requested["semantic_pretrained"] = not bool(args.no_semantic_pretrained)
    mapping = {
        "semantic_dim": "semantic_dim",
        "semantic_layers": "semantic_layers",
        "preprocess": "preprocess",
        "crop_size": "crop_size",
        "augment": "augment",
        "branches": "branches",
        "max_train_images": "max_train_images",
        "max_train_per_class": "max_train_per_class",
        "subset_seed": "subset_seed",
        "p_blur": "p_blur",
        "p_jpeg": "p_jpeg",
        "p_downscale": "p_downscale",
        "max_sigma": "max_sigma",
        "min_quality": "min_quality",
        "downscale_min": "downscale_min",
        "downscale_max": "downscale_max",
    }
    for attr, key in mapping.items():
        if hasattr(args, attr) and getattr(args, attr) not in (None, ""):
            value = getattr(args, attr)
            if key == "branches" and not isinstance(value, str):
                value = ",".join(value)
            if key == "semantic_layers" and not isinstance(value, str):
                value = ",".join(str(v) for v in value)
            requested[key] = value
    # Handle downscale_range -> downscale_min, downscale_max
    if hasattr(args, "downscale_range") and getattr(args, "downscale_range"):
        downscale_min, downscale_max = map(float, args.downscale_range.split(","))
        requested["downscale_min"] = downscale_min
        requested["downscale_max"] = downscale_max
    return requested


def write_cache_meta(cache_dir: str | Path, meta: Mapping[str, Any]) -> None:
    path = Path(cache_dir) / "meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(meta), indent=2) + "\n", encoding="utf-8")


def check_cache_meta(
    cache_dir: str | Path,
    requested: Mapping[str, Any],
    legacy: bool = False,
) -> Dict[str, Any]:
    """Raise if the cache was built with a different feature-affecting config.

    ``legacy`` only enforces ``semantic_model``, matching the previous trainer.
    """
    meta = load_cache_meta(cache_dir)
    if legacy:
        cached_model = meta.get("semantic_model")
        requested_model = requested.get("semantic_model")
        if requested_model is not None and cached_model != requested_model:
            raise RuntimeError(
                f"Feature cache was built with semantic model {cached_model!r}, "
                f"not {requested_model!r}."
            )
        return meta

    mismatches = []
    for key in FEATURE_AFFECTING_KEYS:
        if key not in requested:
            continue
        if key not in meta:
            mismatches.append(f"{key}: cache missing, requested {requested[key]!r}")
            continue
        if not meta_values_equal(key, meta[key], requested[key]):
            mismatches.append(f"{key}: cache={meta[key]!r} requested={requested[key]!r}")
    if mismatches:
        raise RuntimeError(
            "Feature cache meta.json does not match the requested config:\n  "
            + "\n  ".join(mismatches)
            + "\nRebuild the cache or pass --legacy to use the old semantic_model-only check."
        )
    return meta
