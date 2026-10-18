"""Helpers for DEV checkpoint selection and last-N epoch snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Optional


def parse_seed_list(args: Any) -> List[int]:
    raw = getattr(args, "seeds", "") or ""
    if str(raw).strip():
        return [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    return [int(getattr(args, "seed", 42))]


def save_epoch_snapshot(
    output_dir: Path,
    mode: str,
    epoch: int,
    keep: int,
    save_fn: Callable[[Path], None],
    rotate_fn: Callable[[Path, str, int], None],
) -> Optional[Path]:
    keep = int(keep or 0)
    if keep <= 0:
        return None
    path = Path(output_dir) / f"{mode}_epoch_{int(epoch):03d}.pt"
    save_fn(path)
    rotate_fn(Path(output_dir), mode, keep)
    return path
