"""Per-image prediction CSV schema used by evaluate / calibrate / stacking."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

PREDICTION_FIELDS: List[str] = [
    "path",
    "folder",
    "source",
    "label",
    "p_ai",
    "p_ai_deep",
    "p_ai_physics",
    "p_ai_semantic",
    "p_ai_freq",
    "p_ai_prnu",
    "logit_real",
    "logit_ai",
    "split",
]


def empty_row(**overrides: Any) -> Dict[str, Any]:
    row = {field: "" for field in PREDICTION_FIELDS}
    row.update(overrides)
    return row


def write_predictions_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(PREDICTION_FIELDS)
    extra = []
    for row in rows:
        for key in row:
            if key not in fieldnames and key not in extra:
                extra.append(key)
    fieldnames.extend(extra)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_predictions_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
