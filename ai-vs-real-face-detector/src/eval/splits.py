"""DEV / REPORT folder partitions for ForenSynths-style evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import yaml

def normalize_source(source: str) -> str:
    return str(source).replace("\\", "/").strip().strip("/").lower()

DEFAULT_SPLITS_PATH = Path(__file__).resolve().parents[2] / "eval" / "splits.yaml"




def matches_folder(source: str, folder: str) -> bool:
    """True if ``source`` is ``folder`` or a descendant (slash-delimited).

    ``stylegan`` matches ``stylegan/car`` but not ``stylegan2``.
    """
    src = normalize_source(source)
    fol = normalize_source(folder)
    if not src or not fol:
        return False
    return src == fol or src.startswith(fol + "/")


def matches_any(source: str, folders: Sequence[str]) -> bool:
    return any(matches_folder(source, folder) for folder in folders)


def folder_of(source: str) -> str:
    """Top-level generator/family name used as the per-generator key."""
    src = normalize_source(source)
    if not src:
        return "unknown"
    return src.split("/", 1)[0]


@dataclass
class EvalSplitSpec:
    dev: List[str] = field(default_factory=list)
    report: List[str] = field(default_factory=list)
    all_folders: List[str] = field(default_factory=list)
    fold_a: List[str] = field(default_factory=list)
    fold_b: List[str] = field(default_factory=list)
    progan: List[str] = field(default_factory=list)
    path: Optional[Path] = None

    def validate(self) -> None:
        if not self.dev:
            raise ValueError("DEV folder list is empty.")
        if not self.report:
            raise ValueError("REPORT folder list is empty.")
        overlap = sorted(
            {
                folder
                for folder in self.dev
                if matches_any(folder, self.report) or any(matches_folder(other, folder) for other in self.report)
            }
        )
        extra = []
        for d in self.dev:
            for r in self.report:
                if matches_folder(d, r) or matches_folder(r, d):
                    extra.append((d, r))
        if extra:
            raise ValueError(f"DEV and REPORT folder sets overlap: {extra}")

        # Assert CRN and IMLE are in the same fold (required because they share real images)
        crn_in_dev = matches_any("crn", self.dev)
        imle_in_dev = matches_any("imle", self.dev)
        if crn_in_dev != imle_in_dev:
            raise ValueError("CRN and IMLE must be in the same fold because they share real images.")

    def assign(self, source: str) -> str:
        if matches_any(source, self.dev):
            return "dev"
        if matches_any(source, self.report):
            return "report"
        if matches_any(source, self.progan):
            return "progan"
        return "unassigned"


def load_split_spec(path: Optional[Path] = None) -> EvalSplitSpec:
    splits_path = Path(path) if path is not None else DEFAULT_SPLITS_PATH
    raw = yaml.safe_load(splits_path.read_text(encoding="utf-8")) or {}
    dev = list(raw.get("dev") or raw.get("fold_a") or [])
    report = list(raw.get("report") or raw.get("fold_b") or [])
    spec = EvalSplitSpec(
        dev=dev,
        report=report,
        all_folders=list(raw.get("all_folders") or []),
        fold_a=list(raw.get("fold_a") or []),
        fold_b=list(raw.get("fold_b") or []),
        progan=list(raw.get("progan") or ["progan"]),
        path=splits_path,
    )
    spec.validate()
    return spec


def partition_sources(
    sources: Iterable[str],
    spec: EvalSplitSpec,
) -> Tuple[List[str], List[str], List[str]]:
    """Return (dev_sources, report_sources, unassigned_sources) unique, sorted."""
    dev, report, unassigned = set(), set(), set()
    for source in sources:
        bucket = spec.assign(source)
        if bucket == "dev":
            dev.add(source)
        elif bucket == "report":
            report.add(source)
        else:
            unassigned.add(source)
    return sorted(dev), sorted(report), sorted(unassigned)


def filter_samples(
    samples: Sequence[Tuple[str, int, str]],
    folders: Sequence[str],
) -> List[Tuple[str, int, str]]:
    return [sample for sample in samples if matches_any(sample[2], folders)]
