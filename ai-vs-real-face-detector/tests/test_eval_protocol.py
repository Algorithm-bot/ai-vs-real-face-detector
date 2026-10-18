"""DEV/REPORT leakage, bootstrap CIs, cache-meta contract, dims-from-config."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.cache_meta import check_cache_meta, write_cache_meta
from src.eval.metrics import (
    best_balanced_accuracy_threshold,
    bootstrap_ci,
    expected_calibration_error,
    metric_summary,
)
from src.eval.splits import load_split_spec, matches_folder
from src.features.spec import branch_dims, fused_dim, parse_branches


def test_default_dev_report_are_disjoint_and_cover_required_types():
    spec = load_split_spec()
    spec.validate()
    for d in spec.dev:
        for r in spec.report:
            assert not matches_folder(d, r)
            assert not matches_folder(r, d)
    joined_dev = " ".join(spec.dev)
    joined_report = " ".join(spec.report)
    assert "stylegan" in joined_dev
    assert "biggan" in joined_dev
    assert "cyclegan" in joined_dev
    assert "san" in joined_dev
    assert "deepfake" in joined_dev
    assert "seeingdark" in joined_report
    assert "stylegan2" in joined_report
    # Assert crn and imle are in the same fold
    crn_dev = any("crn" in f for f in spec.dev)
    imle_dev = any("imle" in f for f in spec.dev)
    assert crn_dev == imle_dev



def test_stylegan_prefix_does_not_match_stylegan2():
    assert matches_folder("stylegan/car", "stylegan")
    assert not matches_folder("stylegan2/car", "stylegan")
    assert matches_folder("stylegan2/car", "stylegan2")


def test_bootstrap_resamples_within_folder():
    labels = [0, 1, 0, 1]
    scores = [0.1, 0.9, 0.2, 0.8]
    folders = ["a", "a", "b", "b"]
    ci = bootstrap_ci(labels, scores, folders, n_resamples=50, seed=0)
    assert "roc_auc" in ci
    assert ci["accuracy"]["low"] <= ci["accuracy"]["high"]


def test_threshold_search_log_grid():
    labels = [0, 0, 1, 1]
    scores = [1e-6, 1e-4, 0.8, 0.9]
    thr, bacc = best_balanced_accuracy_threshold(labels, scores)
    assert 1e-9 <= thr <= 1.0
    assert bacc >= 0.5


def test_ece_finite():
    ece = expected_calibration_error([0, 1, 0, 1], [0.1, 0.9, 0.4, 0.6])
    assert 0.0 <= ece["expected_calibration_error"] <= 1.0


def test_cache_meta_refuses_mismatch(tmp_path: Path):
    write_cache_meta(tmp_path, {"semantic_model": "vit_small_patch16_224", "preprocess": "resize"})
    check_cache_meta(tmp_path, {"semantic_model": "vit_small_patch16_224"}, legacy=True)
    with pytest.raises(RuntimeError):
        check_cache_meta(
            tmp_path,
            {"semantic_model": "vit_small_patch16_224", "preprocess": "crop"},
            legacy=False,
        )


def test_dims_derived_from_config_not_1695():
    cfg = {
        "branches": "deep,physics,semantic",
        "semantic_model": "vit_small_patch16_224",
        "semantic_dim": 384,
        "physics_dim": 22,
        "deep_dim": 1280,
    }
    dims = branch_dims(cfg)
    assert dims["semantic"] == 384
    assert fused_dim(cfg) == 1280 + 22 + 384
    clip = dict(cfg)
    clip["semantic_model"] = "vit_large_patch14_clip_224.openai"
    clip["semantic_dim"] = 768
    clip["semantic_layers"] = "12,16,20,24"
    # explicit semantic_dim wins unless layers multiply; spec uses semantic_dim if set
    parse_branches("deep,physics,semantic,freq")
    assert "prnu" not in parse_branches("deep,physics,semantic,freq")
