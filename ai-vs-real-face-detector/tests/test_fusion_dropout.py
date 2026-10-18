"""
Unit tests for branch dropout, stacked logistic regression, and multi-seed fusion comparison.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.fusion.fuse import FeatureFusion, FusionMode, GatedFusion
from src.stack_fusion import fit_stacker_on_dev, multi_seed_aggregation, run_stacking_evaluation


def test_branch_dropout_training_vs_eval():
    modality_dims = {
        "deep": 1280,
        "physics": 22,
        "semantic": 384,
        "freq": 64,
    }
    gated = GatedFusion(modality_dims=modality_dims, branch_dropout=0.5)

    bs = 100
    mods = {
        "deep": torch.ones(bs, 1280),
        "physics": torch.ones(bs, 22),
        "semantic": torch.ones(bs, 384),
        "freq": torch.ones(bs, 64),
    }

    # Test eval mode: no dropout should ever occur
    gated.eval()
    out_eval = gated.apply_branch_dropout(mods)
    for k in mods:
        assert torch.all(out_eval[k] == mods[k]), f"Eval mode modified {k}"

    # Test training mode: deep must NEVER be dropped
    gated.train()
    out_train = gated.apply_branch_dropout(mods)
    assert torch.all(out_train["deep"] == mods["deep"]), "Deep branch was modified during dropout"

    # Auxiliary branches should have some zeroed entries
    phys_zeros = (out_train["physics"].sum(dim=1) == 0).float().mean().item()
    sem_zeros = (out_train["semantic"].sum(dim=1) == 0).float().mean().item()
    freq_zeros = (out_train["freq"].sum(dim=1) == 0).float().mean().item()

    assert 0.2 < phys_zeros < 0.8, f"Unexpected physics drop rate: {phys_zeros}"
    assert 0.2 < sem_zeros < 0.8, f"Unexpected semantic drop rate: {sem_zeros}"
    assert 0.2 < freq_zeros < 0.8, f"Unexpected freq drop rate: {freq_zeros}"


def test_branch_dropout_zero_rate():
    # If branch_dropout=0.0, no dropout should occur even in training mode
    modality_dims = {"deep": 1280, "physics": 22}
    gated = GatedFusion(modality_dims=modality_dims, branch_dropout=0.0)
    gated.train()
    mods = {"deep": torch.ones(10, 1280), "physics": torch.ones(10, 22)}
    out = gated.apply_branch_dropout(mods)
    assert torch.all(out["deep"] == mods["deep"])
    assert torch.all(out["physics"] == mods["physics"])


def test_fit_stacker_on_dev():
    n_samples = 100
    X = np.random.rand(n_samples, 4)
    # y depends strongly on first two features
    y = ((X[:, 0] + X[:, 1]) > 1.0).astype(int)
    groups = [f"gen_{i % 5}" for i in range(n_samples)]

    clf, info = fit_stacker_on_dev(X, y, groups=groups, cv_folds=3, seed=42)
    assert hasattr(clf, "coef_")
    assert clf.coef_.shape == (1, 4)
    assert "best_C" in info


def test_run_stacking_evaluation_synthetic(tmp_path: Path):
    n_samples = 120
    # 60 DEV (stylegan/car), 60 REPORT (stylegan2/car)
    sources = ["stylegan/car"] * 60 + ["stylegan2/car"] * 60
    labels = np.array([0, 1] * 60)

    # Synthetic probabilities
    rng = np.random.RandomState(42)
    p_deep = np.clip(labels * 0.7 + rng.uniform(0, 0.3, n_samples), 0.01, 0.99)
    p_phys = np.clip(labels * 0.6 + rng.uniform(0, 0.4, n_samples), 0.01, 0.99)
    p_sem = np.clip(labels * 0.8 + rng.uniform(0, 0.2, n_samples), 0.01, 0.99)
    p_freq = np.clip(labels * 0.65 + rng.uniform(0, 0.35, n_samples), 0.01, 0.99)
    p_fused = (p_deep + p_sem) / 2.0

    df = pd.DataFrame({
        "path": [f"img_{i}.jpg" for i in range(n_samples)],
        "source": sources,
        "folder": sources,
        "label": labels,
        "p_ai_deep": p_deep,
        "p_ai_physics": p_phys,
        "p_ai_semantic": p_sem,
        "p_ai_freq": p_freq,
        "p_ai_fused": p_fused,
    })

    res = run_stacking_evaluation(df, n_bootstrap=50, seed=42)
    assert "branch_weights" in res
    assert "comparison_table" in res
    comp = res["comparison_table"]
    models = comp["Model"].tolist()
    assert "Deep alone" in models
    assert "Mean probability fusion" in models
    assert "Stacked logistic regression" in models
    assert "Gated hybrid" in models


def test_multi_seed_aggregation(tmp_path: Path):
    n_samples = 80
    sources = ["stylegan/car"] * 40 + ["stylegan2/car"] * 40
    labels = np.array([0, 1] * 40)

    results_per_seed = []
    for s in range(3):
        rng = np.random.RandomState(s)
        p_deep = np.clip(labels * 0.7 + rng.uniform(0, 0.3, n_samples), 0.01, 0.99)
        p_phys = np.clip(labels * 0.6 + rng.uniform(0, 0.4, n_samples), 0.01, 0.99)
        p_sem = np.clip(labels * 0.8 + rng.uniform(0, 0.2, n_samples), 0.01, 0.99)
        p_freq = np.clip(labels * 0.65 + rng.uniform(0, 0.35, n_samples), 0.01, 0.99)

        df = pd.DataFrame({
            "path": [f"img_{i}.jpg" for i in range(n_samples)],
            "source": sources,
            "folder": sources,
            "label": labels,
            "p_ai_deep": p_deep,
            "p_ai_physics": p_phys,
            "p_ai_semantic": p_sem,
            "p_ai_freq": p_freq,
        })
        res = run_stacking_evaluation(df, n_bootstrap=20, seed=s)
        results_per_seed.append(res)

    agg = multi_seed_aggregation(results_per_seed)
    assert len(agg) == 3
    assert "AUC (3 seeds)" in agg.columns
    assert "+/-" in agg.iloc[0]["AUC (3 seeds)"]
