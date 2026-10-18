"""Small contract tests for cache/train selection and cached-branch checkpoints."""

from argparse import Namespace
import random

import torch

from src.classifier.head import BranchOnlyClassifier
from src.inference import load_model
from src.train import subsample_balanced


def test_cached_file_list_and_uncached_enumeration_choose_identical_subset():
    # Both cache_features.py and FaceBinaryDataset preserve this ordered file list,
    # then call the same sampler with the same per-class cap and seed.
    samples = [(f"/data/{label}/{i}.png", label, "source") for label in (0, 1) for i in range(8)]
    uncached = subsample_balanced(list(samples), per_class=3, seed=42)
    cached_file_list = [tuple(row) for row in samples]
    cached = subsample_balanced(cached_file_list, per_class=3, seed=42)
    assert cached == uncached


def test_subset_seed_is_independent_of_training_seed():
    samples = [(f"/data/{label}/{i}.png", label, "source") for label in (0, 1) for i in range(12)]
    selections = {}
    for training_seed in (5, 117):
        random.seed(training_seed)
        selections[training_seed] = subsample_balanced(list(samples), per_class=4, seed=42)
    assert selections[5] == selections[117]


def _checkpoint(path, mode, dim):
    model = BranchOnlyClassifier(input_dim=dim, normalize=mode in {"physics_only", "freq_only"})
    torch.save(
        {"mode": mode, "args": {"semantic_dim": dim, "freq_dim": dim}, "model_state_dict": model.state_dict()},
        path,
    )


def test_load_model_accepts_cached_branch_checkpoint_types(tmp_path):
    for mode, dim in (("physics_only", 22), ("semantic_only", 8), ("freq_only", 64)):
        path = tmp_path / f"{mode}.pt"
        _checkpoint(path, mode, dim)
        model, loaded_mode = load_model(str(path), torch.device("cpu"))
        assert loaded_mode == mode
        assert model is not None
