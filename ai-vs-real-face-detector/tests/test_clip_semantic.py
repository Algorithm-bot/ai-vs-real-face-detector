"""Test CLIP semantic branch (random weights / CPU).

Verifies:
1. Native timm ViT-S/16 (384-d) vs CLIP ViT-L/14 (768-d final embedding)
2. Configurable intermediate layer CLS concatenation (e.g. layers 12, 16, 20, 24 -> 4096-d)
3. Dynamic dimension calculation in spec.py
4. Linear probe single-layer architecture
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.classifier.head import BranchOnlyClassifier
from src.features.spec import semantic_dim_from_config
from src.semantic_branch.encoder import SemanticEncoder, parse_layers_list


def test_vit_small_semantic_encoder():
    enc = SemanticEncoder(
        model_name="vit_small_patch16_224",
        pretrained=False,
        device=torch.device("cpu"),
    )
    assert enc.embedding_dim == 384
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    res = enc.extract(img)
    assert res.features.shape == (384,)
    assert enc.embedding_dim == semantic_dim_from_config({"semantic_model": "vit_small_patch16_224"})


def test_clip_vit_l14_default_final_embedding():
    enc = SemanticEncoder(
        model_name="vit_large_patch14_clip_224.openai",
        pretrained=False,
        device=torch.device("cpu"),
    )
    assert enc.embedding_dim == 768
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    res = enc.extract(img)
    assert res.features.shape == (768,)
    assert enc.embedding_dim == semantic_dim_from_config({"semantic_model": "vit_large_patch14_clip_224"})


def test_clip_intermediate_layers_concatenation():
    layers = [12, 16, 20, 24]
    enc = SemanticEncoder(
        model_name="vit_large_patch14_clip_224.openai",
        pretrained=False,
        device=torch.device("cpu"),
        intermediate_layers=layers,
    )
    # 4 layers * 1024-d per block CLS token = 4096-d
    assert enc.embedding_dim == 4096
    derived = semantic_dim_from_config({
        "semantic_model": "vit_large_patch14_clip_224",
        "semantic_layers": "12,16,20,24",
    })
    assert derived == 4096

    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    res = enc.extract(img)
    assert res.features.shape == (4096,)


def test_linear_probe_classifier():
    # Linear probe has exactly 1 linear layer: (input_dim -> 2), no hidden layers
    model = BranchOnlyClassifier(input_dim=768, linear_probe=True)
    assert isinstance(model.classifier, torch.nn.Linear)
    assert model.classifier.in_features == 768
    assert model.classifier.out_features == 2

    x = torch.randn(4, 768)
    logits, _ = model(x)
    assert logits.shape == (4, 2)
