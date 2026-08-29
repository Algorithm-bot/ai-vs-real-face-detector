"""End-to-end inference test with real checkpoint."""

import pytest

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sample_image():
    for sub in ("real", "fake"):
        folder = PROJECT_ROOT / "data" / sub
        if folder.exists():
            for path in folder.iterdir():
                if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                    return str(path)
    pytest.skip("No sample images in data/real or data/fake")


@pytest.fixture
def checkpoint():
    ckpt = PROJECT_ROOT / "models" / "hybrid_best.pt"
    if not ckpt.exists():
        pytest.skip("hybrid_best.pt not found")
    return str(ckpt)


def test_e2e_predict(sample_image, checkpoint):
    from src.inference import predict

    result = predict(
        sample_image,
        checkpoint,
        include_heatmap=False,
        include_full_explainability=False,
        align_face=False,
    )
    assert "label" in result
    assert "probability_distribution" in result
    assert result["probability_distribution"]["real"] >= 0
    assert result["model_mode"] == "hybrid"


def test_e2e_full_explainability(sample_image, checkpoint):
    from src.inference import predict

    result = predict(
        sample_image,
        checkpoint,
        include_heatmap=False,
        include_full_explainability=True,
        align_face=False,
    )
    assert "explanation" in result
    assert "prnu" in result
    assert result["prnu"]["reliability"] == "limited"
    assert "fusion_weights" in result
