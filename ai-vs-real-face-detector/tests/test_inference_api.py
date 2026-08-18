import torch

from src.api.schemas import PredictionResponse
import src.inference as inference


def test_prediction_schema_includes_physics_features():
    response = PredictionResponse(
        label="real",
        confidence_score=99.0,
        probability_distribution={"real": 99.0, "ai": 1.0},
        physics_features={"face_detected": 1.0},
    )
    assert response.physics_features == {"face_detected": 1.0}


def test_get_model_caches_checkpoint_load(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"placeholder")
    calls = []
    sentinel = object()

    def fake_load(path, device):
        calls.append((path, device))
        return sentinel, "hybrid"

    monkeypatch.setattr(inference, "load_model", fake_load)
    inference._MODEL_CACHE.clear()
    try:
        first = inference.get_model(str(checkpoint), torch.device("cpu"))
        second = inference.get_model(str(checkpoint), torch.device("cpu"))
        assert first == second == (sentinel, "hybrid")
        assert len(calls) == 1
    finally:
        inference._MODEL_CACHE.clear()
