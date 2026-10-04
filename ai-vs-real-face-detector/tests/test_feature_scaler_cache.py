import pickle

import numpy as np

from src.features import feature_scalers
from src.physics_branch.normalization import PhysicsNormalizer


def _normalizer(value):
    return PhysicsNormalizer(
        mean=np.array([value], dtype=np.float32),
        std=np.array([1.0], dtype=np.float32),
        feature_names=["feature"],
    )


def test_scaler_cache_round_trip_and_configuration_matching(tmp_path):
    cache_path = tmp_path / "physics_prnu_scalers.pkl"
    physics = _normalizer(1.0)
    prnu = _normalizer(2.0)

    feature_scalers.save_physics_and_prnu_scaler_cache(
        cache_path, physics, prnu, sample_count=10, seed=42
    )

    with cache_path.open("rb") as cache_file:
        payload = pickle.load(cache_file)
    assert payload["cache_version"] == feature_scalers.SCALER_CACHE_VERSION
    assert payload["sample_count"] == 10
    assert payload["seed"] == 42
    assert "physics_scaler" in payload
    assert "prnu_scaler" in payload
    assert not cache_path.with_suffix(".pkl.tmp").exists()

    loaded = feature_scalers.load_physics_and_prnu_scaler_cache(cache_path, 10, 42)
    assert loaded is not None
    assert np.array_equal(loaded[0].mean, physics.mean)
    assert np.array_equal(loaded[1].mean, prnu.mean)
    assert feature_scalers.load_physics_and_prnu_scaler_cache(cache_path, 20, 42) is None
    assert feature_scalers.load_physics_and_prnu_scaler_cache(cache_path, 10, 7) is None


def test_corrupt_scaler_cache_is_ignored(tmp_path):
    cache_path = tmp_path / "physics_prnu_scalers.pkl"
    cache_path.write_bytes(b"not a pickle")

    assert feature_scalers.load_physics_and_prnu_scaler_cache(cache_path, 10, 42) is None


def test_zero_max_samples_fits_all_paths(monkeypatch):
    processed = []

    class FakePhysicsExtractor:
        def extract(self, rgb):
            processed.append(int(rgb[0]))
            return type("Result", (), {"vector": np.array([rgb[0]], dtype=np.float32)})()

        def close(self):
            pass

    class FakePRNUExtractor:
        def extract(self, rgb):
            return type("Result", (), {"vector": np.array([rgb[0]], dtype=np.float32)})()

    monkeypatch.setattr(feature_scalers, "PhysicsFeatureExtractor", FakePhysicsExtractor)
    monkeypatch.setattr(feature_scalers, "PRNUExtractor", FakePRNUExtractor)
    monkeypatch.setattr(feature_scalers, "_load_rgb", lambda path: np.array([int(path)]))

    physics, prnu = feature_scalers.fit_physics_and_prnu_scalers(
        ["1", "2", "3"], max_samples=0, seed=42
    )

    assert processed == [1, 2, 3]
    assert physics.mean[0] == 2.0
    assert prnu.mean[0] == 2.0
