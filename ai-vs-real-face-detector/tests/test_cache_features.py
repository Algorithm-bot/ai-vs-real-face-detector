import json

from src import cache_features
from src.train import FaceBinaryDataset


def test_main_reuses_cached_file_list(tmp_path, monkeypatch):
    cached_samples = [("image.jpg", 1, "synthetic")]
    collected_samples = []
    processed_samples = []

    class DummyEncoder:
        embedding_dim = 3

        def eval(self):
            return self

    monkeypatch.setattr(cache_features.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cache_features, "SemanticEncoder", lambda **kwargs: DummyEncoder())
    monkeypatch.setattr(
        cache_features,
        "collect_labeled_images",
        lambda _split_root: collected_samples.append(cached_samples) or cached_samples,
    )
    monkeypatch.setattr(
        cache_features,
        "process_split",
        lambda _split, samples, *_args: processed_samples.append(samples),
    )

    args = [
        "cache_features",
        "--data-dir",
        str(tmp_path / "data"),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--splits",
        "train",
    ]
    monkeypatch.setattr("sys.argv", args)
    cache_features.main()

    file_list_path = tmp_path / "cache" / "train_file_list.json"
    assert json.loads(file_list_path.read_text(encoding="utf-8")) == [list(cached_samples[0])]

    monkeypatch.setattr("sys.argv", args)
    cache_features.main()

    assert len(collected_samples) == 1
    assert processed_samples == [cached_samples, cached_samples]


def test_face_dataset_reuses_cached_file_list(tmp_path, monkeypatch, capsys):
    data_root = tmp_path / "data"
    for split in ("train", "val", "test"):
        for label in ("real", "fake"):
            (data_root / split / label).mkdir(parents=True)

    cached_samples = [
        (str(data_root / "train" / "real" / "cached.jpg"), 0, "lsun"),
        (str(data_root / "train" / "fake" / "cached.jpg"), 1, "synthetic"),
    ]
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "train_file_list.json").write_text(
        json.dumps(cached_samples),
        encoding="utf-8",
    )

    def unexpected_collect(_split_root):
        raise AssertionError("Cached file list should avoid live enumeration")

    monkeypatch.setattr("src.train.collect_labeled_images", unexpected_collect)
    dataset = FaceBinaryDataset(
        str(data_root),
        split="train",
        use_physics=False,
        feature_cache_dir=str(cache_dir),
    )

    assert dataset.samples == cached_samples
    assert "Loading cached file list for train..." in capsys.readouterr().out


def test_collect_labeled_images_reports_enumeration_progress(tmp_path, capsys):
    split_root = tmp_path / "train"
    real_root = split_root / "real"
    (split_root / "fake").mkdir(parents=True)
    for index in range(100):
        (real_root / f"source-{index:03d}").mkdir(parents=True)
    image_path = real_root / "source-099" / "image.jpg"
    image_path.write_bytes(b"")

    samples = cache_features.collect_labeled_images(split_root)

    assert samples == [(str(image_path), 0, "source-099")]
    assert "processed 100 directories" in capsys.readouterr().out
