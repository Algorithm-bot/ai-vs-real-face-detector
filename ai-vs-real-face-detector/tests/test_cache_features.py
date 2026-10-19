import json
import shutil
import numpy as np
import torch

from src import cache_features
from src.train import FaceBinaryDataset
from src.physics_branch.feature_vector import PHYSICS_FEATURE_DIM
from src.prnu_branch.extractor import PRNU_FEATURE_DIM
from src.freq_branch.extractor import FREQ_FEATURE_DIM


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


def test_face_dataset_attaches_feature_cache_from_argument(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    for split in ("train", "val", "test"):
        for label in ("real", "fake"):
            label_root = data_root / split / label
            label_root.mkdir(parents=True)
            (label_root / "image.jpg").write_bytes(b"image")

    attached = []
    monkeypatch.setattr(
        FaceBinaryDataset,
        "attach_feature_cache",
        lambda self, cache_dir, split: attached.append((cache_dir, split)),
    )
    cache_dir = tmp_path / "cache"
    dataset = FaceBinaryDataset(
        str(data_root),
        split="train",
        use_semantic=True,
        feature_cache_dir=str(cache_dir),
    )

    assert len(dataset.samples) == 2
    assert attached == [(str(cache_dir), "train")]


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


def test_exclude_ffhq_from_splits(tmp_path, monkeypatch):
    # Create synthetic dataset with ffhq folders in train, val, test
    for split in ("train", "val", "test"):
        split_dir = tmp_path / split
        for label in ("real", "fake"):
            label_dir = split_dir / label
            # Create an ffhq subdirectory
            ffhq_dir = label_dir / "ffhq"
            ffhq_dir.mkdir(parents=True)
            # Create a dummy image inside ffhq
            (ffhq_dir / "img.jpg").write_bytes(b"fake")
            # Also create a non-ffhq image directly under label dir to ensure we have some data
            (label_dir / "other.jpg").write_bytes(b"real")

    # Import and set EXCLUDE_SOURCES
    from src.train import EXCLUDE_SOURCES, collect_labeled_images
    # Save original to restore after test? We'll use monkeypatch to set it.
    monkeypatch.setattr("src.train.EXCLUDE_SOURCES", ("ffhq",))

    # Check each split
    for split in ("train", "val", "test"):
        split_root = tmp_path / split
        samples = collect_labeled_images(split_root)
        # Ensure no sample has source equal to "ffhq"
        for path, label, source in samples:
            assert source != "ffhq", f"Found ffhq in source: {source} for path {path}"
        # Also ensure we have some samples (the non-ffhq ones)
        assert len(samples) > 0, f"No samples found in split {split}"


def test_process_split_creates_shard_dir_if_missing_after_compute(tmp_path, monkeypatch):
    """Test that shard directory is created if missing after compute step."""
    # Setup
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    split = "train"
    # Create a few samples
    samples = [("img1.jpg", 0, "source1"), ("img2.jpg", 1, "source2")]
    # Create a mock args object
    class Args:
        batch_size = 2
        workers = 0
        shard_size = 2  # so we have one shard
        seed = 42
        augment = False
        preprocess = "crop"
    args = Args()

    # Mock encoder
    class DummyEncoder:
        embedding_dim = 4  # must match the dummy extractor output? We'll adjust
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])

        def eval(self):
            return self

        def __call__(self, x):
            # Return dummy embeddings of shape (batch_size, embedding_dim)
            return torch.ones((x.shape[0], self.embedding_dim), dtype=torch.float32)

    # Mock FeatureDataset to return dummy tensors
    class MockFeatureDataset:
        def __init__(self, samples, preprocess_mode="crop", crop_size=224, enable_augment=False, seed=42):
            self.samples = samples
            self.preprocess_mode = preprocess_mode
            self.crop_size = crop_size
            self.enable_augment = enable_augment
            self.seed = seed

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            # Return dummy tensors: physics, prnu, freq, sem, ok
            return (
                torch.ones(PHYSICS_FEATURE_DIM, dtype=torch.float32),
                torch.ones(PRNU_FEATURE_DIM, dtype=torch.float32),
                torch.ones(FREQ_FEATURE_DIM, dtype=torch.float32),
                torch.ones(224, 224, 3, dtype=torch.uint8),
                torch.tensor(1, dtype=torch.int8),
            )

    # Now monkeypatch the dependencies
    monkeypatch.setattr(cache_features.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cache_features, "SemanticEncoder", lambda **kwargs: DummyEncoder())
    monkeypatch.setattr(cache_features, "FeatureDataset", MockFeatureDataset)

    # We also need to mock the DataLoader to avoid using multiple workers and to return our data in a single batch
    # We'll create a mock dataset that wraps the MockFeatureDataset? Actually, the DataLoader will use our mocked FeatureDataset.
    # We'll mock the DataLoader class to return an iterator that yields a single batch containing all the samples.
    class MockDataLoader:
        def __init__(self, dataset, batch_size, shuffle, num_workers, pin_memory, prefetch_factor):
            self.dataset = dataset
            self.batch_size = batch_size

        def __iter__(self):
            # Collect all items in the dataset
            batch = [self.dataset[i] for i in range(len(self.dataset))]
            # Now we need to collate the batch: we'll stack each component
            physics_batch = torch.stack([item[0] for item in batch])
            prnu_batch = torch.stack([item[1] for item in batch])
            freq_batch = torch.stack([item[2] for item in batch])
            sem_batch = torch.stack([item[3] for item in batch])
            ok_batch = torch.stack([item[4] for item in batch])
            # Yield a single batch
            yield (physics_batch, prnu_batch, freq_batch, sem_batch, ok_batch)

    monkeypatch.setattr(cache_features, "DataLoader", MockDataLoader)

    # Now we need to remove the shard directory after the compute loop and before the save.
    # We'll remove the shard directory before calling process_split and see if it still works.
    # (Our function has mkdir at the start and before each save, so it should recreate it.)

    shard_dir = cache_dir / f"{split}_shards"
    # Ensure the shard directory does not exist
    if shard_dir.exists():
        shutil.rmtree(shard_dir)

    # Call process_split
    cache_features.process_split(split, samples, DummyEncoder(), torch.device("cuda"), cache_dir, args)

    # Check that the shard file was created
    shard_file = shard_dir / "00000.npz"
    assert shard_file.exists(), "Shard file was not created"