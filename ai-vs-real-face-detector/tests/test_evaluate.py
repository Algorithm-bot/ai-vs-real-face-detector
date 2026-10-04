from pathlib import Path

from src.evaluate import _dataset_for_evaluation, metric_summary, write_leakage_report
from src.train import (
    FaceBinaryDataset,
    _has_explicit_split_layout,
    _split_has_labels,
    collect_labeled_images,
)


def test_metric_summary_binary_counts_and_scores():
    summary = metric_summary([0, 0, 1, 1], [0.1, 0.9, 0.2, 0.8])
    assert summary["tn"] == 1
    assert summary["fp"] == 1
    assert summary["fn"] == 1
    assert summary["tp"] == 1
    assert summary["accuracy"] == 0.5
    assert summary["roc_auc"] == 0.5


def test_leakage_report_detects_identical_files(tmp_path: Path):
    train = tmp_path / "train.jpg"
    test = tmp_path / "test.jpg"
    train.write_bytes(b"same image bytes")
    test.write_bytes(b"same image bytes")
    report = tmp_path / "report.txt"
    write_leakage_report([str(train)], [str(test)], report)
    assert "Overlapping test images: 1" in report.read_text(encoding="utf-8")


def test_collect_cnndetection_layout(tmp_path: Path):
    real = tmp_path / "airplane" / "0_real"
    fake = tmp_path / "airplane" / "1_fake"
    real.mkdir(parents=True)
    fake.mkdir(parents=True)
    (real / "a.jpg").write_bytes(b"real")
    (fake / "b.png").write_bytes(b"fake")
    samples = collect_labeled_images(tmp_path)
    labels = {Path(path).name: label for path, label, _ in samples}
    assert labels["a.jpg"] == 0
    assert labels["b.png"] == 1


def test_explicit_layout_detection_uses_direct_directory_checks(tmp_path: Path, monkeypatch):
    import os

    for split in ("train", "val", "test"):
        for label in ("real", "fake"):
            (tmp_path / split / label).mkdir(parents=True)

    def unexpected_rglob(*args, **kwargs):
        raise AssertionError("Explicit layout detection must not recurse")

    def unexpected_walk(*args, **kwargs):
        raise AssertionError("Explicit layout detection must not walk the tree")

    monkeypatch.setattr(Path, "rglob", unexpected_rglob)
    monkeypatch.setattr(os, "walk", unexpected_walk)
    assert _has_explicit_split_layout(tmp_path)


def test_collect_explicit_layout_recurses_only_within_label_roots(
    tmp_path: Path, monkeypatch
):
    real = tmp_path / "real"
    fake = tmp_path / "fake"
    (real / "source_a" / "nested").mkdir(parents=True)
    fake.mkdir()
    (real / "top.jpg").write_bytes(b"real")
    (real / "source_a" / "nested" / "nested.png").write_bytes(b"real nested")
    (fake / "top.webp").write_bytes(b"fake")

    def unexpected_rglob(*args, **kwargs):
        raise AssertionError("Explicit collection must not use Path.rglob")

    monkeypatch.setattr(Path, "rglob", unexpected_rglob)
    samples = collect_labeled_images(tmp_path)

    by_name = {Path(path).name: (label, source) for path, label, source in samples}
    assert by_name == {
        "top.jpg": (0, "lsun"),
        "nested.png": (0, "source_a/nested"),
        "top.webp": (1, "synthetic"),
    }


def test_split_has_labels_direct_children_do_not_rglob(tmp_path: Path, monkeypatch):
    import os

    (tmp_path / "train" / "real").mkdir(parents=True)
    (tmp_path / "train" / "fake").mkdir(parents=True)

    def unexpected_rglob(*args, **kwargs):
        raise AssertionError("Direct label dirs must not recurse")

    def unexpected_walk(*args, **kwargs):
        raise AssertionError("Direct label dirs must not walk the tree")

    monkeypatch.setattr(Path, "rglob", unexpected_rglob)
    monkeypatch.setattr(os, "walk", unexpected_walk)
    assert _split_has_labels(tmp_path, "train")


def test_collect_explicit_layout_ignores_sibling_directories(
    tmp_path: Path, monkeypatch
):
    import os

    real = tmp_path / "real"
    fake = tmp_path / "fake"
    other = tmp_path / "other"
    real.mkdir()
    fake.mkdir()
    other.mkdir()
    (real / "a.jpg").write_bytes(b"real")
    (fake / "b.jpg").write_bytes(b"fake")
    (other / "c.jpg").write_bytes(b"skip")

    walked = []
    original_walk = os.walk

    def tracking_walk(path, *args, **kwargs):
        walked.append(Path(path).name)
        return original_walk(path, *args, **kwargs)

    monkeypatch.setattr(os, "walk", tracking_walk)
    samples = collect_labeled_images(tmp_path)
    names = {Path(path).name for path, _, _ in samples}
    assert names == {"a.jpg", "b.jpg"}
    assert set(walked) == {"real", "fake"}


def test_face_binary_dataset_explicit_layout_does_not_rglob(
    tmp_path: Path, monkeypatch
):
    import os

    for split in ("train", "val", "test"):
        for label, name in (("real", "r.jpg"), ("fake", "f.jpg")):
            folder = tmp_path / split / label
            folder.mkdir(parents=True)
            (folder / name).write_bytes(b"img")
        decoy = tmp_path / split / "other"
        decoy.mkdir()
        (decoy / "skip.jpg").write_bytes(b"skip")

    def unexpected_rglob(*args, **kwargs):
        raise AssertionError("Explicit FaceBinaryDataset must not recurse")

    walked = []
    original_walk = os.walk

    def tracking_walk(path, *args, **kwargs):
        walked.append(Path(path).name)
        assert Path(path).name in {"real", "fake"}
        return original_walk(path, *args, **kwargs)

    monkeypatch.setattr(Path, "rglob", unexpected_rglob)
    monkeypatch.setattr(os, "walk", tracking_walk)
    ds = FaceBinaryDataset(
        str(tmp_path),
        split="train",
        seed=42,
        use_physics=False,
    )
    assert len(ds.samples) == 2
    labels = {label for _, label, _ in ds.samples}
    assert labels == {0, 1}
    assert set(walked) == {"real", "fake"}


def test_cnndetection_nested_layout_still_collected(tmp_path: Path):
    real = tmp_path / "car" / "0_real" / "nested"
    fake = tmp_path / "car" / "1_fake"
    real.mkdir(parents=True)
    fake.mkdir(parents=True)
    (real / "a.jpg").write_bytes(b"real")
    (fake / "b.jpg").write_bytes(b"fake")
    samples = collect_labeled_images(tmp_path)
    by_name = {Path(path).name: (label, source) for path, label, source in samples}
    assert by_name["a.jpg"][0] == 0
    assert by_name["b.jpg"] == (1, "car")
    assert by_name["a.jpg"][1] == "nested"


def test_evaluation_uses_explicit_train_val_test(tmp_path: Path):
    for split in ("train", "val", "test"):
        for label, name in (("real", "r.jpg"), ("fake", "f.jpg")):
            folder = tmp_path / split / label / "lsun"
            folder.mkdir(parents=True)
            (folder / name).write_bytes(b"img")
    ds, train_paths, split_name = _dataset_for_evaluation(
        tmp_path,
        {"seed": 42, "no_semantic_pretrained": True},
        "stage1",
    )
    assert split_name == "explicit train/val/test split"
    assert len(ds.samples) == 2
    assert len(train_paths) == 2
