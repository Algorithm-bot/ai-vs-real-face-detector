from pathlib import Path

from src.evaluate import _dataset_for_evaluation, metric_summary, write_leakage_report
from src.train import collect_labeled_images


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
