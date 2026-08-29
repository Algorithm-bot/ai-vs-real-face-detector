from pathlib import Path

from src.evaluate import metric_summary, write_leakage_report


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
