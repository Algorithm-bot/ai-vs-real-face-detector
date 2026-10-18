"""Prediction rows emitted by evaluation remain consumable by stack/diagnostics."""

import pandas as pd

from src.diagnose_branches import run_diagnostics_from_predictions
from src.evaluate import _tables_by_eval_split
from src.eval.forward import predictions_to_rows
from src.eval.splits import load_split_spec
from src.stack_fusion import run_stacking_evaluation


def test_evaluate_rows_feed_stacking_and_diagnostics(tmp_path):
    spec = load_split_spec()
    dev, report = spec.dev[0], spec.report[0]
    labels = [i % 2 for i in range(20)]
    folders = [dev] * 10 + [report] * 10
    collected = {
        "path": [f"image-{i}" for i in range(20)], "source": folders,
        "folder": folders, "label": labels,
        "p_ai": [0.1 if label == 0 else 0.9 for label in labels],
        "logit_real": [0.0] * 20, "logit_ai": [0.0] * 20,
    }
    rows = predictions_to_rows(collected, split="all-test", mode="semantic_only")
    rows = [
        {
            **row,
            "p_ai__stage1": row["p_ai"],
            "p_ai__semantic_only": row["p_ai"],
            "p_ai__physics_only": row["p_ai"],
            "p_ai__freq_only": row["p_ai"],
            "p_ai__full_hybrid": row["p_ai"],
            "logit_ai__semantic_only": row["logit_ai"],
        }
        for row in rows
    ]
    path = tmp_path / "predictions.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    result = run_stacking_evaluation(pd.read_csv(path), n_bootstrap=2, seed=1)
    assert not result["comparison_table"].empty
    assert "Gated hybrid" in result["comparison_table"]["Model"].values
    assert {"Acc @ 0.5 CI", "Acc @ Calibrated CI"} <= set(result["comparison_table"].columns)
    diagnostics = run_diagnostics_from_predictions(path)
    assert "stage1_auc" in diagnostics.columns
    assert "OVERALL_REPORT" in diagnostics["generator"].values


def test_evaluation_tables_are_grouped_by_named_split():
    spec = load_split_spec()
    sources = [spec.dev[0], spec.report[0], spec.progan[0]]
    rows = [
        {
            "path": f"sample-{index}",
            "folder": source,
            "source": source,
            "label": index % 2,
            "p_ai": 0.1 if index % 2 == 0 else 0.9,
        }
        for index, source in enumerate(sources * 4)
    ]
    tables = _tables_by_eval_split(rows, spec, 0.5, n_bootstrap=2, seed=1)
    assert set(tables) == {"fold_a", "fold_b", "progan"}
    for split in ("fold_a", "fold_b", "progan"):
        assert tables[split]["at_0.5"]["overall"]["samples"] > 0
        assert "roc_auc" in tables[split]["at_0.5"]["overall"]
        assert "ci" in tables[split]["at_0.5"]["overall"]
        assert "ci" in tables[split]["calibrated"]["overall"]
