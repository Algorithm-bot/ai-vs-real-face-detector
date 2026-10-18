"""
Stacking baseline and fusion comparison script.

Fits a cross-validated logistic regression stacker on DEV folders (FOLD_A),
evaluates on REPORT folders (FOLD_B), and compares:
1. Deep alone
2. Simple average of branch probabilities
3. Stacked logistic regression
4. Gated hybrid with branch dropout
5. Gated hybrid without branch dropout

Computes AUC, AP, acc@0.5, and acc@calibrated with bootstrap 95% CIs.
Supports multi-seed aggregation across 3 seeds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.metrics import (
    best_balanced_accuracy_threshold,
    bootstrap_ci,
    metric_summary,
)
from src.eval.splits import DEFAULT_SPLITS_PATH, folder_of, load_split_spec


def extract_branch_feature_matrix(
    df: pd.DataFrame,
    branch_cols: List[str],
    use_logits: bool = False,
) -> np.ndarray:
    """Extract probability or logit feature matrix for the branches."""
    X = df[branch_cols].values.astype(np.float64)
    if use_logits:
        eps = 1e-6
        X = np.clip(X, eps, 1.0 - eps)
        X = np.log(X / (1.0 - X))
    return np.nan_to_num(X, nan=0.5, posinf=10.0, neginf=-10.0)


def fit_stacker_on_dev(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    groups: Optional[List[str]] = None,
    cv_folds: int = 5,
    seed: int = 42,
) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """
    Fit L2-regularized logistic regression stacker on DEV.
    Uses GroupKFold by generator folder if enough groups exist, else StratifiedKFold.
    """
    n_groups = len(set(groups)) if groups is not None else 0
    if groups is not None and n_groups >= cv_folds:
        cv = GroupKFold(n_splits=cv_folds)
        cv_split = list(cv.split(X_dev, y_dev, groups=groups))
    else:
        cv = StratifiedKFold(n_splits=min(cv_folds, 5), shuffle=True, random_state=seed)
        cv_split = list(cv.split(X_dev, y_dev))

    # Search regularization parameter C across log scale
    clf = LogisticRegressionCV(
        Cs=np.logspace(-3, 3, 10),
        cv=cv_split,
        scoring="roc_auc",
        random_state=seed,
        max_iter=1000,
    )
    clf.fit(X_dev, y_dev)
    best_c = float(clf.C_[0]) if hasattr(clf, "C_") else 1.0

    return clf, {"best_C": best_c, "cv_scores": [float(s) for s in clf.scores_[1].mean(axis=0)]}


def evaluate_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    folders: List[str],
    calibrated_threshold: float = 0.5,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute summary metrics and bootstrap 95% CIs."""
    summary_05 = metric_summary(y_true, y_score, threshold=0.5)
    summary_cal = metric_summary(y_true, y_score, threshold=calibrated_threshold)

    ci = bootstrap_ci(
        y_true,
        y_score,
        folders=folders,
        threshold=calibrated_threshold,
        n_resamples=n_bootstrap,
        seed=seed,
    )
    ci_05 = bootstrap_ci(
        y_true,
        y_score,
        folders=folders,
        threshold=0.5,
        n_resamples=n_bootstrap,
        seed=seed,
    )

    return {
        "auc": summary_05["roc_auc"],
        "ap": summary_05["average_precision"],
        "acc_at_0.5": summary_05["accuracy"],
        "acc_at_calibrated": summary_cal["accuracy"],
        "calibrated_threshold": calibrated_threshold,
        "ci": ci,
        "ci_at_0.5": ci_05,
    }


def run_stacking_evaluation(
    pred_df: pd.DataFrame,
    splits_path: Optional[Path] = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
    extra_models: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Run full stacking evaluation and comparison on a predictions DataFrame."""
    spec = load_split_spec(splits_path)

    sources = pred_df["folder"].astype(str).tolist()
    labels = pred_df["label"].astype(int).values

    assignments = [spec.assign(s) for s in sources]
    dev_mask = np.array([a == "dev" for a in assignments])
    report_mask = np.array([a == "report" for a in assignments])

    if not np.any(dev_mask):
        raise ValueError("No DEV samples found in predictions.")
    if not np.any(report_mask):
        raise ValueError("No REPORT samples found in predictions.")

    # The notebook stores one probability per evaluated mode.
    mode_cols = sorted(
        c for c in pred_df.columns
        if c.startswith("p_ai__") and c != "p_ai__full_hybrid"
    )
    legacy_cols = [
        c for c in ["p_ai_deep", "p_ai_physics", "p_ai_semantic", "p_ai_freq", "p_ai_prnu"]
        if c in pred_df.columns
    ]
    branch_cols = mode_cols or legacy_cols
    if not branch_cols:
        raise ValueError(f"No branch probability columns found in: {list(pred_df.columns)}")
    deep_col = (
        "p_ai__stage1" if "p_ai__stage1" in pred_df
        else "p_ai_deep" if "p_ai_deep" in pred_df
        else branch_cols[0]
    )
    hybrid_col = (
        "p_ai__full_hybrid" if "p_ai__full_hybrid" in pred_df
        else "p_ai_fused" if "p_ai_fused" in pred_df
        else None
    )

    df_dev = pred_df[dev_mask].copy()
    df_rep = pred_df[report_mask].copy()

    X_dev = extract_branch_feature_matrix(df_dev, branch_cols)
    y_dev = df_dev["label"].astype(int).values
    groups_dev = [folder_of(s) for s in df_dev["folder"]]

    X_rep = extract_branch_feature_matrix(df_rep, branch_cols)
    y_rep = df_rep["label"].astype(int).values
    folders_rep = [folder_of(s) for s in df_rep["folder"]]

    # 1. Fit stacker on DEV
    stacker, fit_info = fit_stacker_on_dev(X_dev, y_dev, groups=groups_dev, seed=seed)
    weights = {
        col.removeprefix("p_ai__").removeprefix("p_ai_"): round(float(w), 4)
        for col, w in zip(branch_cols, stacker.coef_[0])
    }
    intercept = round(float(stacker.intercept_[0]), 4)

    # Predictions on DEV (to calibrate thresholds)
    p_dev_deep = df_dev[deep_col].values
    p_dev_mean = X_dev.mean(axis=1)
    p_dev_stack = stacker.predict_proba(X_dev)[:, 1]

    # Predictions on REPORT
    p_rep_deep = df_rep[deep_col].values
    p_rep_mean = X_rep.mean(axis=1)
    p_rep_stack = stacker.predict_proba(X_rep)[:, 1]

    # Threshold calibration on DEV
    cal_th_deep, _ = best_balanced_accuracy_threshold(y_dev, p_dev_deep)
    cal_th_mean, _ = best_balanced_accuracy_threshold(y_dev, p_dev_mean)
    cal_th_stack, _ = best_balanced_accuracy_threshold(y_dev, p_dev_stack)

    models_to_eval = {
        "Deep alone": (p_rep_deep, cal_th_deep),
        "Mean probability fusion": (p_rep_mean, cal_th_mean),
        "Stacked logistic regression": (p_rep_stack, cal_th_stack),
    }

    if hybrid_col is not None:
        p_dev_fused = df_dev[hybrid_col].values
        p_rep_fused = df_rep[hybrid_col].values
        cal_th_fused, _ = best_balanced_accuracy_threshold(y_dev, p_dev_fused)
        models_to_eval["Gated hybrid"] = (p_rep_fused, cal_th_fused)

    if extra_models:
        for name, p_rep in extra_models.items():
            models_to_eval[name] = (p_rep, 0.5)

    comparison_rows = []
    for model_name, (scores, cal_th) in models_to_eval.items():
        res = evaluate_predictions(
            y_rep,
            scores,
            folders=folders_rep,
            calibrated_threshold=cal_th,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        ci = res["ci"]
        auc_str = (
            f"{res['auc']:.4f} [{ci['roc_auc']['low']:.4f}, {ci['roc_auc']['high']:.4f}]"
            if res['auc'] is not None and ci['roc_auc']['low'] is not None
            else "N/A"
        )
        ap_str = (
            f"{res['ap']:.4f} [{ci['average_precision']['low']:.4f}, {ci['average_precision']['high']:.4f}]"
            if res['ap'] is not None and ci['average_precision']['low'] is not None
            else "N/A"
        )
        comparison_rows.append({
            "Model": model_name,
            "AUC": auc_str,
            "AP": ap_str,
            "Acc @ 0.5": f"{res['acc_at_0.5'] * 100:.2f}%",
            "Acc @ 0.5 CI": (
                f"[{res['ci_at_0.5']['accuracy']['low'] * 100:.2f}%, "
                f"{res['ci_at_0.5']['accuracy']['high'] * 100:.2f}%]"
                if res["ci_at_0.5"].get("accuracy", {}).get("low") is not None
                else "N/A"
            ),
            "Acc @ Calibrated": f"{res['acc_at_calibrated'] * 100:.2f}% (th={cal_th:.4f})",
            "Acc @ Calibrated CI": (
                f"[{ci['accuracy']['low'] * 100:.2f}%, {ci['accuracy']['high'] * 100:.2f}%]"
                if ci.get("accuracy", {}).get("low") is not None
                else "N/A"
            ),
            "_raw_auc": res['auc'],
            "_raw_ap": res['ap'],
            "_raw_acc_05": res['acc_at_0.5'],
            "_raw_acc_cal": res['acc_at_calibrated'],
            "_cal_th": cal_th,
        })

    comparison_df = pd.DataFrame(comparison_rows)

    return {
        "branch_weights": weights,
        "intercept": intercept,
        "fit_info": fit_info,
        "comparison_table": comparison_df,
        "report_predictions": {
            "p_ai_deep": p_rep_deep,
            "p_ai_mean": p_rep_mean,
            "p_ai_stack": p_rep_stack,
        },
    }


def multi_seed_aggregation(results_per_seed: List[Dict[str, Any]]) -> pd.DataFrame:
    """Aggregate comparison tables across seeds reporting mean +/- std."""
    n_seeds = len(results_per_seed)
    model_names = results_per_seed[0]["comparison_table"]["Model"].tolist()

    agg_rows = []
    for model_name in model_names:
        aucs = [r["comparison_table"].loc[r["comparison_table"]["Model"] == model_name, "_raw_auc"].values[0] for r in results_per_seed]
        aps = [r["comparison_table"].loc[r["comparison_table"]["Model"] == model_name, "_raw_ap"].values[0] for r in results_per_seed]
        acc_05s = [r["comparison_table"].loc[r["comparison_table"]["Model"] == model_name, "_raw_acc_05"].values[0] for r in results_per_seed]
        acc_cals = [r["comparison_table"].loc[r["comparison_table"]["Model"] == model_name, "_raw_acc_cal"].values[0] for r in results_per_seed]

        aucs = [a for a in aucs if a is not None]
        auc_str = f"{np.mean(aucs):.4f} +/- {np.std(aucs):.4f}" if aucs else "N/A"
        ap_str = f"{np.mean(aps):.4f} +/- {np.std(aps):.4f}"
        acc05_str = f"{np.mean(acc_05s) * 100:.2f}% +/- {np.std(acc_05s) * 100:.2f}%"
        acccal_str = f"{np.mean(acc_cals) * 100:.2f}% +/- {np.std(acc_cals) * 100:.2f}%"

        agg_rows.append({
            "Model": model_name,
            f"AUC ({n_seeds} seeds)": auc_str,
            f"AP ({n_seeds} seeds)": ap_str,
            f"Acc @ 0.5 ({n_seeds} seeds)": acc05_str,
            f"Acc @ Calibrated ({n_seeds} seeds)": acccal_str,
        })

    return pd.DataFrame(agg_rows)


def main():
    parser = argparse.ArgumentParser(description="Stacking baseline and fusion comparison.")
    parser.add_argument("--predictions", nargs="+", required=True, help="One or more per-image predictions CSV paths (e.g. 3 seeds)")
    parser.add_argument("--splits-yaml", type=Path, default=DEFAULT_SPLITS_PATH, help="Path to eval/splits.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_outputs/stacking"), help="Output directory")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Number of bootstrap iterations for CI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_results = []
    for i, pred_path in enumerate(args.predictions):
        print(f"\nProcessing prediction set {i + 1}/{len(args.predictions)}: {pred_path}")
        df = pd.read_csv(pred_path)
        res = run_stacking_evaluation(
            df,
            splits_path=args.splits_yaml,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + i,
        )
        seed_results.append(res)
        print(f"Learned branch weights: {res['branch_weights']} (intercept: {res['intercept']})")

    # Save single or multi-seed comparison
    if len(seed_results) == 1:
        comp_df = seed_results[0]["comparison_table"]
        clean_cols = [c for c in comp_df.columns if not c.startswith("_")]
        display_df = comp_df[clean_cols]
    else:
        display_df = multi_seed_aggregation(seed_results)

    # Save to CSV and Markdown
    csv_path = args.output_dir / "comparison_table.csv"
    md_path = args.output_dir / "comparison_table.md"
    display_df.to_csv(csv_path, index=False)
    md_content = display_df.to_markdown(index=False)
    md_path.write_text(md_content + "\n", encoding="utf-8")

    # Save metadata and learned weights
    summary = {
        "predictions_files": [str(p) for p in args.predictions],
        "n_seeds": len(seed_results),
        "branch_weights_per_seed": [r["branch_weights"] for r in seed_results],
        "intercept_per_seed": [r["intercept"] for r in seed_results],
    }
    (args.output_dir / "stacking_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 80)
    print("FUSION COMPARISON TABLE (REPORT Folders)")
    print("=" * 80)
    print(display_df.to_string(index=False))
    print("=" * 80)
    print(f"\nSaved Markdown table to: {md_path}")
    print(f"Saved CSV table to: {csv_path}\n")


if __name__ == "__main__":
    main()
