"""
Branch diagnostic script: evaluate each branch alone on ForenSynths REPORT folders.

Fits a simple probe / LogisticRegression on DEV folders (FOLD_A),
evaluates per-generator AUC on REPORT folders (FOLD_B),
and flags any branch with AUC < 0.5 (inverted ranking).
Outputs results to CSV and prints a diagnostic summary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.splits import folder_of, load_split_spec


def compute_generator_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Compute AUC if both classes are present, else None."""
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def run_diagnostics_from_cache(
    cache_dir: Path,
    splits_path: Optional[Path] = None,
    branches: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run branch diagnostics using cached feature arrays."""
    spec = load_split_spec(splits_path)
    test_path = cache_dir / "test.npz"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing {test_path}")

    z = np.load(test_path)
    sources = [str(s) for s in z["sources"]]
    labels = np.asarray(z["labels"], dtype=np.int64)

    # Assign samples to dev, report, progan
    assignments = [spec.assign(s) for s in sources]
    dev_mask = np.array([a == "dev" for a in assignments])
    report_mask = np.array([a == "report" for a in assignments])

    if not np.any(dev_mask):
        raise ValueError("No DEV samples found in test cache.")
    if not np.any(report_mask):
        raise ValueError("No REPORT samples found in test cache.")

    available_branches = [b for b in ["physics", "prnu", "freq", "semantic"] if b in z.files]
    if branches:
        available_branches = [b for b in branches if b in available_branches]

    print(f"Running diagnostics on {len(available_branches)} branches: {available_branches}")
    print(f"DEV samples: {int(dev_mask.sum())}, REPORT samples: {int(report_mask.sum())}")

    # Fit a probe on DEV for each branch, predict probabilities on REPORT
    branch_probs: Dict[str, np.ndarray] = {}
    for branch in available_branches:
        X = np.asarray(z[branch], dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        X_dev, y_dev = X[dev_mask], labels[dev_mask]
        X_rep = X[report_mask]

        scaler = StandardScaler()
        X_dev_scaled = scaler.fit_transform(X_dev)
        X_rep_scaled = scaler.transform(X_rep)

        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf.fit(X_dev_scaled, y_dev)

        # Probability of class 1 (fake/AI)
        probs_rep = clf.predict_proba(X_rep_scaled)[:, 1]
        branch_probs[branch] = probs_rep

    # Group REPORT samples by generator folder
    report_sources = [sources[i] for i, m in enumerate(report_mask) if m]
    report_labels = labels[report_mask]
    report_folders = [folder_of(s) for s in report_sources]
    unique_folders = sorted(list(set(report_folders)))

    rows = []
    for folder in unique_folders:
        f_mask = np.array([f == folder for f in report_folders])
        y_f = report_labels[f_mask]
        row: Dict[str, object] = {
            "generator": folder,
            "n_images": int(len(y_f)),
            "n_real": int((y_f == 0).sum()),
            "n_fake": int((y_f == 1).sum()),
        }
        inverted = []
        for branch in available_branches:
            p_f = branch_probs[branch][f_mask]
            auc = compute_generator_auc(y_f, p_f)
            row[f"{branch}_auc"] = round(auc, 4) if auc is not None else np.nan
            if auc is not None and auc < 0.5:
                inverted.append(branch)
        row["inverted_branches"] = ", ".join(inverted) if inverted else "none"
        rows.append(row)

    # Overall / pooled row
    overall_row: Dict[str, object] = {
        "generator": "OVERALL_REPORT",
        "n_images": int(len(report_labels)),
        "n_real": int((report_labels == 0).sum()),
        "n_fake": int((report_labels == 1).sum()),
    }
    overall_inverted = []
    for branch in available_branches:
        p_all = branch_probs[branch]
        auc = compute_generator_auc(report_labels, p_all)
        overall_row[f"{branch}_auc"] = round(auc, 4) if auc is not None else np.nan
        if auc is not None and auc < 0.5:
            overall_inverted.append(branch)
    overall_row["inverted_branches"] = ", ".join(overall_inverted) if overall_inverted else "none"
    rows.append(overall_row)

    df = pd.DataFrame(rows)
    return df


def run_diagnostics_from_predictions(
    pred_csv: Path,
    splits_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Run branch diagnostics from per-image predictions CSV."""
    spec = load_split_spec(splits_path)
    df = pd.read_csv(pred_csv)

    sources = df["folder"].astype(str).tolist()
    labels = df["label"].astype(int).values

    assignments = [spec.assign(s) for s in sources]
    report_mask = np.array([a == "report" for a in assignments])

    df_rep = df[report_mask].copy()
    if len(df_rep) == 0:
        raise ValueError("No REPORT samples found in predictions CSV.")

    branch_cols = [c for c in df.columns if c.startswith("p_ai_") and c != "p_ai_fused"]
    branches = [c.replace("p_ai_", "") for c in branch_cols]

    report_sources = df_rep["folder"].astype(str).tolist()
    report_labels = df_rep["label"].astype(int).values
    report_folders = [folder_of(s) for s in report_sources]
    unique_folders = sorted(list(set(report_folders)))

    rows = []
    for folder in unique_folders:
        f_mask = np.array([f == folder for f in report_folders])
        y_f = report_labels[f_mask]
        row: Dict[str, object] = {
            "generator": folder,
            "n_images": int(len(y_f)),
            "n_real": int((y_f == 0).sum()),
            "n_fake": int((y_f == 1).sum()),
        }
        inverted = []
        for b, col in zip(branches, branch_cols):
            p_f = df_rep[col].values[f_mask]
            auc = compute_generator_auc(y_f, p_f)
            row[f"{b}_auc"] = round(auc, 4) if auc is not None else np.nan
            if auc is not None and auc < 0.5:
                inverted.append(b)
        row["inverted_branches"] = ", ".join(inverted) if inverted else "none"
        rows.append(row)

    # Overall
    overall_row: Dict[str, object] = {
        "generator": "OVERALL_REPORT",
        "n_images": int(len(report_labels)),
        "n_real": int((report_labels == 0).sum()),
        "n_fake": int((report_labels == 1).sum()),
    }
    overall_inverted = []
    for b, col in zip(branches, branch_cols):
        p_all = df_rep[col].values
        auc = compute_generator_auc(report_labels, p_all)
        overall_row[f"{b}_auc"] = round(auc, 4) if auc is not None else np.nan
        if auc is not None and auc < 0.5:
            overall_inverted.append(b)
    overall_row["inverted_branches"] = ", ".join(overall_inverted) if overall_inverted else "none"
    rows.append(overall_row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Branch diagnostic tool: evaluate per-branch AUC on REPORT folders.")
    parser.add_argument("--feature-cache", type=str, default="", help="Path to cache directory with test.npz")
    parser.add_argument("--predictions", type=str, default="", help="Path to per-image predictions CSV")
    parser.add_argument("--splits", type=str, default="", help="Path to eval/splits.yaml")
    parser.add_argument("--output", type=str, default="eval_out/branch_diagnostics.csv", help="Output CSV path")
    args = parser.parse_args()

    splits_path = Path(args.splits) if args.splits else None

    if args.predictions:
        df = run_diagnostics_from_predictions(Path(args.predictions), splits_path=splits_path)
    elif args.feature_cache:
        df = run_diagnostics_from_cache(Path(args.feature_cache), splits_path=splits_path)
    else:
        sys.exit("Please provide either --feature-cache or --predictions.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 80)
    print("BRANCH DIAGNOSTIC REPORT (AUC on REPORT folders)")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    print(f"Saved diagnostic report to: {out_path}\n")

    # Flag any inverted rankings
    inverted_rows = df[df["inverted_branches"] != "none"]
    if len(inverted_rows) > 0:
        print("[WARNING] Inverted rankings detected (AUC < 0.5):")
        for _, r in inverted_rows.iterrows():
            print(f"  Generator '{r['generator']}': {r['inverted_branches']}")
    else:
        print("[INFO] No inverted rankings detected (all branch AUCs >= 0.5).")


if __name__ == "__main__":
    main()
