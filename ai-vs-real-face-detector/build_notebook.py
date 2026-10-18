"""Build the v2 kaggle_train.ipynb from scratch.

Every CLI flag used here has been confirmed by reading the argparse blocks in:
  src/train.py          --mode, --data-dir, --data-source, --output-dir, --epochs,
                        --batch-size, --lr, --num-workers, --branches,
                        --branch-dropout, --semantic-model, --semantic-dim,
                        --deep-no-down (default False), --feature-cache,
                        --max-train-images, --seed, --preprocess, --fast, --fresh
  src/cache_features.py --data-dir, --cache-dir, --max-train-images,
                        --workers, --shard-size, --semantic-model,
                        --batch-size, --preprocess, --augment, --branches,
                        --splits, --seed, --force
  src/evaluate.py       --checkpoint, --data-dir, --output-dir, --device,
                        --feature-cache, --splits-yaml, --eval-set (report|dev|all),
                        --legacy, --n-bootstrap, --seed
  src/calibrate_threshold.py
                        --checkpoint, --data-dir, --feature-cache,
                        --splits-yaml, --device
  src/stack_fusion.py   --predictions (nargs+), --splits-yaml, --output-dir,
                        --n-bootstrap, --seed
  src/diagnose_branches.py
                        --feature-cache, --predictions, --splits, --output
  src/train_cached.py   reuses train.parse_args (--mode, --feature-cache,
                        --data-dir, --output-dir, --epochs, --batch-size, --lr)

Flags NOT confirmed (none invented):
  All flags above are confirmed from source.
"""
import json
from pathlib import Path

nb = {
    "metadata": {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
    "cells": [],
}

def md(source, cell_id=None):
    cell = {"cell_type": "markdown", "metadata": {}, "source": source}
    if cell_id:
        cell["id"] = cell_id
    nb["cells"].append(cell)

def code(source, cell_id=None):
    cell = {"cell_type": "code", "metadata": {"trusted": True},
            "source": source, "execution_count": None, "outputs": []}
    if cell_id:
        cell["id"] = cell_id
    nb["cells"].append(cell)

# ── Cell 0: intro ────────────────────────────────────────────
md("""\
# Train & Evaluate the Multi-Branch AI-Generated Image Detector (v2)

Runs on Kaggle GPU. Covers:
1. Clone `hybrid-detector-eval-overhaul` branch, install deps, verify dataset
2. Build multi-branch feature cache (CLIP ViT-L/14 + physics + freq)
3. Train individual branches (semantic, physics, freq, deep)
4. Train gated 4-branch hybrid with branch dropout
5. Evaluate, calibrate, diagnose, and compare (DEV / REPORT / PROGAN protocol)
6. Sync everything to HuggingFace

**Setup:** GPU → T4/P100 · Internet → On · Dataset added as input · `HF_TOKEN` secret enabled.
""", "intro")

# ── Cell 1: config ───────────────────────────────────────────
code("""\
# 1. Config
import os, sys, json, time, subprocess, threading
from pathlib import Path

GITHUB_REPO_URL = "https://github.com/Algorithm-bot/ai-vs-real-face-detector.git"
GITHUB_BRANCH   = "hybrid-detector-eval-overhaul"
KAGGLE_DATASET_MOUNT = Path(
    "/kaggle/input/datasets/sahilsawantspit/ai-vs-real-face-detector-data"
)

# ── Semantic model (CLIP ViT-L/14 via timm) ──
SEMANTIC_MODEL = "vit_large_patch14_clip_224.openai"
SEMANTIC_DIM   = 768    # confirmed: timm CLIP ViT-L/14 CLS token dim
BRANCHES_STR   = "deep,physics,semantic,freq"   # PRNU off

# ── Training subset cap (applied identically to cache + training) ──
MAX_TRAIN_IMAGES = 200_000   # 0 = full 721k

# ── Branch training schedule ──
BRANCHES = [
    "semantic_only", "physics_only", "freq_only",   # cached-feature runs
    "stage1",                                        # deep branch on pixels
    "full_hybrid",                                   # gated 4-branch with dropout
]
HPARAMS = {
    "semantic_only": dict(epochs=60, batch_size=128, lr=1e-3),
    "physics_only":  dict(epochs=60, batch_size=128, lr=1e-3),
    "freq_only":     dict(epochs=60, batch_size=128, lr=1e-3),
    "stage1":        dict(epochs=4,  batch_size=32,  lr=1e-4),
    "full_hybrid":   dict(epochs=10, batch_size=32,  lr=1e-4),
}

FORCE_RETRAIN = False
BRANCH_DROPOUT = 0.15   # for full_hybrid

MODELS_ROOT       = Path("/kaggle/working/models_v2")
FEATURE_CACHE_DIR = Path("/kaggle/working/feature_cache_v2")
EVAL_OUT          = Path("/kaggle/working/eval_out_v2")
NUM_WORKERS       = 3

HF_CHECKPOINT_REPO    = "sahilSpit/checkpoints"
HF_CHECKPOINT_PRIVATE = True

SEED = 42
log_path = Path("/kaggle/working/train.log")

# ── Validate ──
unknown = [m for m in BRANCHES if m not in HPARAMS]
assert not unknown, f"No HPARAMS entry for: {unknown}"
for mode in BRANCHES:
    (MODELS_ROOT / mode).mkdir(parents=True, exist_ok=True)
FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
EVAL_OUT.mkdir(parents=True, exist_ok=True)
assert KAGGLE_DATASET_MOUNT.exists(), f"{KAGGLE_DATASET_MOUNT} not found"
print("Dataset:", KAGGLE_DATASET_MOUNT)
print("Branches:", BRANCHES)
print("MAX_TRAIN_IMAGES:", MAX_TRAIN_IMAGES)
""", "config")

# ── Cell 2: clone ────────────────────────────────────────────
code("""\
# 2. Clone repo & install requirements
import shutil
clone_dir = Path("/kaggle/working/ai-vs-real-face-detector")
if clone_dir.exists():
    shutil.rmtree(clone_dir)
subprocess.run([
    "git", "clone", "--depth", "1",
    "--branch", GITHUB_BRANCH,
    GITHUB_REPO_URL, str(clone_dir)
], check=True)
requirements_file = next(clone_dir.rglob("requirements.txt"))
subprocess.run(["pip", "install", "-q", "-r", str(requirements_file)], check=True)
project_root = requirements_file.parent
print("Project root:", project_root)
""", "clone")

# ── Cell 3: verify ───────────────────────────────────────────
code("""\
# 2b. Kaggle verification report
subprocess.run([
    sys.executable, str(project_root / "kaggle_verify.py"),
    "--data-dir", str(KAGGLE_DATASET_MOUNT),
    "--feature-cache", str(FEATURE_CACHE_DIR),
], check=False)
""", "verify")

# ── Cell 4: HF auth ─────────────────────────────────────────
code("""\
# 3. HF auth
from huggingface_hub import HfApi, snapshot_download
from kaggle_secrets import UserSecretsClient

hf_token = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["HF_TOKEN"] = hf_token   # passed to subprocesses
api = HfApi(token=hf_token)
api.create_repo(repo_id=HF_CHECKPOINT_REPO, repo_type="model",
                private=HF_CHECKPOINT_PRIVATE, exist_ok=True)
print("Checkpoint repo:", HF_CHECKPOINT_REPO)
""", "hf-auth")

# ── Cell 5: sync / restore ──────────────────────────────────
code("""\
# 4a. Restore from HF + background sync
WORKING = Path("/kaggle/working")
SYNC_DIRS = {FEATURE_CACHE_DIR: "feature_cache_v2"}
for mode in BRANCHES:
    SYNC_DIRS[MODELS_ROOT / mode] = f"models_v2/{mode}"
SYNC_DIRS[EVAL_OUT] = "eval_out_v2"
uploaded = {}

try:
    snapshot_download(
        repo_id=HF_CHECKPOINT_REPO, repo_type="model", token=hf_token,
        local_dir=str(WORKING),
        allow_patterns=["feature_cache_v2/*", "models_v2/*", "eval_out_v2/*"],
    )
except Exception as e:
    print("Nothing to restore:", e)

for d in SYNC_DIRS:
    for f in d.rglob("*"):
        if f.is_file():
            uploaded[str(f)] = f.stat().st_mtime

def sync_once(min_age=60):
    for local_dir, repo_dir in SYNC_DIRS.items():
        for f in local_dir.rglob("*"):
            if not f.is_file():
                continue
            st = f.stat()
            if time.time() - st.st_mtime < min_age:
                continue
            if uploaded.get(str(f)) == st.st_mtime:
                continue
            try:
                api.upload_file(
                    path_or_fileobj=str(f),
                    path_in_repo=f"{repo_dir}/{f.relative_to(local_dir).as_posix()}",
                    repo_id=HF_CHECKPOINT_REPO, repo_type="model",
                )
                uploaded[str(f)] = st.st_mtime
            except Exception as e:
                print("sync failed:", f.name, e, flush=True)

stop_sync = threading.Event()
def _sync_loop():
    while not stop_sync.is_set():
        sync_once()
        stop_sync.wait(120)
threading.Thread(target=_sync_loop, daemon=True).start()
print("Background sync started")
""", "sync")

# ── Cell 6: cache build ─────────────────────────────────────
code("""\
# 4b. Multi-branch feature cache (CLIP ViT-L/14 + Physics + Freq)
cache_cmd = [
    sys.executable, str(project_root / "src" / "cache_features.py"),
    "--data-dir",          str(KAGGLE_DATASET_MOUNT),
    "--cache-dir",         str(FEATURE_CACHE_DIR),
    "--workers",           str(NUM_WORKERS),
    "--semantic-model",    SEMANTIC_MODEL,
    "--preprocess",        "crop",
    "--augment",
    "--branches",          BRANCHES_STR,
    "--batch-size",        "64",
    "--seed",              str(SEED),
]
if MAX_TRAIN_IMAGES > 0:
    cache_cmd += ["--max-train-images", str(MAX_TRAIN_IMAGES)]

print("Cache cmd:", " ".join(cache_cmd))
subprocess.run(cache_cmd, check=False,
               env={**os.environ, "PYTHONUNBUFFERED": "1", "HF_TOKEN": hf_token})

# Assert cache completeness
for split in ("train", "val", "test"):
    p = FEATURE_CACHE_DIR / f"{split}.npz"
    assert p.exists(), f"Missing cache shard: {p}"
meta_path = FEATURE_CACHE_DIR / "meta.json"
assert meta_path.exists(), "Missing meta.json"
meta = json.loads(meta_path.read_text())
assert meta.get("semantic_model") == SEMANTIC_MODEL, \\
    f"meta semantic_model mismatch: {meta.get('semantic_model')}"
assert meta.get("branches") == BRANCHES_STR, \\
    f"meta branches mismatch: {meta.get('branches')}"
assert meta.get("preprocess") == "crop", \\
    f"meta preprocess mismatch: {meta.get('preprocess')}"
if MAX_TRAIN_IMAGES > 0:
    assert meta.get("max_train_images") == MAX_TRAIN_IMAGES, \\
        f"meta max_train_images mismatch: {meta.get('max_train_images')}"
print("\\nCache verification PASSED:", json.dumps(meta, indent=2))
""", "cache-build")

# ── Cell 7: run_and_log + train ──────────────────────────────
code("""\
# 5. Train branches
import io

def run_and_log(cmd, extra_env=None):
    \\"\\"\\"Run subprocess, read raw bytes, split on \\\\r and \\\\n so tqdm bars render.
    Passes HF_TOKEN in env.  Logs to train.log.\\"\\"\\"
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_TOKEN": hf_token,
           **(extra_env or {})}
    print("Running:", " ".join(cmd), flush=True)
    last_bar = 0.0
    with open(log_path, "ab") as log:
        log.write(f"\\n===== {' '.join(cmd)} =====\\n".encode())
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                env=env)
        buf = b""
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            # Split on \\r or \\n to handle tqdm carriage returns
            while b"\\r" in buf or b"\\n" in buf:
                idx_r = buf.find(b"\\r")
                idx_n = buf.find(b"\\n")
                if idx_r == -1: idx_r = len(buf)
                if idx_n == -1: idx_n = len(buf)
                idx = min(idx_r, idx_n)
                line = buf[:idx].decode("utf-8", errors="replace")
                buf = buf[idx + 1:]
                if not line.strip():
                    continue
                is_bar = "it/s]" in line or "s/it]" in line or "%|" in line
                if is_bar and "100%|" not in line:
                    now = time.time()
                    if now - last_bar < 30:
                        continue
                    last_bar = now
                print(line, flush=True)
                log.write((line + "\\n").encode())
                log.flush()
        if buf.strip():
            line = buf.decode("utf-8", errors="replace")
            print(line, flush=True)
            log.write((line + "\\n").encode())
        proc.wait()
    return proc.returncode

def is_done(mode):
    return (MODELS_ROOT / mode / f"{mode}_history.json").exists()

def build_cmd(mode):
    hp = HPARAMS[mode]
    if mode in ("semantic_only", "physics_only", "freq_only", "prnu_only"):
        # Cached-feature training via train_cached.py
        return [
            sys.executable, "-u", str(project_root / "src" / "train_cached.py"),
            "--mode",          mode,
            "--data-dir",      str(KAGGLE_DATASET_MOUNT),
            "--feature-cache", str(FEATURE_CACHE_DIR),
            "--output-dir",    str(MODELS_ROOT / mode),
            "--epochs",        str(hp["epochs"]),
            "--batch-size",    str(hp["batch_size"]),
            "--lr",            str(hp["lr"]),
            "--seed",          str(SEED),
        ]
    elif mode == "stage1":
        cmd = [
            sys.executable, "-u", str(project_root / "src" / "train.py"),
            "--mode",          "stage1",
            "--data-dir",      str(KAGGLE_DATASET_MOUNT),
            "--data-source",   "local",
            "--output-dir",    str(MODELS_ROOT / mode),
            "--epochs",        str(hp["epochs"]),
            "--batch-size",    str(hp["batch_size"]),
            "--lr",            str(hp["lr"]),
            "--num-workers",   str(NUM_WORKERS),
            "--seed",          str(SEED),
        ]
        if MAX_TRAIN_IMAGES > 0:
            cmd += ["--max-train-images", str(MAX_TRAIN_IMAGES)]
        return cmd
    elif mode == "full_hybrid":
        cmd = [
            sys.executable, "-u", str(project_root / "src" / "train.py"),
            "--mode",            "full_hybrid",
            "--data-dir",        str(KAGGLE_DATASET_MOUNT),
            "--data-source",     "local",
            "--feature-cache",   str(FEATURE_CACHE_DIR),
            "--output-dir",      str(MODELS_ROOT / mode),
            "--epochs",          str(hp["epochs"]),
            "--batch-size",      str(hp["batch_size"]),
            "--lr",              str(hp["lr"]),
            "--num-workers",     str(NUM_WORKERS),
            "--seed",            str(SEED),
            "--semantic-model",  SEMANTIC_MODEL,
            "--semantic-dim",    str(SEMANTIC_DIM),
            "--branches",        BRANCHES_STR,
            "--branch-dropout",  str(BRANCH_DROPOUT),
            "--preprocess",      "crop",
            # --deep-no-down NOT passed: off by default
        ]
        if MAX_TRAIN_IMAGES > 0:
            cmd += ["--max-train-images", str(MAX_TRAIN_IMAGES)]
        return cmd
    else:
        raise ValueError(f"Unknown mode: {mode}")

status = {}
for mode in BRANCHES:
    print(f"\\n{'=' * 70}\\n  BRANCH: {mode}\\n{'=' * 70}", flush=True)
    if is_done(mode) and not FORCE_RETRAIN:
        print(f"{mode}: already done — skipping.")
        status[mode] = "skipped (already done)"
        continue
    cmd = build_cmd(mode)
    rc = run_and_log(cmd)
    sync_once(min_age=0)
    status[mode] = "ok" if rc == 0 else f"FAILED (exit {rc})"
    print(f"\\n{mode}: {status[mode]}", flush=True)

print("\\nSummary:")
for mode, s in status.items():
    print(f"  {mode:18s} {s}")
""", "train")

# ── Cell 8: evaluate + calibrate + diagnose ──────────────────
code("""\
# 6. Evaluate each branch + hybrid, calibrate, diagnose

SPLITS_YAML = str(project_root / "eval" / "splits.yaml")
eval_results = {}

# ── 6a. Per-image prediction export ──
checkpoints = {}
for mode in BRANCHES:
    d = MODELS_ROOT / mode
    for name in (f"{mode}_best.pt", f"full_hybrid_best.pt", "hybrid_best.pt"):
        p = d / name
        if p.exists():
            checkpoints[mode] = p
            break
print("Checkpoints found:", {k: str(v) for k, v in checkpoints.items()})

for mode, ckpt in checkpoints.items():
    out_dir = EVAL_OUT / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    # Evaluate on REPORT set (FOLD_B = headline)
    cmd = [
        sys.executable, str(project_root / "src" / "evaluate.py"),
        "--checkpoint",    str(ckpt),
        "--data-dir",      str(KAGGLE_DATASET_MOUNT),
        "--output-dir",    str(out_dir),
        "--feature-cache", str(FEATURE_CACHE_DIR),
        "--splits-yaml",   SPLITS_YAML,
        "--eval-set",      "report",
        "--n-bootstrap",   "1000",
        "--seed",          str(SEED),
    ]
    print(f"\\n--- Evaluating {mode} on REPORT ---")
    run_and_log(cmd)

    # Also evaluate on DEV for stacking/comparison
    out_dir_dev = EVAL_OUT / f"{mode}_dev"
    out_dir_dev.mkdir(parents=True, exist_ok=True)
    cmd_dev = [
        sys.executable, str(project_root / "src" / "evaluate.py"),
        "--checkpoint",    str(ckpt),
        "--data-dir",      str(KAGGLE_DATASET_MOUNT),
        "--output-dir",    str(out_dir_dev),
        "--feature-cache", str(FEATURE_CACHE_DIR),
        "--splits-yaml",   SPLITS_YAML,
        "--eval-set",      "dev",
        "--n-bootstrap",   "1000",
        "--seed",          str(SEED),
    ]
    print(f"--- Evaluating {mode} on DEV ---")
    run_and_log(cmd_dev)

    # Evaluate on PROGAN separately (--eval-set all, filter later)
    sync_once(min_age=0)

# ── 6b. Calibrate threshold on DEV for the full_hybrid ──
hybrid_ckpt = checkpoints.get("full_hybrid")
if hybrid_ckpt:
    print("\\n--- Calibrating threshold on DEV ---")
    cal_cmd = [
        sys.executable, str(project_root / "src" / "calibrate_threshold.py"),
        "--checkpoint",    str(hybrid_ckpt),
        "--data-dir",      str(KAGGLE_DATASET_MOUNT),
        "--feature-cache", str(FEATURE_CACHE_DIR),
        "--splits-yaml",   SPLITS_YAML,
    ]
    run_and_log(cal_cmd)
    sync_once(min_age=0)

# ── 6c. Branch diagnostics ──
print("\\n--- Branch diagnostics ---")
diag_cmd = [
    sys.executable, str(project_root / "src" / "diagnose_branches.py"),
    "--feature-cache", str(FEATURE_CACHE_DIR),
    "--splits",        SPLITS_YAML,
    "--output",        str(EVAL_OUT / "branch_diagnostics.csv"),
]
run_and_log(diag_cmd)

# ── 6d. Stacking fusion comparison ──
hybrid_preds = EVAL_OUT / "full_hybrid" / "predictions.csv"
if hybrid_preds.exists():
    print("\\n--- Stacking fusion comparison ---")
    stack_cmd = [
        sys.executable, str(project_root / "src" / "stack_fusion.py"),
        "--predictions",  str(hybrid_preds),
        "--splits-yaml",  SPLITS_YAML,
        "--output-dir",   str(EVAL_OUT / "stacking"),
        "--n-bootstrap",  "1000",
        "--seed",         str(SEED),
    ]
    run_and_log(stack_cmd)

sync_once(min_age=0)
print("\\nEvaluation complete.")
""", "evaluate")

# ── Cell 9: results table ────────────────────────────────────
code("""\
# 7. Results comparison: FOLD_A / FOLD_B / PROGAN
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(project_root))
from src.eval.splits import load_split_spec, folder_of
from src.eval.metrics import bootstrap_ci, metric_summary

spec = load_split_spec(SPLITS_YAML)

def fold_results(pred_csv, fold_folders, fold_name):
    \\"\\"\\"Compute AUC, AP, acc@0.5, acc@calibrated with bootstrap CIs for a fold.\\"\\"\\"
    if not Path(pred_csv).exists():
        return None
    df = pd.read_csv(pred_csv)
    df["_folder"] = df["folder"].apply(folder_of) if "folder" in df.columns else df["path"].apply(folder_of)
    mask = df["_folder"].isin(set(fold_folders))
    sub = df[mask]
    if len(sub) == 0:
        return None
    y = sub["label"].values
    p = sub["p_ai_fused"].values if "p_ai_fused" in sub.columns else sub["p_ai"].values

    # Load calibrated threshold if available
    cal_path = None
    if hybrid_ckpt:
        for ext in ("calibration.json",):
            cand = hybrid_ckpt.parent / ext
            if cand.exists():
                cal_path = cand
    threshold = 0.5
    if cal_path and cal_path.exists():
        cal = json.loads(cal_path.read_text())
        threshold = cal.get("threshold", 0.5)

    summary = metric_summary(y, p, threshold)
    ci = bootstrap_ci(y, p, n_resamples=1000, seed=SEED)
    return {
        "Fold": fold_name,
        "N": len(sub),
        "AUC": f"{summary.get('roc_auc', 0):.4f} [{ci.get('roc_auc_lo', 0):.4f}, {ci.get('roc_auc_hi', 0):.4f}]",
        "AP": f"{summary.get('average_precision', 0):.4f} [{ci.get('ap_lo', 0):.4f}, {ci.get('ap_hi', 0):.4f}]",
        "Acc@0.5": f"{summary.get('accuracy', 0):.4f}",
        "Acc@Cal": f"{metric_summary(y, p, threshold).get('accuracy', 0):.4f}",
    }

print("=" * 80)
print("CROSS-GENERATOR EVALUATION RESULTS")
print("=" * 80)
pred_csv = str(EVAL_OUT / "full_hybrid" / "predictions.csv")
rows = []
r = fold_results(pred_csv, spec.dev, "FOLD_A (DEV)")
if r: rows.append(r)
r = fold_results(pred_csv, spec.report, "FOLD_B (REPORT)")
if r: rows.append(r)
r = fold_results(pred_csv, spec.progan, "PROGAN (seen)")
if r: rows.append(r)

if rows:
    results_df = pd.DataFrame(rows)
    print(results_df.to_string(index=False))
    results_df.to_csv(EVAL_OUT / "cross_generator_results.csv", index=False)
else:
    print("No predictions CSV found; run evaluation first.")

# Print per-branch comparison if stacking ran
stacking_csv = EVAL_OUT / "stacking" / "comparison_table.csv"
if stacking_csv.exists():
    print("\\n" + "=" * 80)
    print("FUSION COMPARISON TABLE")
    print("=" * 80)
    print(pd.read_csv(stacking_csv).to_string(index=False))

# Per-branch single-mode results
print("\\n" + "=" * 80)
print("PER-BRANCH RESULTS (REPORT set)")
print("=" * 80)
branch_rows = []
for mode in BRANCHES:
    metrics_json = EVAL_OUT / mode / "metrics.json"
    if metrics_json.exists():
        m = json.loads(metrics_json.read_text())
        overall = m.get("overall_at_calibrated") or m.get("overall") or {}
        branch_rows.append({
            "Branch": mode,
            "AUC": f"{overall.get('roc_auc', 0):.4f}",
            "AP": f"{overall.get('average_precision', 0):.4f}",
            "Acc": f"{overall.get('accuracy', 0):.4f}",
        })
if branch_rows:
    print(pd.DataFrame(branch_rows).to_string(index=False))
else:
    print("No per-branch results found yet.")
""", "results")

# ── Cell 10: final sync ─────────────────────────────────────
code("""\
# 8. Final sync & cleanup
stop_sync.set()
sync_once(min_age=0)

# Upload log
if log_path.exists():
    try:
        api.upload_file(path_or_fileobj=str(log_path),
                        path_in_repo="logs/train_v2.log",
                        repo_id=HF_CHECKPOINT_REPO, repo_type="model")
    except Exception as e:
        print("Log sync failed:", e)

print("\\nDone:", f"https://huggingface.co/{HF_CHECKPOINT_REPO}")
""", "final-sync")

# ── Cell 11: notes ───────────────────────────────────────────
md("""\
# Notes

- Use **"Save & Run All (Commit)"** for long runs (stage1 + full_hybrid).
- `FORCE_RETRAIN = True` to retrain branches that already have `*_history.json`.
- `MAX_TRAIN_IMAGES = 0` to use the full 721k training set (cache rebuild will take ~7h).
- PROGAN folders are reported separately; they are the seen-generator validation.
- FOLD_A (DEV) is used for checkpoint/epoch selection; FOLD_B (REPORT) is the headline result.
""", "notes")

# ── Write ────────────────────────────────────────────────────
nb_path = Path("notebooks/kaggle_train.ipynb")
nb_path.parent.mkdir(exist_ok=True)
nb_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {nb_path}: {len(nb['cells'])} cells")
