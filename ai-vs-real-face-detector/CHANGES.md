# ForenSynths Overhaul & Multi-Branch Architecture Changes

This document details the architectural overhaul, cross-generator evaluation protocol, single-source-of-truth preprocessing, semantic branch upgrade, frequency branch addition, and stacking/fusion enhancements implemented in `ai-vs-real-face-detector`.

---

## 1. Summary of Changes Across the 5 Items

### Item 1: Cross-Generator DEV/REPORT Partition & Calibration Protocol
- **DEV/REPORT Split (`eval/splits.yaml`)**: Partitions the 42 ForenSynths generator folders into:
  - **FOLD_A (DEV / selection set, 14 folders, 38,437 images)**: `stylegan_bedroom`, `stylegan_car`, `stylegan_cat`, `gaugan`, `cyclegan_apple`, `cyclegan_horse`, `cyclegan_orange`, `cyclegan_summer`, `cyclegan_winter`, `cyclegan_zebra`, `san`, `deepfake`, `biggan`, `stargan`.
  - **FOLD_B (REPORT / headline set, 8 folders, 43,848 images)**: `stylegan2_car`, `stylegan2_cat`, `stylegan2_church`, `stylegan2_horse`, `crn`, `imle`, `seeingdark`, `whichfaceisreal`.
  - **PROGAN (Seen generator, 20 folders, 7,960 images)**: Kept separate; reported as seen-generator validation.
  - *No-leakage rule*: `crn` and `imle` share real photos and are strictly placed together in FOLD_B. Thresholds applied to REPORT are fitted on DEV (and vice versa for cross-fitting).
- **DEV Checkpoint Selection**: Best model checkpoint is selected via mean DEV score (average of DEV AUC and balanced accuracy at the calibrated DEV threshold) instead of ProGAN validation accuracy.
- **Per-Image Predictions CSV**: `src/evaluate.py` outputs per-image records (`path`, `folder`, `label`, `p_ai_<branch>`, `p_ai_fused`) enabling offline recomputation of any metric, threshold, or confidence interval.
- **Calibration Tool (`src/calibrate_threshold.py`)**: Computes temperature scaling and searches log-spaced threshold grid on DEV. Saves calibration parameters to `calibration.json`.
- **Bootstrap 95% Confidence Intervals**: Resampling with replacement within each generator folder (1,000 resamples by default) for AUC, AP, and accuracy.

### Item 2: Preprocessing Single Source of Truth & Augmentation
- **Shared Preprocessing Module (`src/preprocess.py`)**: Single source of truth across training, caching, evaluation, and live inference:
  - Training: Deterministic random crop 224x224 (with edge reflection padding if smaller).
  - Evaluation / Inference: Center crop 224x224.
  - Legacy mode: Resize 224x224 with `cv2.INTER_AREA` (selectable via `--preprocess resize` or `--legacy`).
- **Raw-Resolution Augmentations (`src/augment.py`)**:
  - Gaussian blur ($\sigma \in [0.5, 3.0]$) and JPEG compression ($Q \in [30, 95]$) applied at native image resolution before any cropping.
  - Deterministic per-path MD5 hash seeding: ensuring cached and live forward passes produce bit-for-bit identical augmented pixels for any image path.
- **Deep Stem Resolution (`--deep-no-down`)**:
  - Optional flag changing EfficientNet-B0 stem convolution stride from 2 to 1 (retaining spatial resolution for fine artifact extraction).

### Item 3: Configurable CLIP ViT-L/14 Semantic Branch
- **Model Upgrade**: Replaced ViT-Small with frozen `vit_large_patch14_clip_224.openai` (via `timm`).
- **Intermediate CLS Token Extraction**: Optional `--semantic-layers 12,16,20,24` extracting and concatenating CLS tokens across transformer blocks (4 x 1024 = 4096-d).
- **Linear Probe Mode**: Added `clip_linear` training mode in `src/train_cached.py` with a single linear classification head.
- **GPU fp16 Batch Processing**: `src/cache_features.py` extracts semantic embeddings using GPU batching (batch size 32 or 64) with `torch.autocast(dtype=torch.float16)`.
- *Note*: Pretrained CLIP weights download requires internet access in Kaggle notebook (already enabled for HF dataset sync).

### Item 4: Frequency/Upsampling Branch & Diagnostics
- **New Frequency Branch (`src/freq_branch/extractor.py`)**: 64-dimensional feature vector extracted from full-resolution raw image before 224 crop:
  1. *Radially averaged magnitude spectrum* (32 bins, normalized so $\sum = 1.0$).
  2. *Periodic peak scores* (20 bins across radial bands: ratio of top 1% peak energy to median energy, capturing upsampling and transposed-conv grid checkerboards).
  3. *NPR residual statistics* (12 features: mean, std, skewness, excess kurtosis across R, G, B channels of high-pass median difference).
- **Configurable Branch Set (`--branches`)**:
  - Default: `deep,physics,semantic,freq` (PRNU is disabled by default).
  - Supports any comma-separated combination (e.g. `--branches deep,freq` or `--branches deep,physics,prnu,semantic,freq`).
- **One-off Diagnostic Tool (`src/diagnose_branches.py`)**:
  - Evaluates each branch alone using probes fitted on DEV.
  - Reports per-generator AUC on REPORT folders.
  - Flags any branch with inverted ranking ($\text{AUC} < 0.5$) and exports to CSV.

### Item 5: Fusion Enhancements, Branch Dropout & Stacking Baseline
- **Branch Dropout ($p=0.15$)**:
  - During training, auxiliary branches (`physics`, `semantic`, `freq`, `prnu`) are independently zeroed with probability $p=0.15$, while the deep branch is **always active**.
  - Prevents the classification head from over-relying on any single modality.
  - Automatically disabled at evaluation time (all branches active).
  - Mean gate activations per branch are logged on REPORT folders.
- **Stacking Baseline (`src/stack_fusion.py`)**:
  - Cross-validated $L_2$ Logistic Regression on DEV folders using GroupKFold (grouped by generator folder) or StratifiedKFold.
  - Evaluates on REPORT folders and outputs learned weights per branch and intercept.
- **Multi-Seed Comparison Tool**:
  - Compares: (1) Deep alone, (2) Simple average of branch probabilities, (3) Stacked logistic regression, (4) Gated hybrid with dropout, (5) Gated hybrid without dropout.
  - Reports AUC, AP, acc@0.5, and acc@calibrated with bootstrap 95% CIs, supporting aggregation across 3 random seeds.

---

## 2. Command-Line Interface (CLI) Guide

### Rebuild Feature Cache
```bash
python src/cache_features.py \
    --data-dir /kaggle/input/forensynths \
    --cache-dir /kaggle/working/feature_cache \
    --semantic-model vit_large_patch14_clip_224.openai \
    --batch-size 64 \
    --preprocess crop \
    --augment
```
*Legacy cache rebuild:*
```bash
python src/cache_features.py \
    --data-dir /kaggle/input/forensynths \
    --cache-dir /kaggle/working/feature_cache_legacy \
    --semantic-model vit_small_patch16_224 \
    --preprocess resize \
    --legacy
```

### Train Gated Hybrid Detector
```bash
python src/train.py \
    --mode full_hybrid \
    --data-dir /kaggle/input/forensynths \
    --feature-cache /kaggle/working/feature_cache \
    --semantic-model vit_large_patch14_clip_224.openai \
    --semantic-dim 768 \
    --branches deep,physics,semantic,freq \
    --branch-dropout 0.15 \
    --output-dir models/full_hybrid_seed42 \
    --epochs 10 \
    --batch-size 32 \
    --seed 42
```

### Run Linear Probe on CLIP Features
```bash
python src/train_cached.py \
    --mode clip_linear \
    --feature-cache /kaggle/working/feature_cache \
    --semantic-dim 768 \
    --output-dir models/clip_linear \
    --epochs 20 \
    --batch-size 256
```

### Calibrate Threshold on DEV Set
```bash
python src/calibrate_threshold.py \
    --predictions evaluation_outputs/predictions.csv \
    --splits-yaml eval/splits.yaml \
    --output evaluation_outputs/calibration.json
```

### Evaluate on REPORT Folders
```bash
python src/evaluate.py \
    --checkpoint models/full_hybrid_seed42/full_hybrid_best.pt \
    --data-dir /kaggle/input/forensynths \
    --feature-cache /kaggle/working/feature_cache \
    --splits-yaml eval/splits.yaml \
    --eval-set report \
    --output-dir evaluation_outputs/seed42
```

### Run Branch Diagnostics
```bash
python src/diagnose_branches.py \
    --feature-cache /kaggle/working/feature_cache \
    --splits eval/splits.yaml \
    --output eval_out/branch_diagnostics.csv
```

### Run Stacking Baseline & Multi-Seed Comparison
```bash
python src/stack_fusion.py \
    --predictions evaluation_outputs/seed42/predictions.csv \
                  evaluation_outputs/seed43/predictions.csv \
                  evaluation_outputs/seed44/predictions.csv \
    --splits-yaml eval/splits.yaml \
    --output-dir evaluation_outputs/stacking
```

---

## 3. Backward Compatibility & Legacy Reproduction

To reproduce legacy behavior and evaluate older checkpoints trained on the 1,695-dimensional feature vector without changes:
- Pass the `--legacy` flag to `train.py`, `cache_features.py`, or `evaluate.py`.
- In legacy mode:
  - Active branches revert to `deep,physics,prnu,semantic` (1695 dimensions).
  - Preprocessing uses legacy `resize` (224x224 INTER_AREA resize) instead of cropping.
  - Threshold calibration defaults to 0.5 without DEV fold cross-fitting.
  - Evaluation reports across the unpartitioned test tree.
  - Cache metadata verification ignores hash/dimension mismatches.

---

## 4. Unverified GPU Items & Kaggle Notebook Migration

### A. Notebook Migration Guide (`notebooks/kaggle_train.ipynb`)
To run on Kaggle with the new architecture:
1. **Update Clone Cell (Cell 2)**:
   Change the git clone command to pull the new evaluation overhaul branch:
   ```python
   subprocess.run([
       "git", "clone",
       "--depth", "1",
       "--branch", "hybrid-detector-eval-overhaul",
       GITHUB_REPO_URL,
       str(clone_dir)
   ], check=True)
   ```
2. **Delete Legacy Monkey-Patch Cells**:
   The following notebook cells applied temporary monkey-patches that are now natively integrated into the repository. **Delete these cells entirely**:
   - **Cell 2b / Patch Cell 3** (`Make stage1 (deep branch) fully resumable`): Native per-epoch atomic saving with `stage1_history_partial.json` and resilient checkpointing are now in `src/train.py`.
   - **Cell 2c / Patch Cell 4** (`Blur+JPEG augmentation for all branches`): Pre-crop Blur+JPEG augmentation is now in `src/augment.py` and natively invoked by `src/cache_features.py` and `src/train.py`.
   - **Cell 4c / Patch Cell 8** (`Launcher for train_cached.py that fits scalers from cache`): Fast scaler fitting directly from cached `.npz` files is built into `src/train_cached.py`.
3. **Run Verification Cell**:
   Add a cell immediately after cloning to run `kaggle_verify.py` and verify environment and dataset integrity before starting training or caching:
   ```python
   !python kaggle_verify.py --data-dir $KAGGLE_DATASET_MOUNT --feature-cache /kaggle/working/feature_cache
   ```

### B. Unverified Items Deferred to Kaggle
The following items cannot be fully verified locally without GPU accelerators and the real Kaggle dataset storage:
1. **Full Kaggle Dataset Image Counts**: Exact per-folder image counts vs `eval/expected_counts.json` (90,245 total images across 42 folders).
2. **Kaggle GPU Memory & Real Throughput**:
   - EfficientNet-B0 stride 1 vs stride 2 across 90k images.
   - Batch fp16 extraction speed of CLIP ViT-L/14 with Kaggle's T4 or P100 GPUs.
3. **Pretrained Weights Checksum & Internet Download**: Actual download and validation of `vit_large_patch14_clip_224.openai` via `timm` (requires notebook internet enabled).
4. **End-to-End Real Image Parity**: Exact `np.allclose` tolerance check on 20 real ForenSynths images drawn from `crn`, `stylegan_car`, `whichfaceisreal`, and `progan_car`.

---

## 5. Compute Cost & Resource Estimates

### A. Cache Rebuild Time Formula
The total cache rebuild time depends on whether the GPU or CPU pipeline is the bottleneck:
$$\text{Time (hours)} = \frac{N_{\text{images}}}{3600 \times \min\left(\text{Throughput}_{\text{CLIP}}, \, N_{\text{workers}} \times \text{Throughput}_{\text{Freq+Physics}}\right)}$$

- $N_{\text{images}} = 90{,}245$ (for the entire ForenSynths suite).
- $\text{Throughput}_{\text{CLIP}}$: Measured in img/s on GPU (e.g. ~35–45 img/s on T4 in fp16 with batch size 64).
- $\text{Throughput}_{\text{Freq+Physics}}$: Measured in img/s per CPU core (e.g. ~16 img/s per core).
- With 2–3 CPU workers on Kaggle:
  $$\text{Effective Throughput} \approx \min(40, 2 \times 16) \approx 32\text{ img/s}$$
  $$\text{Total Time} \approx \frac{90{,}245}{3600 \times 32} \approx 0.78\text{ hours (~47 minutes)}$$
- If throughput on Kaggle is constrained, use `--max-train-images N` (e.g. `--max-train-images 10000`) on `src/cache_features.py` to balance and cap the training subset.

### B. Component Resource Breakdown

| Component | Hardware | Memory (VRAM) | Estimated Throughput / Time |
| :--- | :--- | :--- | :--- |
| **EfficientNet-B0 (default stride 2)** | NVIDIA T4 / RTX | ~1.2 GB | ~385 img/s (~4 mins for 90k images) |
| **EfficientNet-B0 (`--deep-no-down` stride 1)** | NVIDIA T4 / RTX | ~4.7 GB | ~35 img/s (~42 mins for 90k images) |
| **CLIP ViT-L/14 (fp16, batch size 64)** | NVIDIA T4 / RTX | ~2.3 GB | ~35-45 img/s (~35-45 mins for 90k images) |
| **Frequency Branch Extraction (CPU)** | 1 CPU Core | < 100 MB | ~16 img/s (vectorized NumPy) |
| **Full Hybrid Cache Generation** | 1x T4 + 2 CPU Workers | ~3.5 GB | ~1.0 - 1.5 hours total for ForenSynths |
| **Stacking Logistic Regression (DEV)** | CPU | < 500 MB | < 15 seconds |
