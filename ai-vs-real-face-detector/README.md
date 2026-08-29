# AI vs Real Face Detector

Research prototype for classifying portrait images as **real**, **AI-generated**, or **uncertain**. The repository currently contains three model modes:

- `stage1`: EfficientNet image-only baseline.
- `hybrid`: EfficientNet plus a 20-feature physics/computer-vision branch.
- `full_hybrid`: EfficientNet plus physics, PRNU noise statistics, and a frozen ViT semantic encoder, combined with configurable fusion.

The system is an experimental authenticity signal, not a forensic guarantee. Results depend heavily on the training generators, dataset construction, image quality, and checkpoint used.

## Current architecture

```text
Input image
  -> RGB conversion and optional dlib 5-point face alignment/crop
  -> 224 x 224 ImageNet-normalized tensor

EfficientNet-B0             1280 features
Dlib/OpenCV physics          20 features
PRNU residual statistics       9 features   (full_hybrid)
ViT-S/16 semantic encoder    384 features  (full_hybrid)
  -> concat, gated, or attention fusion
  -> MLP classifier
  -> calibrated probabilities and REAL / AI_GENERATED / UNCERTAIN decision
```

The physics branch measures bilateral corneal highlights, pupil geometry, iris entropy, light direction, shadow coverage, and illumination consistency. Fresnel reflectance is calculated as explainability metadata; it is not one of the 20 classifier features. PRNU is reference-free by default and therefore reports `limited` reliability unless a camera reference residual is supplied.

## Repository layout

```text
src/train.py                         Training entry point
src/inference.py                     Cached single-image inference
src/evaluate.py                      Metrics, plots, leakage report, ablations
src/deep_branch/                     EfficientNet, preprocessing, face alignment
src/physics_branch/                  dlib landmarks and physics/CV features
src/prnu_branch/                     Noise residual and PRNU features
src/semantic_branch/                 ViT embeddings and attention maps
src/fusion/                          Concat, gated, and attention fusion
src/classifier/                      Legacy and full hybrid classifiers
src/calibration/                     Temperature scaling and uncertainty
src/localization/                    Patch scoring and suspicious regions
src/explain/                         Grad-CAM and human-readable explanations
src/api/                             FastAPI routes and Pydantic responses
src/db/                              PostgreSQL/SQLAlchemy persistence
src/config/                          YAML configuration and Pydantic models
tests/                               Unit and integration tests
notebooks/                           Run 2 dataset and Colab training notebooks
dashboard/                           Static upload dashboard
docker/                              API container and Compose deployment
```

## Setup

From the project directory on Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest tests -v
```

The requirements file includes dlib, tqdm, PyTorch, timm, FastAPI, PostgreSQL support, scikit-learn, and plotting dependencies. Pretrained EfficientNet and ViT weights may be downloaded by timm on first use.

## Dataset layout

The preferred layout is an explicit held-out split:

```text
data/
  train/real/        train/fake/
  val/real/          val/fake/
  test/real/         test/fake/
```

Images may be nested below `fake/`, for example `fake/stylegan2/` and `fake/diffusion/`. The loader records the first nested directory as the source name. Supported extensions are `.jpg`, `.jpeg`, `.png`, and `.webp` (the Run 2 notebook also counts `.bmp`).

For legacy `stage1` and `hybrid` runs, `data/real` and `data/fake` are accepted and split deterministically into train/validation groups by label and source. A legacy layout cannot provide a safe held-out test split. The Run 2 preparation notebook creates balanced 5,000-real / 5,000-fake data from FFHQ, StyleGAN2, and DeepFakeFace diffusion sources, then creates 70/15/15 train/validation/test splits and reproducibility manifests.

## Training

Training is intended for a Google Colab or Kaggle CUDA GPU. `--allow-cpu` is available for smoke tests only.

```powershell
# Image-only baseline
python src/train.py --mode stage1 --data-dir data --epochs 10 --batch-size 32 --output-dir models

# EfficientNet + physics
python src/train.py --mode hybrid --data-dir data --epochs 15 --batch-size 32 --output-dir models

# Full Run 2 model; requires explicit train/val/test directories
python src/train.py --mode full_hybrid --data-dir data --output-dir models/run2_full_hybrid --epochs 15 --batch-size 16 --fusion-mode gated
```

Full hybrid training extracts all modalities for every sample, fits physics and PRNU normalizers on training images only, supports checkpoint resume, saves a last and best checkpoint each epoch, and evaluates the best checkpoint on the held-out test loader. Its default semantic model is `vit_small_patch16_224` with a 384-dimensional embedding. Fusion choices are `concat`, `gated`, and `attention`.

Important options include `--backbone`, `--freeze-blocks`, `--semantic-model`, `--semantic-dim`, `--fusion-mode`, `--no-pretrained`, `--no-semantic-pretrained`, `--epochs`, `--batch-size`, `--lr`, `--weight-decay`, `--val-ratio`, `--num-workers`, and `--seed`.

## Inference

```powershell
python src/inference.py path\to\face.jpg --checkpoint models\hybrid_best.pt
python src/inference.py path\to\face.jpg --checkpoint models\run2_full_hybrid\full_hybrid_best.pt --no-heatmap
```

Inference caches one model per checkpoint/device pair. It can align faces by default, or use `--no-align` for legacy full-frame preprocessing. Results can include probabilities, confidence, uncertainty, rejection reason, model mode, face status, physics features, PRNU status, fusion weights, explanation text, Grad-CAM, ViT attention, patch scores, and suspicious regions.

The API-facing label is `real`, `ai`, or `uncertain`; the detailed decision is `REAL`, `AI_GENERATED`, or `UNCERTAIN`. Temperature scaling is applied using the configured temperature. Uncertainty rejection uses entropy, probability margin, and confidence thresholds from `src/config/config.yaml`.

## Evaluation

```powershell
python src/evaluate.py --checkpoint models\hybrid_best.pt --data-dir data
python src/evaluate.py --checkpoint models\hybrid_best.pt --data-dir data --ablation
```

Evaluation writes `metrics.json`, `predictions.csv`, confusion/ROC/precision-recall plots, and `data_leakage_report.txt` under `evaluation_outputs/`. It reports accuracy, precision, recall, F1, specificity, ROC-AUC, confusion counts, and calibration error. SHA-256 comparison detects exact file duplicates between training and selected evaluation images, not visually similar images.

Current limitation: `evaluate.py` still calls the legacy two-input model path and has not been updated to evaluate `full_hybrid` checkpoints with PRNU and semantic tensors. Use the held-out test evaluation performed by `train.py` for full-hybrid runs until that evaluator path is extended.

## API and dashboard

```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Endpoints are mounted under `/api/v1`: `GET /health`, `POST /predict`, `GET /predictions`, and `GET /predictions/{prediction_id}`. The API validates file format, file size, and image dimensions, stores predictions through SQLAlchemy/PostgreSQL, and serves the static dashboard at `/dashboard/`.

Configure the database, model path, upload limit, CORS, logging, and inference thresholds in `src/config/config.yaml`. The application currently reads the YAML database URL rather than overriding it from `DATABASE_URL`.

```powershell
cd docker
docker compose up --build
```

## Known limitations and risks

- Training data and model checkpoints are not included by default.
- Training data is generator-specific; unseen-generator performance is not established.
- Physics, PRNU, ViT, and patch scoring are CPU/GPU intensive.
- PRNU is only a limited noise-statistics signal without reference camera images.
- Fresnel and shadow measurements are heuristic estimates, not calibrated simulators.
- The current `stage1` runner calls the full-hybrid epoch helper with stage1 batches; repair and test that path before relying on stage1 training.
- The evaluator has the full-hybrid compatibility gap described above.
- Checkpoint loading uses PyTorch pickle serialization; only load trusted checkpoints.
- Default CORS is permissive, authentication and rate limiting are absent, alerts are configuration-only, and uploads are not malware-scanned.

See [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) for the detailed implementation reference and [EVALUATION.md](EVALUATION.md) for evaluation guidance.
