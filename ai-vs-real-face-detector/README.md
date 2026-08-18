# AI vs Real Face Detector

Hybrid classifier combining **EfficientNet-B0**, **physics-based forensics** (dlib landmarks, Fresnel corneal analysis, iris/pupil, shadow geometry), optional **PRNU camera forensics**, and **ViT semantic features** with configurable fusion (concat / gated / attention).

## Architecture

```
Input Image
    ├── Preprocessing (face detect, align, crop, 224×224, ImageNet norm)
    ├── Deep Branch (EfficientNet-B0) ──► 1280-d embedding
    ├── Physics Branch (dlib + CV) ──► 20-d features (+ Fresnel/shadow metadata)
    ├── PRNU Branch (noise residual) ──► 9-d features [explainability; limited without reference]
    └── Semantic Branch (ViT-S/16) ──► 384-d embedding + attention map
                │
        Fusion (concat | gated | attention)
                │
          MLP classifier + calibration
                │
    REAL / AI_GENERATED / UNCERTAIN + explainability bundle
```

## Implemented vs experimental

| Feature | Status |
|---------|--------|
| EfficientNet + Physics hybrid (concat) | **Implemented** — `hybrid_best.pt` checkpoint |
| Face alignment before deep branch | **Implemented** — config `inference.align_face` |
| Fresnel reflection analysis | **Implemented** — metadata + explainability |
| Shadow/illumination geometry | **Implemented** |
| PRNU noise forensics | **Implemented** — reliability `limited` without reference camera |
| ViT semantic encoder | **Implemented** — explainability; classification requires `full_hybrid` training |
| Gated / attention fusion | **Implemented** — requires `full_hybrid` checkpoint |
| Patch localization heatmaps | **Implemented** |
| Probability calibration + UNCERTAIN | **Implemented** |
| Full ablation study | **Implemented** — `evaluate.py --ablation` |
| Full hybrid trained checkpoint | **Experimental** — train with `--mode full_hybrid` on Colab |

## Quick start (local)

```bash
cd ai-vs-real-face-detector
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
pytest tests/ -v
```

### Inference (requires `models/hybrid_best.pt`)

```bash
python src/inference.py path/to/face.jpg --checkpoint models/hybrid_best.pt
```

Use `--no-align` for legacy full-frame resize (matches original training).

### Evaluation

```bash
python src/evaluate.py --checkpoint models/hybrid_best.pt --data-dir data
python src/evaluate.py --ablation  # ablation comparison
```

### API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## Hardware note

Training requires **Google Colab / Kaggle GPU**. Local 8GB RAM is fine for single-image inference and tests.

## Project structure

- `src/deep_branch/` — preprocessing, face alignment, EfficientNet
- `src/physics_branch/` — landmarks, Fresnel, iris/pupil, shadows
- `src/prnu_branch/` — PRNU noise forensics
- `src/semantic_branch/` — ViT encoder + attention maps
- `src/fusion/` — concat, gated, attention fusion
- `src/localization/` — patch-level scoring
- `src/calibration/` — temperature scaling, uncertainty
- `src/explain/` — Grad-CAM, human-readable explanations
- `src/classifier/` — HybridClassifier, FullHybridClassifier, ablation configs

See `EVALUATION.md` and `PROJECT_DOCUMENTATION.md` for details.
