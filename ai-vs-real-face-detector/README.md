# AI vs Real Face Detector

Hybrid classifier that combines a **deep learning branch** (EfficientNet-B0 via `timm`) with a **physics-based forensic branch** (corneal specular highlights, iris/pupil analysis, light-direction consistency) to detect AI-generated vs real face portraits.

## Architecture

```
Input Image
    ├── Deep Branch (EfficientNet-B0) ──► embedding vector
    └── Physics Branch (MediaPipe + CV) ──► handcrafted vector
                    │
            Concatenate (fusion)
                    │
              MLP classifier
                    │
         label + confidence + Grad-CAM
```

## Hardware note

**Do not train on an 8GB RAM laptop without GPU.** Training scripts are written for **Google Colab / Kaggle GPU runtimes**. Locally you can run:

- Physics branch analysis and unit tests
- Single-image inference with a downloaded checkpoint
- API / DB / Docker infrastructure

## Project structure

See the repository layout under `src/`, `data/`, `models/`, `tests/`, `docker/`, and `notebooks/`.

## Quick start (local)

```bash
cd ai-vs-real-face-detector
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
pytest tests/ -v
```

### Physics branch on one image (debug)

```python
import cv2
from src.physics_branch.feature_vector import PhysicsFeatureExtractor

img = cv2.imread("path/to/portrait.jpg")
with PhysicsFeatureExtractor() as ext:
    result = ext.extract(img)
print(result.to_dict())
```

### Inference (requires trained checkpoint)

Train on Colab first, download `models/hybrid_best.pt`, then:

```bash
python src/inference.py path/to/face.jpg --checkpoint models/hybrid_best.pt
```

## Training (Colab / Kaggle only)

Open `notebooks/train_colab.ipynb` or upload the project and run:

```bash
# Stage 1 — deep branch baseline
python src/train.py --mode stage1 --data-dir data --epochs 10 --output-dir models

# Stage 3 — hybrid fusion training (after stage1 checkpoint exists)
python src/train.py --mode hybrid --data-dir data --epochs 15 --output-dir models
```

### Dataset layout

```
data/
  real/    # FFHQ subset
  fake/    # StyleGAN2/3 faces (add diffusion faces in pass 2)
```

**Pass 1:** FFHQ vs StyleGAN2  
**Pass 2:** Add diffusion faces (e.g. Purdue AI-Face-FairnessBench or Kaggle deepfake set)

## API & Docker (Stage 5)

```bash
cd docker
docker compose up --build
```

- API docs: http://localhost:8000/docs  
- Dashboard: http://localhost:8000/dashboard/

Place a trained checkpoint at `models/hybrid_best.pt` before starting the API.

## Outputs

Inference returns JSON:

```json
{
  "label": "ai",
  "confidence_score": 87.5,
  "probability_distribution": { "real": 12.5, "ai": 87.5 },
  "heatmap": "<base64 PNG>"
}
```

## Configuration

Edit `src/config/config.yaml` for thresholds, model paths, DB URL, and logging.

## References

- Hu, Li & Lyu (ICASSP 2021) — corneal specular highlight inconsistency in GAN faces  
- Reference implementation: [discovershu/gan_detect_iris](https://github.com/discovershu/gan_detect_iris)

## License

MIT (add your license as needed)
