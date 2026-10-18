

from __future__ import annotations
from tqdm import tqdm
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import cv2
cv2.setNumThreads(1)

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score


# ============================================================
# EARLY STOPPING
# ============================================================

class EarlyStopping:
    """Stop when the monitored validation metric fails to improve for N epochs."""

    def __init__(self, patience: int = 5, best_score: float = -1.0) -> None:
        if patience < 1:
            raise ValueError("Early-stopping patience must be at least 1.")
        self.patience = patience
        self.best_score = best_score
        self.bad_epochs = 0

    def step(self, score: float, epoch: int) -> bool:
        # The training code uses validation accuracy for best-checkpoint selection,
        # so early stopping monitors the same metric for consistent behavior.
        if score > self.best_score:
            self.best_score = score
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        print(
            f"[EarlyStopping] No validation-accuracy improvement for "
            f"{self.bad_epochs}/{self.patience} epoch(s)."
        )
        if self.bad_epochs >= self.patience:
            print(
                f"[EarlyStopping] Stopping after epoch {epoch}; "
                f"best validation accuracy={self.best_score:.4f}."
            )
            return True
        return False


# ============================================================
# PROJECT PATH
# ============================================================

# src/train.py
#     ↑
# parents[0] = src
# parents[1] = project root

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.classifier.head import BranchOnlyClassifier, FullHybridClassifier, HybridClassifier
from src.deep_branch.feature_extractor import DeepClassifier
from src.deep_branch.preprocessing import (
    FacePreprocessor,
    get_train_transforms,
    get_val_transforms,
)
from src.physics_branch.feature_vector import (
    PhysicsFeatureExtractor,
    PHYSICS_FEATURE_DIM,
)
from src.physics_branch.normalization import PhysicsNormalizer
from src.prnu_branch.extractor import PRNUExtractor, PRNU_FEATURE_DIM
from src.semantic_branch.encoder import DEFAULT_VIT_MODEL, SemanticEncoder
from src.fusion.fuse import FusionMode
from src.eval.cache_meta import check_cache_meta, requested_meta_from_args
from src.eval.splits import DEFAULT_SPLITS_PATH, filter_samples, load_split_spec
from src.eval.train_select import parse_seed_list, save_epoch_snapshot


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible training."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        # Deterministic behavior where possible.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# GPU
# ============================================================

def require_gpu(allow_cpu: bool = False) -> torch.device:
    """Select MPS or CUDA for training; permit CPU only for smoke tests."""

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        device = torch.device("mps")
        print("=" * 60)
        print("ACCELERATOR INFORMATION")
        print("=" * 60)
        print("Device:", device)
        print("MPS available:", mps_backend.is_available())
        print("MPS built:", mps_backend.is_built())
        print("=" * 60)
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("=" * 60)
        print("ACCELERATOR INFORMATION")
        print("=" * 60)
        print("Device:", device)
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA:", torch.version.cuda)
        print("=" * 60)
        return device

    if not allow_cpu:
        raise RuntimeError(
            "No MPS or CUDA accelerator available.\n"
            "Run this script on Apple Silicon, Google Colab, or Kaggle with an accelerator enabled.\n"
            "Do NOT train on CPU without --allow-cpu."
        )

    device = torch.device("cpu")
    print("WARNING: CPU mode enabled.")
    print("This should ONLY be used for smoke tests.")
    return device


# ============================================================
# DATASET
# ============================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CNN_REAL_DIRNAMES = {"real", "0_real"}
CNN_FAKE_DIRNAMES = {"fake", "1_fake"}
PHYSICS_MODES = {"hybrid", "full_hybrid", "physics_only"}
PRNU_MODES = {"full_hybrid", "prnu_only"}
SEMANTIC_MODES = {"full_hybrid", "semantic_only"}
NEEDS_TEST_LOADER = {"stage1", "full_hybrid", "physics_only", "prnu_only", "semantic_only"}

# Set from --fast in main(): bf16 autocast for the EfficientNet forward passes.
AMP_ENABLED = False


def _amp_ctx():
    return torch.autocast(
        device_type="cuda", dtype=torch.bfloat16,
        enabled=AMP_ENABLED and torch.cuda.is_available(),
    )


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _label_from_dirname(name: str) -> Optional[int]:
    if name in CNN_REAL_DIRNAMES:
        return 0
    if name in CNN_FAKE_DIRNAMES:
        return 1
    return None


def _has_explicit_split_layout(root: Path) -> bool:
    """True when data/{train,val,test}/{real,fake} exist as direct directories."""
    return all(
        (root / split / label).is_dir()
        for split in ("train", "val", "test")
        for label in ("real", "fake")
    )


def _has_explicit_label_dirs(split_root: Path) -> bool:
    """True for the normal split_root/{real,fake} layout (no recursion)."""
    return (split_root / "real").is_dir() and (split_root / "fake").is_dir()


def _split_has_direct_label_dirs(folder: Path) -> bool:
    return any(
        (folder / name).is_dir()
        for name in (*CNN_REAL_DIRNAMES, *CNN_FAKE_DIRNAMES)
    )


def _split_has_nested_cnndetection_labels(folder: Path) -> bool:
    """Walk once looking for 0_real/1_fake (or real/fake) anywhere under folder."""
    for _current_root, dirs, _files in os.walk(folder, followlinks=False):
        for name in dirs:
            if name in CNN_REAL_DIRNAMES or name in CNN_FAKE_DIRNAMES:
                return True
    return False


def _split_has_labels(root: Path, split: str) -> bool:
    folder = root / split
    if not folder.is_dir():
        return False
    if _split_has_direct_label_dirs(folder):
        return True
    return _split_has_nested_cnndetection_labels(folder)


def _source_from_relative(relative: Path, default: str) -> str:
    parts = [p for p in relative.parts if _label_from_dirname(p) is None]
    if not parts:
        return default
    return "/".join(parts)


def _posix_relpath(path: str, start: str) -> str:
    rel = os.path.relpath(path, start)
    if rel in (".", os.curdir):
        return ""
    return rel.replace("\\", "/")


def _collect_under_label_root(
    folder: Path,
    label: int,
    default: str,
    samples: List[Tuple[str, int, str]],
    seen: set,
) -> None:
    """Recursively collect images from one real/ or fake/ root only."""
    folder_str = str(folder)
    directories_processed = 0
    files_processed = 0
    for current_root, dirs, files in os.walk(folder_str, followlinks=False):
        directories_processed += 1
        dirs.sort()
        files.sort()
        rel_parent = _posix_relpath(current_root, folder_str)
        source = rel_parent or default
        for filename in files:
            files_processed += 1
            if files_processed % 10_000 == 0:
                print(
                    f"[enumeration] processed {files_processed:,} files across "
                    f"{directories_processed:,} directories; found {len(samples):,} images",
                    flush=True,
                )
            suffix = os.path.splitext(filename)[1].lower()
            if suffix not in IMAGE_EXTENSIONS:
                continue
            path = os.path.join(current_root, filename)
            if path in seen:
                continue
            seen.add(path)
            samples.append((path, label, source))
        if directories_processed % 100 == 0:
            print(
                f"[enumeration] processed {directories_processed:,} directories; "
                f"found {len(samples):,} images",
                flush=True,
            )


def _collect_cnndetection_images(split_root: Path) -> List[Tuple[str, int, str]]:
    """Single-walk collection for nested 0_real/1_fake (CNNDetection) trees."""
    samples: List[Tuple[str, int, str]] = []
    seen = set()
    split_root_str = str(split_root)
    directories_processed = 0
    files_processed = 0

    for current_root, dirs, files in os.walk(split_root_str, followlinks=False):
        directories_processed += 1
        dirs.sort()
        files.sort()
        if directories_processed % 100 == 0:
            print(
                f"[enumeration] processed {directories_processed:,} directories; "
                f"found {len(samples):,} images",
                flush=True,
            )
        rel = _posix_relpath(current_root, split_root_str)
        parts = Path(rel).parts if rel else ()
        label_index = None
        label = None
        for i, part in enumerate(parts):
            part_label = _label_from_dirname(part)
            if part_label is not None:
                label_index = i
                label = part_label
                break
        if label is None:
            root_label = _label_from_dirname(split_root.name)
            if root_label is not None and not rel:
                label = root_label
                label_index = -1
            else:
                continue

        default = "lsun" if label == 0 else "synthetic"
        if label_index == -1:
            source_from_parent = default
            nested_after_label = rel
        else:
            parent_rel = "/".join(parts[:label_index])
            source_from_parent = _source_from_relative(
                Path(parent_rel) if parent_rel else Path(), default
            )
            nested_after_label = "/".join(parts[label_index + 1 :])
        item_source = nested_after_label or source_from_parent or default

        for filename in files:
            files_processed += 1
            if files_processed % 10_000 == 0:
                print(
                    f"[enumeration] processed {files_processed:,} files across "
                    f"{directories_processed:,} directories; found {len(samples):,} images",
                    flush=True,
                )
            suffix = os.path.splitext(filename)[1].lower()
            if suffix not in IMAGE_EXTENSIONS:
                continue
            path = os.path.join(current_root, filename)
            if path in seen:
                continue
            seen.add(path)
            samples.append((path, label, item_source))
    return samples


def collect_labeled_images(split_root: Path) -> List[Tuple[str, int, str]]:
    """Collect (path, label, source) from real/fake or CNNDetection 0_real/1_fake trees."""
    samples: List[Tuple[str, int, str]] = []
    seen = set()
    if not split_root.is_dir():
        return samples

    if _has_explicit_label_dirs(split_root):
        _collect_under_label_root(split_root / "real", 0, "lsun", samples, seen)
        _collect_under_label_root(split_root / "fake", 1, "synthetic", samples, seen)
        return samples

    return _collect_cnndetection_images(split_root)


def process_face_sample(
    rgb: np.ndarray,
    label: int,
    source: str,
    preprocessor: Optional[FacePreprocessor] = None,
    transform: Optional[Any] = None,
    physics_extractor: Optional[Any] = None,
    prnu_extractor: Optional[Any] = None,
    semantic_extractor: Optional[Any] = None,
    use_physics: bool = False,
    use_prnu: bool = False,
    use_semantic: bool = False,
    cached: Optional[Dict[str, np.ndarray]] = None,
    use_freq: bool = False,
    path: str = "",
    epoch: int = 0,
    seed: int = 42,
    enable_augment: bool = False,
    preprocess_mode: str = "crop",
    crop_size: int = 224,
) -> dict:
    from src.augment import augment_raw_image
    from src.preprocess import PreprocessPipeline

    # Request 2: With feature cache attached, use deterministic per-path seeding (epoch=0)
    # so live deep branch sees the exact same augmented pixels as cached auxiliary features.
    # With NO feature cache attached, draw fresh random blur/JPEG per epoch.
    if enable_augment:
        has_cache = (cached is not None)
        rgb, crop_rng = augment_raw_image(
            rgb,
            path=path if has_cache else "",
            epoch=0 if has_cache else epoch,
            base_seed=seed,
            enable_augment=True,
            deterministic=has_cache,
        )
    else:
        crop_rng = None


    # Preprocessing: crop or resize to 224x224
    if preprocessor is not None:
        pil = preprocessor.preprocess_pil(Image.fromarray(rgb))
        processed_rgb = np.array(pil)
    else:
        pipeline = PreprocessPipeline(
            mode=preprocess_mode, crop_size=crop_size, is_training=enable_augment
        )
        processed_rgb = pipeline.preprocess_image(rgb, rng=crop_rng)
        pil = Image.fromarray(processed_rgb)

    # Branches extract from the consistent processed_rgb
    if use_physics:
        if cached is not None and "physics" in cached:
            physics_vec = cached["physics"]
        else:
            if physics_extractor is None:
                raise RuntimeError("Physics extractor is unavailable.")
            # Use shared unresized center 224x224 crop for physics branch to ensure consistency with cached features.
            from src.preprocess import preprocess_for_freq
            physics_vec = physics_extractor.extract(preprocess_for_freq(rgb)).vector
    else:
        physics_vec = np.zeros(PHYSICS_FEATURE_DIM, dtype=np.float32)

    if use_prnu:
        if cached is not None and "prnu" in cached:
            prnu_vec = cached["prnu"]
        else:
            if prnu_extractor is None:
                raise RuntimeError("PRNU extractor is unavailable.")
            prnu_vec = prnu_extractor.extract(processed_rgb).vector
    else:
        prnu_vec = None

    if use_semantic:
        if cached is not None and "semantic" in cached:
            semantic_vec = cached["semantic"]
        else:
            if semantic_extractor is None:
                raise RuntimeError("Semantic extractor is unavailable.")
            semantic_vec = semantic_extractor.extract(
                processed_rgb, return_attention=False
            ).features
    else:
        semantic_vec = None

    sample = {
        "image": transform(pil),
        "label": torch.tensor(label, dtype=torch.long),
        "physics": torch.from_numpy(np.asarray(physics_vec, dtype=np.float32)),
        "source": source,
    }
    if prnu_vec is not None:
        sample["prnu"] = torch.from_numpy(np.asarray(prnu_vec, dtype=np.float32))
    if semantic_vec is not None:
        sample["semantic"] = torch.from_numpy(np.asarray(semantic_vec, dtype=np.float32))
    if use_freq:
        if cached is not None and cached.get("freq") is not None:
            freq_vec = cached["freq"]
        else:
            from src.freq_branch.extractor import FreqExtractor
            from src.preprocess import preprocess_for_freq
            freq_vec = np.asarray(FreqExtractor().extract(preprocess_for_freq(rgb)).vector, dtype=np.float32)
        sample["freq"] = torch.from_numpy(np.asarray(freq_vec, dtype=np.float32))
    return sample


class FaceBinaryDataset(Dataset):
    """
    Dataset for REAL vs AI-GENERATED images (faces and general scenes).

    Supported directory structures:

    data/{train,val,test}/{real,fake}/...

    CNNDetection / ForenSynths layout:

    data/{train,val,test}/{category}/0_real/
    data/{train,val,test}/{category}/1_fake/

    Nested test generators:

    data/test/{generator}/{class}/0_real
    data/test/{generator}/{class}/1_fake

    Labels:

        0 = REAL
        1 = FAKE / AI-GENERATED
    """

    EXTENSIONS = IMAGE_EXTENSIONS

    def __init__(
        self,
        root: str,
        split: str = "train",
        val_ratio: float = 0.15,
        seed: int = 42,
        use_physics: bool = False,
        use_prnu: bool = False,
        use_semantic: bool = False,
        use_freq: bool = False,
        semantic_model: str = DEFAULT_VIT_MODEL,
        semantic_pretrained: bool = True,
        preprocessor: Optional[FacePreprocessor] = None,
        transform=None,
        feature_cache_dir: Optional[str] = None,
        folder_filter: Optional[List[str]] = None,
        legacy_cache: bool = False,
        cache_requested_meta: Optional[Dict[str, Any]] = None,
    ) -> None:

        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"Invalid split '{split}'. Expected 'train', 'val', or 'test'."
            )

        self.root = Path(root)
        self.use_physics = use_physics
        self.use_prnu = use_prnu
        self.use_semantic = use_semantic
        self.use_freq = use_freq
        self.semantic_model = semantic_model
        self.semantic_pretrained = semantic_pretrained

        self.preprocessor = (
            preprocessor
            if preprocessor is not None
            else FacePreprocessor()
        )

        self.transform = transform
        self.folder_filter = folder_filter
        self.legacy_cache = legacy_cache
        self.cache_requested_meta = cache_requested_meta
        self._cache_row = None
        self._cache_arrays = None
        # Six direct directory checks only. Nested CNNDetection discovery is
        # used later, and only when this explicit layout is absent.
        self.explicit_split_layout = _has_explicit_split_layout(self.root)
        if not self.explicit_split_layout:
            self.explicit_split_layout = all(
                _split_has_labels(self.root, split_name)
                for split_name in ("train", "val", "test")
            )
        if split == "test" and not self.explicit_split_layout:
            raise FileNotFoundError(
                "Held-out testing requires data/{train,val,test} with real/fake "
                "(or CNNDetection 0_real/1_fake) folders. "
                "A legacy data/{real,fake} layout has no safe test split."
            )

        # These extractors return real forensic/semantic measurements.  A
        # failure is deliberately propagated: full_hybrid must never train on
        # fabricated placeholder modalities.
        self.physics_extractor = (
            PhysicsFeatureExtractor()
            if use_physics
            else None
        )
        self.prnu_extractor = PRNUExtractor() if self.use_prnu else None
        self.semantic_extractor = None

        # ----------------------------------------------------
        # Find all images recursively
        # ----------------------------------------------------

        samples: List[Tuple[str, int, str]] = []

        if self.explicit_split_layout:
            file_list_path = (
                Path(feature_cache_dir) / f"{split}_file_list.json"
                if feature_cache_dir
                else None
            )
            if file_list_path is not None and file_list_path.exists():
                print(f"Loading cached file list for {split}...", flush=True)
                samples = [
                    tuple(sample)
                    for sample in json.loads(
                        file_list_path.read_text(encoding="utf-8")
                    )
                ]
            else:
                samples = collect_labeled_images(self.root / split)
        else:
            for label, subdir in ((0, "real"), (1, "fake")):
                folder = self.root / subdir
                if not folder.exists():
                    print(f"WARNING: Directory does not exist: {folder}")
                    continue
                for path in sorted(folder.rglob("*")):
                    if not path.is_file() or path.suffix.lower() not in self.EXTENSIONS:
                        continue
                    relative_path = path.relative_to(folder)
                    source = relative_path.parts[0] if len(relative_path.parts) > 1 else subdir
                    samples.append((str(path), label, source))

        # ----------------------------------------------------
        # Validate dataset
        # ----------------------------------------------------

        if not samples:
            raise FileNotFoundError(
                f"No images found under:\n"
                f"  {self.root / 'real'}\n"
                f"  {self.root / 'fake'}\n\n"
                f"Expected images with extensions:\n"
                f"  {', '.join(sorted(self.EXTENSIONS))}"
            )

        # ----------------------------------------------------
        # Show total dataset composition
        # ----------------------------------------------------

        total_counts: Dict[Tuple[int, str], int] = defaultdict(int)

        for _, label, source in samples:
            total_counts[(label, source)] += 1

        print("\nTotal dataset composition:")

        for (label, source), count in sorted(total_counts.items()):
            label_name = "real" if label == 0 else "fake"

            print(
                f"  {label_name}/{source}: {count}"
            )

        print(f"  TOTAL: {len(samples)}")

        # ----------------------------------------------------
        # Explicit held-out layout (preferred)
        # ----------------------------------------------------
        if self.explicit_split_layout:
            self.samples = samples
        else:
            # ----------------------------------------------------
            # Legacy stratified train/validation split.  Kept for existing
            # stage1/hybrid commands; it is never used for held-out testing.
        # ----------------------------------------------------
        #
        # We split separately for:
        #
        #   real/real
        #   fake/stylegan2
        #   fake/diffusion
        #
        # This prevents one fake generator from accidentally
        # disappearing from validation.
        # ----------------------------------------------------

            groups: Dict[
            Tuple[int, str],
            List[Tuple[str, int, str]]
        ] = defaultdict(list)

            for sample in samples:

                _, label, source = sample

                groups[(label, source)].append(sample)

            train_samples: List[
            Tuple[str, int, str]
        ] = []

            val_samples: List[
            Tuple[str, int, str]
        ] = []

            rng = random.Random(seed)

            for (label, source), group in sorted(groups.items()):

                group = group.copy()

                rng.shuffle(group)

            # Calculate train split.
                split_idx = int(
                    len(group) * (1.0 - val_ratio)
                )

            # Guarantee at least one validation sample
            # when the group contains at least two images.
                if len(group) > 1:

                    split_idx = min(
                        max(split_idx, 1),
                        len(group) - 1,
                    )

                else:

                    split_idx = len(group)

                train_samples.extend(
                    group[:split_idx]
                )

                val_samples.extend(
                    group[split_idx:]
                )

        # Shuffle final datasets.
            rng.shuffle(train_samples)
            rng.shuffle(val_samples)

            if split == "train":
                self.samples = train_samples
            else:
                self.samples = val_samples

        if self.folder_filter:
            self.samples = filter_samples(self.samples, self.folder_filter)
            if not self.samples:
                raise FileNotFoundError(
                    f"No images remained after applying folder_filter={self.folder_filter!r} "
                    f"on split {split!r}."
                )

        # ----------------------------------------------------
        # Print split composition
        # ----------------------------------------------------

        print(
            f"\n{split.upper()} split:"
        )

        split_counts: Dict[
            Tuple[int, str],
            int
        ] = defaultdict(int)

        for _, label, source in self.samples:

            split_counts[
                (label, source)
            ] += 1

        for (label, source), count in sorted(
            split_counts.items()
        ):

            label_name = (
                "real"
                if label == 0
                else "fake"
            )

            print(
                f"  {label_name}/{source}: {count}"
            )

        print(
            f"  TOTAL: {len(self.samples)}"
        )

        cache_dir = feature_cache_dir or os.environ.get("FEATURE_CACHE_DIR")
        if cache_dir and (use_physics or use_prnu or use_semantic or use_freq):
            self.attach_feature_cache(cache_dir, split)

    def attach_feature_cache(self, cache_dir: str, split: str) -> None:
        """Serve physics/PRNU/semantic vectors from cache_features.py output."""
        cache_dir = Path(cache_dir)
        requested = {"semantic_model": self.semantic_model} if self.use_semantic else {}
        extra = getattr(self, "cache_requested_meta", None)
        if extra:
            requested.update(extra)
        check_cache_meta(cache_dir, requested, legacy=self.legacy_cache)
        z = np.load(cache_dir / f"{split}.npz")
        arrays = {k: z[k] for k in z.files if k in {"physics", "prnu", "semantic", "freq"}}
        row = {str(p): i for i, p in enumerate(z["paths"])}
        kept = [s for s in self.samples if s[0] in row]
        if len(kept) != len(self.samples):
            print(f"  [feature cache] {split}: using {len(kept)}/{len(self.samples)} "
                  f"images that are present in the cache")
        if not kept:
            raise RuntimeError(f"Feature cache has none of the {split} images.")
        self.samples = kept
        self._cache_row = row
        self._cache_arrays = arrays
        # Extractors are not needed any more; free them.
        if self.physics_extractor is not None:
            try:
                self.physics_extractor.close()
            except Exception:
                pass
        self.physics_extractor = None
        self.prnu_extractor = None
        print(f"  [feature cache] {split}: serving features from {cache_dir}")

    # --------------------------------------------------------
    # Dataset length
    # --------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    # --------------------------------------------------------
    # Get sample
    # --------------------------------------------------------

    def __getitem__(self, idx: int):

        path, label, source = self.samples[idx]

        # ----------------------------------------------------
        # Load image with OpenCV
        # ----------------------------------------------------

        import cv2

        bgr = cv2.imread(path)

        if bgr is None:
            raise FileNotFoundError(
                f"Could not read image: {path}"
            )

        rgb = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2RGB,
        )

        cached = None
        if self._cache_row is not None:
            row = self._cache_row[path]
            cached = {
                key: self._cache_arrays[key][row]
                for key in self._cache_arrays
            }
        if self.use_semantic and cached is None and self.semantic_extractor is None:
            # Lazy construction avoids loading a ViT in stage1/hybrid modes
            # and ensures each DataLoader worker owns its encoder safely.
            self.semantic_extractor = SemanticEncoder(
                model_name=self.semantic_model,
                pretrained=self.semantic_pretrained,
                device=torch.device("cpu"),
            )
        epoch = getattr(self, "epoch", 0)
        enable_augment = getattr(self, "enable_augment", False)
        preprocess_mode = getattr(self, "preprocess_mode", "crop")
        sample = process_face_sample(
            rgb, label, source, self.preprocessor, self.transform,
            self.physics_extractor, self.prnu_extractor, self.semantic_extractor,
            self.use_physics, self.use_prnu, self.use_semantic, cached=cached,
            use_freq=self.use_freq,
            path=path,
            epoch=epoch,
            seed=getattr(self, "seed", 42),
            enable_augment=enable_augment,
            preprocess_mode=preprocess_mode,
        )
        sample["path"] = path
        return sample



class StreamingFaceBinaryDataset(IterableDataset):
    """Stream labeled WebDataset shards from the Hugging Face Hub.

    Feature caching is deliberately not supported here: cache entries are
    currently indexed by local path, while WebDataset samples have no stable
    local filename.
    """

    IMAGE_KEYS = ("jpg", "jpeg", "png", "webp")
    STREAM_RETRY_ATTEMPTS = 5
    STREAM_RETRY_INITIAL_BACKOFF_SECONDS = 2

    def __init__(
        self,
        repo_id: str,
        split: str,
        seed: int,
        use_physics: bool,
        use_prnu: bool,
        use_semantic: bool,
        semantic_model: str,
        semantic_pretrained: bool,
        preprocessor: FacePreprocessor,
        transform,
    ) -> None:
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split {split!r}; expected train, val, or test.")
        if not repo_id:
            raise ValueError("--hf-dataset-repo is required with --data-source hf-stream.")

        self.split = split
        self.repo_id = repo_id
        self.seed = seed
        self.use_physics = use_physics
        self.use_prnu = use_prnu
        self.use_semantic = use_semantic
        self.semantic_model = semantic_model
        self.semantic_pretrained = semantic_pretrained
        self.preprocessor = preprocessor
        self.transform = transform
        self._stream = self._build_stream()

        # Constructed by the worker that consumes the data, not in the parent
        # process. This is important for the ViT and for DataLoader workers.
        self.physics_extractor = None
        self.prnu_extractor = None
        self.semantic_extractor = None
        self._extractor_worker_id = None

    def _build_stream(self):
        """Create the deterministic source stream used for initial reads and retries."""
        # Keep this import local so the established local-folder workflow can
        # still import train.py before optional streaming dependencies install.
        import datasets

        stream = datasets.load_dataset(
            "webdataset",
            data_files={
                self.split: (
                    f"hf://datasets/{self.repo_id}/{self.split}/{self.split}-*.tar"
                )
            },
            streaming=True,
        )[self.split]
        if self.split == "train":
            stream = stream.shuffle(buffer_size=2000, seed=self.seed)
        return stream

    def reset_stream(self) -> None:
        """Restart the deterministic source stream after a pre-training pass."""
        self._stream = self._build_stream()

    @staticmethod
    def _stream_retryable_errors() -> Tuple[type[BaseException], ...]:
        """Return network exceptions emitted by supported datasets streaming stacks."""
        errors: List[type[BaseException]] = [ConnectionError, TimeoutError]

        # Older datasets/huggingface_hub releases use requests + urllib3.
        try:
            from requests.exceptions import RequestException
            from urllib3.exceptions import HTTPError as Urllib3HTTPError

            errors.extend((RequestException, Urllib3HTTPError))
        except ImportError:
            pass

        # Current datasets retries these transport exceptions internally when
        # opening remote files, but they can still escape during iteration.
        try:
            import asyncio
            import httpx

            errors.extend((asyncio.TimeoutError, httpx.RequestError))
        except ImportError:
            pass
        try:
            from aiohttp.client_exceptions import ClientError

            errors.append(ClientError)
        except ImportError:
            pass
        try:
            from fsspec.exceptions import FSTimeoutError

            errors.append(FSTimeoutError)
        except ImportError:
            pass
        return tuple(errors)

    def _restart_stream_after_failure(self, yielded_samples: int):
        """Recreate a failed iterable and advance it to the last yielded sample."""
        stream = self._build_stream()
        if yielded_samples:
            stream = stream.skip(yielded_samples)
        self._stream = stream
        return iter(stream)

    def _ensure_extractors(self) -> None:
        from torch.utils.data import get_worker_info

        worker = get_worker_info()
        worker_id = worker.id if worker is not None else -1
        if self._extractor_worker_id == worker_id:
            return
        if self.physics_extractor is not None:
            try:
                self.physics_extractor.close()
            except Exception:
                pass
        self.physics_extractor = PhysicsFeatureExtractor() if self.use_physics else None
        self.prnu_extractor = PRNUExtractor() if self.use_prnu else None
        self.semantic_extractor = (
            SemanticEncoder(
                model_name=self.semantic_model,
                pretrained=self.semantic_pretrained,
                device=torch.device("cpu"),
            )
            if self.use_semantic
            else None
        )
        self._extractor_worker_id = worker_id

    @classmethod
    def _decode_sample(cls, item) -> Tuple[np.ndarray, int, str]:
        img_value = next((item[key] for key in cls.IMAGE_KEYS if key in item), None)
        if img_value is None:
            raise KeyError("WebDataset sample has no jpg/jpeg/png/webp image payload.")

        if isinstance(img_value, Image.Image):
            # datasets' webdataset builder auto-decodes recognized image
            # extensions (png/jpg/webp) into PIL Images rather than raw bytes.
            rgb = np.array(img_value.convert("RGB"))
        else:
            bgr = cv2.imdecode(np.frombuffer(img_value, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("Could not decode streamed WebDataset image.")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        metadata = item.get("json")
        if metadata is None:
            raise KeyError("WebDataset sample has no json metadata payload.")
        if isinstance(metadata, (bytes, bytearray)):
            metadata = metadata.decode("utf-8")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        if not isinstance(metadata, dict) or "label" not in metadata or "source" not in metadata:
            raise ValueError("WebDataset json metadata must contain label and source.")
        return rgb, int(metadata["label"]), str(metadata["source"])

    def iter_decoded_samples(self) -> Iterator[Tuple[np.ndarray, int, str]]:
        """Yield raw RGB samples for streaming-only preprocessing such as scaler fitting."""
        stream_iterator = iter(self._stream)
        yielded_samples = 0
        retry_attempt = 0
        retryable_errors = self._stream_retryable_errors()

        while True:
            try:
                item = next(stream_iterator)
            except StopIteration:
                return
            except retryable_errors as exc:
                retry_attempt += 1
                if retry_attempt > self.STREAM_RETRY_ATTEMPTS:
                    raise

                delay = self.STREAM_RETRY_INITIAL_BACKOFF_SECONDS * (2 ** (retry_attempt - 1))
                print(
                    "WARNING: Hugging Face stream fetch failed "
                    f"({type(exc).__name__}: {exc}). Retrying in {delay}s "
                    f"[{retry_attempt}/{self.STREAM_RETRY_ATTEMPTS}]...",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
                stream_iterator = self._restart_stream_after_failure(yielded_samples)
                continue

            retry_attempt = 0
            yielded_samples += 1
            yield self._decode_sample(item)

    def __iter__(self) -> Iterator[dict]:
        self._ensure_extractors()
        for rgb, label, source in self.iter_decoded_samples():
            yield process_face_sample(
                rgb, label, source, self.preprocessor, self.transform,
                self.physics_extractor, self.prnu_extractor, self.semantic_extractor,
                self.use_physics, self.use_prnu, self.use_semantic,
                use_freq=getattr(self, "use_freq", False),
            )


# ============================================================
# DATA LOADERS
# ============================================================

def subsample_balanced(samples, per_class: int, seed: int):
    """Keep at most `per_class` images per label (0=real, 1=fake), chosen at random.

    Random choice keeps the mix of LSUN categories / generators roughly
    proportional to the full training set.
    """
    rng = random.Random(seed)
    out = []
    for label in (0, 1):
        group = [s for s in samples if s[1] == label]
        rng.shuffle(group)
        out.extend(group[:per_class])
    rng.shuffle(out)
    return out


def _loader_extras(args, persistent: bool = False) -> dict:
    if args.num_workers <= 0:
        return {}
    extras = {"prefetch_factor": 4}
    if persistent:
        extras["persistent_workers"] = True
    return extras


def _known_loader_batches(args, split: str) -> int:
    """Return a safe batch count for local or length-less streaming loaders."""
    if args.data_source == "hf-stream":
        known_sizes = {
            "train": args.known_train_size,
            "val": args.known_val_size,
            "test": args.known_test_size,
        }
        try:
            known_size = known_sizes[split]
        except KeyError as exc:
            raise ValueError(f"Unknown streamed split: {split!r}") from exc
        return known_size // args.batch_size
    raise ValueError("Known batch counts are only needed for hf-stream loaders.")


def _loader_batches(loader, args, split: str) -> int:
    return _known_loader_batches(args, split) if args.data_source == "hf-stream" else len(loader)


def fit_streaming_feature_scalers(
    dataset: StreamingFaceBinaryDataset,
    max_samples: int = 20_000,
    total_samples: Optional[int] = None,
) -> Tuple[PhysicsNormalizer, PhysicsNormalizer]:
    """Fit full-hybrid scalers without requiring local image paths."""
    from src.prnu_branch.extractor import PRNU_FEATURE_NAMES

    if max_samples < 0:
        raise ValueError("max_samples must be >= 0 (0 means all training images).")

    physics_extractor = PhysicsFeatureExtractor()
    prnu_extractor = PRNUExtractor()
    physics_vectors = []
    prnu_vectors = []
    try:
        for rgb, _, _ in tqdm(
            dataset.iter_decoded_samples(),
            total=max_samples if max_samples else total_samples,
            desc="Fitting streamed physics/PRNU scalers",
            unit="image",
            dynamic_ncols=True,
        ):
            physics_vectors.append(physics_extractor.extract(rgb).vector)
            prnu_vectors.append(prnu_extractor.extract(rgb).vector)
            if max_samples and len(physics_vectors) >= max_samples:
                break
    finally:
        physics_extractor.close()
        # Scaler fitting is a pre-training pass. Reset the iterable so the
        # training DataLoader starts from the first deterministic sample and
        # therefore still sees the complete training split.
        dataset.reset_stream()
    if not physics_vectors:
        raise RuntimeError("No streamed training samples were available to fit feature scalers.")
    return (
        PhysicsNormalizer.fit(np.stack(physics_vectors, axis=0)),
        PhysicsNormalizer.fit(
            np.stack(prnu_vectors, axis=0), feature_names=list(PRNU_FEATURE_NAMES)
        ),
    )


def build_loaders(args, device: torch.device, folder_filter: Optional[List[str]] = None, split: Optional[str] = None):

    preprocessor = FacePreprocessor()
    streaming = args.data_source == "hf-stream"
    dataset_discovery_seconds = 0.0

    def make_dataset(split_name: str, transform, folders=None):
        branches_str = getattr(args, "branches", "deep,physics,semantic,freq")
        if getattr(args, "legacy", False) and getattr(args, "branches", None) is None:
            branches_str = "deep,physics,prnu,semantic"
        branch_set = {b.strip().lower() for b in branches_str.split(",")} if isinstance(branches_str, str) else set(branches_str)
        common = dict(
            use_physics="physics" in branch_set if args.mode == "full_hybrid" else (args.mode in PHYSICS_MODES),
            use_prnu="prnu" in branch_set if args.mode == "full_hybrid" else (args.mode in PRNU_MODES),
            use_semantic="semantic" in branch_set if args.mode == "full_hybrid" else (args.mode in SEMANTIC_MODES),
            use_freq="freq" in branch_set if args.mode == "full_hybrid" else (args.mode == "freq_only"),
            semantic_model=args.semantic_model,
            semantic_pretrained=not args.no_semantic_pretrained,
            preprocessor=preprocessor,
            transform=transform,
        )
        if streaming:
            return StreamingFaceBinaryDataset(
                repo_id=args.hf_dataset_repo,
                split=split_name,
                seed=args.seed,
                **common,
            )
        return FaceBinaryDataset(
            root=args.data_dir,
            split=split_name,
            val_ratio=args.val_ratio,
            seed=args.seed,
            feature_cache_dir=args.feature_cache,
            folder_filter=folders,
            legacy_cache=bool(getattr(args, "legacy", False)),
            cache_requested_meta=requested_meta_from_args(args),
            **common,
        )

    def make_timed_dataset(split_name: str, transform, folders=None):
        nonlocal dataset_discovery_seconds
        started_at = time.perf_counter()
        dataset = make_dataset(split_name, transform, folders=folders)
        elapsed = time.perf_counter() - started_at
        dataset_discovery_seconds += elapsed
        print(f"[dataset] {split_name} discovery: {elapsed:.1f} sec")
        return dataset

    if split is not None:
        ds = make_timed_dataset(split, get_val_transforms(), folders=folder_filter)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            **_loader_extras(args),
        )
        return loader

    # --------------------------------------------------------
    # Training dataset
    # --------------------------------------------------------

    train_ds = make_timed_dataset("train", get_train_transforms())

    effective_cap = getattr(args, "max_train_per_class", 0)
    if getattr(args, "max_train_images", 0) > 0:
        effective_cap = max(1, args.max_train_images // 2)

    if not streaming and effective_cap > 0:
        before = len(train_ds.samples)
        train_ds.samples = subsample_balanced(
            train_ds.samples, effective_cap, args.subset_seed
        )
        print(f"Training subset: {before} -> {len(train_ds.samples)} images "
              f"(max_train_images={getattr(args, 'max_train_images', 0)}, "
              f"max {effective_cap} per class, subset_seed={args.subset_seed})")
    elif streaming and effective_cap > 0:
        print("WARNING: --max-train-images / --max-train-per-class is ignored for hf-stream datasets.")

    # --------------------------------------------------------
    # Validation dataset
    # --------------------------------------------------------

    val_ds = make_timed_dataset("val", get_val_transforms())

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=not streaming,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        **_loader_extras(args, persistent=True),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        **_loader_extras(args),
    )

    if args.mode not in NEEDS_TEST_LOADER:
        print(f"[dataset] total discovery: {dataset_discovery_seconds:.1f} sec")
        return train_loader, val_loader

    test_ds = make_timed_dataset("test", get_val_transforms())
    print(f"[dataset] total discovery: {dataset_discovery_seconds:.1f} sec")
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        **_loader_extras(args),
    )
    return train_loader, val_loader, test_loader


def report_deep_no_down_estimate(args, train_loader, val_loader, test_loader, elapsed_seconds):
    """Report a whole-split estimate using the measured train+val image rate."""
    if not getattr(args, "deep_no_down", False) or elapsed_seconds <= 0:
        return

    def split_size(loader, split_name):
        try:
            return len(loader.dataset)
        except TypeError:
            return int(getattr(args, f"known_{split_name}_size", 0))

    split_sizes = [
        split_size(train_loader, "train"),
        split_size(val_loader, "val"),
        split_size(test_loader, "test"),
    ]
    observed_images = split_sizes[0] + split_sizes[1]
    if observed_images <= 0:
        return
    observed_rate = observed_images / elapsed_seconds
    total_images = sum(split_sizes)
    estimated_seconds = total_images / observed_rate
    print("\n" + "!" * 72)
    print("WARNING: --deep-no-down is ENABLED (EfficientNet stem stride 1 instead of 2).")
    print("  - VRAM usage will increase significantly (~4.7 GB vs ~1.2 GB).")
    print(f"  - Observed train+val rate: {observed_rate:.1f} img/s ({observed_images:,} images in {elapsed_seconds / 60:.1f} min).")
    print(f"  - One pass over train+val+test ({total_images:,} images): ~{estimated_seconds / 60:.1f} min ({estimated_seconds / 3600:.2f} h).")
    print("!" * 72 + "\n")


# ============================================================
# STAGE 1 — DEEP BRANCH
# ============================================================

def train_epoch_stage1(
    model,
    loader,
    criterion,
    optimizer,
    device,
) -> Dict[str, float]:

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        labels = batch["label"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with _amp_ctx():
            logits, _ = model(images)
        logits = logits.float()

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += images.size(0)

    return {
        "loss": total_loss / total,
        "acc": correct / total,
    }


@torch.no_grad()
def eval_stage1(
    model,
    loader,
    criterion,
    device,
) -> Dict[str, float]:

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        labels = batch["label"].to(
            device,
            non_blocking=True,
        )

        with _amp_ctx():
            logits, _ = model(images)
        logits = logits.float()

        loss = criterion(
            logits,
            labels,
        )

        total_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += images.size(0)

    return {
        "loss": total_loss / total,
        "acc": correct / total,
    }


# ============================================================
# HYBRID — DEEP + PHYSICS
# ============================================================

def train_epoch_hybrid(model, loader, criterion, optimizer, device) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        physics = batch["physics"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        logits, _ = model(images, physics)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        pbar.set_postfix(loss=loss.item(), acc=correct / total)
    return {"loss": total_loss / total, "acc": correct / total}

@torch.no_grad()
def eval_hybrid(
    model,
    loader,
    criterion,
    device,
) -> Dict[str, float]:

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        physics = batch["physics"].to(
            device,
            non_blocking=True,
        )

        labels = batch["label"].to(
            device,
            non_blocking=True,
        )

        logits, _ = model(
            images,
            physics,
        )

        loss = criterion(
            logits,
            labels,
        )

        total_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += images.size(0)

    return {
        "loss": total_loss / total,
        "acc": correct / total,
    }


# ============================================================
# FULL HYBRID — DEEP + PHYSICS + PRNU + SEMANTIC
# ============================================================

def _full_hybrid_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    epoch=None,
    total_epochs=None,
    start_batch: int = 0,
    checkpoint_callback=None,
    total_batches: Optional[int] = None,
  ) -> Dict[str, float]:

    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    labels_all = []
    preds_all = []
    scores_all = []

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    # ------------------------------------------------------------
    # PROGRESS BAR
    # ------------------------------------------------------------

    if epoch is not None and total_epochs is not None:
        phase = "Training" if training else "Validation"

        progress_desc = (
            f"Epoch {epoch}/{total_epochs} | {phase}"
        )
    else:
        progress_desc = (
            "Training" if training else "Validation"
        )

    progress_bar = tqdm(
        loader,
        desc=progress_desc,
        total=total_batches,
        leave=True,
        dynamic_ncols=True,
    )

    # ------------------------------------------------------------
    # PROCESS BATCHES
    # ------------------------------------------------------------

    with context:

        for batch_idx, batch in enumerate(progress_bar, start=1):

            if batch_idx <= start_batch:
                progress_bar.set_postfix(
                    status=f"resuming: skipped {batch_idx}/{start_batch}"
                )
                continue

            try:

                images = batch["image"].to(
                    device,
                    non_blocking=True,
                )

                physics = batch["physics"].to(
                    device,
                    non_blocking=True,
                )

                prnu = batch["prnu"].to(
                    device,
                    non_blocking=True,
                )

                semantic = batch["semantic"].to(
                    device,
                    non_blocking=True,
                )

                labels = batch["label"].to(
                    device,
                    non_blocking=True,
                )

            except KeyError as exc:

                raise RuntimeError(
                    "full_hybrid requires image, physics, "
                    "PRNU, and semantic features; "
                    f"missing {exc.args[0]!r}."
                ) from exc

            # ----------------------------------------------------
            # FORWARD + BACKWARD
            # ----------------------------------------------------

            if training:

                optimizer.zero_grad(
                    set_to_none=True
                )

            with _amp_ctx():
                logits, _ = model(
                    images,
                    physics,
                    prnu,
                    semantic,
                )
            logits = logits.float()

            loss = criterion(
                logits,
                labels,
            )

            if training:

                loss.backward()

                optimizer.step()

            # ----------------------------------------------------
            # METRICS
            # ----------------------------------------------------

            batch_size = images.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            batch_preds = logits.argmax(
                dim=1
            )

            batch_scores = torch.softmax(
                logits,
                dim=1,
            )[:, 1]

            labels_all.extend(
                labels.detach()
                .cpu()
                .tolist()
            )

            preds_all.extend(
                batch_preds.detach()
                .cpu()
                .tolist()
            )

            scores_all.extend(
                batch_scores.detach()
                .cpu()
                .tolist()
            )

            # ----------------------------------------------------
            # RUNNING METRICS
            # ----------------------------------------------------

            running_acc = (
                sum(
                    p == y
                    for p, y in zip(
                        preds_all,
                        labels_all,
                    )
                )
                / len(labels_all)
            )

            running_loss = (
                total_loss
                / len(labels_all)
            )

            # ----------------------------------------------------
            # UPDATE PROGRESS BAR
            # ----------------------------------------------------

            progress_bar.set_postfix(
                loss=f"{running_loss:.4f}",
                acc=f"{running_acc:.4f}",
                batch=f"{batch_idx}/{total_batches}" if total_batches is not None else str(batch_idx),
            )

            if (
                training
                and checkpoint_callback is not None
                and batch_idx % checkpoint_callback["interval"] == 0
            ):
                checkpoint_callback["save"](
                    batch_idx=batch_idx,
                    metrics={
                        "loss": running_loss,
                        "acc": running_acc,
                    },
                )

    # ------------------------------------------------------------
    # FINAL METRICS
    # ------------------------------------------------------------

    if not labels_all:

        raise RuntimeError(
            "No samples were available for "
            "full_hybrid evaluation."
        )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels_all,
            preds_all,
            average="binary",
            zero_division=0,
        )
    )

    metrics = {

        "loss":
            total_loss
            / len(labels_all),

        "acc":
            float(
                accuracy_score(
                    labels_all,
                    preds_all,
                )
            ),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "confusion_matrix":
            confusion_matrix(
                labels_all,
                preds_all,
                labels=[0, 1],
            ).tolist(),
    }

    # ROC-AUC
    metrics["roc_auc"] = (
        float(
            roc_auc_score(
                labels_all,
                scores_all,
            )
        )
        if len(set(labels_all)) == 2
        else None
    )

    return metrics


def train_epoch_full_hybrid(
    model,
    loader,
    criterion,
    optimizer,
    device,
    epoch,
    total_epochs,
    start_batch: int = 0,
    checkpoint_callback=None,
    total_batches: Optional[int] = None,
):
    return _full_hybrid_epoch(
        model,
        loader,
        criterion,
        device,
        optimizer,
        epoch,
        total_epochs,
        start_batch=start_batch,
        checkpoint_callback=checkpoint_callback,
        total_batches=total_batches,
    )


@torch.no_grad()
def eval_full_hybrid(
    model,
    loader,
    criterion,
    device,
    epoch=None,
    total_epochs=None,
    total_batches: Optional[int] = None,
):
    return _full_hybrid_epoch(
        model,
        loader,
        criterion,
        device,
        None,
        epoch,
        total_epochs,
        total_batches=total_batches,
    )


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer,
    epoch: int,
    metrics: dict,
    mode: str,
    args,
    physics_normalizer: Optional[PhysicsNormalizer] = None,
    prnu_normalizer: Optional[PhysicsNormalizer] = None,
    scheduler=None,
    batch_idx: int = 0,
    epoch_complete: bool = True,
    best_acc: Optional[float] = None,
) -> None:
    """Save a resumable training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": int(epoch),
        "batch_idx": int(batch_idx),
        "epoch_complete": bool(epoch_complete),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "mode": mode,
        "args": vars(args),
        "physics_dim": PHYSICS_FEATURE_DIM,
    }

    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()

    if best_acc is not None:
        payload["best_acc"] = float(best_acc)

    if physics_normalizer is not None:
        payload["physics_normalizer"] = physics_normalizer.to_dict()

    if prnu_normalizer is not None:
        payload["prnu_normalizer"] = prnu_normalizer.to_dict()

    torch.save(payload, path)

    print(
        f"Saved checkpoint: {path} "
        f"(epoch={epoch}, batch={batch_idx}, "
        f"epoch_complete={epoch_complete})"
    )


def _epoch_checkpoint_index(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except ValueError:
        return -1


def rotate_epoch_checkpoints(output_dir: Path, mode: str, keep: int) -> None:
    if keep <= 0:
        return
    existing = sorted(
        Path(output_dir).glob(f"{mode}_epoch_*.pt"),
        key=_epoch_checkpoint_index,
    )
    for old in existing[:-keep]:
        try:
            old.unlink()
        except OSError:
            pass


def _splits_path(args):
    raw = getattr(args, "splits_yaml", None) or ""
    return raw if raw else None


def resolve_selection_loader(args, device, val_loader):
    """DEV loader from the test tree, or ProGAN-val when --legacy / unavailable."""
    if getattr(args, "legacy", False):
        return val_loader, "val"
    try:
        spec = load_split_spec(_splits_path(args) or DEFAULT_SPLITS_PATH)
        loader = build_loaders(args, device, folder_filter=spec.dev, split="test")
        print(f"Checkpoint selection uses DEV folders from {spec.path}")
        return loader, "dev"
    except Exception as exc:
        print(f"WARNING: DEV split unavailable ({exc}); selecting on validation.")
        return val_loader, "val"


def report_eval_loader(args, device, test_loader):
    """Headline metrics must not include DEV folders unless --legacy."""
    if getattr(args, "legacy", False) or test_loader is None:
        return test_loader, "test"
    try:
        spec = load_split_spec(_splits_path(args) or DEFAULT_SPLITS_PATH)
        loader = build_loaders(args, device, folder_filter=spec.report, split="test")
        print(f"Final metrics use REPORT folders from {spec.path}")
        return loader, "report"
    except Exception as exc:
        print(f"WARNING: REPORT split unavailable ({exc}); using full test loader.")
        return test_loader, "test"


def score_checkpoint(args, model, device, mode, val_metrics, selection_loader, selection_name):
    if getattr(args, "legacy", False) or selection_name == "val":
        return float(val_metrics.get("acc", 0.0)), val_metrics
    score, details, _collected = eval_selection_score(model, selection_loader, device, mode)
    merged = dict(val_metrics)
    merged.update(details)
    merged["selection_score"] = float(score)
    print(
        f"[DEV] auc={details.get('auc')}  bacc={details.get('balanced_accuracy'):.4f}  "
        f"score={score:.4f}"
    )
    return float(score), merged


def eval_selection_score(model, loader, device, mode: str):
    from src.eval.forward import predict_loader
    from src.eval.metrics import selection_score as _selection_score

    collected = predict_loader(model, mode, loader, device)
    score, details = _selection_score(collected["label"], collected["p_ai"])
    details["acc"] = details.get("balanced_accuracy", 0.0)
    details["roc_auc"] = details.get("auc")
    return score, details, collected


# ============================================================
# STAGE 1 TRAINING
# ============================================================

def run_stage1(
    args,
    device,
) -> None:

    print("\n")
    print("=" * 60)
    print("STAGE 1 — DEEP BRANCH TRAINING")
    print("=" * 60)

    train_loader, val_loader, test_loader = build_loaders(args, device)
    selection_loader, selection_name = val_loader, "val"
    if not getattr(args, "legacy", False):
        try:
            spec = load_split_spec(getattr(args, "splits_yaml", None) or DEFAULT_SPLITS_PATH)
            selection_loader = build_loaders(
                args, device, folder_filter=spec.dev, split="test"
            )
            selection_name = "dev"
            print(f"Checkpoint selection uses DEV folders from {spec.path}")
        except Exception as exc:
            print(f"WARNING: DEV split unavailable ({exc}); selecting on validation.")

    print(
        f"\nTraining batches: {_loader_batches(train_loader, args, 'train')}"
    )

    print(
        f"Validation batches: {_loader_batches(val_loader, args, 'val')}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = DeepClassifier(
        model_name=args.backbone,
        pretrained=True,
        freeze_blocks=args.freeze_blocks,
    ).to(device)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_acc = -1.0

    history = []

    # --------------------------------------------------------
    # RESUME FROM CHECKPOINT (epoch-level; stage1 has no mid-epoch
    # checkpointing, so a resume repeats at most the current epoch).
    # stage1_last.pt -> latest resumable training state
    # stage1_best.pt -> best validation model for final testing
    # --------------------------------------------------------

    last_path = Path(args.output_dir) / "stage1_last.pt"
    best_path = Path(args.output_dir) / "stage1_best.pt"

    start_epoch = 1

    if last_path.exists() and not args.fresh:

        print("\n" + "=" * 60)
        print("RESUMING FROM LAST TRAINING CHECKPOINT")
        print("=" * 60)
        print("Checkpoint:", last_path)

        checkpoint = torch.load(last_path, map_location=device, weights_only=False)

        model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        previous_epoch = int(checkpoint.get("epoch", 0))
        stored_best_acc = checkpoint.get("best_acc", None)
        if stored_best_acc is not None:
            best_acc = float(stored_best_acc)

        start_epoch = previous_epoch + 1

        print("Checkpoint epoch:", previous_epoch)
        print("Starting from epoch:", start_epoch)
        print("Best validation accuracy so far:", best_acc)
        print("=" * 60)

    elif last_path.exists() and args.fresh:
        print("\n--fresh set: ignoring existing stage1_last.pt, starting from Epoch 1.")
    else:
        print("\nNo last training checkpoint found. Starting training from Epoch 1.")

    early_stopper = EarlyStopping(
        patience=args.early_stopping_patience,
        best_score=best_acc,
    )

    if start_epoch > args.epochs:

        print("\nTraining already reached requested epoch count.")
        print(f"Checkpoint epoch = {start_epoch - 1}, requested epochs = {args.epochs}")

    else:

        for epoch in range(
            start_epoch,
            args.epochs + 1,
        ):

            epoch_started_at = time.perf_counter()

            train_metrics = train_epoch_stage1(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )

            val_metrics = eval_stage1(
                model,
                val_loader,
                criterion,
                device,
            )

            report_deep_no_down_estimate(
                args, train_loader, val_loader, test_loader,
                time.perf_counter() - epoch_started_at,
            )

            scheduler.step()

            sel_score, sel_metrics = score_checkpoint(
                args, model, device, "stage1", val_metrics, selection_loader, selection_name
            )
            record = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "selection": sel_metrics,
            }
            history.append(record)
            print(
                f"\n[Stage1 Epoch {epoch}/{args.epochs}] "
                f"train loss={train_metrics['loss']:.4f} "
                f"acc={train_metrics['acc']:.4f} | "
                f"val loss={val_metrics['loss']:.4f} "
                f"acc={val_metrics['acc']:.4f} | "
                f"{selection_name}_score={sel_score:.4f}"
            )

            def _save(path):
                save_checkpoint(
                    path, model, optimizer, epoch, sel_metrics, "stage1", args,
                    scheduler=scheduler, best_acc=best_acc,
                )

            _save(last_path)
            save_epoch_snapshot(
                Path(args.output_dir), "stage1", epoch,
                getattr(args, "keep_last_epochs", 0), _save, rotate_epoch_checkpoints,
            )
            if sel_score > best_acc:
                best_acc = sel_score
                _save(best_path)
            if early_stopper.step(sel_score, epoch):
                break

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    if not best_path.exists():
        raise RuntimeError("No stage1 checkpoint was saved.")

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    headline_loader, headline_name = report_eval_loader(args, device, test_loader)
    test_metrics = eval_stage1(model, headline_loader, criterion, device)
    test_metrics["split"] = headline_name

    history_path = Path(args.output_dir) / "stage1_history.json"
    with open(history_path, "w") as f:
        json.dump({"epochs": history, "test": test_metrics, "best_selection_score": best_acc}, f, indent=2)

    print("\n")
    print("=" * 60)
    print("STAGE 1 COMPLETE")
    print("=" * 60)
    print("Best validation accuracy:", best_acc)
    print("Held-out test:", json.dumps(test_metrics, indent=2))
    print("Checkpoint:", best_path)


# ============================================================
# HYBRID TRAINING
# ============================================================

def run_hybrid(
    args,
    device,
) -> None:

    print("\n")
    print("=" * 60)
    print("HYBRID TRAINING — DEEP + PHYSICS")
    print("=" * 60)

    train_loader, val_loader = build_loaders(
        args, device
    )
    selection_loader, selection_name = resolve_selection_loader(args, device, val_loader)

    print(
        f"\nTraining batches: {_loader_batches(train_loader, args, 'train')}"
    )

    print(
        f"Validation batches: {_loader_batches(val_loader, args, 'val')}"
    )

    # --------------------------------------------------------
    # Hybrid model
    # --------------------------------------------------------

    model = HybridClassifier(
        model_name=args.backbone,
        pretrained=True,
        freeze_blocks=args.freeze_blocks,
    ).to(device)

    # --------------------------------------------------------
    # Warm start from Stage 1
    # --------------------------------------------------------

    stage1_ckpt = (
        Path(args.output_dir)
        / "stage1_best.pt"
    )

    if stage1_ckpt.exists():

        print(
            "\nLoading Stage 1 checkpoint..."
        )

        ckpt = torch.load(
            stage1_ckpt,
            map_location=device,
        )

        state = ckpt[
            "model_state_dict"
        ]

        # Extract only feature extractor
        # weights from Stage 1.

        feature_state = {
            key.replace(
                "feature_extractor.",
                ""
            ): value

            for key, value in state.items()

            if key.startswith(
                "feature_extractor."
            )
        }

        if feature_state:

            model.feature_extractor.load_state_dict(
                feature_state,
                strict=False,
            )

            print(
                "Loaded Stage 1 backbone weights."
            )

        else:

            print(
                "WARNING: No feature extractor "
                "weights found in Stage 1 checkpoint."
            )

    else:

        print(
            "\nWARNING: Stage 1 checkpoint not found:"
        )

        print(
            stage1_ckpt
        )

        print(
            "Hybrid training will start "
            "from the pretrained backbone."
        )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_acc = 0.0

    history = []
    early_stopper = EarlyStopping(
        patience=args.early_stopping_patience,
        best_score=best_acc,
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        train_metrics = train_epoch_hybrid(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_metrics = eval_hybrid(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step()

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }

        history.append(record)

        print(
            f"\n[Hybrid Epoch {epoch}/{args.epochs}] "
            f"train loss={train_metrics['loss']:.4f} "
            f"acc={train_metrics['acc']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['acc']:.4f}"
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if val_metrics["acc"] > best_acc:

            best_acc = val_metrics["acc"]

            save_checkpoint(
                Path(args.output_dir)
                / "hybrid_best.pt",

                model,

                optimizer,

                epoch,

                val_metrics,

                "hybrid",

                args,
            )

        if early_stopper.step(val_metrics["acc"], epoch):
            break

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history_path = (
        Path(args.output_dir)
        / "hybrid_history.json"
    )

    with open(
        history_path,
        "w",
    ) as f:

        json.dump(
            history,
            f,
            indent=2,
        )

    print("\n")
    print("=" * 60)
    print("HYBRID TRAINING COMPLETE")
    print("=" * 60)
    print("Best validation accuracy:", best_acc)

    print(
        "Checkpoint:",
        Path(args.output_dir)
        / "hybrid_best.pt",
    )


def run_full_hybrid(args, device) -> None:
    """Train/resume the four-modality model using explicit train/val/test sets."""

    print("\n" + "=" * 60)
    print("FULL HYBRID TRAINING — EFFICIENTNET + PHYSICS + PRNU + ViT")
    print("=" * 60)

    print("[1/5] Building datasets")
    train_loader, val_loader, test_loader = build_loaders(args, device)

    # ------------------------------------------------------------
    # Fit feature scalers ONLY on training images
    # ------------------------------------------------------------
    from src.features.feature_scalers import (
        fit_physics_and_prnu_scalers,
        load_physics_and_prnu_scaler_cache,
        save_physics_and_prnu_scaler_cache,
    )

    print("[2/5] Preparing physics/PRNU scalers")
    scaler_cache_path = Path(args.output_dir) / "physics_prnu_scalers.pkl"
    scaler_started_at = time.perf_counter()
    cached_scalers = None
    if not args.fresh and scaler_cache_path.exists():
        print("Loading cached physics/PRNU scalers...")
        cached_scalers = load_physics_and_prnu_scaler_cache(
            scaler_cache_path,
            args.scaler_samples,
            args.seed,
        )

    if cached_scalers is not None:
        physics_normalizer, prnu_normalizer = cached_scalers
        print(f"Scalers loaded in {time.perf_counter() - scaler_started_at:.1f} seconds")
    else:
        if scaler_cache_path.exists() and not args.fresh:
            print("Scaler cache is incompatible or unreadable; refitting scalers.")
        elif args.fresh:
            print("--fresh set: ignoring cached physics/PRNU scalers.")

        if args.data_source == "hf-stream":
            physics_normalizer, prnu_normalizer = fit_streaming_feature_scalers(
                train_loader.dataset,
                max_samples=args.scaler_samples,
                total_samples=args.known_train_size,
            )
        else:
            # Path strings are inexpensive; images are decoded one at a time
            # in the fitter, and the DataLoader still owns the full dataset.
            train_paths = [path for path, _, _ in train_loader.dataset.samples]
            physics_normalizer, prnu_normalizer = fit_physics_and_prnu_scalers(
                train_paths,
                max_samples=args.scaler_samples,
                seed=args.seed,
            )
        save_physics_and_prnu_scaler_cache(
            scaler_cache_path,
            physics_normalizer,
            prnu_normalizer,
            args.scaler_samples,
            args.seed,
        )
        print(f"Scalers fitted and cached in {time.perf_counter() - scaler_started_at:.1f} seconds")

    # ------------------------------------------------------------
    # Create model
    # ------------------------------------------------------------
    branches_str = getattr(args, "branches", "deep,physics,semantic,freq")
    if getattr(args, "legacy", False) and getattr(args, "branches", None) is None:
        branches_str = "deep,physics,prnu,semantic"
    branches_list = [b.strip().lower() for b in branches_str.split(",") if b.strip()]
    model = FullHybridClassifier(
        model_name=args.backbone,
        semantic_model=args.semantic_model,
        semantic_pretrained=not args.no_semantic_pretrained,
        semantic_dim=args.semantic_dim,
        pretrained=not args.no_pretrained,
        freeze_blocks=args.freeze_blocks,
        fusion_mode=FusionMode(args.fusion_mode),
        normalize_physics=True,
        normalize_prnu=True,
        branches=branches_list,
        no_down=getattr(args, "deep_no_down", False),
        branch_dropout=getattr(args, "branch_dropout", 0.0),
    ).to(device)

    model.set_feature_scalers(
        physics_normalizer,
        prnu_normalizer,
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    # ------------------------------------------------------------
    # RESUME FROM CHECKPOINT
    # ------------------------------------------------------------
    # full_hybrid_last.pt  -> latest resumable training state
    # full_hybrid_best.pt  -> best validation model for final testing
    # full_hybrid_resume.pt -> emergency mid-epoch recovery state
    # ------------------------------------------------------------
    print("[4/5] Loading checkpoint")
    resume_checkpoint_path = Path(args.output_dir) / "full_hybrid_last.pt"
    best_checkpoint_path = Path(args.output_dir) / "full_hybrid_best.pt"

    start_epoch = 1
    start_batch = 0
    best_acc = -1.0
    history = []

    if resume_checkpoint_path.exists() and not args.fresh:

        print("\n" + "=" * 60)
        print("RESUMING FROM LAST TRAINING CHECKPOINT")
        print("=" * 60)
        print("Checkpoint:", resume_checkpoint_path)

        checkpoint = torch.load(
            resume_checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        # Restore scheduler state directly. Do NOT call scheduler.step()
        # repeatedly before optimizer.step().
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        previous_epoch = int(checkpoint.get("epoch", 0))
        epoch_complete = bool(checkpoint.get("epoch_complete", True))
        saved_batch = int(checkpoint.get("batch_idx", 0))

        previous_metrics = checkpoint.get("metrics", {})
        stored_best_acc = checkpoint.get("best_acc", None)

        if stored_best_acc is not None:
            best_acc = float(stored_best_acc)
        else:
            best_acc = float(
                previous_metrics.get("acc", -1.0)
            )

        if epoch_complete:
            start_epoch = previous_epoch + 1
            start_batch = 0
        else:
            start_epoch = previous_epoch
            start_batch = saved_batch

        print("Checkpoint epoch:", previous_epoch)
        print("Checkpoint batch:", saved_batch)
        print("Epoch complete:", epoch_complete)
        print("Starting from epoch:", start_epoch)
        print("Starting after batch:", start_batch)
        print("Best validation accuracy:", best_acc)
        print("=" * 60)

    elif resume_checkpoint_path.exists() and args.fresh:
        print("\n--fresh set: ignoring existing full_hybrid_last.pt, starting from Epoch 1.")
    else:
        print("\nNo last training checkpoint found.")
        print("Starting training from Epoch 1.")

    early_stopper = EarlyStopping(
        patience=args.early_stopping_patience,
        best_score=best_acc,
    )

    # ------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------

    print("[5/5] Starting training")

    if start_epoch > args.epochs:

        print("\nTraining already reached requested epoch count.")
        print(
            f"Checkpoint epoch = {start_epoch - 1}, "
            f"requested epochs = {args.epochs}"
        )

    else:

        checkpoint_interval = max(
            1,
            int(args.checkpoint_interval),
        )

        for epoch in range(
            start_epoch,
            args.epochs + 1,
        ):

            print(
                f"\nStarting Epoch {epoch}/{args.epochs}"
            )
            epoch_started_at = time.perf_counter()

            epoch_start_batch = (
                start_batch
                if epoch == start_epoch
                else 0
            )

            def save_mid_epoch_checkpoint(batch_idx, metrics):
                mid_checkpoint = (
                    Path(args.output_dir)
                    / "full_hybrid_resume.pt"
                )

                save_checkpoint(
                    mid_checkpoint,
                    model,
                    optimizer,
                    epoch,
                    metrics,
                    "full_hybrid",
                    args,
                    physics_normalizer,
                    prnu_normalizer,
                    scheduler=scheduler,
                    batch_idx=batch_idx,
                    epoch_complete=False,
                    best_acc=best_acc,
                )

            checkpoint_callback = {
                "interval": checkpoint_interval,
                "save": save_mid_epoch_checkpoint,
            }

            train_metrics = train_epoch_full_hybrid(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                epoch,
                args.epochs,
                start_batch=epoch_start_batch,
                checkpoint_callback=checkpoint_callback,
                total_batches=_loader_batches(train_loader, args, "train"),
            )

            val_metrics = eval_full_hybrid(
                model,
                val_loader,
                criterion,
                device,
                epoch,
                args.epochs,
                total_batches=_loader_batches(val_loader, args, "val"),
            )

            report_deep_no_down_estimate(
                args, train_loader, val_loader, test_loader,
                time.perf_counter() - epoch_started_at,
            )

            # Correct scheduler order: optimizer.step() happens inside the
            # training epoch; scheduler.step() happens once afterward.
            scheduler.step()

            record = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }

            history.append(record)

            print(
                f"[FullHybrid Epoch {epoch}/{args.epochs}] "
                f"train loss={train_metrics['loss']:.4f} "
                f"acc={train_metrics['acc']:.4f} | "
                f"val loss={val_metrics['loss']:.4f} "
                f"acc={val_metrics['acc']:.4f} "
                f"f1={val_metrics['f1']:.4f}"
            )

            # ----------------------------------------------------
            # SAVE LAST CHECKPOINT EVERY EPOCH
            # ----------------------------------------------------
            last_checkpoint = (
                Path(args.output_dir)
                / "full_hybrid_last.pt"
            )

            save_checkpoint(
                last_checkpoint,
                model,
                optimizer,
                epoch,
                val_metrics,
                "full_hybrid",
                args,
                physics_normalizer,
                prnu_normalizer,
                scheduler=scheduler,
                batch_idx=_loader_batches(train_loader, args, "train"),
                epoch_complete=True,
                best_acc=best_acc,
            )

            # A complete epoch supersedes the mid-epoch checkpoint.
            mid_checkpoint = (
                Path(args.output_dir)
                / "full_hybrid_resume.pt"
            )
            if mid_checkpoint.exists():
                try:
                    mid_checkpoint.unlink()
                except OSError:
                    pass

            # ----------------------------------------------------
            # SAVE BEST CHECKPOINT
            # ----------------------------------------------------
            if val_metrics["acc"] > best_acc:

                best_acc = val_metrics["acc"]

                save_checkpoint(
                    best_checkpoint_path,
                    model,
                    optimizer,
                    epoch,
                    val_metrics,
                    "full_hybrid",
                    args,
                    physics_normalizer,
                    prnu_normalizer,
                    scheduler=scheduler,
                    batch_idx=_loader_batches(train_loader, args, "train"),
                    epoch_complete=True,
                    best_acc=best_acc,
                )

                print("New best checkpoint saved.")

            start_batch = 0
            if early_stopper.step(val_metrics["acc"], epoch):
                break

    # ------------------------------------------------------------
    # TEST BEST MODEL
    # ------------------------------------------------------------

    if not best_checkpoint_path.exists():

        raise RuntimeError(
            "No full_hybrid checkpoint was saved."
        )

    print("\n" + "=" * 60)
    print("LOADING BEST CHECKPOINT FOR FINAL TEST")
    print("=" * 60)

    best_checkpoint = torch.load(
        best_checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        best_checkpoint["model_state_dict"]
    )

    test_metrics = eval_full_hybrid(
        model,
        test_loader,
        criterion,
        device,
        epoch="test",
        total_epochs="test",
    )

    history_payload = {
        "epochs": history,
        "test": test_metrics,
    }

    with open(
        Path(args.output_dir)
        / "full_hybrid_history.json",
        "w",
    ) as f:

        json.dump(
            history_payload,
            f,
            indent=2,
        )

    print(
        "\nHeld-out test metrics:"
    )

    print(
        json.dumps(
            test_metrics,
            indent=2,
        )
    )

    print(
        "\nBest checkpoint:",
        best_checkpoint_path,
    )

    print(
        "Last checkpoint:",
        Path(args.output_dir)
        / "full_hybrid_last.pt",
    )


def _feature_key_for_mode(mode: str) -> str:
    return {
        "physics_only": "physics",
        "prnu_only": "prnu",
        "semantic_only": "semantic",
    }[mode]


def _run_feature_epoch(
    model,
    loader,
    criterion,
    device,
    feature_key: str,
    optimizer=None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels_all: List[int] = []
    preds_all: List[int] = []
    scores_all: List[float] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            feats = batch[feature_key].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits, _ = model(feats)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            probs = torch.softmax(logits.detach(), dim=1)
            total_loss += float(loss.item()) * labels.size(0)
            labels_all.extend(labels.detach().cpu().tolist())
            preds_all.extend(probs.argmax(dim=1).cpu().tolist())
            scores_all.extend(probs[:, 1].cpu().tolist())
    n = max(len(labels_all), 1)
    acc = accuracy_score(labels_all, preds_all) if labels_all else 0.0
    try:
        auc = roc_auc_score(labels_all, scores_all) if len(set(labels_all)) == 2 else None
    except ValueError:
        auc = None
    return {"loss": total_loss / n, "acc": float(acc), "roc_auc": auc}


def run_branch_only(args, device) -> None:
    """Train a single-branch MLP so ablations are comparable to the full model."""
    mode = args.mode
    feature_key = _feature_key_for_mode(mode)
    print("\n" + "=" * 60)
    print(f"SINGLE-BRANCH TRAINING — {mode.upper()}")
    print("=" * 60)

    train_loader, val_loader, test_loader = build_loaders(args, device)
    physics_normalizer, prnu_normalizer = None, None
    if mode in {"physics_only", "prnu_only"}:
        from src.features.feature_scalers import fit_physics_and_prnu_scalers
        train_paths = [path for path, _, _ in train_loader.dataset.samples]
        physics_normalizer, prnu_normalizer = fit_physics_and_prnu_scalers(train_paths)

    dims = {
        "physics_only": PHYSICS_FEATURE_DIM,
        "prnu_only": PRNU_FEATURE_DIM,
        "semantic_only": args.semantic_dim,
    }
    model = BranchOnlyClassifier(
        input_dim=dims[mode],
        normalize=mode in {"physics_only", "prnu_only"},
    ).to(device)
    if mode == "physics_only":
        model.set_normalizer(physics_normalizer)
    elif mode == "prnu_only":
        model.set_normalizer(prnu_normalizer)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = -1.0
    best_path = Path(args.output_dir) / f"{mode}_best.pt"
    last_path = Path(args.output_dir) / f"{mode}_last.pt"
    history = []

    # ------------------------------------------------------------
    # RESUME FROM CHECKPOINT (epoch-level)
    # {mode}_last.pt -> latest resumable training state
    # {mode}_best.pt -> best validation model for final testing
    # ------------------------------------------------------------

    start_epoch = 1

    if last_path.exists() and not args.fresh:

        print("\n" + "=" * 60)
        print("RESUMING FROM LAST TRAINING CHECKPOINT")
        print("=" * 60)
        print("Checkpoint:", last_path)

        checkpoint = torch.load(last_path, map_location=device, weights_only=False)

        model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        previous_epoch = int(checkpoint.get("epoch", 0))
        stored_best_acc = checkpoint.get("best_acc", None)
        if stored_best_acc is not None:
            best_acc = float(stored_best_acc)

        start_epoch = previous_epoch + 1

        print("Checkpoint epoch:", previous_epoch)
        print("Starting from epoch:", start_epoch)
        print("Best validation accuracy so far:", best_acc)
        print("=" * 60)

    elif last_path.exists() and args.fresh:
        print(f"\n--fresh set: ignoring existing {last_path.name}, starting from Epoch 1.")
    else:
        print("\nNo last training checkpoint found. Starting training from Epoch 1.")

    early_stopper = EarlyStopping(
        patience=args.early_stopping_patience,
        best_score=best_acc,
    )

    if start_epoch > args.epochs:
        print("\nTraining already reached requested epoch count.")
        print(f"Checkpoint epoch = {start_epoch - 1}, requested epochs = {args.epochs}")

    # range(start_epoch, args.epochs + 1) is naturally empty when the
    # checkpoint already covers the requested epoch count.
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = _run_feature_epoch(
            model, train_loader, criterion, device, feature_key, optimizer
        )
        val_metrics = _run_feature_epoch(
            model, val_loader, criterion, device, feature_key
        )
        scheduler.step()
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"[{mode} Epoch {epoch}/{args.epochs}] "
            f"train_acc={train_metrics['acc']:.4f} val_acc={val_metrics['acc']:.4f}"
        )
        save_checkpoint(
            last_path, model, optimizer, epoch, val_metrics, mode, args,
            physics_normalizer=physics_normalizer if mode == "physics_only" else None,
            prnu_normalizer=prnu_normalizer if mode == "prnu_only" else None,
            scheduler=scheduler, best_acc=best_acc,
        )
        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            save_checkpoint(
                best_path, model, optimizer, epoch, val_metrics, mode, args,
                physics_normalizer=physics_normalizer if mode == "physics_only" else None,
                prnu_normalizer=prnu_normalizer if mode == "prnu_only" else None,
                scheduler=scheduler, best_acc=best_acc,
            )

        if early_stopper.step(val_metrics["acc"], epoch):
            break

    if not best_path.exists():
        raise RuntimeError(f"No {mode} checkpoint was saved.")

    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = _run_feature_epoch(model, test_loader, criterion, device, feature_key)
    history_path = Path(args.output_dir) / f"{mode}_history.json"
    history_path.write_text(
        json.dumps({"epochs": history, "test": test_metrics, "best_val_acc": best_acc}, indent=2)
    )
    print(json.dumps(test_metrics, indent=2))
    print("Best checkpoint:", best_path)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train AI vs Real image detector "
            "(Colab/Kaggle GPU)"
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "stage1",
            "hybrid",
            "full_hybrid",
            "physics_only",
            "prnu_only",
            "semantic_only",
            "freq_only",
            "clip_linear",
        ],
        default="stage1",
        help=(
            "stage1 = deep branch only; "
            "physics_only / prnu_only / semantic_only / freq_only / clip_linear = single-branch MLPs; "
            "hybrid = deep + physics; "
            "full_hybrid = deep + physics + PRNU + ViT"
        ),
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help=(
            "For full_hybrid and branch-only modes: data/{train,val,test} with "
            "real/fake or CNNDetection 0_real/1_fake folders. "
            "Legacy stage1/hybrid also accept data/{real,fake}."
        ),
    )

    parser.add_argument(
        "--data-source",
        choices=["local", "hf-stream"],
        default="local",
        help="Read local --data-dir folders (default) or stream WebDataset shards from Hugging Face.",
    )

    parser.add_argument(
        "--hf-dataset-repo",
        type=str,
        default="sahilSpit/ai-vs-real-face-detector-data",
        help="Public Hugging Face dataset repo with train/, val/, and test/ WebDataset shards.",
    )

    parser.add_argument(
        "--known-train-size",
        type=int,
        default=721_869,
        help="Known number of streamed training samples; used where IterableDataset has no length.",
    )

    parser.add_argument(
        "--known-val-size",
        type=int,
        default=8_375,
        help="Known number of streamed validation samples; used where IterableDataset has no length.",
    )

    parser.add_argument(
        "--known-test-size",
        type=int,
        default=90_704,
        help="Known number of streamed test samples; used where IterableDataset has no length.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory for checkpoints.",
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="efficientnet_b0",
        help="timm backbone.",
    )

    parser.add_argument(
        "--freeze-blocks",
        type=int,
        default=5,
        help="Number of early backbone blocks to freeze.",
    )

    parser.add_argument(
        "--semantic-model",
        type=str,
        default=DEFAULT_VIT_MODEL,
        help="Frozen timm ViT/CLIP-compatible semantic backbone for full_hybrid.",
    )

    parser.add_argument(
        "--semantic-dim",
        type=int,
        default=384,
        help="Embedding dimension emitted by --semantic-model (384 for vit_small_patch16_224).",
    )

    parser.add_argument(
        "--fusion-mode",
        choices=[mode.value for mode in FusionMode],
        default=FusionMode.GATED.value,
        help="Four-modality fusion method for full_hybrid.",
    )

    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not download/use pretrained EfficientNet weights (smoke tests only).",
    )
    parser.add_argument(
        "--no-semantic-pretrained",
        action="store_true",
        help="Do not download/use pretrained semantic weights (smoke tests only).",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Stop when validation accuracy fails to improve for this many consecutive epochs.",
    )

    parser.add_argument(
        "--preprocess",
        choices=["crop", "resize"],
        default="crop",
        help="crop = random 224x224 crop train, center crop eval; resize = legacy",
    )
    parser.add_argument(
        "--deep-no-down",
        action="store_true",
        default=False,
        help="Set EfficientNet-B0 stem conv stride from 2 to 1",
    )
    parser.add_argument(
        "--branches",
        type=str,
        default="deep,physics,semantic,freq",
        help="Comma-separated branches to use (default: deep,physics,semantic,freq; prnu is off by default)",
    )
    parser.add_argument(
        "--branch-dropout",
        type=float,
        default=0.15,
        help="Probability of dropping auxiliary branches during training (default: 0.15)",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        default=False,
        help="Enable legacy evaluation and training behavior",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of each source group used for validation.",
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=25,
        help=(
            "Save an emergency mid-epoch checkpoint every N completed "
            "batches in full_hybrid mode."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=42,
        help="Random seed for balanced subset selection only; training seed controls initialization and ordering.",
    )

    parser.add_argument(
        "--scaler-samples",
        type=int,
        default=20_000,
        help=(
            "Number of training images used only to fit physics/PRNU scalers "
            "for full_hybrid; 0 = all training images. Training still uses "
            "the complete dataset."
        ),
    )

    parser.add_argument(
        "--max-train-per-class",
        type=int,
        default=0,
        help=(
            "If > 0, randomly keep at most this many real and this many fake "
            "TRAIN images (val/test are untouched). 0 = use everything."
        ),
    )
    parser.add_argument(
        "--max-train-images",
        type=int,
        default=0,
        help=(
            "If > 0, randomly keep at most this many total TRAIN images, "
            "balanced per class (N//2 real, N//2 fake), seeded. 0 = use everything."
        ),
    )

    parser.add_argument(
        "--feature-cache",
        type=str,
        default="",
        help="Directory written by src/cache_features.py. Physics/PRNU/ViT vectors are "
             "read from it instead of being recomputed for every image every epoch.",
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="bf16 autocast + cudnn.benchmark for the EfficientNet runs "
             "(faster, but not bit-for-bit reproducible).",
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Ignore any existing <mode>_last.pt checkpoint in --output-dir and "
            "start this mode from epoch 1, instead of auto-resuming. Applies to "
            "stage1, full_hybrid, and the branch-only modes."
        ),
    )

    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help=(
            "Allow CPU execution for smoke tests only. "
            "Do NOT use for real training."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    global AMP_ENABLED
    args = parse_args()

    if args.data_source == "hf-stream" and args.mode != "full_hybrid":
        raise ValueError("--data-source hf-stream is currently supported only with --mode full_hybrid.")
    if args.data_source == "hf-stream" and not args.hf_dataset_repo:
        raise ValueError("--hf-dataset-repo is required with --data-source hf-stream.")
    if args.data_source == "hf-stream" and (
        args.known_train_size < args.batch_size
        or args.known_val_size < args.batch_size
        or args.known_test_size < args.batch_size
    ):
        raise ValueError("Known streamed train/val/test sizes must each be at least one batch.")
    if args.scaler_samples < 0:
        raise ValueError("--scaler-samples must be >= 0 (0 means all training images).")

    set_seed(
        args.seed
    )

    if args.feature_cache:
        os.environ["FEATURE_CACHE_DIR"] = str(args.feature_cache)
    if args.fast:
        AMP_ENABLED = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = require_gpu(args.allow_cpu)

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    Path(
        args.output_dir
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Print configuration
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)

    print("Mode:", args.mode)
    print("Data:", args.hf_dataset_repo if args.data_source == "hf-stream" else args.data_dir)
    print("Data source:", args.data_source)
    print("Output:", args.output_dir)
    print("Backbone:", args.backbone)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Learning rate:", args.lr)
    print("Validation ratio:", args.val_ratio)
    print("Workers:", args.num_workers)
    print("Checkpoint interval:", args.checkpoint_interval, "batches")
    print("Seed:", args.seed)
    print("Scaler samples:", args.scaler_samples, "(0 = all training images)")
    print("Fresh start (ignore existing checkpoint):", args.fresh)
    print("Deep no-down (stem stride 1):", args.deep_no_down)
    print("=" * 60)

    # --------------------------------------------------------
    # Run selected mode
    # --------------------------------------------------------

    if args.mode == "stage1":

        run_stage1(
            args,
            device,
        )

    elif args.mode == "hybrid":

        run_hybrid(
            args,
            device,
        )

    elif args.mode == "full_hybrid":

        run_full_hybrid(args, device)

    elif args.mode in {"physics_only", "prnu_only", "semantic_only"}:

        run_branch_only(args, device)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
