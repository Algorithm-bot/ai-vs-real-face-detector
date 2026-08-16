"""
Training script for Colab/Kaggle.

NOT intended for local 8GB RAM machines.

Usage in Colab:
    !python src/train.py --mode stage1 --data-dir data --epochs 10 --batch-size 32 --output-dir models

    !python src/train.py --mode hybrid --data-dir data --epochs 15 --batch-size 32 --output-dir models

Local:
    Do NOT run training loops on an 8GB RAM laptop.
    Use inference.py with a downloaded checkpoint.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset


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

from src.classifier.head import HybridClassifier
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


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible training."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic behavior where possible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# GPU
# ============================================================

def require_gpu() -> torch.device:
    """Require CUDA GPU for real training."""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not available.\n"
            "Run this script on Google Colab or Kaggle with GPU runtime enabled.\n"
            "Do NOT train on an 8GB RAM laptop."
        )

    device = torch.device("cuda")

    print("=" * 60)
    print("GPU INFORMATION")
    print("=" * 60)
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)
    print("=" * 60)

    return device


# ============================================================
# DATASET
# ============================================================

class FaceBinaryDataset(Dataset):
    """
    Dataset for REAL vs AI-GENERATED faces.

    Supported directory structures:

    data/
    ├── real/
    │   ├── image1.jpg
    │   └── image2.jpg
    │
    └── fake/
        ├── image1.jpg
        └── image2.jpg

    AND recursively:

    data/
    ├── real/
    │   └── ...
    │
    └── fake/
        ├── stylegan2/
        │   ├── image1.jpg
        │   └── image2.jpg
        │
        └── diffusion/
            ├── image3.jpg
            └── image4.jpg

    Labels:

        0 = REAL
        1 = FAKE / AI-GENERATED
    """

    EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        val_ratio: float = 0.15,
        seed: int = 42,
        use_physics: bool = False,
        preprocessor: Optional[FacePreprocessor] = None,
        transform=None,
    ) -> None:

        if split not in {"train", "val"}:
            raise ValueError(
                f"Invalid split '{split}'. Expected 'train' or 'val'."
            )

        self.root = Path(root)
        self.use_physics = use_physics

        self.preprocessor = (
            preprocessor
            if preprocessor is not None
            else FacePreprocessor()
        )

        self.transform = transform

        # Physics extractor is only created for hybrid training.
        self.physics_extractor = (
            PhysicsFeatureExtractor()
            if use_physics
            else None
        )

        # ----------------------------------------------------
        # Find all images recursively
        # ----------------------------------------------------

        samples: List[Tuple[str, int, str]] = []

        # label:
        #   0 = real
        #   1 = fake
        #
        # subdir:
        #   real
        #   fake

        for label, subdir in (
            (0, "real"),
            (1, "fake"),
        ):

            folder = self.root / subdir

            if not folder.exists():
                print(
                    f"WARNING: Directory does not exist: {folder}"
                )
                continue

            # rglob allows:
            #
            # fake/image.jpg
            #
            # AND
            #
            # fake/stylegan2/image.jpg
            # fake/diffusion/image.jpg

            for path in sorted(folder.rglob("*")):

                if not path.is_file():
                    continue

                if path.suffix.lower() not in self.EXTENSIONS:
                    continue

                relative_path = path.relative_to(folder)

                # Example:
                #
                # real/abc.jpg
                # -> source = real
                #
                # fake/stylegan2/abc.jpg
                # -> source = stylegan2
                #
                # fake/diffusion/abc.jpg
                # -> source = diffusion

                if len(relative_path.parts) > 1:
                    source = relative_path.parts[0]
                else:
                    source = subdir

                samples.append(
                    (
                        str(path),
                        label,
                        source,
                    )
                )

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
        # Stratified train/validation split
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

        # ----------------------------------------------------
        # Physics branch
        # ----------------------------------------------------

        if self.use_physics:

            physics_result = (
                self.physics_extractor.extract(rgb)
            )

            physics_vec = physics_result.vector

        else:

            physics_vec = np.zeros(
                PHYSICS_FEATURE_DIM,
                dtype=np.float32,
            )

        # ----------------------------------------------------
        # Deep-learning preprocessing
        # ----------------------------------------------------

        pil = self.preprocessor.preprocess_pil(
            Image.fromarray(rgb)
        )

        tensor = self.transform(pil)

        # ----------------------------------------------------
        # Return sample
        # ----------------------------------------------------

        return {
            "image": tensor,

            "label": torch.tensor(
                label,
                dtype=torch.long,
            ),

            "physics": torch.from_numpy(
                np.asarray(
                    physics_vec,
                    dtype=np.float32,
                )
            ),

            "path": path,

            "source": source,
        }


# ============================================================
# DATA LOADERS
# ============================================================

def build_loaders(
    args,
) -> Tuple[DataLoader, DataLoader]:

    preprocessor = FacePreprocessor()

    # --------------------------------------------------------
    # Training dataset
    # --------------------------------------------------------

    train_ds = FaceBinaryDataset(
        root=args.data_dir,
        split="train",
        val_ratio=args.val_ratio,
        seed=args.seed,
        use_physics=args.mode == "hybrid",
        preprocessor=preprocessor,
        transform=get_train_transforms(),
    )

    # --------------------------------------------------------
    # Validation dataset
    # --------------------------------------------------------

    val_ds = FaceBinaryDataset(
        root=args.data_dir,
        split="val",
        val_ratio=args.val_ratio,
        seed=args.seed,
        use_physics=args.mode == "hybrid",
        preprocessor=preprocessor,
        transform=get_val_transforms(),
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


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

        logits, _ = model(images)

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

        logits, _ = model(images)

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

def train_epoch_hybrid(
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

        physics = batch["physics"].to(
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

        logits, _ = model(
            images,
            physics,
        )

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
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "metrics":
                metrics,

            "mode":
                mode,

            "args":
                vars(args),

            "physics_dim":
                PHYSICS_FEATURE_DIM,
        },
        path,
    )

    print(
        f"Saved checkpoint: {path}"
    )


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

    train_loader, val_loader = build_loaders(
        args
    )

    print(
        f"\nTraining batches: {len(train_loader)}"
    )

    print(
        f"Validation batches: {len(val_loader)}"
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

    best_acc = 0.0

    history = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):

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

        scheduler.step()

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }

        history.append(record)

        print(
            f"\n[Stage1 Epoch {epoch}/{args.epochs}] "
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
                / "stage1_best.pt",

                model,

                optimizer,

                epoch,

                val_metrics,

                "stage1",

                args,
            )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history_path = (
        Path(args.output_dir)
        / "stage1_history.json"
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
    print("STAGE 1 COMPLETE")
    print("=" * 60)
    print("Best validation accuracy:", best_acc)
    print(
        "Checkpoint:",
        Path(args.output_dir)
        / "stage1_best.pt",
    )


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
        args
    )

    print(
        f"\nTraining batches: {len(train_loader)}"
    )

    print(
        f"Validation batches: {len(val_loader)}"
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


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train AI vs Real Face Detector "
            "(Colab/Kaggle GPU)"
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "stage1",
            "hybrid",
        ],
        default="stage1",
        help=(
            "stage1 = deep branch only; "
            "hybrid = deep + physics"
        ),
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help=(
            "Root directory containing "
            "real/ and fake/"
        ),
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
        "--epochs",
        type=int,
        default=10,
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

    args = parse_args()

    set_seed(
        args.seed
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if args.allow_cpu:

        device = torch.device(
            "cpu"
        )

        print(
            "WARNING: CPU mode enabled."
        )

        print(
            "This should ONLY be used "
            "for smoke tests."
        )

    else:

        device = require_gpu()

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
    print("Data:", args.data_dir)
    print("Output:", args.output_dir)
    print("Backbone:", args.backbone)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Learning rate:", args.lr)
    print("Validation ratio:", args.val_ratio)
    print("Workers:", args.num_workers)
    print("Seed:", args.seed)

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


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()