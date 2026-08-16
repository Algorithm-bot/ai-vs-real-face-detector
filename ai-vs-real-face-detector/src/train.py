"""
Training script for Colab/Kaggle — NOT intended for local 8GB RAM machines.

Usage (Colab):
  !python src/train.py --mode stage1 --data-dir /content/data --epochs 10
  !python src/train.py --mode hybrid --data-dir /content/data --epochs 15

Local: do NOT run training loops. Use inference.py with a downloaded checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Allow running from project root or Colab upload folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier.head import HybridClassifier
from src.deep_branch.feature_extractor import DeepClassifier
from src.deep_branch.preprocessing import FacePreprocessor, get_train_transforms, get_val_transforms
from src.physics_branch.feature_vector import PhysicsFeatureExtractor, PHYSICS_FEATURE_DIM


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def require_gpu() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not available. Run this script on Google Colab or Kaggle "
            "with GPU runtime enabled. Do NOT train on an 8GB RAM laptop."
        )
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    return device


class FaceBinaryDataset(Dataset):
    """Expects data/real/*.jpg and data/fake/*.jpg layout."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

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
        self.root = Path(root)
        self.use_physics = use_physics
        self.preprocessor = preprocessor or FacePreprocessor()
        self.transform = transform
        self.physics_extractor = PhysicsFeatureExtractor() if use_physics else None

        samples: List[Tuple[str, int]] = []
        for label, subdir in ((0, "real"), (1, "fake")):
            folder = self.root / subdir
            if not folder.exists():
                continue
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() in self.EXTENSIONS:
                    samples.append((str(path), label))

        if not samples:
            raise FileNotFoundError(
                f"No images found under {self.root}/real or {self.root}/fake. "
                "See notebooks/train_colab.ipynb for dataset setup."
            )

        rng = random.Random(seed)
        rng.shuffle(samples)
        split_idx = int(len(samples) * (1.0 - val_ratio))
        if split == "train":
            self.samples = samples[:split_idx]
        else:
            self.samples = samples[split_idx:]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        bgr = __import__("cv2").imread(path)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = __import__("cv2").cvtColor(bgr, __import__("cv2").COLOR_BGR2RGB)

        if self.use_physics:
            physics_vec = self.physics_extractor.extract(rgb).vector
        else:
            physics_vec = np.zeros(PHYSICS_FEATURE_DIM, dtype=np.float32)

        pil = self.preprocessor.preprocess_pil(Image.fromarray(rgb))
        tensor = self.transform(pil)
        return {
            "image": tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "physics": torch.from_numpy(physics_vec),
            "path": path,
        }


def build_loaders(args) -> Tuple[DataLoader, DataLoader]:
    pre = FacePreprocessor()
    train_ds = FaceBinaryDataset(
        args.data_dir,
        split="train",
        val_ratio=args.val_ratio,
        use_physics=args.mode == "hybrid",
        preprocessor=pre,
        transform=get_train_transforms(),
    )
    val_ds = FaceBinaryDataset(
        args.data_dir,
        split="val",
        val_ratio=args.val_ratio,
        use_physics=args.mode == "hybrid",
        preprocessor=pre,
        transform=get_val_transforms(),
    )
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


def train_epoch_stage1(model, loader, criterion, optimizer, device) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        logits, _ = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return {"loss": total_loss / total, "acc": correct / total}


@torch.no_grad()
def eval_stage1(model, loader, criterion, device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits, _ = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return {"loss": total_loss / total, "acc": correct / total}


def train_epoch_hybrid(model, loader, criterion, optimizer, device) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
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
    return {"loss": total_loss / total, "acc": correct / total}


@torch.no_grad()
def eval_hybrid(model, loader, criterion, device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        images = batch["image"].to(device)
        physics = batch["physics"].to(device)
        labels = batch["label"].to(device)
        logits, _ = model(images, physics)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return {"loss": total_loss / total, "acc": correct / total}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer,
    epoch: int,
    metrics: dict,
    mode: str,
    args,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "mode": mode,
            "args": vars(args),
            "physics_dim": PHYSICS_FEATURE_DIM,
        },
        path,
    )
    print(f"Saved checkpoint: {path}")


def run_stage1(args, device) -> None:
    train_loader, val_loader = build_loaders(args)
    model = DeepClassifier(
        model_name=args.backbone,
        pretrained=True,
        freeze_blocks=args.freeze_blocks,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_m = train_epoch_stage1(model, train_loader, criterion, optimizer, device)
        val_m = eval_stage1(model, val_loader, criterion, device)
        scheduler.step()
        record = {"epoch": epoch, "train": train_m, "val": val_m}
        history.append(record)
        print(f"[Stage1 Epoch {epoch}] train loss={train_m['loss']:.4f} acc={train_m['acc']:.4f} | "
              f"val loss={val_m['loss']:.4f} acc={val_m['acc']:.4f}")
        if val_m["acc"] > best_acc:
            best_acc = val_m["acc"]
            save_checkpoint(
                Path(args.output_dir) / "stage1_best.pt",
                model,
                optimizer,
                epoch,
                val_m,
                "stage1",
                args,
            )

    with open(Path(args.output_dir) / "stage1_history.json", "w") as f:
        json.dump(history, f, indent=2)


def run_hybrid(args, device) -> None:
    train_loader, val_loader = build_loaders(args)
    model = HybridClassifier(
        model_name=args.backbone,
        pretrained=True,
        freeze_blocks=args.freeze_blocks,
    ).to(device)

    # Optionally warm-start from stage1 backbone weights
    stage1_ckpt = Path(args.output_dir) / "stage1_best.pt"
    if stage1_ckpt.exists():
        ckpt = torch.load(stage1_ckpt, map_location=device)
        state = ckpt["model_state_dict"]
        # Load matching feature extractor weights
        fe_state = {
            k.replace("feature_extractor.", ""): v
            for k, v in state.items()
            if k.startswith("feature_extractor.")
        }
        model.feature_extractor.load_state_dict(fe_state, strict=False)
        print(f"Loaded stage1 backbone weights from {stage1_ckpt}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_m = train_epoch_hybrid(model, train_loader, criterion, optimizer, device)
        val_m = eval_hybrid(model, val_loader, criterion, device)
        scheduler.step()
        record = {"epoch": epoch, "train": train_m, "val": val_m}
        history.append(record)
        print(f"[Hybrid Epoch {epoch}] train loss={train_m['loss']:.4f} acc={train_m['acc']:.4f} | "
              f"val loss={val_m['loss']:.4f} acc={val_m['acc']:.4f}")
        if val_m["acc"] > best_acc:
            best_acc = val_m["acc"]
            save_checkpoint(
                Path(args.output_dir) / "hybrid_best.pt",
                model,
                optimizer,
                epoch,
                val_m,
                "hybrid",
                args,
            )

    with open(Path(args.output_dir) / "hybrid_history.json", "w") as f:
        json.dump(history, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Train AI vs Real face detector (Colab/Kaggle only)")
    parser.add_argument("--mode", choices=["stage1", "hybrid"], default="stage1")
    parser.add_argument("--data-dir", type=str, default="data", help="Root with real/ and fake/ subdirs")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0")
    parser.add_argument("--freeze-blocks", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Skip GPU check (for smoke tests only — not for real training)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cpu") if args.allow_cpu else require_gpu()
    if args.allow_cpu:
        print("WARNING: CPU mode — smoke test only. Use GPU on Colab for real training.")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.mode == "stage1":
        run_stage1(args, device)
    else:
        run_hybrid(args, device)


if __name__ == "__main__":
    main()
