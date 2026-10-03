"""
Train the single-branch models (physics_only, prnu_only, semantic_only) from the
feature cache. No image decoding, no feature extraction: the whole run is a few
minutes on the GPU.

Uses the same CLI as train.py (it reuses train.parse_args), so checkpoints carry
the same 'args', normalizers and keys, and are written with train.save_checkpoint.

    python src/train_cached.py --mode physics_only --feature-cache /mnt/data/feature_cache \
        --data-dir /mnt/data --output-dir ~/models/physics_only \
        --epochs 30 --batch-size 512 --lr 1e-3 --max-train-per-class 100000
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.train as T
from src.classifier.head import BranchOnlyClassifier
from src.features.feature_scalers import fit_physics_and_prnu_scalers
from src.physics_branch.feature_vector import PHYSICS_FEATURE_DIM
from src.prnu_branch.extractor import PRNU_FEATURE_DIM

KEY = {"physics_only": "physics", "prnu_only": "prnu", "semantic_only": "semantic"}


def load_split(cache_dir, split):
    z = np.load(Path(cache_dir) / f"{split}.npz")
    return {k: z[k] for k in z.files}


def _binary_metrics(y, pred, prob):
    out = {"samples": int(len(y)), "acc": float(accuracy_score(y, pred))}
    out["accuracy"] = out["acc"]
    if len(set(y.tolist())) == 2:
        pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        out.update(precision=float(pr), recall=float(rc), f1=float(f1), f1_score=float(f1))
        out["roc_auc"] = float(roc_auc_score(y, prob))
    else:
        out["roc_auc"] = None
    return out


@torch.no_grad()
def evaluate(model, X, y, sources=None, bs=8192):
    model.eval()
    logits = torch.cat([model.classifier(X[i:i + bs]) for i in range(0, len(X), bs)])
    loss = float(F.cross_entropy(logits, y).item())
    prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    pred = logits.argmax(dim=1).cpu().numpy()
    yt = y.cpu().numpy()
    metrics = _binary_metrics(yt, pred, prob)
    metrics["loss"] = loss
    if sources is not None:
        by_source = {}
        for s in np.unique(sources):
            m = sources == s
            by_source[str(s)] = _binary_metrics(yt[m], pred[m], prob[m])
        metrics["by_source"] = by_source
    return metrics


def main():
    args = T.parse_args()
    mode = args.mode
    if mode not in KEY:
        sys.exit("train_cached.py handles physics_only / prnu_only / semantic_only. "
                 "Use train.py for stage1 and full_hybrid.")
    if not args.feature_cache:
        sys.exit("Pass --feature-cache <dir written by cache_features.py>")

    T.set_seed(args.seed)
    device = torch.device("cpu") if args.allow_cpu else T.require_gpu()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    key = KEY[mode]

    tr = load_split(args.feature_cache, "train")
    va = load_split(args.feature_cache, "val")
    te = load_split(args.feature_cache, "test")

    rows = np.arange(len(tr["labels"]))
    if args.max_train_per_class > 0:
        chosen = T.subsample_balanced(
            [(int(i), int(l), "") for i, l in zip(rows, tr["labels"])],
            args.max_train_per_class, args.seed,
        )
        rows = np.array(sorted(s[0] for s in chosen))
    print(f"[{mode}] train rows: {len(rows)}  val: {len(va['labels'])}  test: {len(te['labels'])}")

    physics_norm = prnu_norm = norm = None
    if mode in {"physics_only", "prnu_only"}:
        physics_norm, prnu_norm = fit_physics_and_prnu_scalers([str(tr["paths"][i]) for i in rows])
        norm = physics_norm if mode == "physics_only" else prnu_norm

    dims = {"physics_only": PHYSICS_FEATURE_DIM, "prnu_only": PRNU_FEATURE_DIM,
            "semantic_only": args.semantic_dim}

    def prep(d, idx=None):
        X = d[key] if idx is None else d[key][idx]
        if norm is not None:  # identical math to BranchOnlyClassifier._prepare, done once on bulk data
            X = norm.normalize_batch(X)
        return torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(device)

    def labels(d, idx=None):
        y = d["labels"] if idx is None else d["labels"][idx]
        return torch.from_numpy(y.astype(np.int64)).to(device)

    Xtr, ytr = prep(tr, rows), labels(tr, rows)
    Xva, yva = prep(va), labels(va)
    Xte, yte = prep(te), labels(te)
    if Xtr.shape[1] != dims[mode]:
        sys.exit(f"Feature width {Xtr.shape[1]} != expected {dims[mode]} for {mode}")

    model = BranchOnlyClassifier(input_dim=dims[mode], normalize=norm is not None).to(device)
    if mode == "physics_only":
        model.set_normalizer(physics_norm)
    elif mode == "prnu_only":
        model.set_normalizer(prnu_norm)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_path = out_dir / f"{mode}_best.pt"
    last_path = out_dir / f"{mode}_last.pt"
    best_acc, history = -1.0, []

    def save(path, epoch, val_metrics):
        T.save_checkpoint(
            path, model, optimizer, epoch, val_metrics, mode, args,
            physics_normalizer=physics_norm if mode == "physics_only" else None,
            prnu_normalizer=prnu_norm if mode == "prnu_only" else None,
            scheduler=scheduler, best_acc=best_acc,
        )

    n, bs = len(Xtr), args.batch_size
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        total, correct, loss_sum = 0, 0, 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:  # BatchNorm needs >1 sample
                continue
            logits = model.classifier(Xtr[idx])
            loss = F.cross_entropy(logits, ytr[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(idx)
            correct += int((logits.argmax(1) == ytr[idx]).sum().item())
            total += len(idx)
        scheduler.step()
        val = evaluate(model, Xva, yva)
        history.append({"epoch": epoch,
                        "train": {"loss": loss_sum / total, "acc": correct / total},
                        "val": val})
        print(f"[{mode} Epoch {epoch}/{args.epochs}] train_acc={correct / total:.4f} "
              f"val_acc={val['acc']:.4f}")
        if val["acc"] > best_acc:
            best_acc = val["acc"]
            save(best_path, epoch, val)
        save(last_path, epoch, val)

    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    test = evaluate(model, Xte, yte, sources=te["sources"])
    (out_dir / f"{mode}_history.json").write_text(
        json.dumps({"epochs": history, "test": test, "best_val_acc": best_acc}, indent=2))
    print(json.dumps({k: v for k, v in test.items() if k != "by_source"}, indent=2))
    print("Best checkpoint:", best_path)


if __name__ == "__main__":
    main()
