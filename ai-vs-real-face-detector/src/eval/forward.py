"""Dispatch a batch through a loaded checkpoint for evaluation/calibration."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch


def logits_from_batch(
    model: torch.nn.Module,
    mode: str,
    batch: Dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    image = batch.get("image")
    if image is not None:
        image = image.to(device, non_blocking=True)
    if mode == "stage1":
        logits, _ = model(image)
    elif mode == "hybrid":
        logits, _ = model(image, batch["physics"].to(device, non_blocking=True))
    elif mode == "full_hybrid":
        extras = {}
        if batch.get("freq") is not None:
            extras["freq"] = batch["freq"].to(device, non_blocking=True)
        try:
            logits, _ = model(
                image,
                batch["physics"].to(device, non_blocking=True),
                batch["prnu"].to(device, non_blocking=True) if batch.get("prnu") is not None else None,
                batch["semantic"].to(device, non_blocking=True) if batch.get("semantic") is not None else None,
                **extras,
            )
        except TypeError:
            logits, _ = model(
                image,
                batch["physics"].to(device, non_blocking=True),
                batch["prnu"].to(device, non_blocking=True),
                batch["semantic"].to(device, non_blocking=True),
            )
    elif mode == "physics_only":
        logits, _ = model(batch["physics"].to(device, non_blocking=True))
    elif mode == "prnu_only":
        logits, _ = model(batch["prnu"].to(device, non_blocking=True))
    elif mode in {"semantic_only", "clip_linear"}:
        logits, _ = model(batch["semantic"].to(device, non_blocking=True))
    elif mode == "freq_only":
        logits, _ = model(batch["freq"].to(device, non_blocking=True))
    else:
        raise ValueError(f"Unsupported checkpoint mode: {mode}")
    return logits


@torch.no_grad()
def predict_loader(
    model: torch.nn.Module,
    mode: str,
    loader,
    device: torch.device,
) -> Dict[str, Any]:
    from src.eval.splits import folder_of

    model.eval()
    paths, sources, labels, p_ai, logit_real, logit_ai = [], [], [], [], [], []
    gate_sums: Dict[str, float] = {}
    gate_count = 0
    for batch in loader:
        logits = logits_from_batch(model, mode, batch, device).float()
        probs = torch.softmax(logits, dim=1)
        n = logits.size(0)
        batch_paths = batch.get("path")
        if batch_paths is None:
            batch_paths = [""] * n
        elif isinstance(batch_paths, torch.Tensor):
            batch_paths = [str(p) for p in batch_paths]
        batch_sources = batch.get("source")
        if batch_sources is None:
            batch_sources = ["unknown"] * n
        for i in range(n):
            paths.append(str(batch_paths[i]))
            sources.append(str(batch_sources[i]))
            labels.append(int(batch["label"][i].item()))
            p_ai.append(float(probs[i, 1].item()))
            logit_real.append(float(logits[i, 0].item()))
            logit_ai.append(float(logits[i, 1].item()))
        if hasattr(model, "get_fusion_weights") and mode == "full_hybrid":
            try:
                physics = batch["physics"].to(device)
                prnu = batch.get("prnu")
                semantic = batch.get("semantic")
                weights = model.get_fusion_weights(
                    batch["image"].to(device),
                    physics,
                    prnu.to(device) if prnu is not None else None,
                    semantic.to(device) if semantic is not None else None,
                )
                for name, value in weights.items():
                    gate_sums[name] = gate_sums.get(name, 0.0) + float(value) * n
                gate_count += n
            except Exception:
                pass
    mean_gates = (
        {name: value / max(gate_count, 1) for name, value in gate_sums.items()}
        if gate_count
        else {}
    )
    folders = [folder_of(source) for source in sources]
    return {
        "path": paths,
        "source": sources,
        "folder": folders,
        "label": labels,
        "p_ai": p_ai,
        "logit_real": logit_real,
        "logit_ai": logit_ai,
        "mean_gates": mean_gates,
    }


def predictions_to_rows(collected: Dict[str, Any], split: str = "report") -> list:
    from src.eval.predictions import empty_row

    rows = []
    n = len(collected["label"])
    for i in range(n):
        rows.append(
            empty_row(
                path=collected["path"][i],
                folder=collected["folder"][i],
                source=collected["source"][i],
                label=collected["label"][i],
                p_ai=collected["p_ai"][i],
                logit_real=collected["logit_real"][i],
                logit_ai=collected["logit_ai"][i],
                split=split,
            )
        )
    return rows
