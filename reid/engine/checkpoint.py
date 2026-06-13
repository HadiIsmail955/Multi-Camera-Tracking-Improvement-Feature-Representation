# src/engine/checkpoint.py

import os
import torch


def save_checkpoint(
    path: str,
    model,
    epoch: int,
    backbone: str,
    rank1: float,
    mAP: float,
    extra: dict | None = None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "backbone": backbone,
        "state_dict": model.state_dict(),
        "rank1": rank1,
        "mAP": mAP,
    }

    if extra:
        checkpoint.update(extra)

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    device,
):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return torch.load(path, map_location=device)
