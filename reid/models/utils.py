import random

import numpy as np
import torch
import torch.nn as nn


def param_summary(model: nn.Module, name: str) -> dict:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    frozen = total - trainable
    pct = 100.0 * trainable / total if total > 0 else 0.0

    print(f"\n  [{name}] Parameter summary")
    print(f"    Total      : {total:>12,}")
    print(f"    Trainable  : {trainable:>12,}  ({pct:.2f}%)")
    print(f"    Frozen     : {frozen:>12,}")

    return {
        "model": name,
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": round(pct, 4),
    }

def enable_determinism(seed: int = 42, use_deterministic: bool = False) -> None:
    """Enable determinism across Python, Pytorch and Numpy.
    Args:
        seed (int): Sets the seed for determinism. Defaults to 42.
        use_deterministic (bool): Use deterministic algorithms only.
        A `RuntimeError` will be thrown when non-deterministic algorithms are
        applied. Defaults to True.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(use_deterministic)
