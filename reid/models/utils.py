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
