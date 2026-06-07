import torch
import torch.nn.functional as F

def get_current_lr(optimizer) -> float:
    return optimizer.param_groups[0]["lr"]

def get_all_lrs(optimizer):
    return [group["lr"] for group in optimizer.param_groups]

def build_optimizer(model, args):
    head_lr = getattr(args, "head_lr", getattr(args, "lr", 3e-4))
    backbone_lr = getattr(args, "backbone_lr", 1e-5)

    head_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("backbone."):
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = []

    if len(head_params) > 0:
        param_groups.append({"params": head_params, "lr": head_lr})

    if len(backbone_params) > 0:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})

    if len(param_groups) == 0:
        raise RuntimeError("No trainable parameters found.")

    return torch.optim.AdamW(
        param_groups,
        weight_decay=args.weight_decay,
    )


def build_scheduler(optimizer, args):
    scheduler_name = str(getattr(args, "scheduler", "warmup_cosine")).lower()
    warmup_epochs = int(getattr(args, "warmup_epochs", 3))
    min_lr = float(getattr(args, "min_lr", 1e-6))
    epochs = int(args.epochs)

    if scheduler_name in {"none", "constant", "off"}:
        return None

    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(getattr(args, "plateau_factor", 0.5)),
            patience=int(getattr(args, "plateau_patience", 2)),
            min_lr=min_lr,
        )

    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(epochs, 1),
            eta_min=min_lr,
        )

    if scheduler_name == "cosine_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(getattr(args, "restart_t0", 10)),
            T_mult=int(getattr(args, "restart_tmult", 2)),
            eta_min=min_lr,
        )

    if scheduler_name != "warmup_cosine":
        raise ValueError(
            "Unknown scheduler. Use one of: "
            "warmup_cosine, cosine, cosine_restarts, plateau, none"
        )

    if warmup_epochs <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(epochs, 1),
            eta_min=min_lr,
        )

    warmup_epochs = min(warmup_epochs, max(epochs - 1, 1))

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=float(getattr(args, "warmup_start_factor", 0.1)),
        total_iters=warmup_epochs,
    )

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs - warmup_epochs, 1),
        eta_min=min_lr,
    )

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def step_scheduler_after_epoch(scheduler, args, val_metrics=None):
    if scheduler is None:
        return

    scheduler_name = str(getattr(args, "scheduler", "warmup_cosine")).lower()

    if scheduler_name == "plateau":
        if val_metrics is not None and "mAP" in val_metrics:
            scheduler.step(float(val_metrics["mAP"]))
        return

    scheduler.step()
