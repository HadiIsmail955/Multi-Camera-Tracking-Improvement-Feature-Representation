from typing import Dict, Tuple
import torch
from pathlib import Path
from .helper import log_info


def checkpoint_payload(
    model,
    optimizer,
    scheduler,
    epoch,
    train_dataset,
    args,
    train_metrics,
    val_metrics=None,
    scaler=None,
    best_val_map=-1.0,
    best_val_rank1=-1.0,
    best_train_loss=float("inf"),
):
    return {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "id_to_label": train_dataset.id_to_label,
        "label_to_id": train_dataset.label_to_id,
        "camera_to_id": train_dataset.camera_to_id,
        "args": vars(args),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "best_val_map": float(best_val_map),
        "best_val_rank1": float(best_val_rank1),
        "best_train_loss": float(best_train_loss),
    }


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    train_dataset,
    args,
    train_metrics,
    val_metrics=None,
    scaler=None,
    best_val_map=-1.0,
    best_val_rank1=-1.0,
    best_train_loss=float("inf"),
):
    torch.save(
        checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            train_dataset=train_dataset,
            args=args,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            scaler=scaler,
            best_val_map=best_val_map,
            best_val_rank1=best_val_rank1,
            best_train_loss=best_train_loss,
        ),
        path,
    )


def save_checkpoint_with_logger(
    logger,
    model,
    optimizer,
    scheduler,
    epoch,
    train_dataset,
    args,
    train_metrics,
    val_metrics,
    name,
):
    logger.save_checkpoint(
        model=model,
        name=name,
        epoch=epoch,
        optimizer=optimizer,
        scheduler=scheduler,
        id_to_label=train_dataset.id_to_label,
        label_to_id=train_dataset.label_to_id,
        camera_to_id=train_dataset.camera_to_id,
        args=vars(args),
        train_metrics=train_metrics,
        val_metrics=val_metrics,
    )


def save_any_checkpoint(
    logger,
    out_dir,
    model,
    optimizer,
    scheduler,
    epoch,
    train_dataset,
    args,
    train_metrics,
    val_metrics,
    name,
    scaler=None,
    best_val_map=-1.0,
    best_val_rank1=-1.0,
    best_train_loss=float("inf"),
):
    if logger is not None and hasattr(logger, "save_checkpoint"):
        save_checkpoint_with_logger(
            logger=logger,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            train_dataset=train_dataset,
            args=args,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            name=name,
        )

    save_checkpoint(
        path=out_dir / name,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        train_dataset=train_dataset,
        args=args,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        scaler=scaler,
        best_val_map=best_val_map,
        best_val_rank1=best_val_rank1,
        best_train_loss=best_train_loss,
    )


def _strip_state_dict_prefixes(state_dict):
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected state_dict dict, got {type(state_dict)}")

    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def _extract_model_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(checkpoint)}")

    for key in ("model", "state_dict", "model_state_dict"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return _strip_state_dict_prefixes(checkpoint[key])

    if all(torch.is_tensor(v) for v in checkpoint.values()):
        return _strip_state_dict_prefixes(checkpoint)

    raise KeyError(
        "Could not find model weights in checkpoint. Expected key 'model', "
        "'state_dict', 'model_state_dict', or a raw state_dict."
    )


def load_model_weights(
    model,
    checkpoint_path,
    device,
    logger=None,
    strict=True,
    ignore_shape_mismatch=False,
):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = _extract_model_state_dict(checkpoint)

    if ignore_shape_mismatch:
        model_state = model.state_dict()
        filtered = {}
        skipped = []

        for key, value in state_dict.items():
            if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
                filtered[key] = value
            else:
                skipped.append(key)

        missing, unexpected = model.load_state_dict(filtered, strict=False)
        log_info(logger, f"Loaded model weights from: {checkpoint_path}")
        log_info(logger, f"Loaded tensors: {len(filtered)}")
        if skipped:
            log_info(logger, f"Skipped shape-mismatch/unmatched tensors: {skipped[:20]}")
        if missing:
            log_info(logger, f"Missing tensors after partial load: {list(missing)[:20]}")
        if unexpected:
            log_info(logger, f"Unexpected tensors after partial load: {list(unexpected)[:20]}")
        return checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    log_info(logger, f"Loaded model weights from: {checkpoint_path}")

    if not strict:
        if missing:
            log_info(logger, f"Missing tensors: {list(missing)[:20]}")
        if unexpected:
            log_info(logger, f"Unexpected tensors: {list(unexpected)[:20]}")

    return checkpoint


def maybe_load_training_state(
    model,
    optimizer,
    scheduler,
    scaler,
    args,
    device,
    logger=None,
):
    start_epoch = 1
    best_val_map = -1.0
    best_val_rank1 = -1.0
    best_train_loss = float("inf")

    resume_path = getattr(args, "resume", None)
    load_model_path = getattr(args, "load_model", None)

    if resume_path:
        checkpoint = load_model_weights(
            model=model,
            checkpoint_path=resume_path,
            device=device,
            logger=logger,
            strict=bool(getattr(args, "resume_strict", True)),
            ignore_shape_mismatch=bool(getattr(args, "resume_ignore_shape_mismatch", False)),
        )

        ckpt_epoch = int(checkpoint.get("epoch", 0))
        best_val_map = float(checkpoint.get("best_val_map", -1.0))
        best_val_rank1 = float(checkpoint.get("best_val_rank1", -1.0))
        best_train_loss = float(checkpoint.get("best_train_loss", float("inf")))

        if getattr(args, "resume_add_epochs", 0) and int(args.resume_add_epochs) > 0:
            args.epochs = ckpt_epoch + int(args.resume_add_epochs)
            log_info(logger, f"resume_add_epochs={args.resume_add_epochs}; total epochs set to {args.epochs}")

        if not bool(getattr(args, "resume_reset_optimizer", False)):
            if checkpoint.get("optimizer") is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
                log_info(logger, "Optimizer state restored.")
            else:
                log_info(logger, "Checkpoint has no optimizer state; using fresh optimizer.")

            if scheduler is not None and checkpoint.get("scheduler") is not None:
                try:
                    scheduler.load_state_dict(checkpoint["scheduler"])
                    log_info(logger, "Scheduler state restored.")
                except Exception as exc:
                    log_info(logger, f"Could not restore scheduler state: {exc}. Using fresh scheduler.")
            elif scheduler is not None:
                log_info(logger, "Checkpoint has no scheduler state; using fresh scheduler.")

            if scaler is not None and checkpoint.get("scaler") is not None:
                try:
                    scaler.load_state_dict(checkpoint["scaler"])
                    log_info(logger, "AMP scaler state restored.")
                except Exception as exc:
                    log_info(logger, f"Could not restore AMP scaler: {exc}. Using fresh scaler.")

            start_epoch = ckpt_epoch + 1
        else:
            log_info(logger, "resume_reset_optimizer=True: loaded model only; optimizer/scheduler/scaler reset.")
            start_epoch = 1

        log_info(logger, f"Resume start_epoch={start_epoch}, target epochs={args.epochs}")
        return start_epoch, best_val_map, best_val_rank1, best_train_loss

    if load_model_path:
        load_model_weights(
            model=model,
            checkpoint_path=load_model_path,
            device=device,
            logger=logger,
            strict=bool(getattr(args, "load_model_strict", False)),
            ignore_shape_mismatch=bool(getattr(args, "load_model_ignore_shape_mismatch", True)),
        )
        log_info(logger, "Model-only checkpoint loaded. Optimizer and scheduler are fresh.")

    return start_epoch, best_val_map, best_val_rank1, best_train_loss
