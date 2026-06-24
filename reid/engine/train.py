import time
import math
from contextlib import nullcontext

import torch
import torch.nn as nn

from reid.engine.logging import (
    make_log_row,
    write_training_log,
)
from reid.engine.validate import validate
from reid.engine.checkpoint import save_checkpoint


def cosine_with_warmup(
    optimizer,
    warmup_epochs: int,
    total_epochs: int,
):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs

        progress = (epoch - warmup_epochs) / max(
            1,
            total_epochs - warmup_epochs,
        )

        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
    )


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scaler,
    ce_loss,
    triplet_loss,
    supcon_loss,
    arcface_loss,
    occlusion_loss,
    device,
    amp_enabled: bool = False,
    ce_weight: float = 1.0,
    triplet_weight: float = 1.0,
    supcon_weight: float = 0.2,
    arcface_weight: float = 0.0,
    occlusion_weight: float = 0.0,
    max_grad_norm: float = 1.0,
    epoch: int | None = None,
    print_every: int = 1,
) -> dict:
    model.train()

    total_loss = 0.0
    ce_running = 0.0
    tri_running = 0.0
    supcon_running = 0.0
    arcface_running = 0.0
    occlusion_running = 0.0
    n_batches = 0

    for imgs, labels, _ in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )

        with amp_ctx:
            embs, logits = model.forward_train(imgs)

            loss_ce = (
                ce_loss(logits, labels) if ce_weight > 0.0 else embs.new_tensor(0.0)
            )
            loss_tri = (
                triplet_loss(embs, labels)
                if triplet_weight > 0.0
                else embs.new_tensor(0.0)
            )
            loss_supcon = (
                supcon_loss(embs, labels)
                if (supcon_loss is not None and supcon_weight > 0.0)
                else embs.new_tensor(0.0)
            )

            if arcface_loss is not None and arcface_weight > 0.0:
                classifier = getattr(model, "classifier", None)
                if classifier is None or not hasattr(classifier, "weight"):
                    raise RuntimeError(
                        "ArcFace is enabled but model.classifier.weight is missing."
                    )

                loss_arcface = arcface_loss(
                    embeddings=embs,
                    labels=labels,
                    classifier_weight=classifier.weight,
                )
            else:
                loss_arcface = embs.new_tensor(0.0)

            loss_occlusion = (
                occlusion_loss(embeddings=embs)
                if (occlusion_loss is not None and occlusion_weight > 0.0)
                else embs.new_tensor(0.0)
            )

            loss = (
                ce_weight * loss_ce
                + triplet_weight * loss_tri
                + supcon_weight * loss_supcon
                + arcface_weight * loss_arcface
                + occlusion_weight * loss_occlusion
            )

        optimizer.zero_grad(set_to_none=True)

        if amp_enabled:
            scaler.scale(loss).backward()

            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=max_grad_norm,
                )

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            if max_grad_norm is not None:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=max_grad_norm,
                )

            optimizer.step()

        total_loss += loss.item()
        ce_running += loss_ce.item()
        tri_running += loss_tri.item()
        supcon_running += loss_supcon.item()
        arcface_running += loss_arcface.item()
        occlusion_running += loss_occlusion.item()
        n_batches += 1

        if print_every and n_batches % print_every == 0:
            prefix = f"  Epoch {epoch:03d}" if epoch is not None else "  Epoch"
            print(
                f"\r{prefix}  Batch {n_batches:03d}  "
                f"loss={loss.item():.4f}  "
                f"ce={loss_ce.item():.4f}  "
                f"tri={loss_tri.item():.4f}  "
                f"supcon={loss_supcon.item():.4f}  "
                f"arc={loss_arcface.item():.4f}  "
                f"occ={loss_occlusion.item():.4f}",
                end="",
            )

    if n_batches == 0:
        raise RuntimeError("Training loader produced zero batches.")

    print()

    return {
        "loss": total_loss / n_batches,
        "loss_ce": ce_running / n_batches,
        "loss_triplet": tri_running / n_batches,
        "loss_supcon": supcon_running / n_batches,
        "loss_arcface": arcface_running / n_batches,
        "loss_occlusion": occlusion_running / n_batches,
        "num_batches": n_batches,
    }


def train_model(
    model,
    train_loader,
    query_loader,
    gallery_loader,
    optimizer,
    scheduler,
    ce_loss,
    triplet_loss,
    supcon_loss,
    arcface_loss,
    occlusion_loss,
    device,
    *,
    epochs: int,
    backbone: str,
    ce_weight: float = 1.0,
    triplet_weight: float = 1.0,
    supcon_weight: float = 0.2,
    arcface_weight: float = 0.0,
    occlusion_weight: float = 0.0,
    eval_interval: int = 50,
    checkpoint_dir: str = "checkpoints",
    log_path: str = "training_log.csv",
    max_grad_norm: float = 1.0,
    amp: bool = True,
) -> dict:
    print("Training started...")

    amp_enabled = amp and (device.type == "cuda")
    amp_module = getattr(torch, "amp", None)
    if amp_module is not None and hasattr(amp_module, "GradScaler"):
        scaler = amp_module.GradScaler("cuda", enabled=amp_enabled)
    else:
        scaler = torch.amp.GradScaler(enabled=amp_enabled)

    if amp and not amp_enabled:
        print("  [NOTE] mixed_precision requested but CUDA is unavailable; using FP32.")
    else:
        print(f"  mixed_precision={'on' if amp_enabled else 'off'}")

    log_rows = []
    best_mAP = 0.0
    last_rank1 = 0.0
    last_mAP = 0.0

    best_checkpoint_path = f"{checkpoint_dir}/best_model.pth"
    last_checkpoint_path = f"{checkpoint_dir}/last_model.pth"

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_stats = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            ce_loss=ce_loss,
            triplet_loss=triplet_loss,
            supcon_loss=supcon_loss,
            arcface_loss=arcface_loss,
            occlusion_loss=occlusion_loss,
            device=device,
            amp_enabled=amp_enabled,
            ce_weight=ce_weight,
            triplet_weight=triplet_weight,
            supcon_weight=supcon_weight,
            arcface_weight=arcface_weight,
            occlusion_weight=occlusion_weight,
            max_grad_norm=max_grad_norm,
            epoch=epoch,
        )

        if scheduler is not None:
            scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start_time

        should_validate = epoch % eval_interval == 0 or epoch == epochs

        rank1 = 0.0
        rank5 = 0.0
        rank10 = 0.0
        mAP = 0.0
        is_best = False

        # Save checkpoint just in case.
        save_checkpoint(
            path=last_checkpoint_path,
            model=model,
            epoch=epochs,
            backbone=backbone,
            rank1=last_rank1,
            mAP=last_mAP,
        )

        if should_validate:
            rank1, rank5, rank10, mAP = validate(
                model=model,
                query_loader=query_loader,
                gallery_loader=gallery_loader,
                device=device,
            )

            last_rank1 = rank1
            last_mAP = mAP

            if mAP > best_mAP:
                best_mAP = mAP
                is_best = True

                save_checkpoint(
                    path=best_checkpoint_path,
                    model=model,
                    epoch=epoch,
                    backbone=backbone,
                    rank1=rank1,
                    mAP=mAP,
                )

        log_rows.append(
            make_log_row(
                epoch=epoch,
                loss=train_stats["loss"],
                loss_ce=train_stats["loss_ce"],
                loss_triplet=train_stats["loss_triplet"],
                loss_supcon=train_stats["loss_supcon"],
                loss_arcface=train_stats["loss_arcface"],
                loss_occlusion=train_stats["loss_occlusion"],
                lr=lr_now,
                rank1=rank1,
                mAP=mAP,
            )
        )

        if should_validate:
            star = "  ★" if is_best else ""
            print(
                f"Epoch [{epoch:03d}/{epochs}]  "
                f"loss={train_stats['loss']:.4f}  "
                f"ce={train_stats['loss_ce']:.4f}  "
                f"tri={train_stats['loss_triplet']:.4f}  "
                f"supcon={train_stats['loss_supcon']:.4f}  "
                f"arc={train_stats['loss_arcface']:.4f}  "
                f"occ={train_stats['loss_occlusion']:.4f}  "
                f"lr={lr_now:.2e}  "
                f"Rank-1={rank1:.4f}  "
                f"Rank-5={rank5:.4f}  "
                f"Rank-10={rank10:.4f}  "
                f"mAP={mAP:.4f}  "
                f"[{elapsed:.1f}s]"
                f"{star}"
            )
        else:
            print(
                f"Epoch [{epoch:03d}/{epochs}]  "
                f"loss={train_stats['loss']:.4f}  "
                f"ce={train_stats['loss_ce']:.4f}  "
                f"tri={train_stats['loss_triplet']:.4f}  "
                f"supcon={train_stats['loss_supcon']:.4f}  "
                f"arc={train_stats['loss_arcface']:.4f}  "
                f"occ={train_stats['loss_occlusion']:.4f}  "
                f"lr={lr_now:.2e}  "
                f"[{elapsed:.1f}s]"
            )

    # Overwrite last checkpoint with score info.
    save_checkpoint(
        path=last_checkpoint_path,
        model=model,
        epoch=epochs,
        backbone=backbone,
        rank1=last_rank1,
        mAP=last_mAP,
    )

    write_training_log(
        path=log_path,
        rows=log_rows,
    )

    print("\n[Training Complete]")
    print(f"  Best mAP : {best_mAP:.4f}")
    print(f"  Checkpoints saved in: {checkpoint_dir}/")
    print(f"  Log saved: {log_path}")

    return {
        "best_mAP": best_mAP,
        "last_rank1": last_rank1,
        "last_mAP": last_mAP,
        "log_rows": log_rows,
    }
