import argparse
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ..dataLoader.customData.MTMCCSVDataset import MTMCCSVDataset
from ..dataLoader.transformation.ReIDTransform import ReIDTransform
from ..model.DINOv2ReID import DINOv2ReID
from ..losses.contrastiveLoss import ReIDLoss

from ..utils.optimizerAndScheduler import build_optimizer, build_scheduler, get_current_lr, get_all_lrs, step_scheduler_after_epoch
from ..metrics.similarityMetrics import compute_embedding_similarity_metrics, compute_embedding_retrieval_metrics
from ..utils.loadAndSaveModel import save_checkpoint, save_any_checkpoint, maybe_load_training_state
from ..utils.dataloader import build_train_loader, build_eval_loader, maybe_subset_eval_dataset
from ..utils.embedding import validate_embeddings_with_labels, extract_validation_embeddings
from ..utils.helper import log_info, move_batch_to_device, parse_model_output, tensor_outputs_to_float

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device: str,
    epoch: int,
    logger=None,
    log_every: int = 20,
    grad_clip: float = 5.0,
    scaler=None,
    use_amp: bool = True,
):
    model.train()

    total_loss = 0.0
    total_id = 0.0
    total_triplet = 0.0
    total_contrastive = 0.0
    total_occlusion = 0.0

    total_steps = len(loader)
    amp_enabled = bool(use_amp and device == "cuda")

    start_time = time.time()

    for step, batch in enumerate(loader, start=1):
        images, labels, cameras, is_occluded = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            raw_output = model(images)

        outputs = tensor_outputs_to_float(parse_model_output(raw_output))

        loss_dict = criterion(
            outputs=outputs,
            labels=labels,
            cameras=cameras,
            is_occluded=is_occluded,
        )

        loss = loss_dict["loss"]

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=grad_clip,
            )

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=grad_clip,
            )

            optimizer.step()

        total_loss += loss.item()
        total_id += loss_dict["loss_id"].item()
        total_triplet += loss_dict["loss_triplet"].item()
        total_contrastive += loss_dict["loss_contrastive"].item()
        total_occlusion += loss_dict.get("loss_occlusion", torch.tensor(0.0)).item()

        should_log = step == 1 or step % log_every == 0 or step == total_steps

        if should_log:
            metrics = {
                "loss": loss.item(),
                "id": loss_dict["loss_id"].item(),
                "triplet": loss_dict["loss_triplet"].item(),
                "contrastive": loss_dict["loss_contrastive"].item(),
                "occlusion": loss_dict.get("loss_occlusion", torch.tensor(0.0)).item(),
                "lr": get_current_lr(optimizer),
            }

            if logger is not None and hasattr(logger, "log_step"):
                logger.log_step(
                    epoch=epoch,
                    step=step,
                    total_steps=total_steps,
                    metrics=metrics,
                )
            else:
                log_info(
                    logger,
                    (
                        f"epoch={epoch} | "
                        f"step={step}/{total_steps} | "
                        f"loss={metrics['loss']:.4f} | "
                        f"id={metrics['id']:.4f} | "
                        f"triplet={metrics['triplet']:.4f} | "
                        f"contrastive={metrics['contrastive']:.4f} | "
                        f"occ={metrics['occlusion']:.4f} | "
                        f"lr={metrics['lr']:.8f}"
                    ),
                )

    n = max(total_steps, 1)
    epoch_time = time.time() - start_time

    return {
        "epoch": epoch,
        "loss": total_loss / n,
        "loss_id": total_id / n,
        "loss_triplet": total_triplet / n,
        "loss_contrastive": total_contrastive / n,
        "loss_occlusion": total_occlusion / n,
        "lr": get_current_lr(optimizer),
        "epoch_time_sec": epoch_time,
    }

def main(args, logger=None):
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_info(logger, "=" * 80)
    log_info(logger, "Starting ReID embedding training")
    log_info(logger, f"Device: {device}")

    if torch.cuda.is_available():
        log_info(logger, f"GPU: {torch.cuda.get_device_name(0)}")

    log_info(logger, "=" * 80)

    train_transform = ReIDTransform(
        backbone=args.backbone,
        img_size=args.image_size,
        train=True,
    )

    eval_transform = ReIDTransform(
        backbone=args.backbone,
        img_size=args.image_size,
        train=False,
    )

    train_dataset = MTMCCSVDataset(
        root=args.data_root,
        split="train",
        scene_folders=args.train_scenes,
        transform=train_transform,
        min_images_per_id=args.min_images_per_id,
        min_normal_images_per_id=args.min_normal_images_per_id,
        object_types=args.object_types,
        base_path=args.base_path,
        debug=args.debug_dataset,
        verify_paths=args.verify_paths,
        scene_aware_ids=args.scene_aware_ids,
        min_cameras_per_id=args.min_cameras_per_id,
        include_occlusion_crops=args.include_occlusion_crops,
        only_occlusion_crops=False,
        metadata_filename=args.metadata_filename,
    )

    train_loader = build_train_loader(
        train_dataset=train_dataset,
        args=args,
        logger=logger,
    )

    val_dataset = None
    val_loader = None

    if args.use_validation:
        val_dataset = MTMCCSVDataset(
            root=args.data_root,
            split=args.val_split,
            scene_folders=args.val_scenes,
            transform=eval_transform,
            min_images_per_id=args.min_images_per_id,
            min_normal_images_per_id=args.min_normal_images_per_id,
            object_types=args.object_types,
            base_path=args.base_path,
            debug=args.debug_dataset,
            verify_paths=args.verify_paths,
            scene_aware_ids=args.scene_aware_ids,
            min_cameras_per_id=args.min_cameras_per_id,
            include_occlusion_crops=args.val_include_occlusion_crops,
            only_occlusion_crops=False,
            metadata_filename=args.metadata_filename,
        )

        val_eval_dataset = maybe_subset_eval_dataset(
            val_dataset,
            max_samples=args.max_val_samples,
            seed=args.val_subset_seed,
        )

        val_loader = build_eval_loader(val_eval_dataset, args)

    if logger is not None and hasattr(logger, "log_dataset"):
        logger.log_dataset(train_dataset=train_dataset)
    else:
        log_info(logger, f"Train samples: {len(train_dataset)}")
        log_info(logger, f"Train classes: {train_dataset.num_classes}")
        log_info(logger, f"Train cameras: {len(train_dataset.camera_to_id)}")
        log_info(logger, f"Train scenes: {train_dataset.scene_folders}")

        if val_dataset is not None:
            log_info(logger, f"Val samples full: {len(val_dataset)}")
            log_info(logger, f"Val samples used: {len(val_loader.dataset)}")
            log_info(logger, f"Validation every: {args.val_every} epoch(s)")
            log_info(logger, "Validation mode: label-aware embedding retrieval, no logits")
        else:
            log_info(logger, "Validation: disabled. Set --use_validation to validate embeddings.")

    model = DINOv2ReID(
        num_classes=train_dataset.num_classes,
        dino_type=args.backbone_type,
        freeze_backbone=args.freeze_backbone,
        embedding_dim=args.embedding_dim,
        unfreeze_last_blocks=args.unfreeze_last_blocks,
        dropout=args.dropout,
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    log_info(logger, f"Total parameters: {total_params:,}")
    log_info(logger, f"Trainable parameters: {trainable_params:,}")

    criterion = ReIDLoss(
        id_weight=args.id_weight,
        triplet_weight=args.triplet_weight,
        contrastive_weight=args.contrastive_weight,
        occlusion_consistency_weight=args.occlusion_consistency_weight,
        triplet_margin=args.triplet_margin,
        temperature=args.temperature,
        label_smoothing=args.label_smoothing,
        cross_camera_weight=args.cross_camera_weight,
        same_camera_weight=args.same_camera_weight,
        occlusion_positive_weight=args.occlusion_positive_weight,
        metric_embedding_key=args.metric_embedding_key,
    ).to(device)

    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)

    log_info(logger, f"Optimizer parameter groups: {len(optimizer.param_groups)}")
    for i, group in enumerate(optimizer.param_groups):
        group_params = sum(p.numel() for p in group["params"])
        log_info(logger, f"  group {i}: lr={group['lr']:.8f}, params={group_params:,}")

    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(args.use_amp and device == "cuda")
    )

    start_epoch, best_val_map, best_val_rank1, best_train_loss = maybe_load_training_state(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        args=args,
        device=device,
        logger=logger,
    )

    if start_epoch > args.epochs:
        log_info(
            logger,
            f"Checkpoint epoch is already past target epochs: start_epoch={start_epoch}, epochs={args.epochs}. Nothing to train.",
        )
        return

    for epoch in range(start_epoch, args.epochs + 1):
        log_info(logger, "")
        log_info(logger, f"Epoch {epoch}/{args.epochs}")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            logger=logger,
            log_every=args.log_every,
            grad_clip=args.grad_clip,
            scaler=scaler,
            use_amp=args.use_amp,
        )

        val_metrics = None

        should_validate = (
            val_loader is not None
            and (
                epoch == 1
                or epoch % args.val_every == 0
                or epoch == args.epochs
            )
        )

        if should_validate:
            val_metrics = validate_embeddings_with_labels(
                model=model,
                val_loader=val_loader,
                device=device,
                args=args,
                logger=logger,
            )
        else:
            val_metrics = None
            if val_loader is not None:
                log_info(logger, f"Validation skipped at epoch {epoch}.")

        step_scheduler_after_epoch(
            scheduler=scheduler,
            args=args,
            val_metrics=val_metrics,
        )
        train_metrics["lr_after_scheduler"] = get_current_lr(optimizer)
        train_metrics["all_lrs_after_scheduler"] = get_all_lrs(optimizer)

        if logger is not None and hasattr(logger, "log_epoch"):
            epoch_log = dict(train_metrics)
            if val_metrics is not None:
                epoch_log.update({f"val_{k}": v for k, v in val_metrics.items()})
            logger.log_epoch(epoch_log)
        else:
            log_info(
                logger,
                (
                    f"Train | "
                    f"loss={train_metrics['loss']:.4f} | "
                    f"id={train_metrics['loss_id']:.4f} | "
                    f"triplet={train_metrics['loss_triplet']:.4f} | "
                    f"contrastive={train_metrics['loss_contrastive']:.4f} | "
                    f"occ={train_metrics['loss_occlusion']:.4f} | "
                    f"lr={train_metrics['lr']:.8f} | "
                    f"time={train_metrics['epoch_time_sec']:.2f}s"
                ),
            )

        if val_metrics is not None:
            current_map = val_metrics["mAP"]
            current_rank1 = val_metrics["Rank1"]

            is_best = current_map > best_val_map or (
                current_map == best_val_map and current_rank1 > best_val_rank1
            )

            if is_best:
                best_val_map = current_map
                best_val_rank1 = current_rank1

                save_any_checkpoint(
                    logger=logger,
                    out_dir=out_dir,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    train_dataset=train_dataset,
                    args=args,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    name="best_embedding.pt",
                    scaler=scaler,
                    best_val_map=best_val_map,
                    best_val_rank1=best_val_rank1,
                    best_train_loss=best_train_loss,
                )

                log_info(
                    logger,
                    f"Saved best model by embedding validation mAP={best_val_map:.4f}, Rank1={best_val_rank1:.4f}.",
                )
        elif val_loader is None:
            if train_metrics["loss"] < best_train_loss:
                best_train_loss = train_metrics["loss"]

                save_any_checkpoint(
                    logger=logger,
                    out_dir=out_dir,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    train_dataset=train_dataset,
                    args=args,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    name="best_train_loss.pt",
                    scaler=scaler,
                    best_val_map=best_val_map,
                    best_val_rank1=best_val_rank1,
                    best_train_loss=best_train_loss,
                )

                log_info(logger, "Saved best model by training loss fallback.")

        save_any_checkpoint(
            logger=logger,
            out_dir=out_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            train_dataset=train_dataset,
            args=args,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            name="last.pt",
            scaler=scaler,
            best_val_map=best_val_map,
            best_val_rank1=best_val_rank1,
            best_train_loss=best_train_loss,
        )

    log_info(logger, "Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", default="DataSet/MTMC_Tracking_2025_Preprocessed")
    parser.add_argument("--base_path", default=".")
    parser.add_argument("--out_dir", default="outputs_reid")
    parser.add_argument("--backbone", default="dinov2")
    parser.add_argument("--image_size", type=int, default=224)

    parser.add_argument("--train_scenes", nargs="*", default=None)
    parser.add_argument("--val_scenes", nargs="*", default=None)
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--use_validation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--object_types", nargs="*", default=None)

    parser.add_argument("--min_images_per_id", type=int, default=2)
    parser.add_argument("--min_normal_images_per_id", type=int, default=1)
    parser.add_argument("--min_cameras_per_id", type=int, default=2)
    parser.add_argument("--metadata_filename", default="metadata.csv")
    parser.add_argument("--include_occlusion_crops", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val_include_occlusion_crops", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_queries_per_id", type=int, default=1)
    parser.add_argument("--verify_paths", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scene_aware_ids", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug_dataset", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--backbone_type", default="vit_b", choices=["vit_b", "vit_l", "vit_g"])
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--unfreeze_last_blocks", type=int, default=0)
    
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--grad_clip", type=float, default=5.0)

    parser.add_argument("--use_pk_sampler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pk_identities", type=int, default=8)
    parser.add_argument("--pk_instances", type=int, default=4)
    parser.add_argument("--same_camera_instances", type=int, default=2)
    parser.add_argument("--occlusion_aware_sampler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normal_instances_per_id", type=int, default=1)
    parser.add_argument("--occlusion_instances_per_id", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--max_val_samples", type=int, default=5000)
    parser.add_argument("--val_subset_seed", type=int, default=42)

    parser.add_argument("--head_lr", type=float, default=3e-4)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--scheduler",
        default="warmup_cosine",
        choices=["warmup_cosine", "cosine", "cosine_restarts", "plateau", "none"],
    )
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--warmup_start_factor", type=float, default=0.1)
    parser.add_argument("--plateau_patience", type=int, default=2)
    parser.add_argument("--plateau_factor", type=float, default=0.5)
    parser.add_argument("--restart_t0", type=int, default=10)
    parser.add_argument("--restart_tmult", type=int, default=2)

    parser.add_argument("--resume", default=None, help="Resume full training state from last.pt or another checkpoint.")
    parser.add_argument("--load_model", default=None, help="Load model weights only and train with a fresh optimizer/scheduler.")
    parser.add_argument("--resume_add_epochs", type=int, default=0, help="With --resume, train this many extra epochs beyond the checkpoint epoch.")
    parser.add_argument("--resume_reset_optimizer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume_strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume_ignore_shape_mismatch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load_model_strict", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load_model_ignore_shape_mismatch", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--id_weight", type=float, default=1.0)
    parser.add_argument("--triplet_weight", type=float, default=1.0)
    parser.add_argument("--contrastive_weight", type=float, default=0.2)
    parser.add_argument("--occlusion_consistency_weight", type=float, default=0.1)
    parser.add_argument("--triplet_margin", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--cross_camera_weight", type=float, default=1.0)
    parser.add_argument("--same_camera_weight", type=float, default=0.5)
    parser.add_argument("--occlusion_positive_weight", type=float, default=1.5)
    parser.add_argument(
        "--metric_embedding_key",
        default="bn_embedding",
        choices=["embedding", "bn_embedding"],
    )

    parser.add_argument(
        "--eval_embedding_key",
        default="bn_embedding",
        choices=["embedding", "bn_embedding"],
    )
    parser.add_argument("--max_eval_pairs", type=int, default=50000)

    args = parser.parse_args()
    main(args)
