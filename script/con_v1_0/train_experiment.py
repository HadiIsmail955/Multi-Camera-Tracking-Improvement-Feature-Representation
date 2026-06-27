from argparse import Namespace
import torch

from reid.train.train_reid import main as train_reid_main
from reid.utils.experiment_logger import ExperimentLogger


def build_args():
    experiment_name = "dinov2_reid_embedding_v2"

    args = Namespace(
        data_root="./DataSet/MTMC_Tracking_2025_Preprocessed",
        base_path=".",

        train_scenes=["Warehouse_000", "Warehouse_004", "Warehouse_008"],
        val_scenes=["Warehouse_015"],
        val_split="val",
        use_validation=True,
        object_types=None,

        metadata_filename="metadata.csv",
        include_occlusion_crops=True,
        val_include_occlusion_crops=False,

        min_images_per_id=4,
        min_normal_images_per_id=2,
        min_cameras_per_id=2,

        verify_paths=True,
        scene_aware_ids=True,
        debug_dataset=True,

        out_dir=f"outputs_reid/{experiment_name}",
        experiment_name=experiment_name,

        backbone="dinov2",
        backbone_type="vit_b",
        embedding_dim=512,
        image_size=224,
        dropout=0.1,

        freeze_backbone=True,
        unfreeze_last_blocks=0,

        epochs=100,
        workers=8,

        use_pk_sampler=True,

        pk_identities=15,
        pk_instances=6,
        same_camera_instances=2,
        batch_size=15 * 6,

        eval_batch_size=256,
        val_every=1,
        max_val_samples=5000,
        val_subset_seed=42,
        max_eval_pairs=50000,        

        head_lr=1e-4,
        backbone_lr=5e-6,
        min_lr=1e-6,
        weight_decay=1e-4,

        scheduler="warmup_cosine",
        warmup_epochs=3,
        warmup_start_factor=0.1,

        grad_clip=5.0,
        use_amp=True,
        log_every=20,

        id_weight=1.0,
        triplet_weight=1.0,
        contrastive_weight=0.2,

        triplet_margin=0.3,
        temperature=0.07,
        label_smoothing=0.1,

        cross_camera_weight=1.0,
        same_camera_weight=0.5,

        ignore_same_id_same_camera=True,

        occlusion_aware_sampler=True,
        normal_instances_per_id=1,
        occlusion_instances_per_id=1,

        occlusion_consistency_weight=0.1,
        occlusion_positive_weight=1.5,

        metric_embedding_key="bn_embedding",
        eval_embedding_key="bn_embedding",

        resume=None,
        # resume="./outputs_reid/dinov2_reid_embedding_v2_20260613_211015/checkpoints/last.pt",
        # load_model=None,
        load_model="./outputs_reid/dinov2_reid_embedding_v2_20260613_211015/checkpoints/last.pt",

        resume_add_epochs=0,
        resume_reset_optimizer=False,
        resume_strict=True,
        resume_ignore_shape_mismatch=False,

        load_model_strict=False,
        load_model_ignore_shape_mismatch=True,
    )

    return args


def print_experiment_info(args):
    print("=" * 80, flush=True)
    print("Starting ReID embedding training experiment", flush=True)
    print("=" * 80, flush=True)

    print("CUDA available:", torch.cuda.is_available(), flush=True)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    print("Data root:", args.data_root, flush=True)
    print("Output dir:", args.out_dir, flush=True)
    print("Backbone:", args.backbone, flush=True)
    print("DINO type:", args.backbone_type, flush=True)
    print("Embedding dim:", args.embedding_dim, flush=True)
    print("Image size:", args.image_size, flush=True)
    print("Epochs:", args.epochs, flush=True)
    print("Batch size:", args.batch_size, flush=True)
    print("PK sampler:", args.use_pk_sampler, flush=True)
    print("PK:", args.pk_identities, "x", args.pk_instances, flush=True)
    print("Same-camera instances:", args.same_camera_instances, flush=True)
    print("Freeze backbone:", args.freeze_backbone, flush=True)
    print("Unfreeze last blocks:", args.unfreeze_last_blocks, flush=True)
    print("Metric embedding key:", args.metric_embedding_key, flush=True)
    print("Eval embedding key:", args.eval_embedding_key, flush=True)

    print("=" * 80, flush=True)


def main():
    args = build_args()
    print_experiment_info(args)

    logger = ExperimentLogger(
        base_dir="outputs_reid",
        exp_name=args.experiment_name,
    )

    logger.save_config(vars(args))

    try:
        train_reid_main(args=args, logger=logger)

    except Exception:
        logger.info("=" * 80)
        logger.info("TRAINING CRASHED")
        logger.info("=" * 80)
        logger.exception("Exception traceback:")
        raise

    finally:
        logger.close()


if __name__ == "__main__":
    main()