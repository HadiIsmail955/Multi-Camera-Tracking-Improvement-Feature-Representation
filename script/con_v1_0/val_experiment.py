import argparse

from reid.val.analyze_reid import run_diagnostics


def build_parser():
    parser = argparse.ArgumentParser(
        description="Full ReID validation + DBSCAN/HDBSCAN cluster failure diagnosis."
    )

    # Data
    parser.add_argument("--data_root", default="DataSet/MTMC_Tracking_2025_Preprocessed")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--base_path", default=".")
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--object_types", nargs="*", default=None)
    parser.add_argument("--metadata_filename", default="metadata.csv")

    parser.add_argument("--min_images_per_id", type=int, default=2)
    parser.add_argument("--min_normal_images_per_id", type=int, default=1)
    parser.add_argument("--min_cameras_per_id", type=int, default=2)

    parser.add_argument("--include_occlusion_crops", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--only_occlusion_crops", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--verify_paths", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scene_aware_ids", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug_dataset", action=argparse.BooleanOptionalAction, default=False)

    # Checkpoint / model
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--embedding_key", default="bn_embedding", choices=["embedding", "bn_embedding"])
    parser.add_argument("--embedding_dim", type=int, default=None)
    parser.add_argument("--backbone_type", default=None, choices=["vit_b", "vit_l", "vit_g"])
    parser.add_argument("--backbone", default="dinov2")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--model_type",default="dinov2",choices=["dinov2", "paper_osnet_ain"],)
    parser.add_argument("--paper_model_name",default="osnet_ain_x1_0",)
    parser.add_argument("--paper_pretrained",action=argparse.BooleanOptionalAction,default=True,)
    parser.add_argument("--paper_checkpoint",default=None,)

    # Runtime
    parser.add_argument("--out_dir", default="embedding_validation_full_diagnosis")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    # Evaluation level
    parser.add_argument("--level", default="tracklet", choices=["crop", "tracklet"])
    parser.add_argument(
        "--tracklet_group_mode",
        default="auto",
        choices=["auto", "track_id", "global_id_camera"],
    )
    parser.add_argument(
        "--aggregation",
        default="mean_topk",
        choices=["mean", "medoid", "mean_topk", "quality_mean_topk"],
    )
    parser.add_argument(
        "--identity_col",
        default="identity_key",
        choices=["identity_key", "global_id"],
        help="Use identity_key for scene-aware validation. global_id is only safe inside one scene.",
    )

    # Retrieval / pair metrics
    parser.add_argument("--ranks", nargs="*", type=int, default=[1, 5, 10, 20])
    parser.add_argument("--max_rank_curve", type=int, default=50)
    parser.add_argument("--metric_device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--metric_chunk_size", type=int, default=1024)
    parser.add_argument("--max_pairs", type=int, default=200000)
    parser.add_argument("--pair_sampling", default="balanced", choices=["balanced", "random"])
    parser.add_argument("--max_eval_samples",type=int,default=0,help="Subsample evaluation before expensive metrics. 0 means use all samples.",)
    parser.add_argument("--eval_sampling",default="identity_balanced",choices=["identity_balanced", "random"],help="Sampling strategy used when --max_eval_samples > 0.",)

    # Clustering
    parser.add_argument("--cluster_method", default="dbscan", choices=["dbscan", "hdbscan", "optics"])
    parser.add_argument("--min_cluster_size", type=int, default=3)
    parser.add_argument("--min_samples", type=int, default=2)
    parser.add_argument("--dbscan_eps", type=float, default=0.35)

    parser.add_argument("--run_eps_grid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--eps_values",
        nargs="*",
        type=float,
        default=[0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    )

    # Visualization
    parser.add_argument("--reduce_method", default="tsne", choices=["tsne", "pca", "umap"])
    parser.add_argument("--reduce_3d_method",default="pca",choices=["pca", "tsne", "umap"],)
    parser.add_argument("--max_plot_points", type=int, default=3000)
    parser.add_argument("--make_3d_plots",action=argparse.BooleanOptionalAction,default=False,help="Create interactive 3D Plotly visualizations.",)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_diagnostics(args)


if __name__ == "__main__":
    main()
