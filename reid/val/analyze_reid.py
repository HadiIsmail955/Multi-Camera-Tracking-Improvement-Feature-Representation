from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader

from ..utils.aggregation import aggregate_to_tracklets
from ..clustering.clustering import (
    add_cluster_failure_columns,
    build_cluster_error_tables,
    cluster_unknown_k,
    compute_cluster_failure_metrics,
    compute_clustering_metrics,
    evaluate_dbscan_eps_grid,
)
from ..dataLoader.customData.val_data import build_dataset, extract_crop_embeddings
from ..metrics.health import compute_embedding_health_metrics
from ..model.model_loading import load_model_from_checkpoint
from ..model.paper_osnet_ain import PaperOSNetAINEmbedding
from ..metrics.retrieval import (
    compute_cross_camera_retrieval_metrics,
    compute_grouped_retrieval_metrics,
)
from ..metrics.similarityMetrics import (
    compute_similarity_distribution_metrics,
    compute_threshold_metrics,
    sample_similarity_pairs,
)
from ..utils.helper import ensure_dir, save_metrics, set_seed
from ..visualization.visualization import (
    choose_plot_subset,
    plot_rank_curve,
    plot_scatter,
    plot_similarity_histogram,
    reduce_embeddings,
    reduce_embeddings_3d,
    save_interactive_plots,
    save_interactive_3d_plots,
)
from ..dataLoader.sampler.val_sampler import subsample_eval_set


def run_diagnostics(args):
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    print("=" * 80)
    print("Full ReID validation + DBSCAN/HDBSCAN cluster failure diagnosis")
    print("=" * 80)
    print("Device:", device)
    print("Checkpoint:", args.checkpoint)
    print("Data root:", args.data_root)
    print("Split:", args.split)
    print("Level:", args.level)
    print("Embedding key:", args.embedding_key)
    print("Identity column:", args.identity_col)
    print("Cluster method:", args.cluster_method)
    print("Output dir:", out_dir)
    print("=" * 80)

    dataset = build_dataset(args)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
        persistent_workers=args.workers > 0,
    )

    print("Validation samples:", len(dataset))
    print("Validation classes:", getattr(dataset, "num_classes", "unknown"))
    print("Validation scenes:", getattr(dataset, "scene_folders", "unknown"))

    if getattr(args, "model_type", "dinov2") == "paper_osnet_ain":
        model = PaperOSNetAINEmbedding(
            model_name=getattr(args, "paper_model_name", "osnet_ain_x1_0"),
            num_classes=1000,
            pretrained=getattr(args, "paper_pretrained", True),
            checkpoint=getattr(args, "paper_checkpoint", None),
        )

        model = model.to(device)
        model.eval()

        print("=" * 80)
        print("Using paper OSNet-AIN embedding model")
        print("No DINO checkpoint will be loaded.")
        print("=" * 80)

    else:
        model, _ = load_model_from_checkpoint(args.checkpoint, args, device)

    crop_embeddings, crop_df = extract_crop_embeddings(
        model=model,
        loader=loader,
        device=device,
        embedding_key=args.embedding_key,
        use_amp=args.use_amp,
    )

    print("Extracted crop embeddings:", crop_embeddings.shape)

    if args.level == "crop":
        eval_embeddings = crop_embeddings
        eval_df = crop_df.copy()
        resolved_group_mode = "crop"

    elif args.level == "tracklet":
        eval_embeddings, eval_df, resolved_group_mode = aggregate_to_tracklets(
            embeddings=crop_embeddings,
            df=crop_df,
            group_mode=args.tracklet_group_mode,
            aggregation=args.aggregation,
        )
        print("Aggregated tracklet embeddings:", eval_embeddings.shape)
        print("Resolved tracklet group mode:", resolved_group_mode)

    else:
        raise ValueError(f"Unknown level: {args.level}")

    eval_embeddings = normalize(eval_embeddings.astype(np.float32))

    if args.identity_col not in eval_df.columns:
        raise KeyError(
            f"--identity_col '{args.identity_col}' not found in metadata columns: "
            f"{list(eval_df.columns)}"
        )

    true_ids = sorted(eval_df[args.identity_col].astype(str).unique())
    id_to_label = {identity: idx for idx, identity in enumerate(true_ids)}
    eval_df["true_label"] = (
        eval_df[args.identity_col].astype(str).map(id_to_label).astype(int)
    )

    full_num_samples_before_subsample = len(eval_df)

    if getattr(args, "max_eval_samples", 0) and args.max_eval_samples > 0:
        print("")
        print("=" * 80)
        print("SUBSAMPLING EVALUATION SET")
        print("=" * 80)
        print("Before:", len(eval_df))
        print("Requested max_eval_samples:", args.max_eval_samples)
        print("Mode:", args.eval_sampling)

        eval_embeddings, eval_df = subsample_eval_set(
            embeddings=eval_embeddings,
            df=eval_df,
            identity_col=args.identity_col,
            max_samples=args.max_eval_samples,
            seed=args.seed,
            mode=args.eval_sampling,
        )

        true_ids = sorted(eval_df[args.identity_col].astype(str).unique())
        id_to_label = {identity: idx for idx, identity in enumerate(true_ids)}
        eval_df["true_label"] = (
            eval_df[args.identity_col].astype(str).map(id_to_label).astype(int)
        )

        print("After:", len(eval_df))
        print("Identities after:", eval_df[args.identity_col].nunique())
        print("=" * 80)

    np.save(out_dir / "embeddings.npy", eval_embeddings)

    health_metrics = compute_embedding_health_metrics(eval_embeddings)

    pair_df = sample_similarity_pairs(
        embeddings=eval_embeddings,
        df=eval_df,
        identity_col=args.identity_col,
        max_pairs=args.max_pairs,
        seed=args.seed,
        mode=args.pair_sampling,
    )
    pair_df.to_csv(out_dir / "pair_similarity_sample.csv", index=False)

    similarity_metrics = compute_similarity_distribution_metrics(pair_df)
    threshold_metrics = compute_threshold_metrics(pair_df)

    retrieval_metrics, rank_curve, query_results = compute_cross_camera_retrieval_metrics(
        embeddings=eval_embeddings,
        df=eval_df,
        identity_col=args.identity_col,
        ranks=args.ranks,
        max_rank_curve=args.max_rank_curve,
        metric_device=args.metric_device,
        chunk_size=args.metric_chunk_size,
    )
    rank_curve.to_csv(out_dir / "rank_curve.csv", index=False)
    query_results.to_csv(out_dir / "query_retrieval_results.csv", index=False)

    grouped_by_object = compute_grouped_retrieval_metrics(
        embeddings=eval_embeddings,
        df=eval_df,
        identity_col=args.identity_col,
        group_col="object_type",
        ranks=(1, 5),
    )
    grouped_by_object.to_csv(out_dir / "retrieval_by_object_type.csv", index=False)

    grouped_by_scene = compute_grouped_retrieval_metrics(
        embeddings=eval_embeddings,
        df=eval_df,
        identity_col=args.identity_col,
        group_col="scene",
        ranks=(1, 5),
    )
    grouped_by_scene.to_csv(out_dir / "retrieval_by_scene.csv", index=False)

    if args.run_eps_grid:
        eps_grid_df = evaluate_dbscan_eps_grid(
            embeddings=eval_embeddings,
            df=eval_df,
            identity_col=args.identity_col,
            eps_values=args.eps_values,
            min_samples=args.min_samples,
        )
        eps_grid_df.to_csv(out_dir / "dbscan_eps_grid.csv", index=False)

        print("")
        print("=" * 80)
        print("DBSCAN EPS GRID")
        print("=" * 80)
        cols = [
            "eps",
            "num_found_clusters",
            "noise_ratio",
            "cluster_purity_no_noise",
            "ARI",
            "NMI",
            "cluster_pair_f1",
            "misclustered_sample_rate",
            "merge_error_cluster_rate",
            "fragmented_identity_rate",
        ]
        cols = [c for c in cols if c in eps_grid_df.columns]
        print(eps_grid_df[cols].to_string(index=False))
        print("=" * 80)

    cluster_labels = cluster_unknown_k(
        embeddings=eval_embeddings,
        method=args.cluster_method,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        dbscan_eps=args.dbscan_eps,
    )

    eval_df["cluster_label"] = cluster_labels

    eval_df = add_cluster_failure_columns(
        eval_df,
        true_id_col=args.identity_col,
        cluster_col="cluster_label",
    )

    cluster_summary, merge_errors, fragmentation = build_cluster_error_tables(
        eval_df,
        true_id_col=args.identity_col,
        cluster_col="cluster_label",
    )

    clustering_metrics = compute_clustering_metrics(
        embeddings=eval_embeddings,
        true_labels=eval_df["true_label"].values,
        cluster_labels=cluster_labels,
    )

    cluster_failure_metrics = compute_cluster_failure_metrics(
        eval_df,
        merge_errors,
        fragmentation,
        cluster_col="cluster_label",
    )

    eval_df.to_csv(
        out_dir / "embedding_metadata_with_cluster_diagnosis.csv",
        index=False,
    )
    eval_df.to_csv(out_dir / "embedding_metadata.csv", index=False)
    cluster_summary.to_csv(out_dir / "cluster_summary.csv", index=False)
    merge_errors.to_csv(out_dir / "merge_errors.csv", index=False)
    fragmentation.to_csv(out_dir / "fragmentation_errors.csv", index=False)

    eval_df[eval_df["is_misclustered"]].to_csv(
        out_dir / "misclustered_points.csv",
        index=False,
    )
    eval_df[eval_df["is_noise"]].to_csv(
        out_dir / "noise_points.csv",
        index=False,
    )

    plot_embeddings, plot_df = choose_plot_subset(
        eval_embeddings,
        eval_df,
        max_points=args.max_plot_points,
        seed=args.seed,
    )

    # -------------------------------------------------------------------------
    # 2D visualization
    # Metrics, retrieval, and DBSCAN are computed in original embedding space.
    # This 2D projection is only for human visualization.
    # -------------------------------------------------------------------------
    xy = reduce_embeddings(
        plot_embeddings,
        method=args.reduce_method,
        seed=args.seed,
    )

    plot_df["x"] = xy[:, 0]
    plot_df["y"] = xy[:, 1]

    # -------------------------------------------------------------------------
    # Optional 3D visualization
    # Disabled by default because large crop-level scenes can be slow.
    # -------------------------------------------------------------------------
    make_3d_plots = getattr(args, "make_3d_plots", False)

    if make_3d_plots:
        reduce_3d_method = getattr(args, "reduce_3d_method", "pca")

        xyz = reduce_embeddings_3d(
            plot_embeddings,
            method=reduce_3d_method,
            seed=args.seed,
        )

        plot_df["x3d"] = xyz[:, 0]
        plot_df["y3d"] = xyz[:, 1]
        plot_df["z3d"] = xyz[:, 2]

    plot_df.to_csv(out_dir / "visualization_points.csv", index=False)

    plot_scatter(
        xy,
        plot_df["true_label"].values,
        f"{args.level} embeddings colored by {args.identity_col}",
        out_dir / "embedding_by_real_identity.png",
    )

    plot_scatter(
        xy,
        plot_df["camera"].astype(str).values,
        f"{args.level} embeddings colored by camera",
        out_dir / "embedding_by_camera.png",
    )

    plot_scatter(
        xy,
        plot_df["object_type"].astype(str).values,
        f"{args.level} embeddings colored by object type",
        out_dir / "embedding_by_object_type.png",
    )

    plot_scatter(
        xy,
        plot_df["cluster_label"].values,
        f"{args.level} embeddings colored by discovered cluster",
        out_dir / "embedding_by_cluster.png",
    )

    plot_scatter(
        xy,
        plot_df["is_misclustered"].astype(int).values,
        f"{args.level} embeddings: misclustered points",
        out_dir / "embedding_misclustered_points.png",
    )

    plot_similarity_histogram(pair_df, out_dir / "similarity_histogram.png")
    plot_rank_curve(rank_curve, out_dir / "rank_curve.png")

    save_interactive_plots(
        plot_df=plot_df,
        out_dir=out_dir,
        level=args.level,
        identity_col=args.identity_col,
    )

    if make_3d_plots:
        save_interactive_3d_plots(
            plot_df=plot_df,
            out_dir=out_dir,
            level=args.level,
            identity_col=args.identity_col,
        )

    metrics = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "level": args.level,
            "embedding_key": args.embedding_key,
            "identity_col": args.identity_col,
            "tracklet_group_mode": args.tracklet_group_mode,
            "resolved_group_mode": resolved_group_mode,
            "aggregation": args.aggregation,
            "cluster_method": args.cluster_method,
            "dbscan_eps": args.dbscan_eps,
            "min_samples": args.min_samples,
            "min_cluster_size": args.min_cluster_size,
            "reduce_method": args.reduce_method,
            "pair_sampling": args.pair_sampling,
            "full_num_samples_before_subsample": full_num_samples_before_subsample,
            "used_num_samples": int(len(eval_df)),
            "max_eval_samples": getattr(args, "max_eval_samples", 0),
            "eval_sampling": getattr(args, "eval_sampling", "identity_balanced"),
            "make_3d_plots": make_3d_plots,
            "reduce_3d_method": getattr(args, "reduce_3d_method", "pca"),
        },
        "health": health_metrics,
        "retrieval": retrieval_metrics,
        "similarity": similarity_metrics,
        "threshold": threshold_metrics,
        "clustering": clustering_metrics,
        "cluster_failure": cluster_failure_metrics,
    }

    save_metrics(metrics, out_dir)

    print("")
    print("=" * 80)
    print("MAIN METRICS")
    print("=" * 80)
    print("Retrieval mAP:", retrieval_metrics.get("mAP"))

    for rank in args.ranks:
        print(f"Rank{rank}:", retrieval_metrics.get(f"Rank{rank}"))

    print("Valid queries:", retrieval_metrics.get("valid_queries"))
    print("Same-ID cosine mean:", similarity_metrics.get("same_id_cos_mean"))
    print(
        "Same-ID cross-camera cosine mean:",
        similarity_metrics.get("same_id_cross_camera_cos_mean"),
    )
    print("Different-ID cosine mean:", similarity_metrics.get("diff_id_cos_mean"))
    print("Separation gap:", similarity_metrics.get("embedding_separation_gap"))
    print("Pair ROC-AUC:", threshold_metrics.get("pair_roc_auc"))
    print("Best threshold:", threshold_metrics.get("best_threshold"))
    print("ARI:", clustering_metrics.get("ARI"))
    print("NMI:", clustering_metrics.get("NMI"))
    print("Cluster purity no noise:", clustering_metrics.get("cluster_purity_no_noise"))
    print(
        "Misclustered sample rate:",
        cluster_failure_metrics.get("misclustered_sample_rate"),
    )
    print("Noise sample rate:", cluster_failure_metrics.get("noise_sample_rate"))
    print(
        "Merge error cluster rate:",
        cluster_failure_metrics.get("merge_error_cluster_rate"),
    )
    print(
        "Fragmented identity rate:",
        cluster_failure_metrics.get("fragmented_identity_rate"),
    )
    print("Effective rank:", health_metrics.get("effective_rank"))
    print("=" * 80)

    print("")
    print("Top merge errors:")
    if len(merge_errors) > 0:
        print(merge_errors.head(10).to_string(index=False))
    else:
        print("No merge-error clusters found.")

    print("")
    print("Top fragmented identities:")
    if len(fragmentation) > 0:
        print(fragmentation.head(10).to_string(index=False))
    else:
        print("No fragmented identities found.")

    print("")
    print("Saved outputs:")
    for path in sorted(out_dir.iterdir()):
        print(" ", path)