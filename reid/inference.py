import argparse
import csv

import torch

from reid.config import load_yaml_config
from reid.data.records import load_records
from reid.data.loaders import build_eval_loader
from reid.engine import extract_embeddings, pool_tracklet_embeddings, read_tracklet_ids
from reid.engine.checkpoint import load_checkpoint
from reid.evaluation import (
    cosine_distance_matrix,
    k_reciprocal_rerank,
    rank_gallery_indices,
    ranked_indices_to_pids,
    compute_rank1_map,
)
from reid.models import build_model


def _infer_num_classes_from_checkpoint(checkpoint: dict) -> int:
    """Try to infer the number of classes from the checkpoint's classifier weights."""
    state_dict = checkpoint.get("state_dict", {})

    for key in ("classifier.weight", "module.classifier.weight"):
        weight = state_dict.get(key)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            return int(weight.shape[0])

    return 1


def _checkpoint_uses_lora(checkpoint: dict) -> bool:
    state_dict = checkpoint.get("state_dict", {})
    return any((".lora_A" in key) or (".lora_B" in key) for key in state_dict.keys())


def load_model_from_config(config: dict, device):
    model_cfg = config["model"]
    ckpt_cfg = config["checkpoint"]

    use_pretrained = ckpt_cfg.get("use_pretrained", False)

    if use_pretrained:
        print("\n  Mode     : pretrained weights only")
        print(f"  Backbone : {model_cfg['backbone']}")

        model = build_model(
            backbone=model_cfg["backbone"],
            num_classes=1,
            lora_rank=model_cfg["lora_rank"],
            use_lora=model_cfg.get("use_lora", True),
            osnet_weight_path=model_cfg.get("osnet_weights"),
            dino_feature_mode=model_cfg.get("feature_mode", "cls"),
        ).to(device)

        if model_cfg["backbone"] == "osnet" and hasattr(model, "use_raw_inference"):
            setattr(model, "use_raw_inference", True)

        model.eval()
        return model

    checkpoint_path = ckpt_cfg["path"]

    print("\n  Mode       : fine-tuned checkpoint")
    print(f"  Checkpoint : {checkpoint_path}")

    checkpoint = load_checkpoint(
        path=checkpoint_path,
        device=device,
    )

    checkpoint_backbone = checkpoint.get(
        "backbone",
        model_cfg["backbone"],
    )

    if checkpoint_backbone != model_cfg["backbone"]:
        print(
            f"  [NOTE] Checkpoint backbone '{checkpoint_backbone}' differs "
            f"from config backbone '{model_cfg['backbone']}'. "
            f"Using checkpoint backbone."
        )

    num_classes = _infer_num_classes_from_checkpoint(checkpoint)

    config_use_lora = bool(model_cfg.get("use_lora", True))
    ckpt_use_lora = _checkpoint_uses_lora(checkpoint)
    resolved_use_lora = config_use_lora

    if checkpoint_backbone in {"dinov2", "dinov3"} and (
        config_use_lora != ckpt_use_lora
    ):
        resolved_use_lora = ckpt_use_lora
        print(
            "  [NOTE] Config use_lora "
            f"({config_use_lora}) does not match checkpoint ({ckpt_use_lora}); "
            f"using checkpoint mode ({resolved_use_lora})."
        )

    model = build_model(
        backbone=checkpoint_backbone,
        num_classes=num_classes,
        lora_rank=model_cfg["lora_rank"],
        use_lora=resolved_use_lora,
        osnet_weight_path=model_cfg.get("osnet_weights"),
        dino_feature_mode=model_cfg.get("feature_mode", "cls"),
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])

    print(
        f"  [OK] Loaded epoch={checkpoint.get('epoch', '?')}  "
        f"Rank-1={checkpoint.get('rank1', 0):.4f}  "
        f"mAP={checkpoint.get('mAP', 0):.4f}"
    )

    model.eval()
    return model


def write_matching_results(
    path: str,
    ranked_pids: list[list[int]],
    q_pids: list[int],
    q_camids: list[int],
):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "query_pid",
                "query_cam",
                "top1",
                "top2",
                "top3",
                "top4",
                "top5",
                "top6",
                "top7",
                "top8",
                "top9",
                "top10",
                "top1_correct",
            ]
        )

        for i, ranked in enumerate(ranked_pids):
            top10 = ranked[:10] + [-1] * max(0, 10 - len(ranked))
            top1_correct = int(top10[0] == q_pids[i]) if top10 else 0

            writer.writerow([q_pids[i], q_camids[i]] + top10 + [top1_correct])


def main(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = config["data"]
    eval_cfg = config["eval"]
    rerank_cfg = config["rerank"]
    output_cfg = config["output"]

    print(f"\nCross-Camera Matching  |  device={device}")

    model = load_model_from_config(
        config=config,
        device=device,
    )

    query_records = load_records(data_cfg["query"])
    gallery_records = load_records(data_cfg["gallery"])

    _, query_loader = build_eval_loader(
        records=query_records,
        batch_size=eval_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
    )

    _, gallery_loader = build_eval_loader(
        records=gallery_records,
        batch_size=eval_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
    )

    print("\n  Extracting embeddings...")
    print(f"    Query   : {len(query_records):,} images")
    print(f"    Gallery : {len(gallery_records):,} images")

    q_embs, q_pids, q_camids = extract_embeddings(
        model=model,
        loader=query_loader,
        device=device,
        show_progress=True,
    )

    g_embs, g_pids, g_camids = extract_embeddings(
        model=model,
        loader=gallery_loader,
        device=device,
        show_progress=True,
    )

    print(f"    Embedding dim : {q_embs.shape[1]}")

    # Tracklet pooling
    tracklet_pool = data_cfg.get("tracklet_pool", None)  # "mean" | "max" | None
    if tracklet_pool:
        q_tracklet_ids = read_tracklet_ids(data_cfg["query"])
        g_tracklet_ids = read_tracklet_ids(data_cfg["gallery"])

        if not q_tracklet_ids[0]:
            print("  [WARNING] query CSV has no tracklet_id column; skipping pooling.")
        else:
            q_embs, q_pids, q_camids, q_tids = pool_tracklet_embeddings(
                q_embs,
                q_pids,
                q_camids,
                q_tracklet_ids,
                pool=tracklet_pool,
            )
            g_embs, g_pids, g_camids, g_tids = pool_tracklet_embeddings(
                g_embs,
                g_pids,
                g_camids,
                g_tracklet_ids,
                pool=tracklet_pool,
            )
            print(
                f"  Tracklet pooling ({tracklet_pool}): "
                f"{len(q_tids)} query tracklets, "
                f"{len(g_tids)} gallery tracklets"
            )

    print("\n  Computing distance matrix...")

    if rerank_cfg.get("enabled", False):
        print(
            "  Applying k-reciprocal re-ranking "
            f"(k1={rerank_cfg['k1']}, "
            f"k2={rerank_cfg['k2']}, "
            f"lambda={rerank_cfg['lambda']})"
        )

        dist_matrix = k_reciprocal_rerank(
            q_embs=q_embs,
            g_embs=g_embs,
            k1=rerank_cfg["k1"],
            k2=rerank_cfg["k2"],
            lam=rerank_cfg["lambda"],
        )
    else:
        dist_matrix = cosine_distance_matrix(
            q_embs=q_embs,
            g_embs=g_embs,
        )

    print(
        f"  Distance matrix : {list(dist_matrix.shape)}  "
        f"min={dist_matrix.min():.4f}  "
        f"max={dist_matrix.max():.4f}"
    )

    ranked_indices = rank_gallery_indices(
        dist_matrix=dist_matrix,
        q_pids=q_pids,
        q_camids=q_camids,
        g_pids=g_pids,
        g_camids=g_camids,
        remove_junk=True,
    )

    rank1, mAP = compute_rank1_map(
        ranked_indices=ranked_indices,
        q_pids=q_pids,
        q_camids=q_camids,
        g_pids=g_pids,
        g_camids=g_camids,
    )

    ranked_pids = ranked_indices_to_pids(
        ranked_indices=ranked_indices,
        g_pids=g_pids,
    )

    mode = (
        "pretrained"
        if config["checkpoint"].get("use_pretrained", False)
        else "checkpoint"
    )
    rerank_tag = "+rerank" if rerank_cfg.get("enabled", False) else ""

    print(f"\n  ── Results [{mode}{rerank_tag}] ──────────────────")
    print(f"  Rank-1 : {rank1:.4f}  ({rank1 * 100:.2f}%)")
    print(f"  mAP    : {mAP:.4f}  ({mAP * 100:.2f}%)")
    print("  ──────────────────────────────────────────")

    torch.save(
        {
            "dist_matrix": dist_matrix,
            "q_pids": q_pids,
            "q_camids": q_camids,
            "g_pids": g_pids,
            "g_camids": g_camids,
            "reranked": rerank_cfg.get("enabled", False),
            "mode": mode,
        },
        output_cfg["distance_matrix"],
    )

    write_matching_results(
        path=output_cfg["matching_results"],
        ranked_pids=ranked_pids,
        q_pids=q_pids,
        q_camids=q_camids,
    )

    print(f"\n  Saved: {output_cfg['distance_matrix']}")
    print(f"  Saved: {output_cfg['matching_results']}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ReID inference / cross-camera matching"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML inference config.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_yaml_config(args.config)
    main(config)
