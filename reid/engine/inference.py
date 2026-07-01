# src/engine/inference.py

import csv
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.preprocessing import normalize
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


@torch.no_grad()
def _load_embeddings_cache(
    cache_path: str,
    split_name: str | None = None,
):
    """Load cached embeddings payload from disk and validate structure."""
    cache_file = Path(cache_path)
    if not cache_file.exists():
        return None

    payload = torch.load(cache_file, map_location="cpu")
    required_keys = {"embs", "pids", "camids"}
    if not isinstance(payload, dict) or not required_keys.issubset(payload.keys()):
        raise ValueError(
            f"Invalid embedding cache format at {cache_file}. "
            "Expected keys: embs, pids, camids"
        )

    embs = payload["embs"]
    pids = payload["pids"]
    camids = payload["camids"]

    if not isinstance(embs, torch.Tensor):
        raise ValueError(
            f"Invalid 'embs' type in cache {cache_file}; expected torch.Tensor"
        )

    if len(embs) != len(pids) or len(embs) != len(camids):
        raise ValueError(
            f"Cache length mismatch in {cache_file}: "
            f"len(embs)={len(embs)}, len(pids)={len(pids)}, len(camids)={len(camids)}"
        )

    if split_name:
        print(f"    {split_name:<7}: loaded cached embeddings from {cache_file}")
    else:
        print(f"Loaded cached embeddings from {cache_file}")

    return embs, list(pids), list(camids)


@torch.no_grad()
def _extract_embeddings(
    model,
    loader,
    device,
    show_progress: bool = False,
):
    """Extract embeddings for a full DataLoader without cache handling."""
    model.eval()

    all_embs = []
    all_pids = []
    all_camids = []

    iterator = loader
    if show_progress:
        iterator = tqdm(loader, desc="Extracting embeddings")

    for imgs, pids, camids in iterator:
        imgs = imgs.to(device)
        embs = model(imgs)

        all_embs.append(embs.cpu())
        all_pids.extend(pids.tolist())
        all_camids.extend(camids.tolist())

    return torch.cat(all_embs), all_pids, all_camids


@torch.no_grad()
def extract_embeddings(
    model,
    loader,
    device,
    show_progress: bool = False,
    cache_path: str | None = None,
    split_name: str | None = None,
):
    """
    Extract embeddings for a full DataLoader.

    Returns:
        embeddings: torch.Tensor [N, D]
        pids: list[int]
        camids: list[int]
    """
    if cache_path:
        cached = _load_embeddings_cache(cache_path=cache_path, split_name=split_name)
        if cached is not None:
            return cached

    embs_out, all_pids, all_camids = _extract_embeddings(
        model=model,
        loader=loader,
        device=device,
        show_progress=show_progress,
    )

    if cache_path:
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "embs": embs_out.cpu(),
                "pids": list(all_pids),
                "camids": list(all_camids),
            },
            cache_file,
        )
        if split_name:
            print(f"    {split_name:<7}: saved embeddings cache to {cache_file}")
        else:
            print(f"Saved embeddings cache to {cache_file}")

    return embs_out, all_pids, all_camids


def read_tracklet_ids(csv_path: str) -> list[str]:
    """Read the ``tracklet_id`` column from a manifest CSV in row order.

    Returns an empty string for each row if the column is absent (falls
    back gracefully to image-level evaluation).
    """
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return [row.get("tracklet_id", "") for row in reader]


def _pool_mean(group: torch.Tensor) -> torch.Tensor:
    return group.mean(dim=0)


def _pool_max(group: torch.Tensor) -> torch.Tensor:
    return group.max(dim=0).values


def _pool_weighted(group: torch.Tensor, weighted_temperature: float) -> torch.Tensor:
    if len(group) == 1:
        return group[0]

    feats = normalize(group.detach().cpu().numpy(), norm="l2")
    centrality = cosine_similarity(feats).mean(axis=1)
    centrality = centrality - centrality.max()

    weights = np.exp(centrality * weighted_temperature)
    weights = weights / (weights.sum() + 1e-12)

    weights_t = torch.tensor(weights, dtype=group.dtype, device=group.device)
    return (group * weights_t[:, None]).sum(dim=0)


def _pool_medoid(group: torch.Tensor) -> torch.Tensor:
    if len(group) == 1:
        return group[0]

    feats = normalize(group.detach().cpu().numpy(), norm="l2")
    centrality = cosine_similarity(feats).mean(axis=1)
    return group[int(np.argmax(centrality))]


def _pool_consensus(group: torch.Tensor, consensus_k: int) -> torch.Tensor:
    if len(group) == 1:
        return group[0]

    feats = normalize(group.detach().cpu().numpy(), norm="l2")
    centrality = cosine_similarity(feats).mean(axis=1)
    top_idx = np.argsort(centrality)[-min(consensus_k, len(group)) :]
    return group[top_idx].mean(dim=0)


def _pool_dbscan(
    group: torch.Tensor,
    dbscan_eps: float,
    dbscan_min_samples: int,
) -> tuple[torch.Tensor, bool]:
    """Returns (pooled embedding, did_fallback)."""
    if len(group) < dbscan_min_samples:
        return group.mean(dim=0), True

    feats = normalize(group.detach().cpu().numpy(), norm="l2")
    labels = (
        DBSCAN(
            eps=dbscan_eps,
            min_samples=dbscan_min_samples,
            metric="cosine",
        )
        .fit(feats)
        .labels_
    )

    valid_mask = labels >= 0
    if not np.any(valid_mask):
        return group.mean(dim=0), True

    unique_labels, counts = np.unique(labels[valid_mask], return_counts=True)
    largest_cluster = unique_labels[np.argmax(counts)]
    return group[labels == largest_cluster].mean(dim=0), False


def _pool_hdbscan(
    group: torch.Tensor,
    hdbscan_min_cluster_size: int,
    hdbscan_min_samples: int | None,
) -> tuple[torch.Tensor, bool]:

    if len(group) < hdbscan_min_cluster_size:
        return group.mean(dim=0), True

    feats = normalize(group.detach().cpu().numpy(), norm="l2")
    labels = (
        HDBSCAN(
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
            metric="cosine",
        )
        .fit(feats)
        .labels_
    )

    valid_mask = labels >= 0
    if not np.any(valid_mask):
        return group.mean(dim=0), True

    unique_labels, counts = np.unique(labels[valid_mask], return_counts=True)
    largest_cluster = unique_labels[np.argmax(counts)]
    return group[labels == largest_cluster].mean(dim=0), False


def _pool_gem(group: torch.Tensor, gem_p: float = 3.0) -> torch.Tensor:
    if len(group) == 1:
        return group[0]

    # For signed embeddings: pool magnitudes with GeM, then restore sign.
    sign = torch.sign(group.mean(dim=0))
    mean_abs_pow = torch.abs(group).pow(gem_p).mean(dim=0)
    mag = mean_abs_pow.clamp_min(1e-12).pow(1.0 / gem_p)
    return sign * mag


def pool_tracklet_embeddings(
    embs: torch.Tensor,
    pids: list[int],
    camids: list[int],
    tracklet_ids: list[str],
    pool: str = "mean",
    dbscan_eps: float = 0.2,
    dbscan_min_samples: int = 4,
    hdbscan_min_cluster_size: int = 4,
    hdbscan_min_samples: int | None = None,
    consensus_k: int = 10,
    weighted_temperature: float = 10.0,
):
    """
    Pool per-image embeddings into one descriptor per tracklet.

    Supported pooling modes:

        mean
            Average all embeddings.

        max
            Max pooling across embeddings.

        dbscan
            DBSCAN -> largest non-noise cluster -> mean pool.

        hdbscan
            HDBSCAN -> largest non-noise cluster -> mean pool.

        weighted
            Similarity-weighted pooling using all embeddings.

        medoid
            Use the most central embedding.

        consensus
            Mean of the top-k most central embeddings.

        gem
            Generalised mean pooling (signed variant).

    Returns:
        (
            tracklet_embs,
            tracklet_pids,
            tracklet_camids,
            tracklet_ids
        )
    """

    valid_pools = (
        "mean",
        "max",
        "dbscan",
        "hdbscan",
        "weighted",
        "medoid",
        "consensus",
        "gem",
    )
    if pool not in valid_pools:
        raise ValueError(f"pool must be one of {valid_pools}, got '{pool}'")

    print(f"\n  Pooling {embs.shape[0]:,} image embeddings using '{pool}' pooling...")

    # ----------------------------------------------------------
    # Build tracklet index
    # ----------------------------------------------------------

    order: list[str] = []
    seen: set[str] = set()
    indices_by_tracklet: dict[str, list[int]] = defaultdict(list)
    pid_by_tracklet: dict[str, int] = {}
    camid_by_tracklet: dict[str, int] = {}

    for i, tid in enumerate(tracklet_ids):
        indices_by_tracklet[tid].append(i)
        if tid not in seen:
            seen.add(tid)
            order.append(tid)
            pid_by_tracklet[tid] = pids[i]
            camid_by_tracklet[tid] = camids[i]

    # ----------------------------------------------------------
    # Pool each tracklet
    # ----------------------------------------------------------

    pooled_embs: list[torch.Tensor] = []
    num_dbscan_fallbacks = 0
    num_hdbscan_fallbacks = 0

    for tid in order:
        group = embs[indices_by_tracklet[tid]]  # [k, D]

        if pool == "mean":
            pooled_embs.append(_pool_mean(group))
        elif pool == "max":
            pooled_embs.append(_pool_max(group))
        elif pool == "weighted":
            pooled_embs.append(_pool_weighted(group, weighted_temperature))
        elif pool == "medoid":
            pooled_embs.append(_pool_medoid(group))
        elif pool == "consensus":
            pooled_embs.append(_pool_consensus(group, consensus_k))
        elif pool == "dbscan":
            emb, fallback = _pool_dbscan(group, dbscan_eps, dbscan_min_samples)
            pooled_embs.append(emb)
            if fallback:
                num_dbscan_fallbacks += 1
        elif pool == "hdbscan":
            emb, fallback = _pool_hdbscan(
                group, hdbscan_min_cluster_size, hdbscan_min_samples
            )
            pooled_embs.append(emb)
            if fallback:
                num_hdbscan_fallbacks += 1
        elif pool == "gem":
            pooled_embs.append(_pool_gem(group))

    # ----------------------------------------------------------
    # Logging
    # ----------------------------------------------------------

    if pool == "dbscan":
        print(f"DBSCAN fallbacks: {num_dbscan_fallbacks:,}/{len(order):,} tracklets")
    elif pool == "hdbscan":
        print(
            f"HDBSCAN fallbacks: {num_hdbscan_fallbacks:,}/{len(order):,} tracklets"
        )

    print(
        f"Pooled from {embs.shape[0]:,} images into {len(order):,} tracklet embeddings"
    )

    return (
        torch.stack(pooled_embs),
        [pid_by_tracklet[t] for t in order],
        [camid_by_tracklet[t] for t in order],
        order,
    )
