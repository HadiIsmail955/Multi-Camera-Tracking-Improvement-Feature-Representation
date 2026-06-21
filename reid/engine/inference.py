# src/engine/inference.py

import csv
from collections import defaultdict

import torch
from tqdm import tqdm

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


@torch.no_grad()
def extract_embeddings(
    model,
    loader,
    device,
    show_progress: bool = False,
):
    """
    Extract embeddings for a full DataLoader.

    Returns:
        embeddings: torch.Tensor [N, D]
        pids: list[int]
        camids: list[int]
    """
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


def read_tracklet_ids(csv_path: str) -> list[str]:
    """Read the ``tracklet_id`` column from a manifest CSV in row order.

    Returns an empty string for each row if the column is absent (falls
    back gracefully to image-level evaluation).
    """
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return [row.get("tracklet_id", "") for row in reader]


def pool_tracklet_embeddings(
    embs: torch.Tensor,
    pids: list[int],
    camids: list[int],
    tracklet_ids: list[str],
    pool: str = "mean",
    dbscan_eps: float = 0.2,
    dbscan_min_samples: int = 3,
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

        weighted
            Similarity-weighted pooling using all embeddings.

        medoid
            Use the most central embedding.

        consensus
            Mean of the top-k most central embeddings.

    Returns:
        (
            tracklet_embs,
            tracklet_pids,
            tracklet_camids,
            tracklet_ids
        )
    """

    print(f"\n  Pooling {embs.shape[0]:,} image embeddings using '{pool}' pooling...")

    valid_pools = (
        "mean",
        "max",
        "dbscan",
        "weighted",
        "medoid",
        "consensus",
        "gem",
    )

    if pool not in valid_pools:
        raise ValueError(f"pool must be one of {valid_pools}, got '{pool}'")

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

    pooled_embs: list[torch.Tensor] = []

    num_dbscan_fallbacks = 0

    # ----------------------------------------------------------
    # Pool each tracklet
    # ----------------------------------------------------------

    for tid in order:
        idx = indices_by_tracklet[tid]
        group = embs[idx]  # [k, D]
        centrality = None

        # ----------------------------------------------
        # Mean
        # ----------------------------------------------
        if pool == "mean":
            pooled_embs.append(group.mean(dim=0))
            continue

        # ----------------------------------------------
        # Max
        # ----------------------------------------------
        if pool == "max":
            pooled_embs.append(group.max(dim=0).values)
            continue

        # ----------------------------------------------
        # Shared preprocessing
        # ----------------------------------------------
        if pool in ("weighted", "medoid", "consensus"):
            if len(group) == 1:
                pooled_embs.append(group[0])
                continue

            feats = group.detach().cpu().numpy()
            feats = normalize(feats, norm="l2")

            sim = cosine_similarity(feats)

            # centrality = average similarity to all others
            centrality = sim.mean(axis=1)

        # ----------------------------------------------
        # Similarity-weighted pooling
        # ----------------------------------------------
        if pool == "weighted":
            if centrality is None:
                raise RuntimeError("centrality is not initialized for weighted pooling")

            centrality = centrality - centrality.max()

            weights = np.exp(centrality * weighted_temperature)

            weights = weights / (weights.sum() + 1e-12)

            weights_t = torch.tensor(
                weights,
                dtype=group.dtype,
                device=group.device,
            )

            pooled_embs.append((group * weights_t[:, None]).sum(dim=0))
            continue

        # ----------------------------------------------
        # Medoid pooling
        # ----------------------------------------------
        if pool == "medoid":
            if centrality is None:
                raise RuntimeError("centrality is not initialized for medoid pooling")

            medoid_idx = int(np.argmax(centrality))

            pooled_embs.append(group[medoid_idx])
            continue

        # ----------------------------------------------
        # Consensus pooling
        # ----------------------------------------------
        if pool == "consensus":
            if centrality is None:
                raise RuntimeError(
                    "centrality is not initialized for consensus pooling"
                )

            k = min(consensus_k, len(group))

            top_idx = np.argsort(centrality)[-k:]

            pooled_embs.append(group[top_idx].mean(dim=0))
            continue

        # ----------------------------------------------
        # DBSCAN pooling
        # ----------------------------------------------
        if pool == "dbscan":
            if len(group) < dbscan_min_samples:
                pooled_embs.append(group.mean(dim=0))
                num_dbscan_fallbacks += 1
                continue

            feats = group.detach().cpu().numpy()
            feats = normalize(feats, norm="l2")

            clustering = DBSCAN(
                eps=dbscan_eps,
                min_samples=dbscan_min_samples,
                metric="cosine",
            ).fit(feats)

            labels = clustering.labels_

            valid_mask = labels >= 0

            if not np.any(valid_mask):
                pooled_embs.append(group.mean(dim=0))
                num_dbscan_fallbacks += 1
                continue

            unique_labels, counts = np.unique(
                labels[valid_mask],
                return_counts=True,
            )

            largest_cluster = unique_labels[np.argmax(counts)]

            cluster_mask = labels == largest_cluster

            cluster_embs = group[cluster_mask]

            pooled_embs.append(cluster_embs.mean(dim=0))

            continue

        # ----------------------------------------------
        # GeM pooling
        # ----------------------------------------------
        if pool == "gem":
            gem_p = 3.0
            if len(group) == 1:
                pooled_embs.append(group[0])
                continue

            # GeM assumes positive values.
            # ReID embeddings usually contain negatives,
            # so we use a signed version.

            sign = torch.sign(group.mean(dim=0))

            x_pow = torch.sign(group) * torch.abs(group).pow(gem_p)
            gem_emb = torch.sign(x_pow.mean(dim=0)) * torch.abs(x_pow.mean(dim=0)).pow(1.0 / gem_p)

            pooled_embs.append(gem_emb)

            continue

    # ----------------------------------------------------------
    # Logging
    # ----------------------------------------------------------

    if pool == "dbscan":
        print(f"DBSCAN fallbacks: {num_dbscan_fallbacks:,}/{len(order):,} tracklets")

    print(
        f"Pooled from {embs.shape[0]:,} images into {len(order):,} tracklet embeddings"
    )

    # ----------------------------------------------------------
    # Return
    # ----------------------------------------------------------

    return (
        torch.stack(pooled_embs),
        [pid_by_tracklet[t] for t in order],
        [camid_by_tracklet[t] for t in order],
        order,
    )
