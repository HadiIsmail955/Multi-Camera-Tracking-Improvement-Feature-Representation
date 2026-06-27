from typing import Dict, Iterable
import torch
import torch.nn.functional as F
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import normalize

from .health import stats_np

def compute_embedding_similarity_metrics(
    embeddings: torch.Tensor,
    global_ids,
    cameras,
    max_pairs: int = 200000,
):
    embeddings = F.normalize(embeddings.float(), p=2, dim=1)
    n = embeddings.size(0)

    global_ids = [str(x) for x in global_ids]
    cameras = [str(x) for x in cameras]

    same_id = []
    diff_id = []
    same_id_cross_camera = []
    same_camera_diff_id = []

    total_pairs = n * (n - 1) // 2

    if total_pairs > max_pairs:
        pairs = torch.randint(0, n, (max_pairs, 2))
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        iterable = pairs.tolist()
    else:
        iterable = ((i, j) for i in range(n) for j in range(i + 1, n))

    for i, j in iterable:
        value = float(torch.dot(embeddings[i], embeddings[j]).item())

        is_same_id = global_ids[i] == global_ids[j]
        is_same_camera = cameras[i] == cameras[j]

        if is_same_id:
            same_id.append(value)
            if not is_same_camera:
                same_id_cross_camera.append(value)
        else:
            diff_id.append(value)
            if is_same_camera:
                same_camera_diff_id.append(value)

    def stats(values, prefix):
        if len(values) == 0:
            return {
                f"{prefix}_mean": 0.0,
                f"{prefix}_std": 0.0,
                f"{prefix}_p05": 0.0,
                f"{prefix}_p50": 0.0,
                f"{prefix}_p95": 0.0,
            }

        x = torch.tensor(values, dtype=torch.float32)
        return {
            f"{prefix}_mean": float(x.mean().item()),
            f"{prefix}_std": float(x.std(unbiased=False).item()),
            f"{prefix}_p05": float(torch.quantile(x, 0.05).item()),
            f"{prefix}_p50": float(torch.quantile(x, 0.50).item()),
            f"{prefix}_p95": float(torch.quantile(x, 0.95).item()),
        }

    metrics = {}
    metrics.update(stats(same_id, "same_id_cos"))
    metrics.update(stats(diff_id, "diff_id_cos"))
    metrics.update(stats(same_id_cross_camera, "same_id_cross_camera_cos"))
    metrics.update(stats(same_camera_diff_id, "same_camera_diff_id_cos"))

    metrics["embedding_separation_gap"] = (
        metrics["same_id_cos_mean"] - metrics["diff_id_cos_mean"]
    )

    metrics["cross_camera_gap"] = (
        metrics["same_id_cross_camera_cos_mean"] - metrics["diff_id_cos_mean"]
    )

    metrics["camera_confusion_gap"] = (
        metrics["same_id_cross_camera_cos_mean"]
        - metrics["same_camera_diff_id_cos_mean"]
    )

    return metrics

def compute_embedding_retrieval_metrics(
    embeddings: torch.Tensor,
    global_ids,
    cameras,
    ranks=(1, 5, 10),
):
    embeddings = F.normalize(embeddings.float(), p=2, dim=1)
    sim = torch.matmul(embeddings, embeddings.T)

    global_ids = [str(x) for x in global_ids]
    cameras = [str(x) for x in cameras]

    rank_hits = {rank: [] for rank in ranks}
    average_precisions = []
    valid_queries = 0

    n = embeddings.size(0)

    for i in range(n):
        candidate_indices = [
            j for j in range(n)
            if j != i and cameras[j] != cameras[i]
        ]

        if len(candidate_indices) == 0:
            continue

        scores = sim[i, candidate_indices]
        matches = torch.tensor(
            [global_ids[j] == global_ids[i] for j in candidate_indices],
            dtype=torch.float32,
        )

        if matches.sum().item() == 0:
            continue

        valid_queries += 1

        order = torch.argsort(scores, descending=True)
        sorted_matches = matches[order]

        for rank in ranks:
            topk = sorted_matches[:rank]
            rank_hits[rank].append(float(topk.sum().item() > 0))

        cumulative_matches = sorted_matches.cumsum(dim=0)
        precision_at_k = cumulative_matches / torch.arange(
            1,
            sorted_matches.numel() + 1,
            dtype=torch.float32,
        )

        ap = (precision_at_k * sorted_matches).sum() / sorted_matches.sum().clamp(min=1.0)
        average_precisions.append(float(ap.item()))

    if valid_queries == 0:
        metrics = {f"Rank{rank}": 0.0 for rank in ranks}
        metrics["mAP"] = 0.0
        metrics["valid_queries"] = 0
        return metrics

    metrics = {
        f"Rank{rank}": float(sum(rank_hits[rank]) / len(rank_hits[rank]))
        for rank in ranks
    }
    metrics["mAP"] = float(sum(average_precisions) / len(average_precisions))
    metrics["valid_queries"] = int(valid_queries)

    return metrics

def _pair_row(E, ids, cameras, object_types, i, j):
    sim = float(np.dot(E[i], E[j]))
    same_id = ids[i] == ids[j]
    same_camera = cameras[i] == cameras[j]
    same_object_type = object_types[i] == object_types[j]

    return {
        "i": int(i),
        "j": int(j),
        "cosine": sim,
        "same_id": int(same_id),
        "same_camera": int(same_camera),
        "cross_camera": int(not same_camera),
        "same_object_type": int(same_object_type),
        "id_i": ids[i],
        "id_j": ids[j],
        "camera_i": cameras[i],
        "camera_j": cameras[j],
    }


def sample_similarity_pairs_random(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    identity_col: str,
    max_pairs: int = 200000,
    seed: int = 42,
) -> pd.DataFrame:
    E = normalize(embeddings.astype(np.float32))
    n = len(E)

    ids = df[identity_col].astype(str).values
    cameras = df["camera"].astype(str).values
    object_types = df["object_type"].astype(str).values

    rng = np.random.default_rng(seed)
    total_pairs = n * (n - 1) // 2

    if total_pairs <= max_pairs:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        i = rng.integers(0, n, size=max_pairs)
        j = rng.integers(0, n, size=max_pairs)
        valid = i != j
        i = i[valid]
        j = j[valid]
        pairs = list(zip(i.tolist(), j.tolist()))

    rows = [_pair_row(E, ids, cameras, object_types, i, j) for i, j in pairs]
    return pd.DataFrame(rows)


def sample_similarity_pairs_balanced(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    identity_col: str,
    max_pairs: int = 200000,
    seed: int = 42,
) -> pd.DataFrame:
    E = normalize(embeddings.astype(np.float32))
    n = len(E)

    ids = df[identity_col].astype(str).values
    cameras = df["camera"].astype(str).values
    object_types = df["object_type"].astype(str).values

    rng = np.random.default_rng(seed)

    id_to_indices = {}
    for idx, identity in enumerate(ids):
        id_to_indices.setdefault(identity, []).append(idx)

    all_ids = list(id_to_indices.keys())

    target_pos = max_pairs // 3
    target_cross_pos = max_pairs // 3
    target_neg = max_pairs - target_pos - target_cross_pos

    pairs = set()

    attempts = 0
    while len(pairs) < target_pos and attempts < target_pos * 20:
        attempts += 1
        identity = rng.choice(all_ids)
        pool = id_to_indices[identity]

        if len(pool) < 2:
            continue

        i, j = rng.choice(pool, size=2, replace=False)
        if i > j:
            i, j = j, i

        pairs.add((int(i), int(j)))

    cross_pairs = set()
    attempts = 0
    while len(cross_pairs) < target_cross_pos and attempts < target_cross_pos * 30:
        attempts += 1
        identity = rng.choice(all_ids)
        pool = id_to_indices[identity]

        if len(pool) < 2:
            continue

        i, j = rng.choice(pool, size=2, replace=False)
        if cameras[i] == cameras[j]:
            continue

        if i > j:
            i, j = j, i

        cross_pairs.add((int(i), int(j)))

    pairs.update(cross_pairs)

    attempts = 0
    while len(pairs) < max_pairs and attempts < target_neg * 30:
        attempts += 1
        i, j = rng.integers(0, n, size=2)

        if i == j or ids[i] == ids[j]:
            continue

        if i > j:
            i, j = j, i

        pairs.add((int(i), int(j)))

    rows = [_pair_row(E, ids, cameras, object_types, i, j) for i, j in sorted(pairs)]
    return pd.DataFrame(rows)


def sample_similarity_pairs(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    identity_col: str,
    max_pairs: int = 200000,
    seed: int = 42,
    mode: str = "balanced",
) -> pd.DataFrame:
    if mode == "random":
        return sample_similarity_pairs_random(
            embeddings=embeddings,
            df=df,
            identity_col=identity_col,
            max_pairs=max_pairs,
            seed=seed,
        )

    if mode == "balanced":
        return sample_similarity_pairs_balanced(
            embeddings=embeddings,
            df=df,
            identity_col=identity_col,
            max_pairs=max_pairs,
            seed=seed,
        )

    raise ValueError(f"Unknown pair sampling mode: {mode}")


def compute_similarity_distribution_metrics(pair_df: pd.DataFrame) -> Dict[str, float]:
    same_id = pair_df[pair_df["same_id"] == 1]["cosine"].values
    diff_id = pair_df[pair_df["same_id"] == 0]["cosine"].values
    same_id_cross = pair_df[
        (pair_df["same_id"] == 1) & (pair_df["cross_camera"] == 1)
    ]["cosine"].values
    same_camera_diff = pair_df[
        (pair_df["same_id"] == 0) & (pair_df["same_camera"] == 1)
    ]["cosine"].values

    metrics = {}
    metrics.update(stats_np(same_id, "same_id_cos"))
    metrics.update(stats_np(diff_id, "diff_id_cos"))
    metrics.update(stats_np(same_id_cross, "same_id_cross_camera_cos"))
    metrics.update(stats_np(same_camera_diff, "same_camera_diff_id_cos"))

    metrics["embedding_separation_gap"] = float(
        metrics["same_id_cos_mean"] - metrics["diff_id_cos_mean"]
    )
    metrics["cross_camera_separation_gap"] = float(
        metrics["same_id_cross_camera_cos_mean"] - metrics["diff_id_cos_mean"]
    )
    metrics["camera_confusion_gap"] = float(
        metrics["same_id_cross_camera_cos_mean"]
        - metrics["same_camera_diff_id_cos_mean"]
    )

    return metrics


def compute_threshold_metrics(pair_df: pd.DataFrame) -> Dict[str, float]:
    y_true = pair_df["same_id"].values.astype(int)
    scores = pair_df["cosine"].values.astype(np.float32)

    if len(np.unique(y_true)) < 2:
        return {
            "pair_roc_auc": float("nan"),
            "pair_average_precision": float("nan"),
            "best_f1": float("nan"),
            "best_threshold": float("nan"),
            "best_precision": float("nan"),
            "best_recall": float("nan"),
        }

    metrics = {}
    metrics["pair_roc_auc"] = float(roc_auc_score(y_true, scores))
    metrics["pair_average_precision"] = float(average_precision_score(y_true, scores))

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)

    best_idx = int(np.nanargmax(f1))
    metrics["best_f1"] = float(f1[best_idx])
    metrics["best_precision"] = float(precision[best_idx])
    metrics["best_recall"] = float(recall[best_idx])

    if len(thresholds) == 0:
        metrics["best_threshold"] = float("nan")
    elif best_idx < len(thresholds):
        metrics["best_threshold"] = float(thresholds[best_idx])
    else:
        metrics["best_threshold"] = float(thresholds[-1])

    return metrics
