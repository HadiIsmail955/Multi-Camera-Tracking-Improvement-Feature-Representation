import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    precision_recall_curve,
    roc_auc_score,
    silhouette_score,
    v_measure_score,
)

from reid.dataLoader.customData.MTMCCSVDataset import MTMCCSVDataset
from reid.dataLoader.transformation.ReIDTransform import ReIDTransform
from reid.model.DINOv2ReID import DINOv2ReID


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def save_metrics(metrics: Dict, out_dir: Path):
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(metrics), f, indent=2)

    flat = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if not isinstance(sub_value, (list, tuple, dict)):
                    flat[f"{key}_{sub_key}"] = sub_value
        elif not isinstance(value, (list, tuple, dict)):
            flat[key] = value

    pd.DataFrame([flat]).to_csv(out_dir / "metrics.csv", index=False)


def get_batch_value(batch, key: str, i: int, default=None):
    if key not in batch:
        return default

    value = batch[key]

    if torch.is_tensor(value):
        x = value[i]
        if x.ndim == 0:
            return x.item()
        return x.detach().cpu().numpy().tolist()

    if isinstance(value, (list, tuple)):
        return value[i]

    return value


def safe_int(value, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def string_series(values) -> np.ndarray:
    return np.asarray([str(v) for v in values])


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------


def infer_model_args_from_checkpoint(ckpt: Dict, cli_args):
    ckpt_args = ckpt.get("args", {}) or {}

    def pick(name, default=None):
        if hasattr(cli_args, name) and getattr(cli_args, name) is not None:
            return getattr(cli_args, name)
        return ckpt_args.get(name, default)

    inferred = {
        "embedding_dim": int(pick("embedding_dim", 512)),
        "backbone_type": pick("backbone_type", "vit_b"),
        "dropout": float(pick("dropout", 0.1)),
    }

    return inferred


def get_num_classes_from_checkpoint(ckpt: Dict) -> int:
    if "id_to_label" in ckpt and ckpt["id_to_label"] is not None:
        return len(ckpt["id_to_label"])

    if "label_to_id" in ckpt and ckpt["label_to_id"] is not None:
        return len(ckpt["label_to_id"])

    args = ckpt.get("args", {}) or {}
    if "num_classes" in args:
        return int(args["num_classes"])

    raise KeyError(
        "Could not infer num_classes from checkpoint. Expected one of: "
        "'id_to_label', 'label_to_id', or args['num_classes']."
    )


def load_model_from_checkpoint(checkpoint_path: str, args, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    num_classes = get_num_classes_from_checkpoint(ckpt)
    model_args = infer_model_args_from_checkpoint(ckpt, args)

    model = DINOv2ReID(
        num_classes=num_classes,
        embedding_dim=model_args["embedding_dim"],
        dino_type=model_args["backbone_type"],
        unfreeze_last_blocks=0,
        dropout=model_args["dropout"],
        freeze_backbone=True,
    ).to(device)

    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)

    if len(missing) > 0:
        print("[WARN] Missing keys while loading checkpoint:")
        for key in missing[:20]:
            print("  ", key)
        if len(missing) > 20:
            print(f"  ... {len(missing) - 20} more")

    if len(unexpected) > 0:
        print("[WARN] Unexpected keys while loading checkpoint:")
        for key in unexpected[:20]:
            print("  ", key)
        if len(unexpected) > 20:
            print(f"  ... {len(unexpected) - 20} more")

    model.eval()
    return model, ckpt


# -----------------------------------------------------------------------------
# Embedding extraction, no logits
# -----------------------------------------------------------------------------


@torch.no_grad()
def extract_crop_embeddings(
    model,
    loader,
    device: str,
    embedding_key: str = "bn_embedding",
    use_amp: bool = True,
):
    """
    Extract embeddings only.

    This function intentionally ignores logits. Labels/global IDs are stored
    only for evaluation after embeddings are extracted.
    """
    model.eval()
    amp_enabled = bool(use_amp and device == "cuda")

    all_embeddings = []
    rows = []

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            outputs = model(images)

        if not isinstance(outputs, dict):
            raise TypeError(
                "Expected model(images) to return a dict containing embeddings. "
                "The validation script does not support logits-only outputs."
            )

        if embedding_key not in outputs:
            raise KeyError(
                f"Embedding key '{embedding_key}' not found. "
                f"Available keys: {list(outputs.keys())}"
            )

        embeddings = outputs[embedding_key].float()
        embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings_np = embeddings.cpu().numpy()

        all_embeddings.append(embeddings_np)

        batch_size = embeddings_np.shape[0]
        for i in range(batch_size):
            global_id = get_batch_value(batch, "global_id", i, default=None)
            label = get_batch_value(batch, "label", i, default=None)

            if global_id is None:
                global_id = label

            row = {
                "row_index": len(rows),
                "label": safe_int(label),
                "global_id": str(global_id),
                "camera": str(
                    get_batch_value(
                        batch,
                        "camera",
                        i,
                        default=get_batch_value(batch, "camera_id", i, default="unknown"),
                    )
                ),
                "camera_id": safe_int(get_batch_value(batch, "camera_id", i, default=-1)),
                "frame": safe_int(get_batch_value(batch, "frame", i, default=-1)),
                "object_type": str(get_batch_value(batch, "object_type", i, default="unknown")),
                "scene": str(get_batch_value(batch, "scene", i, default="unknown")),
                "crop_path": str(get_batch_value(batch, "crop_path", i, default="")),
                "track_id": get_batch_value(
                    batch,
                    "track_id",
                    i,
                    default=get_batch_value(batch, "tracklet_id", i, default=None),
                ),
            }

            if row["track_id"] is not None:
                row["track_id"] = str(row["track_id"])

            rows.append(row)

    if len(all_embeddings) == 0:
        raise RuntimeError("No embeddings extracted. Check the validation dataset.")

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    embeddings = normalize(embeddings)
    df = pd.DataFrame(rows)

    true_ids = sorted(df["global_id"].astype(str).unique())
    id_to_label = {gid: idx for idx, gid in enumerate(true_ids)}
    df["true_label"] = df["global_id"].astype(str).map(id_to_label).astype(int)

    return embeddings, df


# -----------------------------------------------------------------------------
# Tracklet aggregation
# -----------------------------------------------------------------------------


def aggregate_group_features(E: np.ndarray, method: str = "mean") -> np.ndarray:
    E = normalize(E.astype(np.float32))

    if method == "mean":
        v = E.mean(axis=0)
        return v / max(np.linalg.norm(v), 1e-12)

    if method == "medoid":
        mean = E.mean(axis=0)
        mean = mean / max(np.linalg.norm(mean), 1e-12)
        scores = E @ mean
        return E[int(np.argmax(scores))]

    if method == "mean_topk":
        mean = E.mean(axis=0)
        mean = mean / max(np.linalg.norm(mean), 1e-12)
        scores = E @ mean
        k = min(20, len(E))
        idx = np.argsort(-scores)[:k]
        v = E[idx].mean(axis=0)
        return v / max(np.linalg.norm(v), 1e-12)

    raise ValueError(f"Unknown aggregation method: {method}")


def aggregate_to_tracklets(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    group_mode: str = "auto",
    aggregation: str = "mean_topk",
):
    """
    Build one vector per tracklet/group.

    group_mode:
        auto:
            Use scene+camera+track_id+object_type if track_id exists,
            otherwise use scene+global_id+camera+object_type.
        track_id:
            Realistic pipeline mode. Requires track_id/tracklet_id in dataset.
        global_id_camera:
            Clean analysis mode. Groups crops by known GT identity per camera.
    """
    emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)
    full_df = pd.concat([df.reset_index(drop=True), emb_df], axis=1)

    has_track_id = (
        "track_id" in full_df.columns
        and full_df["track_id"].notna().any()
        and not (full_df["track_id"].astype(str) == "None").all()
    )

    if group_mode == "auto":
        group_mode = "track_id" if has_track_id else "global_id_camera"

    if group_mode == "track_id":
        if not has_track_id:
            raise ValueError(
                "group_mode='track_id' requested, but no track_id/tracklet_id exists."
            )
        group_cols = ["scene", "camera", "track_id", "object_type"]

    elif group_mode == "global_id_camera":
        group_cols = ["scene", "global_id", "camera", "object_type"]

    else:
        raise ValueError(f"Unknown group_mode: {group_mode}")

    rows = []
    out_embeddings = []

    for keys, group in full_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_dict = dict(zip(group_cols, keys))

        E = group[emb_cols].values.astype(np.float32)
        group_emb = aggregate_group_features(E, method=aggregation)

        row = {
            "num_crops": int(len(group)),
            "global_id": str(group["global_id"].iloc[0]),
            "label": safe_int(group["label"].iloc[0]),
            "true_label": safe_int(group["true_label"].iloc[0]),
            "camera": str(group["camera"].iloc[0]),
            "camera_id": safe_int(group["camera_id"].iloc[0]),
            "scene": str(group["scene"].iloc[0]),
            "object_type": str(group["object_type"].iloc[0]),
            "start_frame": safe_int(group["frame"].min()),
            "end_frame": safe_int(group["frame"].max()),
        }

        if "track_id" in group.columns:
            row["track_id"] = str(group["track_id"].iloc[0])

        row.update({k: str(v) for k, v in key_dict.items()})

        rows.append(row)
        out_embeddings.append(group_emb)

    out_embeddings = np.stack(out_embeddings, axis=0).astype(np.float32)
    out_embeddings = normalize(out_embeddings)
    out_df = pd.DataFrame(rows)

    true_ids = sorted(out_df["global_id"].astype(str).unique())
    id_to_label = {gid: idx for idx, gid in enumerate(true_ids)}
    out_df["true_label"] = out_df["global_id"].astype(str).map(id_to_label).astype(int)

    return out_embeddings, out_df, group_mode


# -----------------------------------------------------------------------------
# Embedding diagnostics
# -----------------------------------------------------------------------------


def stats_np(values: np.ndarray, prefix: str) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float32)

    if values.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_p01": float("nan"),
            f"{prefix}_p05": float("nan"),
            f"{prefix}_p50": float("nan"),
            f"{prefix}_p95": float("nan"),
            f"{prefix}_p99": float("nan"),
        }

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_p01": float(np.percentile(values, 1)),
        f"{prefix}_p05": float(np.percentile(values, 5)),
        f"{prefix}_p50": float(np.percentile(values, 50)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_p99": float(np.percentile(values, 99)),
    }


def compute_embedding_health_metrics(embeddings: np.ndarray) -> Dict[str, float]:
    E = normalize(embeddings.astype(np.float32))
    n, d = E.shape

    norms = np.linalg.norm(E, axis=1)
    dim_std = E.std(axis=0)

    centered = E - E.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(n - 1, 1)
    eigvals = np.linalg.eigvalsh(cov).clip(min=0)
    eig_sum = eigvals.sum() + 1e-12
    probs = eigvals / eig_sum
    entropy = -(probs * np.log(probs + 1e-12)).sum()
    effective_rank = float(np.exp(entropy))
    participation_ratio = float((eig_sum ** 2) / (np.square(eigvals).sum() + 1e-12))

    metrics = {
        "num_embeddings": int(n),
        "embedding_dim": int(d),
        "dim_std_mean": float(dim_std.mean()),
        "dim_std_min": float(dim_std.min()),
        "dim_std_max": float(dim_std.max()),
        "effective_rank": effective_rank,
        "effective_rank_ratio": float(effective_rank / max(d, 1)),
        "participation_ratio": participation_ratio,
        "participation_ratio_ratio": float(participation_ratio / max(d, 1)),
    }
    metrics.update(stats_np(norms, "norm"))
    return metrics


# -----------------------------------------------------------------------------
# Similarity pair sampling and threshold metrics
# -----------------------------------------------------------------------------


def sample_similarity_pairs(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    max_pairs: int = 200000,
    seed: int = 42,
) -> pd.DataFrame:
    E = normalize(embeddings.astype(np.float32))
    n = len(E)

    global_ids = df["global_id"].astype(str).values
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

    rows = []
    for i, j in pairs:
        sim = float(np.dot(E[i], E[j]))
        same_id = global_ids[i] == global_ids[j]
        same_camera = cameras[i] == cameras[j]
        same_object_type = object_types[i] == object_types[j]

        rows.append(
            {
                "i": int(i),
                "j": int(j),
                "cosine": sim,
                "same_id": int(same_id),
                "same_camera": int(same_camera),
                "cross_camera": int(not same_camera),
                "same_object_type": int(same_object_type),
                "global_id_i": global_ids[i],
                "global_id_j": global_ids[j],
                "camera_i": cameras[i],
                "camera_j": cameras[j],
            }
        )

    return pd.DataFrame(rows)


def compute_similarity_distribution_metrics(pair_df: pd.DataFrame) -> Dict[str, float]:
    same_id = pair_df[pair_df["same_id"] == 1]["cosine"].values
    diff_id = pair_df[pair_df["same_id"] == 0]["cosine"].values
    same_id_cross = pair_df[(pair_df["same_id"] == 1) & (pair_df["cross_camera"] == 1)]["cosine"].values
    same_camera_diff = pair_df[(pair_df["same_id"] == 0) & (pair_df["same_camera"] == 1)]["cosine"].values

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
        metrics["same_id_cross_camera_cos_mean"] - metrics["same_camera_diff_id_cos_mean"]
    )

    return metrics


def compute_threshold_metrics(pair_df: pd.DataFrame) -> Dict[str, float]:
    y_true = pair_df["same_id"].values.astype(int)
    scores = pair_df["cosine"].values.astype(np.float32)

    metrics = {}

    if len(np.unique(y_true)) < 2:
        return {
            "pair_roc_auc": float("nan"),
            "pair_average_precision": float("nan"),
            "best_f1": float("nan"),
            "best_threshold": float("nan"),
            "best_precision": float("nan"),
            "best_recall": float("nan"),
        }

    metrics["pair_roc_auc"] = float(roc_auc_score(y_true, scores))
    metrics["pair_average_precision"] = float(average_precision_score(y_true, scores))

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)

    best_idx = int(np.nanargmax(f1))
    metrics["best_f1"] = float(f1[best_idx])
    metrics["best_precision"] = float(precision[best_idx])
    metrics["best_recall"] = float(recall[best_idx])

    if best_idx < len(thresholds):
        metrics["best_threshold"] = float(thresholds[best_idx])
    else:
        metrics["best_threshold"] = float(thresholds[-1])

    return metrics


# -----------------------------------------------------------------------------
# Cross-camera retrieval metrics, no logits
# -----------------------------------------------------------------------------


def compute_cross_camera_retrieval_metrics(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    ranks: Sequence[int] = (1, 5, 10, 20),
    max_rank_curve: int = 50,
    metric_device: str = "auto",
    chunk_size: int = 1024,
):
    """
    For each query, search candidates from different cameras only.
    A retrieved item is correct if it has the same global_id.
    """
    E = torch.from_numpy(normalize(embeddings.astype(np.float32)))

    if metric_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = metric_device

    E = E.to(device)

    global_ids = df["global_id"].astype(str).values
    cameras = df["camera"].astype(str).values

    n = E.size(0)
    max_rank_curve = min(max_rank_curve, max(n - 1, 1))

    rank_hits = {rank: [] for rank in ranks}
    cmc_hits = {rank: [] for rank in range(1, max_rank_curve + 1)}
    aps = []
    valid_queries = 0

    query_records = []

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sims = torch.matmul(E[start:end], E.T).cpu().numpy()

        for local_i, i in enumerate(range(start, end)):
            candidate_mask = cameras != cameras[i]
            candidate_mask[i] = False

            candidate_indices = np.where(candidate_mask)[0]
            if candidate_indices.size == 0:
                continue

            scores = sims[local_i, candidate_indices]
            matches = (global_ids[candidate_indices] == global_ids[i]).astype(np.int32)

            if matches.sum() == 0:
                continue

            valid_queries += 1
            order = np.argsort(-scores)
            sorted_matches = matches[order]
            sorted_scores = scores[order]
            sorted_indices = candidate_indices[order]

            for rank in ranks:
                rank_hits[rank].append(float(sorted_matches[:rank].sum() > 0))

            for rank in range(1, max_rank_curve + 1):
                cmc_hits[rank].append(float(sorted_matches[:rank].sum() > 0))

            ap = average_precision_score(sorted_matches, sorted_scores)
            aps.append(float(ap))

            top_idx = int(sorted_indices[0])
            query_records.append(
                {
                    "query_index": int(i),
                    "query_global_id": global_ids[i],
                    "query_camera": cameras[i],
                    "top1_index": top_idx,
                    "top1_global_id": global_ids[top_idx],
                    "top1_camera": cameras[top_idx],
                    "top1_score": float(sorted_scores[0]),
                    "top1_correct": int(sorted_matches[0]),
                    "ap": float(ap),
                }
            )

    if valid_queries == 0:
        metrics = {f"Rank{rank}": float("nan") for rank in ranks}
        metrics["mAP"] = float("nan")
        metrics["valid_queries"] = 0
        rank_curve = pd.DataFrame({"rank": [], "accuracy": []})
        query_df = pd.DataFrame(query_records)
        return metrics, rank_curve, query_df

    metrics = {f"Rank{rank}": float(np.mean(rank_hits[rank])) for rank in ranks}
    metrics["mAP"] = float(np.mean(aps))
    metrics["valid_queries"] = int(valid_queries)

    rank_curve = pd.DataFrame(
        {
            "rank": list(range(1, max_rank_curve + 1)),
            "accuracy": [float(np.mean(cmc_hits[r])) for r in range(1, max_rank_curve + 1)],
        }
    )

    query_df = pd.DataFrame(query_records)
    return metrics, rank_curve, query_df


# -----------------------------------------------------------------------------
# Clustering metrics
# -----------------------------------------------------------------------------


def cluster_unknown_k(
    embeddings: np.ndarray,
    method: str = "hdbscan",
    min_cluster_size: int = 3,
    min_samples: int = 2,
    dbscan_eps: float = 0.35,
):
    E = normalize(embeddings.astype(np.float32))

    if method == "hdbscan":
        try:
            import hdbscan
        except ImportError as exc:
            raise ImportError(
                "hdbscan is not installed. Install it or use --cluster_method dbscan/optics."
            ) from exc

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        return clusterer.fit_predict(E)

    if method == "dbscan":
        from sklearn.cluster import DBSCAN

        clusterer = DBSCAN(
            eps=dbscan_eps,
            min_samples=min_samples,
            metric="cosine",
        )
        return clusterer.fit_predict(E)

    if method == "optics":
        from sklearn.cluster import OPTICS

        clusterer = OPTICS(
            min_samples=min_samples,
            metric="cosine",
            cluster_method="xi",
        )
        return clusterer.fit_predict(E)

    raise ValueError(f"Unknown cluster method: {method}")


def compute_clustering_metrics(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    cluster_labels: np.ndarray,
) -> Dict[str, float]:
    metrics = {}

    metrics["ARI"] = float(adjusted_rand_score(true_labels, cluster_labels))
    metrics["NMI"] = float(normalized_mutual_info_score(true_labels, cluster_labels))
    metrics["Homogeneity"] = float(homogeneity_score(true_labels, cluster_labels))
    metrics["Completeness"] = float(completeness_score(true_labels, cluster_labels))
    metrics["V_measure"] = float(v_measure_score(true_labels, cluster_labels))

    valid = cluster_labels != -1

    metrics["num_true_ids"] = int(len(np.unique(true_labels)))
    metrics["num_found_clusters"] = int(len(set(cluster_labels) - {-1}))
    metrics["num_noise_samples"] = int((cluster_labels == -1).sum())
    metrics["noise_ratio"] = float((cluster_labels == -1).mean())

    if valid.sum() > 0:
        metrics["ARI_no_noise"] = float(adjusted_rand_score(true_labels[valid], cluster_labels[valid]))
        metrics["NMI_no_noise"] = float(normalized_mutual_info_score(true_labels[valid], cluster_labels[valid]))
    else:
        metrics["ARI_no_noise"] = float("nan")
        metrics["NMI_no_noise"] = float("nan")

    if valid.sum() > 1 and len(np.unique(cluster_labels[valid])) > 1:
        metrics["Silhouette_cosine"] = float(
            silhouette_score(embeddings[valid], cluster_labels[valid], metric="cosine")
        )
    else:
        metrics["Silhouette_cosine"] = float("nan")

    return metrics


# -----------------------------------------------------------------------------
# Dimensionality reduction and visualization
# -----------------------------------------------------------------------------


def choose_plot_subset(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    max_points: int,
    seed: int = 42,
):
    if max_points <= 0 or len(df) <= max_points:
        return embeddings, df.copy()

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=max_points, replace=False)
    idx = np.sort(idx)
    return embeddings[idx], df.iloc[idx].reset_index(drop=True).copy()


def reduce_embeddings(embeddings: np.ndarray, method: str = "tsne", seed: int = 42):
    E = normalize(embeddings.astype(np.float32))

    if len(E) < 3:
        if E.shape[1] >= 2:
            return E[:, :2]
        return np.pad(E, ((0, 0), (0, 2 - E.shape[1])))

    if method == "pca":
        reducer = PCA(n_components=2, random_state=seed)
        return reducer.fit_transform(E)

    if method == "tsne":
        perplexity = min(30, max(2, (len(E) - 1) // 3))
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            random_state=seed,
        )
        return reducer.fit_transform(E)

    if method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise ImportError("umap-learn is not installed. Use --reduce_method tsne or pca.") from exc

        reducer = umap.UMAP(n_components=2, metric="cosine", random_state=seed)
        return reducer.fit_transform(E)

    raise ValueError(f"Unknown reduce method: {method}")


def plot_scatter(xy: np.ndarray, labels, title: str, out_path: Path, size: int = 18):
    labels = np.asarray(labels)

    plt.figure(figsize=(10, 8))
    plt.scatter(
        xy[:, 0],
        xy[:, 1],
        c=labels,
        s=size,
        alpha=0.8,
        cmap="tab20",
    )
    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_similarity_histogram(pair_df: pd.DataFrame, out_path: Path):
    same = pair_df[pair_df["same_id"] == 1]["cosine"].values
    diff = pair_df[pair_df["same_id"] == 0]["cosine"].values
    same_cross = pair_df[(pair_df["same_id"] == 1) & (pair_df["cross_camera"] == 1)]["cosine"].values

    plt.figure(figsize=(10, 6))
    bins = np.linspace(-1, 1, 80)

    if len(diff) > 0:
        plt.hist(diff, bins=bins, alpha=0.55, density=True, label="different ID")
    if len(same) > 0:
        plt.hist(same, bins=bins, alpha=0.55, density=True, label="same ID")
    if len(same_cross) > 0:
        plt.hist(same_cross, bins=bins, alpha=0.55, density=True, label="same ID cross-camera")

    plt.xlabel("cosine similarity")
    plt.ylabel("density")
    plt.title("Embedding similarity distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_rank_curve(rank_curve: pd.DataFrame, out_path: Path):
    if len(rank_curve) == 0:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(rank_curve["rank"].values, rank_curve["accuracy"].values, marker="o", linewidth=2)
    plt.xlabel("Rank")
    plt.ylabel("Retrieval accuracy")
    plt.title("Cross-camera CMC / Rank curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# -----------------------------------------------------------------------------
# Main validation flow
# -----------------------------------------------------------------------------


def build_dataset(args):
    transform = ReIDTransform(
        backbone=args.backbone,
        img_size=args.image_size,
        train=False,
    )

    dataset = MTMCCSVDataset(
        root=args.data_root,
        split=args.split,
        scene_folders=args.scenes,
        transform=transform,
        min_images_per_id=args.min_images_per_id,
        object_types=args.object_types,
        base_path=args.base_path,
        verify_paths=args.verify_paths,
        scene_aware_ids=args.scene_aware_ids,
    )

    return dataset


def main(args):
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("Full ReID embedding validation, no logits")
    print("=" * 80)
    print("Device:", device)
    print("Checkpoint:", args.checkpoint)
    print("Data root:", args.data_root)
    print("Split:", args.split)
    print("Level:", args.level)
    print("Embedding key:", args.embedding_key)
    print("Output dir:", out_dir)
    print("=" * 80)

    dataset = build_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.workers > 0,
    )

    print("Validation samples:", len(dataset))
    print("Validation classes:", getattr(dataset, "num_classes", "unknown"))
    print("Validation scenes:", getattr(dataset, "scene_folders", "unknown"))

    model, ckpt = load_model_from_checkpoint(args.checkpoint, args, device)

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

    np.save(out_dir / "embeddings.npy", eval_embeddings)
    eval_df.to_csv(out_dir / "embedding_metadata.csv", index=False)

    # Health metrics
    health_metrics = compute_embedding_health_metrics(eval_embeddings)

    # Pair metrics and threshold metrics
    pair_df = sample_similarity_pairs(
        embeddings=eval_embeddings,
        df=eval_df,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )
    pair_df.to_csv(out_dir / "pair_similarity_sample.csv", index=False)

    similarity_metrics = compute_similarity_distribution_metrics(pair_df)
    threshold_metrics = compute_threshold_metrics(pair_df)

    # Retrieval metrics
    retrieval_metrics, rank_curve, query_results = compute_cross_camera_retrieval_metrics(
        embeddings=eval_embeddings,
        df=eval_df,
        ranks=args.ranks,
        max_rank_curve=args.max_rank_curve,
        metric_device=args.metric_device,
        chunk_size=args.metric_chunk_size,
    )
    rank_curve.to_csv(out_dir / "rank_curve.csv", index=False)
    query_results.to_csv(out_dir / "query_retrieval_results.csv", index=False)

    # Clustering metrics
    cluster_labels = cluster_unknown_k(
        embeddings=eval_embeddings,
        method=args.cluster_method,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        dbscan_eps=args.dbscan_eps,
    )
    eval_df["cluster_label"] = cluster_labels
    eval_df.to_csv(out_dir / "embedding_metadata.csv", index=False)

    clustering_metrics = compute_clustering_metrics(
        embeddings=eval_embeddings,
        true_labels=eval_df["true_label"].values,
        cluster_labels=cluster_labels,
    )

    # Visualizations
    plot_embeddings, plot_df = choose_plot_subset(
        eval_embeddings,
        eval_df,
        max_points=args.max_plot_points,
        seed=args.seed,
    )
    xy = reduce_embeddings(plot_embeddings, method=args.reduce_method, seed=args.seed)
    plot_df["x"] = xy[:, 0]
    plot_df["y"] = xy[:, 1]
    plot_df.to_csv(out_dir / "visualization_points.csv", index=False)

    plot_scatter(
        xy,
        plot_df["true_label"].values,
        f"{args.level} embeddings colored by global_id",
        out_dir / "embedding_by_global_id.png",
    )

    camera_codes = pd.Categorical(plot_df["camera"].astype(str)).codes
    plot_scatter(
        xy,
        camera_codes,
        f"{args.level} embeddings colored by camera",
        out_dir / "embedding_by_camera.png",
    )

    object_type_codes = pd.Categorical(plot_df["object_type"].astype(str)).codes
    plot_scatter(
        xy,
        object_type_codes,
        f"{args.level} embeddings colored by object type",
        out_dir / "embedding_by_object_type.png",
    )

    cluster_plot_labels = plot_df["cluster_label"].values if "cluster_label" in plot_df.columns else np.zeros(len(plot_df))
    plot_scatter(
        xy,
        cluster_plot_labels,
        f"{args.level} embeddings colored by discovered cluster",
        out_dir / "embedding_by_cluster.png",
    )

    plot_similarity_histogram(pair_df, out_dir / "similarity_histogram.png")
    plot_rank_curve(rank_curve, out_dir / "rank_curve.png")

    # Final combined metrics
    metrics = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "level": args.level,
            "embedding_key": args.embedding_key,
            "tracklet_group_mode": args.tracklet_group_mode,
            "resolved_group_mode": resolved_group_mode,
            "aggregation": args.aggregation,
            "cluster_method": args.cluster_method,
            "reduce_method": args.reduce_method,
        },
        "health": health_metrics,
        "retrieval": retrieval_metrics,
        "similarity": similarity_metrics,
        "threshold": threshold_metrics,
        "clustering": clustering_metrics,
    }

    save_metrics(metrics, out_dir)

    print("\nMain metrics:")
    print("  Retrieval mAP:", retrieval_metrics.get("mAP"))
    for rank in args.ranks:
        print(f"  Rank{rank}:", retrieval_metrics.get(f"Rank{rank}"))
    print("  Valid queries:", retrieval_metrics.get("valid_queries"))
    print("  Same-ID cosine mean:", similarity_metrics.get("same_id_cos_mean"))
    print("  Same-ID cross-camera cosine mean:", similarity_metrics.get("same_id_cross_camera_cos_mean"))
    print("  Different-ID cosine mean:", similarity_metrics.get("diff_id_cos_mean"))
    print("  Separation gap:", similarity_metrics.get("embedding_separation_gap"))
    print("  Pair ROC-AUC:", threshold_metrics.get("pair_roc_auc"))
    print("  Best threshold:", threshold_metrics.get("best_threshold"))
    print("  ARI:", clustering_metrics.get("ARI"))
    print("  NMI:", clustering_metrics.get("NMI"))
    print("  Effective rank:", health_metrics.get("effective_rank"))

    print("\nSaved outputs:")
    for path in sorted(out_dir.iterdir()):
        print(" ", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--data_root", default="DataSet/MTMC_Tracking_2025_Preprocessed")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--base_path", default=".")
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--object_types", nargs="*", default=None)
    parser.add_argument("--min_images_per_id", type=int, default=2)
    parser.add_argument("--verify_paths", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scene_aware_ids", action=argparse.BooleanOptionalAction, default=True)

    # Checkpoint / model
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embedding_key", default="bn_embedding", choices=["embedding", "bn_embedding"])
    parser.add_argument("--embedding_dim", type=int, default=None)
    parser.add_argument("--backbone_type", default=None, choices=[None, "vit_b", "vit_l", "vit_g"])
    parser.add_argument("--backbone", default="dinov2")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=None)

    # Runtime
    parser.add_argument("--out_dir", default="embedding_validation_full")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)

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
        choices=["mean", "medoid", "mean_topk"],
    )

    # Retrieval / pair metrics
    parser.add_argument("--ranks", nargs="*", type=int, default=[1, 5, 10, 20])
    parser.add_argument("--max_rank_curve", type=int, default=50)
    parser.add_argument("--metric_device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--metric_chunk_size", type=int, default=1024)
    parser.add_argument("--max_pairs", type=int, default=200000)

    # Clustering
    parser.add_argument("--cluster_method", default="hdbscan", choices=["hdbscan", "dbscan", "optics"])
    parser.add_argument("--min_cluster_size", type=int, default=3)
    parser.add_argument("--min_samples", type=int, default=2)
    parser.add_argument("--dbscan_eps", type=float, default=0.35)

    # Visualization
    parser.add_argument("--reduce_method", default="tsne", choices=["tsne", "pca", "umap"])
    parser.add_argument("--max_plot_points", type=int, default=3000)

    args = parser.parse_args()
    main(args)
