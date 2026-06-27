from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import normalize


def cluster_unknown_k(
    embeddings: np.ndarray,
    method: str = "dbscan",
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
                "hdbscan is not installed. Install it with `pip install hdbscan` or use --cluster_method dbscan/optics."
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
            n_jobs=-1,
        )
        return clusterer.fit_predict(E)

    if method == "optics":
        from sklearn.cluster import OPTICS

        clusterer = OPTICS(
            min_samples=min_samples,
            metric="cosine",
            cluster_method="xi",
            n_jobs=-1,
        )
        return clusterer.fit_predict(E)

    raise ValueError(f"Unknown cluster method: {method}")


def cluster_purity(y_true: np.ndarray, y_pred: np.ndarray, ignore_noise: bool = True) -> float:
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred)

    if ignore_noise:
        mask = y_pred != -1
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    if len(y_true) == 0:
        return 0.0

    cm = contingency_matrix(y_true, y_pred)

    if cm.size == 0 or cm.sum() == 0:
        return 0.0

    return float(np.sum(np.max(cm, axis=0)) / np.sum(cm))


def _comb2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x * (x - 1.0) / 2.0


def pairwise_cluster_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ignore_noise: bool = True,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred)

    if ignore_noise:
        mask = y_pred != -1
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            "cluster_pair_precision": 0.0,
            "cluster_pair_recall": 0.0,
            "cluster_pair_f1": 0.0,
        }

    cm = contingency_matrix(y_true, y_pred)

    tp = _comb2(cm).sum()
    pred_pairs = _comb2(cm.sum(axis=0)).sum()
    true_pairs = _comb2(cm.sum(axis=1)).sum()

    precision = tp / pred_pairs if pred_pairs > 0 else 0.0
    recall = tp / true_pairs if true_pairs > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "cluster_pair_precision": float(precision),
        "cluster_pair_recall": float(recall),
        "cluster_pair_f1": float(f1),
    }


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
    metrics["cluster_purity_no_noise"] = cluster_purity(true_labels, cluster_labels, ignore_noise=True)

    metrics.update(pairwise_cluster_scores(true_labels, cluster_labels, ignore_noise=True))

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


def add_cluster_failure_columns(
    df: pd.DataFrame,
    true_id_col: str = "identity_key",
    cluster_col: str = "cluster_label",
) -> pd.DataFrame:
    df = df.copy()

    cluster_to_identity = {}
    cluster_to_purity = {}

    non_noise = df[df[cluster_col] != -1]

    for cluster_id, group in non_noise.groupby(cluster_col):
        counts = group[true_id_col].astype(str).value_counts()
        dominant_identity = counts.index[0]
        purity = counts.iloc[0] / counts.sum()

        cluster_to_identity[int(cluster_id)] = dominant_identity
        cluster_to_purity[int(cluster_id)] = float(purity)

    df["cluster_as_identity"] = df[cluster_col].map(cluster_to_identity)
    df["cluster_purity"] = df[cluster_col].map(cluster_to_purity)

    df.loc[df[cluster_col] == -1, "cluster_as_identity"] = "NOISE"
    df.loc[df[cluster_col] == -1, "cluster_purity"] = 0.0

    df["is_noise"] = df[cluster_col].eq(-1)
    df["is_misclustered"] = (
        (~df["is_noise"])
        & (df[true_id_col].astype(str) != df["cluster_as_identity"].astype(str))
    )

    return df


def build_cluster_error_tables(
    df: pd.DataFrame,
    true_id_col: str = "identity_key",
    cluster_col: str = "cluster_label",
):
    non_noise = df[df[cluster_col] != -1].copy()

    if len(non_noise) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    cluster_summary = (
        non_noise.groupby([cluster_col, true_id_col])
        .size()
        .rename("count")
        .reset_index()
    )

    cluster_totals = (
        cluster_summary.groupby(cluster_col)["count"]
        .sum()
        .rename("cluster_total")
        .reset_index()
    )

    cluster_summary = cluster_summary.merge(cluster_totals, on=cluster_col)
    cluster_summary["ratio"] = cluster_summary["count"] / cluster_summary["cluster_total"]

    cluster_stats = (
        cluster_summary.sort_values([cluster_col, "count"], ascending=[True, False])
        .groupby(cluster_col)
        .agg(
            dominant_identity=(true_id_col, "first"),
            dominant_count=("count", "first"),
            cluster_total=("cluster_total", "first"),
            num_real_ids=(true_id_col, "nunique"),
        )
        .reset_index()
    )

    cluster_stats["purity"] = cluster_stats["dominant_count"] / cluster_stats["cluster_total"]

    merge_errors = cluster_stats[cluster_stats["num_real_ids"] > 1].copy()
    merge_errors = merge_errors.sort_values(["num_real_ids", "cluster_total"], ascending=False)

    fragmentation = (
        non_noise.groupby(true_id_col)
        .agg(
            num_clusters_for_id=(cluster_col, "nunique"),
            assigned_count=(cluster_col, "size"),
        )
        .reset_index()
    )

    noise_counts = (
        df[df[cluster_col] == -1]
        .groupby(true_id_col)
        .size()
        .rename("noise_count")
        .reset_index()
    )

    fragmentation = fragmentation.merge(noise_counts, on=true_id_col, how="left")
    fragmentation["noise_count"] = fragmentation["noise_count"].fillna(0).astype(int)
    fragmentation["is_fragmented"] = fragmentation["num_clusters_for_id"] > 1
    fragmentation = fragmentation.sort_values(
        ["num_clusters_for_id", "noise_count", "assigned_count"],
        ascending=False,
    )

    return cluster_summary, merge_errors, fragmentation


def compute_cluster_failure_metrics(
    df: pd.DataFrame,
    merge_errors: pd.DataFrame,
    fragmentation: pd.DataFrame,
    cluster_col: str = "cluster_label",
) -> Dict[str, float]:
    num_clusters = int(len(set(df[cluster_col].values) - {-1}))

    metrics = {
        "misclustered_sample_count": int(df["is_misclustered"].sum()),
        "misclustered_sample_rate": float(df["is_misclustered"].mean()) if len(df) else 0.0,
        "noise_sample_count": int(df["is_noise"].sum()),
        "noise_sample_rate": float(df["is_noise"].mean()) if len(df) else 0.0,
        "merge_error_cluster_count": int(len(merge_errors)),
        "merge_error_cluster_rate": float(len(merge_errors) / max(num_clusters, 1)),
    }

    if len(fragmentation) > 0:
        metrics["fragmented_identity_count"] = int(fragmentation["is_fragmented"].sum())
        metrics["fragmented_identity_rate"] = float(fragmentation["is_fragmented"].mean())
    else:
        metrics["fragmented_identity_count"] = 0
        metrics["fragmented_identity_rate"] = 0.0

    if "is_occluded" in df.columns:
        clean = df[df["is_occluded"].astype(int) == 0]
        occ = df[df["is_occluded"].astype(int) == 1]

        metrics["clean_noise_rate"] = float(clean["is_noise"].mean()) if len(clean) else 0.0
        metrics["occluded_noise_rate"] = float(occ["is_noise"].mean()) if len(occ) else 0.0
        metrics["clean_miscluster_rate"] = float(clean["is_misclustered"].mean()) if len(clean) else 0.0
        metrics["occluded_miscluster_rate"] = float(occ["is_misclustered"].mean()) if len(occ) else 0.0

    return metrics


def evaluate_dbscan_eps_grid(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    identity_col: str,
    eps_values: List[float],
    min_samples: int,
) -> pd.DataFrame:
    rows = []

    for eps in eps_values:
        cluster_labels = cluster_unknown_k(
            embeddings=embeddings,
            method="dbscan",
            min_samples=min_samples,
            dbscan_eps=eps,
        )

        temp_df = df.copy()
        temp_df["cluster_label"] = cluster_labels
        temp_df = add_cluster_failure_columns(
            temp_df,
            true_id_col=identity_col,
            cluster_col="cluster_label",
        )

        _, merge_errors, fragmentation = build_cluster_error_tables(
            temp_df,
            true_id_col=identity_col,
            cluster_col="cluster_label",
        )

        clustering_metrics = compute_clustering_metrics(
            embeddings=embeddings,
            true_labels=temp_df["true_label"].values,
            cluster_labels=cluster_labels,
        )
        failure_metrics = compute_cluster_failure_metrics(
            temp_df,
            merge_errors,
            fragmentation,
            cluster_col="cluster_label",
        )

        row = {"eps": float(eps), "min_samples": int(min_samples)}
        row.update(clustering_metrics)
        row.update(failure_metrics)
        rows.append(row)

    return pd.DataFrame(rows)
