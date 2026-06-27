from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize


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

        reducer = umap.UMAP(
            n_components=2,
            metric="cosine",
            random_state=seed,
        )
        return reducer.fit_transform(E)

    raise ValueError(f"Unknown reduce method: {method}")

def reduce_embeddings_3d(embeddings: np.ndarray, method: str = "pca", seed: int = 42):
    E = normalize(embeddings.astype(np.float32))

    if len(E) < 4:
        if E.shape[1] >= 3:
            return E[:, :3]
        return np.pad(E, ((0, 0), (0, 3 - E.shape[1])))

    if method == "pca":
        reducer = PCA(n_components=3, random_state=seed)
        return reducer.fit_transform(E)

    if method == "tsne":
        perplexity = min(30, max(2, (len(E) - 1) // 3))
        reducer = TSNE(
            n_components=3,
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
            raise ImportError("umap-learn is not installed. Use method='pca' or install umap-learn.") from exc

        reducer = umap.UMAP(
            n_components=3,
            metric="cosine",
            random_state=seed,
        )
        return reducer.fit_transform(E)

    raise ValueError(f"Unknown reduce method: {method}")

def labels_to_numeric(labels):
    labels = np.asarray(labels)

    if np.issubdtype(labels.dtype, np.number):
        return labels

    return pd.Categorical(labels.astype(str)).codes


def plot_scatter(xy: np.ndarray, labels, title: str, out_path: Path, size: int = 18):
    numeric_labels = labels_to_numeric(labels)

    plt.figure(figsize=(10, 8))
    plt.scatter(
        xy[:, 0],
        xy[:, 1],
        c=numeric_labels,
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
    same_cross = pair_df[
        (pair_df["same_id"] == 1) & (pair_df["cross_camera"] == 1)
    ]["cosine"].values

    plt.figure(figsize=(10, 6))
    bins = np.linspace(-1, 1, 80)

    if len(diff) > 0:
        plt.hist(diff, bins=bins, alpha=0.55, density=True, label="different ID")

    if len(same) > 0:
        plt.hist(same, bins=bins, alpha=0.55, density=True, label="same ID")

    if len(same_cross) > 0:
        plt.hist(
            same_cross,
            bins=bins,
            alpha=0.55,
            density=True,
            label="same ID cross-camera",
        )

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
    plt.plot(
        rank_curve["rank"].values,
        rank_curve["accuracy"].values,
        marker="o",
        linewidth=2,
    )
    plt.xlabel("Rank")
    plt.ylabel("Retrieval accuracy")
    plt.title("Cross-camera CMC / Rank curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_interactive_plots(
    plot_df: pd.DataFrame,
    out_dir: Path,
    level: str,
    identity_col: str,
):
    if "x" not in plot_df.columns or "y" not in plot_df.columns:
        return

    try:
        import plotly.express as px
    except ImportError:
        print("[WARN] plotly is not installed; skipping interactive HTML plots.")
        return

    df = plot_df.copy()

    hover_cols = [
        col
        for col in [
            identity_col,
            "global_id",
            "identity_key",
            "cluster_label",
            "cluster_as_identity",
            "cluster_purity",
            "camera",
            "object_type",
            "scene",
            "frame",
            "start_frame",
            "end_frame",
            "num_crops",
            "is_occluded",
            "occluded_crop_ratio",
            "is_misclustered",
            "is_noise",
            "crop_path",
            "example_crop_path",
        ]
        if col in df.columns
    ]

    df["cluster_label_str"] = df["cluster_label"].astype(str)
    df["truth_id_str"] = df[identity_col].astype(str)
    df["miscluster_status"] = np.where(
        df["is_noise"],
        "noise",
        np.where(df["is_misclustered"], "misclustered", "correct"),
    )

    symbol_col = "is_occluded" if "is_occluded" in df.columns else None

    fig_cluster = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster_label_str",
        symbol=symbol_col,
        hover_data=hover_cols,
        title=f"{level} embeddings colored by discovered cluster",
    )
    fig_cluster.write_html(out_dir / "interactive_by_cluster.html")

    fig_truth = px.scatter(
        df,
        x="x",
        y="y",
        color="truth_id_str",
        symbol=symbol_col,
        hover_data=hover_cols,
        title=f"{level} embeddings colored by real identity",
    )
    fig_truth.write_html(out_dir / "interactive_by_real_identity.html")

    fig_error = px.scatter(
        df,
        x="x",
        y="y",
        color="miscluster_status",
        symbol=symbol_col,
        hover_data=hover_cols,
        title=f"{level} embeddings: correct vs misclustered vs noise",
    )
    fig_error.write_html(out_dir / "interactive_miscluster_diagnosis.html")

def save_interactive_3d_plots(
    plot_df: pd.DataFrame,
    out_dir: Path,
    level: str,
    identity_col: str,
):
    if not {"x3d", "y3d", "z3d"}.issubset(plot_df.columns):
        return

    try:
        import plotly.express as px
    except ImportError:
        print("[WARN] plotly is not installed; skipping interactive 3D plots.")
        return

    df = plot_df.copy()

    hover_cols = [
        col
        for col in [
            identity_col,
            "global_id",
            "identity_key",
            "cluster_label",
            "cluster_as_identity",
            "cluster_purity",
            "camera",
            "object_type",
            "scene",
            "frame",
            "is_occluded",
            "is_misclustered",
            "is_noise",
            "crop_path",
            "example_crop_path",
        ]
        if col in df.columns
    ]

    df["cluster_label_str"] = df["cluster_label"].astype(str)
    df["truth_id_str"] = df[identity_col].astype(str)
    df["miscluster_status"] = np.where(
        df["is_noise"],
        "noise",
        np.where(df["is_misclustered"], "misclustered", "correct"),
    )

    fig_cluster = px.scatter_3d(
        df,
        x="x3d",
        y="y3d",
        z="z3d",
        color="cluster_label_str",
        hover_data=hover_cols,
        title=f"{level} embeddings 3D colored by discovered cluster",
    )
    fig_cluster.write_html(out_dir / "interactive_3d_by_cluster.html")

    fig_truth = px.scatter_3d(
        df,
        x="x3d",
        y="y3d",
        z="z3d",
        color="truth_id_str",
        hover_data=hover_cols,
        title=f"{level} embeddings 3D colored by real identity",
    )
    fig_truth.write_html(out_dir / "interactive_3d_by_real_identity.html")

    fig_error = px.scatter_3d(
        df,
        x="x3d",
        y="y3d",
        z="z3d",
        color="miscluster_status",
        hover_data=hover_cols,
        title=f"{level} embeddings 3D: correct vs misclustered vs noise",
    )
    fig_error.write_html(out_dir / "interactive_3d_miscluster_diagnosis.html")