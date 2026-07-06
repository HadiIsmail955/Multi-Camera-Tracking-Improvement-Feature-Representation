import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import umap

"""Synchronized Unified Dual-Manifold Visualization Pipeline for AICity ReID.

This script loads pre-pooled query and gallery tracklet embeddings, reduces
their dimensionality using UMAP and t-SNE, and generates global distribution
and targeted identity cluster plots.
"""

# Global Constants
POOLED_DATA_PATH = "/mnt/nfs/home/st201745/checkpoints/pooled_tracklet_embeddings.pt"
PRESENTATION_PIDS = [3, 12, 22, 36, 41, 908, 923, 980, 993]


def load_pooled_data_and_project(
    pooled_checkpoint_path: str, method: str = "umap"
) -> pd.DataFrame:
  """Loads pre-pooled tracklet data and projects it to a 2D space.

  Args:
      pooled_checkpoint_path: Path to the serialized PyTorch tensor dictionary.
      method: Embedding reduction algorithm to use ('umap' or 'tsne').

  Returns:
      A pandas DataFrame containing 'X', 'Y' coordinates, 'Split' labels,
      'PID' (Person IDs), and the 'Algo' token.

  Raises:
      ValueError: If an unsupported projection method is requested.
  """
  print(f"Loading post-pooling tracklet data from: {pooled_checkpoint_path}")
  data = torch.load(pooled_checkpoint_path, map_location="cpu")

  # Convert tensors/arrays safely to numpy format
  q_embeddings = (
      data["q_embs"].numpy()
      if isinstance(data["q_embs"], torch.Tensor)
      else np.array(data["q_embs"])
  )
  g_embeddings = (
      data["g_embs"].numpy()
      if isinstance(data["g_embs"], torch.Tensor)
      else np.array(data["g_embs"])
  )

  q_pids = (
      data["q_pids"].numpy()
      if isinstance(data["q_pids"], torch.Tensor)
      else np.asarray(data["q_pids"])
  )
  g_pids = (
      data["g_pids"].numpy()
      if isinstance(data["g_pids"], torch.Tensor)
      else np.asarray(data["g_pids"])
  )

  all_embeddings = np.concatenate([q_embeddings, g_embeddings], axis=0)
  split_labels = ["Query"] * len(q_embeddings) + ["Gallery"] * len(g_embeddings)
  all_pids = np.concatenate([q_pids, g_pids])

  method_lower = method.lower()
  if method_lower == "umap":
    print(f"[UMAP] Projecting {len(all_embeddings)} tracklets")
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",  # Alignment with ReID cosine distance rules
        random_state=42,
        n_jobs=-1,
    )
    embeddings_2d = reducer.fit_transform(all_embeddings)
    algo_title = "UMAP"

  elif method_lower == "tsne":
    print(
        "[t-SNE] Applying PCA pre-reduction (1536-D -> 50-D) to speed up"
        " iterations..."
    )
    pca = PCA(n_components=50, random_state=42)
    feats_reduced = pca.fit_transform(all_embeddings)

    print(
        "[t-SNE] Running Barnes-Hut reduction on 50-D principal components..."
    )
    perplexity = min(50, max(5, len(all_embeddings) // 200))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        init="pca",
        metric="cosine",
        method="barnes_hut",
        max_iter=1000,
        n_jobs=-1,
    )
    embeddings_2d = tsne.fit_transform(feats_reduced)
    algo_title = "t-SNE"
  else:
    raise ValueError(
        f"Invalid projection mode '{method}'. Choose 'umap' or 'tsne'."
    )

  plot_df = pd.DataFrame({
      "X": embeddings_2d[:, 0],
      "Y": embeddings_2d[:, 1],
      "Split": split_labels,
      "PID": all_pids,
      "Algo": [algo_title] * len(all_embeddings),
  })

  return plot_df


def plot_global_distribution(df: pd.DataFrame, algo_name: str) -> None:
  """Plots all pooled tracklets to view macro structural density distributions.

  Args:
      df: Source coordinates DataFrame containing X, Y and Split mappings.
      algo_name: Normalized reduction algorithm identifier string.
  """
  plt.figure(figsize=(10, 7), dpi=300)
  df_gallery = df[df["Split"] == "Gallery"]
  df_query = df[df["Split"] == "Query"]

  plt.scatter(
      df_gallery["X"],
      df_gallery["Y"],
      c="#7f7f7f",
      s=12,
      alpha=0.4,
      edgecolors="none",
      label="Gallery",
  )

  # Non-occluding structured query overlay map
  plt.scatter(
      df_query["X"],
      df_query["Y"],
      c="#ff4d4d",
      s=25,
      alpha=0.95,
      edgecolors="#000000",
      linewidths=0.5,
      label="Query",
  )

  plt.title(
      f"AICity 2025: Global Tracklet Manifold ({algo_name})",
      fontsize=14,
      weight="bold",
  )
  plt.xlabel(f"{algo_name} Dimension 1")
  plt.ylabel(f"{algo_name} Dimension 2")
  plt.legend(
      loc="upper right", frameon=True, facecolor='white', edgecolor="none"
  )
  plt.grid(True, linestyle="--", alpha=0.3)
  plt.tight_layout()

  output_path = f"global_space_{algo_name.lower()}.png"
  plt.savefig(output_path)
  print(f"Saved Global Distribution plot to: {os.path.abspath(output_path)}")
  plt.close()


def plot_identity_clusters(
    df: pd.DataFrame, algo_name: str, selected_pids: List[int]
) -> None:
  """Isolates specific target PIDs to display high-contrast micro clusters.

  Args:
      df: Base coordinates DataFrame containing X, Y, Split, and PID keys.
      algo_name: Reduction projection algorithm label string.
      selected_pids: Numeric target identifiers picked for tracking isolation.
  """
  filtered_df = df[df["PID"].isin(selected_pids)].copy()
  filtered_df = filtered_df.sort_values(by="PID")

  df_gal = filtered_df[filtered_df["Split"] == "Gallery"]
  df_q = filtered_df[filtered_df["Split"] == "Query"]

  plt.figure(figsize=(10, 7), dpi=300)

  # Layer 1: Plot background gallery target bounds
  if not df_gal.empty:
    sns.scatterplot(
        data=df_gal,
        x="X",
        y="Y",
        hue="PID",
        marker="o",
        s=100,
        palette="tab10",
        edgecolors="black",
        linewidth=0.5,
        alpha=0.7,
        legend="brief",
    )

  # Layer 2: Overlay thick probe query outlines
  if not df_q.empty:
    sns.scatterplot(
        data=df_q,
        x="X",
        y="Y",
        hue="PID",
        marker="X",
        s=160,
        palette="tab10",
        edgecolors="#000000",
        linewidth=2.0,
        alpha=1.0,
        legend=False,
    )

  plt.title(
      f"Target Cluster Tracking ({algo_name}: Synchronized Validation Slices)",
      fontsize=13,
      weight="bold",
  )
  plt.xlabel(f"{algo_name} Dimension 1")
  plt.ylabel(f"{algo_name} Dimension 2")
  plt.grid(True, linestyle="--", alpha=0.3)
  plt.tight_layout()

  output_path = f"identity_clusters_{algo_name.lower()}.png"
  plt.savefig(output_path)
  print(f"Saved identity cluster plot to: {os.path.abspath(output_path)}")
  plt.close()


def main() -> None:
  """Main execution function block."""
  print("Starting Synchronized Unified Dual-Manifold Pipeline")

  for current_algo in ["umap", "tsne"]:
    print(f"\n🎬 Processing Engine: {current_algo.upper()}")
    print("-" * 50)
    try:
      df_coords = load_pooled_data_and_project(
          POOLED_DATA_PATH, method=current_algo
      )

      # Macro global projection rendering
      plot_global_distribution(df_coords, algo_name=df_coords["Algo"].iloc[0])

      # Micro targeted projection tracking slice
      plot_identity_clusters(
          df_coords,
          algo_name=df_coords["Algo"].iloc[0],
          selected_pids=PRESENTATION_PIDS,
      )

    except (FileNotFoundError, ValueError, RuntimeError) as e:
      print(f"Processing failed for engine {current_algo.upper()}: {e}")

  print("\n Aligned visual plots updated and compiled cleanly!")


if __name__ == "__main__":
  main()