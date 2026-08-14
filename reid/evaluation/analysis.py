# reid/evaluation/analysis.py

"""
Embedding Quality and Clustering Noise Analysis Module.

1. Same-ID Cosine Similarity:
   Cosine similarity between valid positive pairs (same Person ID, different camera).
2. Different-ID Cosine Similarity:
   Cosine similarity between valid negative pairs (different Person ID).
3. Separation Gap:
   Difference between Mean(Same-ID) and Mean(Different-ID).
4. Pair ROC-AUC:
   Measures how accurately positive pairs are ranked above negative pairs.
5. DBSCAN / HDBSCAN Noise Statistics:
   Summary of noise frame embeddings.
"""

from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve


def compute_embedding_metrics(
    sim_matrix: torch.Tensor,
    q_pids: List[int],
    q_camids: List[int],
    g_pids: List[int],
    g_camids: List[int],
    noise_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Args:
        sim_matrix: Tensor of shape [N_q, N_g] containing pairwise cosine similarities.
        q_pids: Person IDs for query tracklets.
        q_camids: Camera IDs for query tracklets.
        g_pids: Person IDs for gallery tracklets.
        g_camids: Camera IDs for gallery tracklets.
        noise_info: Pre-collected noise stats from DBSCAN/HDBSCAN tracklet pooling

    """
    device = sim_matrix.device

    q_pids_t = torch.tensor(q_pids, device=device)
    g_pids_t = torch.tensor(g_pids, device=device)
    q_camids_t = torch.tensor(q_camids, device=device)
    g_camids_t = torch.tensor(g_camids, device=device)

    # Positive pair: Same Person ID AND Different Camera (Same-ID same-camera is ignored as junk)
    same_id_mask = q_pids_t[:, None] == g_pids_t[None, :]
    diff_cam_mask = q_camids_t[:, None] != g_camids_t[None, :]
    pos_mask = same_id_mask & diff_cam_mask

    # Negative pair: Different Person ID
    neg_mask = q_pids_t[:, None] != g_pids_t[None, :]

    pos_sims = sim_matrix[pos_mask]
    neg_sims = sim_matrix[neg_mask]

    # Same-ID Cosine Similarity
    if pos_sims.numel() > 0:
        pos_mean = float(pos_sims.mean().item())
        pos_std = float(pos_sims.std(unbiased=False).item())
        pos_count = int(pos_sims.numel())
    else:
        pos_mean = 0.0
        pos_std = 0.0
        pos_count = 0

    # Different-ID Cosine Similarity
    if neg_sims.numel() > 0:
        neg_mean = float(neg_sims.mean().item())
        neg_std = float(neg_sims.std(unbiased=False).item())
        neg_count = int(neg_sims.numel())
    else:
        neg_mean = 0.0
        neg_std = 0.0
        neg_count = 0

    separation_gap = pos_mean - neg_mean

    # Pair ROC-AUC & ROC Curve
    roc_auc = 0.0
    if pos_count > 0 and neg_count > 0:
        y_true = np.concatenate([
            np.ones(pos_count, dtype=np.int32),
            np.zeros(neg_count, dtype=np.int32),
        ])
        y_score = np.concatenate([
            pos_sims.detach().cpu().numpy(),
            neg_sims.detach().cpu().numpy(),
        ])
        try:
            roc_auc = float(roc_auc_score(y_true, y_score))
        except ValueError:
            roc_auc = 0.0

    metrics = {
        "same_id": {
            "mean": pos_mean,
            "std": pos_std,
            "count": pos_count,
        },
        "diff_id": {
            "mean": neg_mean,
            "std": neg_std,
            "count": neg_count,
        },
        "separation_gap": separation_gap,
        "roc_auc": roc_auc,
        "noise_stats": None,
    }

    # DBSCAN / HDBSCAN Noise
    if noise_info and "noise_rates" in noise_info and len(noise_info["noise_rates"]) > 0:
        noise_samples = np.array(noise_info["noise_samples"], dtype=np.float64)
        noise_rates = np.array(noise_info["noise_rates"], dtype=np.float64)

        metrics["noise_stats"] = {
            "avg_noise_samples": float(np.mean(noise_samples)),
            "avg_noise_rate": float(np.mean(noise_rates)),
            "median_noise_rate": float(np.median(noise_rates)),
            "max_noise_rate": float(np.max(noise_rates)),
        }

    return metrics
