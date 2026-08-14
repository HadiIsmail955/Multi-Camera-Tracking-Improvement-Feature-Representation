# src/evaluation/evaluate.py

import torch

from reid.evaluation.distances import cosine_similarity_matrix
from reid.evaluation.ranking import rank_gallery_indices
from reid.evaluation.metrics import compute_rank1_map
from reid.evaluation.analysis import compute_embedding_metrics


def evaluate_reid(
    q_embs: torch.Tensor,
    q_pids: list[int],
    q_camids: list[int],
    g_embs: torch.Tensor,
    g_pids: list[int],
    g_camids: list[int],
    return_analysis: bool = True,
    noise_info: dict | None = None,
):
    """
    Standard ReID evaluation using cosine similarity.

    Higher similarity is better, so this converts similarity to distance
    before ranking.

    If return_analysis=True, additionally computes and returns embedding quality analysis metrics.
    """
    sim_matrix = cosine_similarity_matrix(q_embs, g_embs)
    dist_matrix = 1.0 - sim_matrix

    ranked_indices = rank_gallery_indices(
        dist_matrix=dist_matrix,
        q_pids=q_pids,
        q_camids=q_camids,
        g_pids=g_pids,
        g_camids=g_camids,
        remove_junk=True,
    )

    rank1, rank5, rank10, mAP = compute_rank1_map(
        ranked_indices=ranked_indices,
        q_pids=q_pids,
        q_camids=q_camids,
        g_pids=g_pids,
        g_camids=g_camids,
    )

    if return_analysis:
        analysis_metrics = compute_embedding_metrics(
            sim_matrix=sim_matrix,
            q_pids=q_pids,
            q_camids=q_camids,
            g_pids=g_pids,
            g_camids=g_camids,
            noise_info=noise_info,
        )
        return rank1, rank5, rank10, mAP, analysis_metrics

    return rank1, rank5, rank10, mAP

