# src/evaluation/evaluate.py

import torch

from reid.evaluation.distances import cosine_similarity_matrix
from reid.evaluation.ranking import rank_gallery_indices
from reid.evaluation.metrics import compute_rank1_map


def evaluate_reid(
    q_embs: torch.Tensor,
    q_pids: list[int],
    q_camids: list[int],
    g_embs: torch.Tensor,
    g_pids: list[int],
    g_camids: list[int],
) -> tuple[float, float]:
    """
    Standard ReID evaluation using cosine similarity.

    Higher similarity is better, so this converts similarity to distance
    before ranking.
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

    return compute_rank1_map(
        ranked_indices=ranked_indices,
        q_pids=q_pids,
        q_camids=q_camids,
        g_pids=g_pids,
        g_camids=g_camids,
    )
