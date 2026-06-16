import torch


def cosine_similarity_matrix(
    q_embs: torch.Tensor,
    g_embs: torch.Tensor,
) -> torch.Tensor:
    """
    Pairwise cosine similarity.

    Assumes both inputs are L2-normalized.
    Returns:
        Tensor [Nq, Ng]
    """
    return q_embs @ g_embs.T


def cosine_distance_matrix(
    q_embs: torch.Tensor,
    g_embs: torch.Tensor,
) -> torch.Tensor:
    """
    Pairwise cosine distance: d = 1 - cosine_similarity.

    Assumes both inputs are L2-normalized.
    Returns:
        Tensor [Nq, Ng]
    """
    return (1.0 - cosine_similarity_matrix(q_embs, g_embs)).clamp(min=0.0)
