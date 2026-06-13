# src/evaluation/rerank.py

import torch

from reid.evaluation.distances import cosine_distance_matrix


def k_reciprocal_rerank(
    q_embs: torch.Tensor,
    g_embs: torch.Tensor,
    k1: int = 20,
    k2: int = 6,
    lam: float = 0.3,
) -> torch.Tensor:
    """
    k-reciprocal re-ranking.

    Returns:
        Re-ranked distance matrix [Nq, Ng]
    """
    n_query = q_embs.shape[0]
    n_total = n_query + g_embs.shape[0]

    all_embs = torch.cat([q_embs, g_embs], dim=0)

    original_dist = cosine_distance_matrix(all_embs, all_embs)
    sorted_indices = torch.argsort(original_dist, dim=1)

    v = torch.zeros(n_total, n_total)

    for i in range(n_total):
        forward_neighbors = sorted_indices[i, 1 : k1 + 1].tolist()

        reciprocal = [
            j for j in forward_neighbors if i in sorted_indices[j, 1 : k1 + 1].tolist()
        ]

        expanded = list(reciprocal)

        for neighbor in reciprocal:
            neighbor_k2 = sorted_indices[
                neighbor,
                1 : k2 // 2 + 1,
            ].tolist()

            for second_neighbor in neighbor_k2:
                if (
                    neighbor
                    in sorted_indices[
                        second_neighbor,
                        1 : k2 // 2 + 1,
                    ].tolist()
                ):
                    expanded.append(second_neighbor)

        expanded = list(set(expanded))

        if expanded:
            distances = original_dist[i, expanded]
            weights = torch.exp(-distances)
            weights = weights / weights.sum()

            for gallery_idx, weight in zip(expanded, weights.tolist()):
                v[i, gallery_idx] = weight

    jaccard_sim = v[:n_query] @ v[n_query:].T

    denominator = (
        v[:n_query].sum(dim=1, keepdim=True)
        + v[n_query:].sum(dim=1, keepdim=True).T
        + 1e-6
    )

    jaccard_dist = (1.0 - 2.0 * jaccard_sim / denominator).clamp(min=0.0)

    original_qg = original_dist[:n_query, n_query:]

    return (1.0 - lam) * jaccard_dist + lam * original_qg
