import torch


def is_junk_match(
    q_pid: int,
    q_camid: int,
    g_pid: int,
    g_camid: int,
) -> bool:
    return g_pid == q_pid and g_camid == q_camid


def rank_gallery_indices(
    dist_matrix: torch.Tensor,
    q_pids: list[int],
    q_camids: list[int],
    g_pids: list[int],
    g_camids: list[int],
    remove_junk: bool = True,
) -> list[list[int]]:
    """
    Rank gallery indices for each query by ascending distance.

    Returns:
        ranked_indices[i] = list of gallery indices sorted best-to-worst
    """
    ranked_results = []

    for i in range(len(q_pids)):
        order = torch.argsort(dist_matrix[i]).tolist()

        if remove_junk:
            order = [
                j
                for j in order
                if not is_junk_match(
                    q_pid=q_pids[i],
                    q_camid=q_camids[i],
                    g_pid=g_pids[j],
                    g_camid=g_camids[j],
                )
            ]

        ranked_results.append(order)

    return ranked_results


def ranked_indices_to_pids(
    ranked_indices: list[list[int]],
    g_pids: list[int],
) -> list[list[int]]:
    return [[g_pids[j] for j in ranked] for ranked in ranked_indices]
