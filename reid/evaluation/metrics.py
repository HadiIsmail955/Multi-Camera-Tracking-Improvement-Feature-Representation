# src/evaluation/metrics.py


def compute_rank1_map(
    ranked_indices: list[list[int]],
    q_pids: list[int],
    q_camids: list[int],
    g_pids: list[int],
    g_camids: list[int],
) -> tuple[float, float, float, float]:
    """
    Compute Rank-1 / Rank-5 / Rank-10 and mAP for person ReID.

    ranked_indices must contain gallery indices sorted best-to-worst.
    Same-camera same-identity junk should already be removed from ranking.
    """
    rank1_hits = 0
    rank5_hits = 0
    rank10_hits = 0
    ap_list = []

    for i, ranked in enumerate(ranked_indices):
        q_pid = q_pids[i]
        q_camid = q_camids[i]

        if ranked and g_pids[ranked[0]] == q_pid:
            rank1_hits += 1

        if any(g_pids[idx] == q_pid for idx in ranked[:5]):
            rank5_hits += 1

        if any(g_pids[idx] == q_pid for idx in ranked[:10]):
            rank10_hits += 1

        num_gt = sum(
            1
            for g_pid, g_camid in zip(g_pids, g_camids)
            if g_pid == q_pid and not (g_pid == q_pid and g_camid == q_camid)
        )

        if num_gt == 0:
            continue

        hits = 0
        precision_sum = 0.0

        for rank, gallery_idx in enumerate(ranked):
            if g_pids[gallery_idx] == q_pid:
                hits += 1
                precision_sum += hits / (rank + 1)

        ap_list.append(precision_sum / num_gt)

    rank1 = rank1_hits / len(q_pids) if q_pids else 0.0
    rank5 = rank5_hits / len(q_pids) if q_pids else 0.0
    rank10 = rank10_hits / len(q_pids) if q_pids else 0.0
    mAP = sum(ap_list) / len(ap_list) if ap_list else 0.0

    return rank1, rank5, rank10, mAP
