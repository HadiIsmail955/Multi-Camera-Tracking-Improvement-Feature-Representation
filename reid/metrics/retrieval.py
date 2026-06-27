from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize


def compute_cross_camera_retrieval_metrics(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    identity_col: str,
    ranks: Sequence[int] = (1, 5, 10, 20),
    max_rank_curve: int = 50,
    metric_device: str = "auto",
    chunk_size: int = 1024,
):
    E = torch.from_numpy(normalize(embeddings.astype(np.float32)))

    if metric_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = metric_device

    E = E.to(device)

    ids = df[identity_col].astype(str).values
    cameras = df["camera"].astype(str).values

    n = E.size(0)
    max_rank_curve = min(max_rank_curve, max(n - 1, 1))

    rank_hits = {rank: [] for rank in ranks}
    cmc_hits = {rank: [] for rank in range(1, max_rank_curve + 1)}
    aps = []
    valid_queries = 0

    query_records = []

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sims = torch.matmul(E[start:end], E.T).detach().cpu().numpy()

        for local_i, i in enumerate(range(start, end)):
            candidate_mask = cameras != cameras[i]
            candidate_mask[i] = False

            candidate_indices = np.where(candidate_mask)[0]

            if candidate_indices.size == 0:
                continue

            scores = sims[local_i, candidate_indices]
            matches = (ids[candidate_indices] == ids[i]).astype(np.int32)

            if matches.sum() == 0:
                continue

            valid_queries += 1

            order = np.argsort(-scores)
            sorted_matches = matches[order]
            sorted_scores = scores[order]
            sorted_indices = candidate_indices[order]

            for rank in ranks:
                rank_hits[rank].append(float(sorted_matches[:rank].sum() > 0))

            for rank in range(1, max_rank_curve + 1):
                cmc_hits[rank].append(float(sorted_matches[:rank].sum() > 0))

            ap = average_precision_score(sorted_matches, sorted_scores)
            aps.append(float(ap))

            top_idx = int(sorted_indices[0])
            query_records.append(
                {
                    "query_index": int(i),
                    "query_id": ids[i],
                    "query_camera": cameras[i],
                    "top1_index": top_idx,
                    "top1_id": ids[top_idx],
                    "top1_camera": cameras[top_idx],
                    "top1_score": float(sorted_scores[0]),
                    "top1_correct": int(sorted_matches[0]),
                    "ap": float(ap),
                }
            )

    if valid_queries == 0:
        metrics = {f"Rank{rank}": float("nan") for rank in ranks}
        metrics["mAP"] = float("nan")
        metrics["valid_queries"] = 0
        rank_curve = pd.DataFrame({"rank": [], "accuracy": []})
        query_df = pd.DataFrame(query_records)
        return metrics, rank_curve, query_df

    metrics = {f"Rank{rank}": float(np.mean(rank_hits[rank])) for rank in ranks}
    metrics["mAP"] = float(np.mean(aps))
    metrics["valid_queries"] = int(valid_queries)

    rank_curve = pd.DataFrame(
        {
            "rank": list(range(1, max_rank_curve + 1)),
            "accuracy": [
                float(np.mean(cmc_hits[r])) for r in range(1, max_rank_curve + 1)
            ],
        }
    )

    query_df = pd.DataFrame(query_records)

    return metrics, rank_curve, query_df


def compute_grouped_retrieval_metrics(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    identity_col: str,
    group_col: str,
    ranks: Sequence[int] = (1, 5),
    min_group_size: int = 2,
) -> pd.DataFrame:
    rows = []

    if group_col not in df.columns:
        return pd.DataFrame(rows)

    for group_value, group_df in df.groupby(group_col):
        if len(group_df) < min_group_size:
            continue

        idx = group_df.index.values
        group_embeddings = embeddings[idx]
        reset_df = group_df.reset_index(drop=True).copy()

        metrics, _, _ = compute_cross_camera_retrieval_metrics(
            embeddings=group_embeddings,
            df=reset_df,
            identity_col=identity_col,
            ranks=ranks,
            max_rank_curve=max(ranks),
            metric_device="cpu",
            chunk_size=512,
        )

        row = {
            "group_col": group_col,
            "group_value": str(group_value),
            "num_samples": int(len(group_df)),
        }
        row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows)
