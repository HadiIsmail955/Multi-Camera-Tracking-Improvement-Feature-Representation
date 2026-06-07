from typing import Dict, Iterable
import torch
import torch.nn.functional as F

def compute_embedding_similarity_metrics(
    embeddings: torch.Tensor,
    global_ids,
    cameras,
    max_pairs: int = 200000,
):
    embeddings = F.normalize(embeddings.float(), p=2, dim=1)
    n = embeddings.size(0)

    global_ids = [str(x) for x in global_ids]
    cameras = [str(x) for x in cameras]

    same_id = []
    diff_id = []
    same_id_cross_camera = []
    same_camera_diff_id = []

    total_pairs = n * (n - 1) // 2

    if total_pairs > max_pairs:
        pairs = torch.randint(0, n, (max_pairs, 2))
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        iterable = pairs.tolist()
    else:
        iterable = ((i, j) for i in range(n) for j in range(i + 1, n))

    for i, j in iterable:
        value = float(torch.dot(embeddings[i], embeddings[j]).item())

        is_same_id = global_ids[i] == global_ids[j]
        is_same_camera = cameras[i] == cameras[j]

        if is_same_id:
            same_id.append(value)
            if not is_same_camera:
                same_id_cross_camera.append(value)
        else:
            diff_id.append(value)
            if is_same_camera:
                same_camera_diff_id.append(value)

    def stats(values, prefix):
        if len(values) == 0:
            return {
                f"{prefix}_mean": 0.0,
                f"{prefix}_std": 0.0,
                f"{prefix}_p05": 0.0,
                f"{prefix}_p50": 0.0,
                f"{prefix}_p95": 0.0,
            }

        x = torch.tensor(values, dtype=torch.float32)
        return {
            f"{prefix}_mean": float(x.mean().item()),
            f"{prefix}_std": float(x.std(unbiased=False).item()),
            f"{prefix}_p05": float(torch.quantile(x, 0.05).item()),
            f"{prefix}_p50": float(torch.quantile(x, 0.50).item()),
            f"{prefix}_p95": float(torch.quantile(x, 0.95).item()),
        }

    metrics = {}
    metrics.update(stats(same_id, "same_id_cos"))
    metrics.update(stats(diff_id, "diff_id_cos"))
    metrics.update(stats(same_id_cross_camera, "same_id_cross_camera_cos"))
    metrics.update(stats(same_camera_diff_id, "same_camera_diff_id_cos"))

    metrics["embedding_separation_gap"] = (
        metrics["same_id_cos_mean"] - metrics["diff_id_cos_mean"]
    )

    metrics["cross_camera_gap"] = (
        metrics["same_id_cross_camera_cos_mean"] - metrics["diff_id_cos_mean"]
    )

    metrics["camera_confusion_gap"] = (
        metrics["same_id_cross_camera_cos_mean"]
        - metrics["same_camera_diff_id_cos_mean"]
    )

    return metrics

def compute_embedding_retrieval_metrics(
    embeddings: torch.Tensor,
    global_ids,
    cameras,
    ranks=(1, 5, 10),
):
    embeddings = F.normalize(embeddings.float(), p=2, dim=1)
    sim = torch.matmul(embeddings, embeddings.T)

    global_ids = [str(x) for x in global_ids]
    cameras = [str(x) for x in cameras]

    rank_hits = {rank: [] for rank in ranks}
    average_precisions = []
    valid_queries = 0

    n = embeddings.size(0)

    for i in range(n):
        candidate_indices = [
            j for j in range(n)
            if j != i and cameras[j] != cameras[i]
        ]

        if len(candidate_indices) == 0:
            continue

        scores = sim[i, candidate_indices]
        matches = torch.tensor(
            [global_ids[j] == global_ids[i] for j in candidate_indices],
            dtype=torch.float32,
        )

        if matches.sum().item() == 0:
            continue

        valid_queries += 1

        order = torch.argsort(scores, descending=True)
        sorted_matches = matches[order]

        for rank in ranks:
            topk = sorted_matches[:rank]
            rank_hits[rank].append(float(topk.sum().item() > 0))

        cumulative_matches = sorted_matches.cumsum(dim=0)
        precision_at_k = cumulative_matches / torch.arange(
            1,
            sorted_matches.numel() + 1,
            dtype=torch.float32,
        )

        ap = (precision_at_k * sorted_matches).sum() / sorted_matches.sum().clamp(min=1.0)
        average_precisions.append(float(ap.item()))

    if valid_queries == 0:
        metrics = {f"Rank{rank}": 0.0 for rank in ranks}
        metrics["mAP"] = 0.0
        metrics["valid_queries"] = 0
        return metrics

    metrics = {
        f"Rank{rank}": float(sum(rank_hits[rank]) / len(rank_hits[rank]))
        for rank in ranks
    }
    metrics["mAP"] = float(sum(average_precisions) / len(average_precisions))
    metrics["valid_queries"] = int(valid_queries)

    return metrics