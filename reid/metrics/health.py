from typing import Dict

import numpy as np
from sklearn.preprocessing import normalize


def stats_np(values: np.ndarray, prefix: str) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float32)

    if values.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_p01": float("nan"),
            f"{prefix}_p05": float("nan"),
            f"{prefix}_p50": float("nan"),
            f"{prefix}_p95": float("nan"),
            f"{prefix}_p99": float("nan"),
        }

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_p01": float(np.percentile(values, 1)),
        f"{prefix}_p05": float(np.percentile(values, 5)),
        f"{prefix}_p50": float(np.percentile(values, 50)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_p99": float(np.percentile(values, 99)),
    }


def compute_embedding_health_metrics(embeddings: np.ndarray) -> Dict[str, float]:
    E = normalize(embeddings.astype(np.float32))
    n, d = E.shape

    norms = np.linalg.norm(E, axis=1)
    dim_std = E.std(axis=0)

    centered = E - E.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(n - 1, 1)

    eigvals = np.linalg.eigvalsh(cov).clip(min=0)
    eig_sum = eigvals.sum() + 1e-12
    probs = eigvals / eig_sum

    entropy = -(probs * np.log(probs + 1e-12)).sum()
    effective_rank = float(np.exp(entropy))
    participation_ratio = float((eig_sum ** 2) / (np.square(eigvals).sum() + 1e-12))

    metrics = {
        "num_embeddings": int(n),
        "embedding_dim": int(d),
        "dim_std_mean": float(dim_std.mean()),
        "dim_std_min": float(dim_std.min()),
        "dim_std_max": float(dim_std.max()),
        "effective_rank": effective_rank,
        "effective_rank_ratio": float(effective_rank / max(d, 1)),
        "participation_ratio": participation_ratio,
        "participation_ratio_ratio": float(participation_ratio / max(d, 1)),
    }

    metrics.update(stats_np(norms, "norm"))

    return metrics
