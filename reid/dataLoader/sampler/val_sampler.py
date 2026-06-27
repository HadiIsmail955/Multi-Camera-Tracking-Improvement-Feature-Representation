import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


def subsample_eval_set(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    identity_col: str = "identity_key",
    max_samples: int = 0,
    seed: int = 42,
    mode: str = "identity_balanced",
):
    embeddings = normalize(embeddings.astype("float32"))
    df = df.reset_index(drop=True).copy()

    n = len(df)

    if max_samples is None or max_samples <= 0 or n <= max_samples:
        df["eval_original_index"] = np.arange(n)
        return embeddings, df

    rng = np.random.default_rng(seed)

    if mode == "random":
        idx = rng.choice(n, size=max_samples, replace=False)
        idx = np.sort(idx)
        out_df = df.iloc[idx].reset_index(drop=True).copy()
        out_df["eval_original_index"] = idx
        return embeddings[idx], out_df

    if mode != "identity_balanced":
        raise ValueError(f"Unknown sampling mode: {mode}")

    ids = df[identity_col].astype(str).values
    unique_ids = np.array(sorted(set(ids)))

    selected = []

    for identity in unique_ids:
        pool = np.where(ids == identity)[0]
        if len(pool) == 0:
            continue

        selected.append(int(rng.choice(pool)))

        if len(selected) >= max_samples:
            break

    selected = list(dict.fromkeys(selected))

    remaining_budget = max_samples - len(selected)

    if remaining_budget > 0:
        selected_set = set(selected)
        remaining_indices = np.array([i for i in range(n) if i not in selected_set])

        if len(remaining_indices) > 0:
            fill = rng.choice(
                remaining_indices,
                size=min(remaining_budget, len(remaining_indices)),
                replace=False,
            )
            selected.extend([int(x) for x in fill])

    idx = np.array(sorted(selected[:max_samples]), dtype=np.int64)

    out_df = df.iloc[idx].reset_index(drop=True).copy()
    out_df["eval_original_index"] = idx

    return embeddings[idx], out_df