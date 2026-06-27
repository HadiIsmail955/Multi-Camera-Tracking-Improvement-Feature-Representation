import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from .helper import safe_int


def aggregate_group_features(E: np.ndarray, method: str = "mean_topk") -> np.ndarray:
    E = normalize(E.astype(np.float32))

    if method == "mean":
        v = E.mean(axis=0)
        return v / max(np.linalg.norm(v), 1e-12)

    if method == "medoid":
        mean = E.mean(axis=0)
        mean = mean / max(np.linalg.norm(mean), 1e-12)
        scores = E @ mean
        return E[int(np.argmax(scores))]

    if method in {"mean_topk", "quality_mean_topk"}:
        mean = E.mean(axis=0)
        mean = mean / max(np.linalg.norm(mean), 1e-12)
        scores = E @ mean
        k = min(20, len(E))
        idx = np.argsort(-scores)[:k]
        v = E[idx].mean(axis=0)
        return v / max(np.linalg.norm(v), 1e-12)

    raise ValueError(f"Unknown aggregation method: {method}")


def aggregate_to_tracklets(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    group_mode: str = "auto",
    aggregation: str = "mean_topk",
):
    emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)
    full_df = pd.concat([df.reset_index(drop=True), emb_df], axis=1)

    has_track_id = (
        "track_id" in full_df.columns
        and full_df["track_id"].notna().any()
        and not (full_df["track_id"].astype(str).str.lower().isin(["none", "nan", ""])).all()
    )

    if group_mode == "auto":
        group_mode = "track_id" if has_track_id else "global_id_camera"

    if group_mode == "track_id":
        if not has_track_id:
            raise ValueError("group_mode='track_id' requested, but no track_id/tracklet_id exists.")
        group_cols = ["scene", "camera", "track_id", "object_type"]

    elif group_mode == "global_id_camera":
        # Analysis mode. Uses GT identity to simulate one tracklet per object per camera.
        group_cols = ["scene", "identity_key", "camera", "object_type"]

    else:
        raise ValueError(f"Unknown group_mode: {group_mode}")

    rows = []
    out_embeddings = []

    for keys, group in full_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        key_dict = dict(zip(group_cols, keys))
        E = group[emb_cols].values.astype(np.float32)
        group_emb = aggregate_group_features(E, method=aggregation)

        row = {
            "num_crops": int(len(group)),
            "global_id": str(group["global_id"].iloc[0]),
            "identity_key": str(group["identity_key"].iloc[0]),
            "label": safe_int(group["label"].iloc[0]),
            "true_label": safe_int(group["true_label"].iloc[0]),
            "camera": str(group["camera"].iloc[0]),
            "camera_id": safe_int(group["camera_id"].iloc[0]),
            "scene": str(group["scene"].iloc[0]),
            "object_type": str(group["object_type"].iloc[0]),
            "is_occluded": int(group["is_occluded"].astype(int).max()) if "is_occluded" in group else 0,
            "occluded_crop_ratio": float(group["is_occluded"].astype(int).mean()) if "is_occluded" in group else 0.0,
            "start_frame": safe_int(group["frame"].min()),
            "end_frame": safe_int(group["frame"].max()),
            "example_crop_path": str(group["crop_path"].iloc[0]) if "crop_path" in group else "",
        }

        if "track_id" in group.columns:
            row["track_id"] = str(group["track_id"].iloc[0])

        row.update({k: str(v) for k, v in key_dict.items()})

        rows.append(row)
        out_embeddings.append(group_emb)

    out_embeddings = np.stack(out_embeddings, axis=0).astype(np.float32)
    out_embeddings = normalize(out_embeddings)

    out_df = pd.DataFrame(rows)
    true_ids = sorted(out_df["identity_key"].astype(str).unique())
    id_to_label = {gid: idx for idx, gid in enumerate(true_ids)}
    out_df["true_label"] = out_df["identity_key"].astype(str).map(id_to_label).astype(int)

    return out_embeddings, out_df, group_mode
