import csv
import torch

from pathlib import Path
from reid.data.records import Record


def format_crop_path(path: str) -> str:
    path_str = str(path).replace("\\", "/")
    if "crops/" in path_str:
        idx = path_str.find("crops/")
        return "./" + path_str[idx:]
    if path_str.startswith(".//"):
        return "./" + path_str[3:]
    return path_str


def export_retrieval_examples(
    output_csv_path: str | Path,
    correct_top5_csv_path: str | Path,
    dist_matrix: torch.Tensor,
    ranked_indices: list[list[int]],
    q_pids: list[int],
    q_camids: list[int],
    g_pids: list[int],
    g_camids: list[int],
    query_records: list[Record],
    gallery_records: list[Record],
    q_tids: list[str] | None = None,
    g_tids: list[str] | None = None,
    top_k: int = 5,
) -> tuple[str, str]:
    """
    Export Top-K retrieval results for every query to CSV files.

    Args:
        dist_matrix: Distance matrix Tensor [num_queries, num_gallery].
        ranked_indices: List of sorted gallery indices for each query (after junk removal).
        q_pids: Query PIDs.
        q_camids: Query camera IDs.
        g_pids: Gallery PIDs.
        g_camids: Gallery camera IDs.
        top_k: Number of ranked matches to export per query (default 5).

    """
    out_path = Path(output_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    correct_path = Path(correct_top5_csv_path)
    correct_path.parent.mkdir(parents=True, exist_ok=True)

    has_tracklet_ids = False
    if q_tids is not None and g_tids is not None:
        has_tracklet_ids = any(q_tids) or any(g_tids)
    elif query_records and gallery_records:
        has_tracklet_ids = any(r.tracklet_id for r in query_records) or any(
            r.tracklet_id for r in gallery_records
        )

    q_tracklet_map: dict[str, list[Record]] = {}
    g_tracklet_map: dict[str, list[Record]] = {}

    if q_tids is not None or g_tids is not None:
        for r in query_records:
            if r.tracklet_id:
                if r.tracklet_id not in q_tracklet_map:
                    q_tracklet_map[r.tracklet_id] = []
                q_tracklet_map[r.tracklet_id].append(r)

        for r in gallery_records:
            if r.tracklet_id:
                if r.tracklet_id not in g_tracklet_map:
                    g_tracklet_map[r.tracklet_id] = []
                g_tracklet_map[r.tracklet_id].append(r)

    header = []
    if has_tracklet_ids:
        header = ["query_pid", "query_tracklet_id", "query_camid", "query_crop_path"]
        for k in range(1, top_k + 1):
            header.extend(
                [
                    f"rank{k}_pid",
                    f"rank{k}_tracklet_id",
                    f"rank{k}_camid",
                    f"rank{k}_crop_path",
                    f"rank{k}_distance",
                ]
            )
    else:
        header = ["query_pid", "query_camid", "query_crop_path"]
        for k in range(1, top_k + 1):
            header.extend(
                [
                    f"rank{k}_pid",
                    f"rank{k}_camid",
                    f"rank{k}_crop_path",
                    f"rank{k}_distance",
                ]
            )

    all_rows = []
    correct_rows = []
    num_queries = len(q_pids)

    for i in range(num_queries):
        row = []
        q_pid = q_pids[i]
        q_camid = q_camids[i]

        if q_tids is not None and i < len(q_tids):
            q_tid = q_tids[i]
            q_recs = q_tracklet_map.get(q_tid, [])
            q_crop = format_crop_path(q_recs[0].filepath) if q_recs else ""
        else:
            q_rec = query_records[i] if i < len(query_records) else None
            q_tid = q_rec.tracklet_id if q_rec else ""
            q_crop = format_crop_path(q_rec.filepath) if q_rec else ""

        if has_tracklet_ids:
            row.extend([q_pid, q_tid, q_camid, q_crop])
        else:
            row.extend([q_pid, q_camid, q_crop])

        ranked = ranked_indices[i] if i < len(ranked_indices) else []
        is_rank1_correct = False

        for k in range(top_k):
            if k < len(ranked):
                g_idx = ranked[k]
                g_pid = g_pids[g_idx]
                g_camid = g_camids[g_idx]
                dist = dist_matrix[i, g_idx].item()

                if k == 0 and g_pid == q_pid:
                    is_rank1_correct = True

                if g_tids is not None and g_idx < len(g_tids):
                    g_tid = g_tids[g_idx]
                    g_recs = g_tracklet_map.get(g_tid, [])
                    g_crop = format_crop_path(g_recs[0].filepath) if g_recs else ""
                else:
                    g_rec = (
                        gallery_records[g_idx] if g_idx < len(gallery_records) else None
                    )
                    g_tid = g_rec.tracklet_id if g_rec else ""
                    g_crop = format_crop_path(g_rec.filepath) if g_rec else ""

                if has_tracklet_ids:
                    row.extend([g_pid, g_tid, g_camid, g_crop, float(dist)])
                else:
                    row.extend([g_pid, g_camid, g_crop, float(dist)])
            else:
                if has_tracklet_ids:
                    row.extend(["", "", "", "", ""])
                else:
                    row.extend(["", "", "", ""])

        all_rows.append(row)
        if is_rank1_correct:
            correct_rows.append(row)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_rows)

    with open(correct_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(correct_rows)

    return str(out_path), str(correct_path)
