"""Preprocessing script for AICITY MTMC Re-ID dataset.

Extracts person crops from annotated videos, builds per-tracklet records,
and writes train/val/query/gallery manifests as CSV files.
"""

import argparse
import csv
import json
import os
import random
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm


IMAGE_EXT = ".jpg"
JPEG_QUALITY = 95
FAST_SEEK_THRESHOLD = 60  # frames: use cap.set() for gaps larger than this


def parse_camera_id(camera_name: str) -> int:
    """Extracts integer camera index from a camera name string.

    Args:
        camera_name: Camera label, e.g. ``"Camera_0001"`` or ``"0001"``.

    Returns:
        Integer camera index, e.g. ``1``.

    Raises:
        ValueError: If no numeric suffix can be extracted.
    """
    if camera_name == "Camera":
        return 0

    # Try splitting by underscore first (e.g., "Camera_0001")
    parts = camera_name.split("_")
    if len(parts) > 1:
        try:
            return int(parts[-1])
        except ValueError:
            pass

    # Try extracting all digits from the string
    digits = re.findall(r"\d+", camera_name)
    if digits:
        return int(digits[-1])

    # Fallback: try direct conversion
    try:
        return int(camera_name)
    except ValueError:
        raise ValueError(
            f"Could not parse camera ID from '{camera_name}'. "
            f"Expected format like 'Camera_0001' or '0001'."
        )


def find_video_path(scene_dir: Path, camera: str) -> Path:
    """Locates a video file for the given scene directory and camera.

    Args:
        scene_dir: Path to the scene directory.
        camera: Camera name, e.g. ``"Camera_0001"``.

    Returns:
        Path to the video file.

    Raises:
        FileNotFoundError: If no video file is found for the given inputs.
    """
    extensions = (".mp4", ".avi", ".mov")
    candidates = [scene_dir / "videos" / f"{camera}{ext}" for ext in extensions] + [
        scene_dir / f"{camera}{ext}" for ext in extensions
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(f"Could not find video for {scene_dir}/{camera}")


def clip_bbox(
    bbox: list[float],
    width: int,
    height: int,
    padding: float = 0.0,
) -> tuple[int, int, int, int]:
    """Clips a bounding box to image boundaries with optional padding.

    Args:
        bbox: Raw bounding box ``[x1, y1, x2, y2]``.
        width: Frame width in pixels.
        height: Frame height in pixels.
        padding: Fractional padding added on each side relative to box size.

    Returns:
        Clipped integer bounding box ``(x1, y1, x2, y2)``.
    """
    x1, y1, x2, y2 = bbox

    if padding > 0:
        bw = x2 - x1
        bh = y2 - y1
        x1 -= bw * padding
        y1 -= bh * padding
        x2 += bw * padding
        y2 += bh * padding

    return (
        max(0, int(round(x1))),
        max(0, int(round(y1))),
        min(width, int(round(x2))),
        min(height, int(round(y2))),
    )


def valid_bbox(
    bbox: tuple[int, int, int, int],
    min_width: int,
    min_height: int,
    min_area: int,
) -> bool:
    """Checks whether a bounding box meets minimum size requirements.

    Args:
        bbox: Integer bounding box ``(x1, y1, x2, y2)``.
        min_width: Minimum acceptable width in pixels.
        min_height: Minimum acceptable height in pixels.
        min_area: Minimum acceptable area in square pixels.

    Returns:
        ``True`` if the bounding box satisfies all size constraints.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    return (
        w > 0 and h > 0 and w >= min_width and h >= min_height and (w * h) >= min_area
    )


def split_into_tracklets(
    frame_ids: list[int],
    max_gap: int,
) -> list[list[int]]:
    """Splits a list of frame IDs into continuous tracklet segments.

    Args:
        frame_ids: List of integer frame indices (may be unsorted/duplicate).
        max_gap: Maximum allowed gap between consecutive frames before
            starting a new tracklet.

    Returns:
        List of tracklet segments, each a sorted list of frame IDs.
    """
    if not frame_ids:
        return []

    frame_ids = sorted(set(frame_ids))
    tracklets: list[list[int]] = []
    current = [frame_ids[0]]

    for prev_frame, curr_frame in zip(frame_ids[:-1], frame_ids[1:]):
        if curr_frame - prev_frame > max_gap:
            tracklets.append(current)
            current = [curr_frame]
        else:
            current.append(curr_frame)

    tracklets.append(current)
    return tracklets


def load_scene_annotations(
    scene_dir: Path,
    split: str,
) -> dict[str, list[dict[str, Any]]]:
    """Loads per-camera annotation rows from a scene ground-truth file.

    Args:
        scene_dir: Path to the scene directory containing
            ``ground_truth.json``.
        split: Dataset split label, e.g. ``"train"`` or ``"val"``.

    Returns:
        Mapping from camera name to a list of annotation dicts, each with
        keys ``split``, ``scene``, ``camera``, ``frame_id``, ``object_id``,
        ``object_type``, ``pid_key``, and ``bbox``.
    """
    scene = scene_dir.name

    with open(scene_dir / "ground_truth.json") as f:
        gt = json.load(f)

    rows_by_camera: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for frame_key, objects in gt.items():
        frame_id = int(frame_key)

        for obj in objects:
            object_id = int(obj["object id"])
            object_type = str(obj["object type"])
            # Prefix with scene name to avoid identity collisions across scenes.
            pid_key = f"{scene}_{object_id}"

            for camera, bbox in obj.get("2d bounding box visible", {}).items():
                rows_by_camera[camera].append(
                    {
                        "split": split,
                        "scene": scene,
                        "camera": camera,
                        "frame_id": frame_id,
                        "object_id": object_id,
                        "object_type": object_type,
                        "pid_key": pid_key,
                        "bbox": bbox,
                    }
                )

    return rows_by_camera


def build_pid_map(
    raw_root: Path,
    splits: list[str],
) -> dict[str, int]:
    """Builds a stable global identity-to-index mapping across splits.

    Args:
        raw_root: Root directory containing per-split subdirectories.
        splits: List of split names to include, e.g. ``["train", "val"]``.

    Returns:
        Mapping from ``"<scene>_<object_id>"`` strings to integer pids,
        sorted alphabetically for reproducibility.
    """
    pid_keys: set[str] = set()

    for split in splits:
        split_dir = raw_root / split
        if not split_dir.exists():
            continue

        scene_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
        for scene_dir in tqdm(scene_dirs, desc=f"pid map [{split}]", unit="scene"):
            gt_path = scene_dir / "ground_truth.json"
            if not gt_path.exists():
                continue

            with open(gt_path) as f:
                gt = json.load(f)

            scene = scene_dir.name
            for _, objects in gt.items():
                for obj in objects:
                    pid_keys.add(f"{scene}_{int(obj['object id'])}")

    return {key: idx for idx, key in enumerate(sorted(pid_keys))}


def build_tracklet_index(
    rows: list[dict[str, Any]],
    max_gap: int,
    min_tracklet_len: int,
) -> dict[tuple[str, int], int]:
    """Builds a frame-level mapping to tracklet index for one camera video.

    Args:
        rows: Annotation rows for a single scene-camera combination.
        max_gap: Maximum frame gap before splitting a tracklet.
        min_tracklet_len: Minimum number of frames for a tracklet to be kept.

    Returns:
        Mapping from ``(pid_key, frame_id)`` to integer tracklet index.
    """
    frames_by_pid: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        frames_by_pid[row["pid_key"]].append(row["frame_id"])

    frame_to_tracklet: dict[tuple[str, int], int] = {}

    for pid_key, frame_ids in frames_by_pid.items():
        tracklets = split_into_tracklets(frame_ids, max_gap=max_gap)
        valid_idx = 0

        for frames in tracklets:
            if len(frames) < min_tracklet_len:
                continue
            for frame_id in frames:
                frame_to_tracklet[(pid_key, frame_id)] = valid_idx
            valid_idx += 1

    return frame_to_tracklet


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    """Reads a CSV file into a list of row dicts."""
    if not path.exists():
        return []

    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def task_output_paths(task: dict[str, Any]) -> tuple[Path, Path, Path]:
    """Returns checkpoint paths for one scene-camera processing task."""
    partial_dir = Path(task["partial_dir"])
    task_id = task["task_id"]
    return (
        partial_dir / f"{task_id}_images.csv",
        partial_dir / f"{task_id}_tracklets.csv",
        partial_dir / f"{task_id}.done",
    )


def task_is_complete(task: dict[str, Any]) -> bool:
    """Checks whether a scene-camera task already has resume artifacts."""
    image_partial_path, tracklet_partial_path, done_path = task_output_paths(task)
    return (
        done_path.exists()
        and image_partial_path.exists()
        and tracklet_partial_path.exists()
    )


def process_one_video(
    task: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Processes a single scene-camera video and extracts crop records.

    Args:
        task: Dict of task parameters. Expected keys: ``split``, ``scene``,
            ``scene_dir``, ``camera``, ``rows``, ``out_root``, ``pid_map``,
            ``min_width``, ``min_height``, ``min_area``, ``padding``,
            ``max_gap``, ``min_tracklet_len``, ``frame_stride``,
            ``resize_h``, ``resize_w``, ``frame_progress``.

    Returns:
        A tuple ``(image_rows, tracklet_rows)`` where each element is a list
        of record dicts ready for CSV output.

    Raises:
        RuntimeError: If the video file cannot be opened.
    """
    split = task["split"]
    scene = task["scene"]
    scene_dir = Path(task["scene_dir"])
    camera = task["camera"]
    rows: list[dict[str, Any]] = task["rows"]
    out_root = Path(task["out_root"])
    pid_map: dict[str, int] = task["pid_map"]

    min_width: int = task["min_width"]
    min_height: int = task["min_height"]
    min_area: int = task["min_area"]
    padding: float = task["padding"]
    max_gap: int = task["max_gap"]
    min_tracklet_len: int = task["min_tracklet_len"]
    frame_stride: int = task["frame_stride"]
    resize_h: int = task["resize_h"]
    resize_w: int = task["resize_w"]
    frame_progress: bool = task.get("frame_progress", False)

    image_partial_path, tracklet_partial_path, done_path = task_output_paths(task)
    if task.get("resume", True) and task_is_complete(task):
        return (
            read_csv_rows(image_partial_path),
            read_csv_rows(tracklet_partial_path),
        )

    video_path = find_video_path(scene_dir, camera)
    rows = sorted(rows, key=lambda r: r["frame_id"])

    rows_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_frame[row["frame_id"]].append(row)

    needed_frames = sorted(rows_by_frame.keys())
    if frame_stride > 1:
        needed_frames = [
            frame_id for frame_id in needed_frames if frame_id % frame_stride == 0
        ]
    if not needed_frames:
        return [], []

    frame_to_tracklet = build_tracklet_index(
        rows,
        max_gap=max_gap,
        min_tracklet_len=min_tracklet_len,
    )

    image_rows: list[dict[str, Any]] = []
    tracklet_accumulator: dict[tuple, list[int]] = defaultdict(list)
    created_dirs: set[Path] = set()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    current_frame = 0

    frame_iterator = needed_frames
    if frame_progress:
        frame_iterator = tqdm(
            needed_frames,
            total=len(needed_frames),
            desc=f"Frames {scene}/{camera}",
            leave=False,
            unit="frame",
        )

    for target_frame in frame_iterator:
        if target_frame < current_frame:
            continue

        gap = target_frame - current_frame
        if gap > FAST_SEEK_THRESHOLD:
            # Use direct seek for large jumps instead of sequential grab.
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            current_frame = target_frame
        else:
            # Fast skip: grab (no decode) until the target frame.
            while current_frame < target_frame:
                if not cap.grab():
                    break
                current_frame += 1

        ok, frame = cap.read()
        if not ok:
            break
        current_frame += 1

        height, width = frame.shape[:2]

        for row in rows_by_frame[target_frame]:
            pid_key = row["pid_key"]
            lookup_key = (pid_key, target_frame)

            if lookup_key not in frame_to_tracklet:
                continue

            bbox = clip_bbox(
                row["bbox"],
                width=width,
                height=height,
                padding=padding,
            )

            if not valid_bbox(bbox, min_width, min_height, min_area):
                continue

            x1, y1, x2, y2 = bbox
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            if resize_h > 0 and resize_w > 0:
                crop = cv2.resize(
                    crop,
                    (resize_w, resize_h),
                    interpolation=cv2.INTER_LINEAR,
                )

            pid = pid_map[pid_key]
            camid = parse_camera_id(camera)
            tracklet_idx = frame_to_tracklet[lookup_key]
            tracklet_id = (
                f"{scene}_{camera}_obj{row['object_id']:06d}_trk{tracklet_idx:03d}"
            )

            crop_dir = (
                out_root
                / "crops"
                / split
                / scene
                / camera
                / f"pid_{pid:06d}"
                / f"tracklet_{tracklet_idx:03d}"
            )
            if crop_dir not in created_dirs:
                crop_dir.mkdir(parents=True, exist_ok=True)
                created_dirs.add(crop_dir)

            crop_path = (
                crop_dir / f"{target_frame:06d}_obj{row['object_id']:06d}{IMAGE_EXT}"
            )

            if not cv2.imwrite(
                str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            ):
                continue

            rel_crop_path = str(crop_path.relative_to(out_root))
            rel_tracklet_dir = str(crop_dir.relative_to(out_root))

            image_rows.append(
                {
                    "filepath": rel_crop_path,
                    "pid": pid,
                    "camid": camid,
                    "scene": scene,
                    "camera": camera,
                    "object_id": row["object_id"],
                    "object_type": row["object_type"],
                    "frame_id": target_frame,
                    "tracklet_id": tracklet_id,
                }
            )

            tracklet_key = (
                tracklet_id,
                pid,
                camid,
                scene,
                camera,
                row["object_id"],
                row["object_type"],
                rel_tracklet_dir,
            )
            tracklet_accumulator[tracklet_key].append(target_frame)

    cap.release()

    tracklet_rows: list[dict[str, Any]] = []
    for key, frame_ids in tracklet_accumulator.items():
        (
            tracklet_id,
            pid,
            camid,
            scene,
            camera,
            object_id,
            object_type,
            rel_tracklet_dir,
        ) = key
        frame_ids = sorted(frame_ids)
        tracklet_rows.append(
            {
                "tracklet_id": tracklet_id,
                "pid": pid,
                "camid": camid,
                "scene": scene,
                "camera": camera,
                "object_id": object_id,
                "object_type": object_type,
                "start_frame": frame_ids[0],
                "end_frame": frame_ids[-1],
                "num_frames": len(frame_ids),
                "tracklet_dir": rel_tracklet_dir,
            }
        )

    image_partial_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(
        image_partial_path,
        image_rows,
        [
            "filepath",
            "pid",
            "camid",
            "scene",
            "camera",
            "object_id",
            "object_type",
            "frame_id",
            "tracklet_id",
        ],
    )
    write_csv(
        tracklet_partial_path,
        tracklet_rows,
        [
            "tracklet_id",
            "pid",
            "camid",
            "scene",
            "camera",
            "object_id",
            "object_type",
            "start_frame",
            "end_frame",
            "num_frames",
            "tracklet_dir",
        ],
    )
    done_path.write_text("ok\n")

    return image_rows, tracklet_rows


def collect_tasks(
    raw_root: Path,
    out_root: Path,
    manifest_dir: Path,
    splits: list[str],
    pid_map: dict[str, int],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Collects per-video processing tasks from all scenes and cameras.

    Args:
        raw_root: Root directory of the raw dataset.
        out_root: Output directory for crops and manifests.
        splits: Dataset split names to process.
        pid_map: Global identity-to-index mapping.
        args: Parsed CLI arguments.

    Returns:
        List of task dicts, one per scene-camera video.
    """
    tasks: list[dict[str, Any]] = []
    partial_dir = manifest_dir / "partials"

    for split in splits:
        split_dir = raw_root / split
        if not split_dir.exists():
            print(f"Skipping missing split: {split_dir}")
            continue

        scene_dirs = sorted(
            d
            for d in split_dir.iterdir()
            if d.is_dir() and (d / "ground_truth.json").exists()
        )
        for scene_dir in tqdm(scene_dirs, desc=f"tasks [{split}]", unit="scene"):
            rows_by_camera = load_scene_annotations(scene_dir, split=split)

            for camera, rows in rows_by_camera.items():
                if not rows:
                    continue
                tasks.append(
                    {
                        "task_id": f"{split}__{scene_dir.name}__{camera}",
                        "split": split,
                        "scene": scene_dir.name,
                        "scene_dir": str(scene_dir),
                        "camera": camera,
                        "partial_dir": str(partial_dir),
                        "rows": rows,
                        "out_root": str(out_root),
                        "pid_map": pid_map,
                        "min_width": args.min_width,
                        "min_height": args.min_height,
                        "min_area": args.min_area,
                        "padding": args.padding,
                        "max_gap": args.max_gap,
                        "min_tracklet_len": args.min_tracklet_len,
                        "frame_stride": args.frame_stride,
                        "resize_h": args.resize_h,
                        "resize_w": args.resize_w,
                        "frame_progress": args.workers == 1,
                    }
                )

    return tasks


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    """Writes records to a CSV file, sorted by all field values.

    Args:
        path: Output file path (parent directory is created if needed).
        rows: List of record dicts to write.
        fieldnames: Ordered column names for the CSV header.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: tuple(str(r.get(k, "")) for k in fieldnames))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            tqdm(rows, desc=f"Writing {path.name}", unit="row", leave=False)
        )


def build_query_gallery(
    tracklet_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Splits validation tracklets into query and gallery sets.

    Selects one tracklet per pid-camera pair for the query set (longest
    tracklet first). The gallery contains all validation tracklets.

    During evaluation:
        - positive: same pid, different camera
        - ignored: same pid, same camera

    Args:
        tracklet_rows: All validation tracklet records.

    Returns:
        A tuple ``(query_rows, gallery_rows)``.
    """
    sorted_rows = sorted(
        tracklet_rows,
        key=lambda r: (
            int(r["pid"]),
            int(r["camid"]),
            -int(r["num_frames"]),
            str(r["tracklet_id"]),
        ),
    )

    query_rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()

    for row in sorted_rows:
        key = (int(row["pid"]), int(row["camid"]))
        if key not in seen:
            query_rows.append(row)
            seen.add(key)

    return query_rows, list(tracklet_rows)


def build_representative_manifest(
    input_csv: Path,
    output_csv: Path,
    fraction: float = 0.1,
    seed: int = 42,
    coverage_fields: tuple[str, ...] = ("pid", "camid", "tracklet_id"),
) -> dict[str, int]:
    """Builds a deterministic representative subset manifest.

    Sampling strategy:
    1) Keep one row per unique ``tracklet_id`` (if present).
    2) Ensure all values for ``coverage_fields`` are represented.
    3) Fill remaining rows uniformly at random to reach ``fraction`` size.

    The output is reproducible for a fixed ``seed``.

    Args:
        input_csv: Source manifest CSV path.
        output_csv: Target CSV path for the sampled subset.
        fraction: Target fraction of rows to keep (e.g. ``0.1`` for 1/10).
        seed: Random seed used for deterministic sampling.
        coverage_fields: Columns for which full value coverage is enforced
            when present in the CSV.

    Returns:
        Summary dict with ``input_rows``, ``target_rows``, ``output_rows``,
        and ``min_required_rows``.
    """
    if not (0 < fraction <= 1):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    rows = read_csv_rows(input_csv)
    if not rows:
        write_csv(output_csv, [], [])
        return {
            "input_rows": 0,
            "target_rows": 0,
            "output_rows": 0,
            "min_required_rows": 0,
        }

    fieldnames = list(rows[0].keys())
    active_coverage_fields = [field for field in coverage_fields if field in fieldnames]

    n_rows = len(rows)
    target_rows = max(1, int(n_rows * fraction))

    selected_indices: set[int] = set()

    if "tracklet_id" in fieldnames:
        first_by_tracklet: dict[str, int] = {}
        for idx, row in enumerate(rows):
            tracklet_id = str(row["tracklet_id"])
            if tracklet_id not in first_by_tracklet:
                first_by_tracklet[tracklet_id] = idx
        selected_indices.update(first_by_tracklet.values())

    def covered_values(field: str) -> set[str]:
        return {str(rows[idx][field]) for idx in selected_indices}

    def sorted_values(values: set[str]) -> list[str]:
        try:
            return sorted(values, key=lambda x: int(x))
        except ValueError:
            return sorted(values)

    for field in active_coverage_fields:
        all_values = {str(row[field]) for row in rows}
        missing = all_values - covered_values(field)
        if not missing:
            continue

        first_by_value: dict[str, int] = {}
        for idx, row in enumerate(rows):
            value = str(row[field])
            if value not in first_by_value:
                first_by_value[value] = idx

        for value in sorted_values(missing):
            selected_indices.add(first_by_value[value])

    min_required_rows = len(selected_indices)
    effective_target = max(target_rows, min_required_rows)

    if len(selected_indices) < effective_target:
        remaining_indices = [
            idx for idx in range(n_rows) if idx not in selected_indices
        ]
        rng = random.Random(seed)
        rng.shuffle(remaining_indices)
        needed = effective_target - len(selected_indices)
        selected_indices.update(remaining_indices[:needed])

    sampled_rows = [rows[idx] for idx in sorted(selected_indices)]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sampled_rows)

    return {
        "input_rows": n_rows,
        "target_rows": target_rows,
        "output_rows": len(sampled_rows),
        "min_required_rows": min_required_rows,
    }


def create_small_query_gallery_manifests(
    manifest_dir: Path,
    fraction: float = 0.1,
    seed: int = 42,
    include_train: bool = False,
) -> dict[str, dict[str, int]]:
    """Creates representative small query/gallery image manifests.

    Reads ``image_gallery.csv`` and ``image_query.csv`` from ``manifest_dir``
    and writes ``image_gallery_small.csv`` and ``image_query_small.csv``.
    If ``include_train`` is set and ``image_train.csv`` exists, also writes
    ``image_train_small.csv``.

    Args:
        manifest_dir: Directory containing manifest CSVs.
        fraction: Fraction of rows to keep.
        seed: Random seed for deterministic sampling.
        include_train: Whether to also create ``image_train_small.csv``.

    Returns:
        Per-file sampling summaries keyed by output filename.
    """
    pairs = [
        (manifest_dir / "image_gallery.csv", manifest_dir / "image_gallery_small.csv"),
        (manifest_dir / "image_query.csv", manifest_dir / "image_query_small.csv"),
    ]

    if include_train and (manifest_dir / "image_train.csv").exists():
        pairs.append(
            (manifest_dir / "image_train.csv", manifest_dir / "image_train_small.csv")
        )

    summaries: dict[str, dict[str, int]] = {}
    for src, dst in pairs:
        summaries[dst.name] = build_representative_manifest(
            input_csv=src,
            output_csv=dst,
            fraction=fraction,
            seed=seed,
            coverage_fields=("pid", "camid", "tracklet_id"),
        )

    return summaries


def main() -> None:
    """Parses CLI arguments and runs the full preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description=("Preprocess AICITY MTMC dataset into ReID crops and manifests.")
    )
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
    )
    parser.add_argument("--min-width", type=int, default=16)
    parser.add_argument("--min-height", type=int, default=16)
    parser.add_argument("--min-area", type=int, default=500)
    parser.add_argument("--padding", type=float, default=0.10)
    parser.add_argument("--max-gap", type=int, default=30)
    parser.add_argument("--min-tracklet-len", type=int, default=8)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=30,
        help="Process every Nth frame (1 = process all frames).",
    )
    # Use 0,0 to keep original crop size.
    parser.add_argument("--resize-h", type=int, default=384)
    parser.add_argument("--resize-w", type=int, default=192)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore resume checkpoints and rebuild all scene-camera tasks.",
    )
    parser.add_argument(
        "--make-small-manifests",
        action="store_true",
        help=(
            "Create representative image_gallery_small.csv and "
            "image_query_small.csv after manifests are prepared."
        ),
    )
    parser.add_argument(
        "--small-fraction",
        type=float,
        default=0.1,
        help="Fraction of rows to keep for small manifests (default: 0.1).",
    )
    parser.add_argument(
        "--small-seed",
        type=int,
        default=42,
        help="Random seed for deterministic small-manifest sampling.",
    )
    parser.add_argument(
        "--small-include-train",
        action="store_true",
        help="Also create image_train_small.csv when --make-small-manifests is used.",
    )

    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    manifest_dir = out_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    print("Building global pid map...")
    pid_map = build_pid_map(raw_root, args.splits)

    with open(manifest_dir / "pid_map.json", "w") as f:
        json.dump(pid_map, f, indent=2)

    print(f"Found {len(pid_map)} unique identities.")

    print("Collecting video tasks...")
    tasks = collect_tasks(
        raw_root=raw_root,
        out_root=out_root,
        manifest_dir=manifest_dir,
        splits=args.splits,
        pid_map=pid_map,
        args=args,
    )
    print(f"Found {len(tasks)} scene-camera video tasks.")
    print(f"Using {args.workers} workers.")

    image_rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tracklet_rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)

    resumable_tasks = [task for task in tasks if task_is_complete(task)]
    pending_tasks = (
        tasks
        if args.overwrite
        else [task for task in tasks if not task_is_complete(task)]
    )

    if args.overwrite:
        print("Overwrite enabled: ignoring existing resume checkpoints.")
    else:
        print(
            f"Resume: {len(resumable_tasks)} completed tasks found, "
            f"{len(pending_tasks)} tasks remaining."
        )
        for task in tqdm(
            resumable_tasks,
            desc="Loading checkpoints",
            unit="video",
        ):
            image_rows, tracklet_rows = process_one_video(task)

            for row in image_rows:
                split = row["filepath"].split(os.sep)[1]
                image_rows_by_split[split].append(row)

            for row in tracklet_rows:
                split = row["tracklet_dir"].split(os.sep)[1]
                tracklet_rows_by_split[split].append(row)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(process_one_video, {**task, "resume": not args.overwrite})
            for task in pending_tasks
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Videos",
            unit="video",
        ):
            image_rows, tracklet_rows = future.result()

            for row in image_rows:
                # Infer split from filepath: crops/<split>/...
                split = row["filepath"].split(os.sep)[1]
                image_rows_by_split[split].append(row)

            for row in tracklet_rows:
                split = row["tracklet_dir"].split(os.sep)[1]
                tracklet_rows_by_split[split].append(row)

    image_fieldnames = [
        "filepath",
        "pid",
        "camid",
        "scene",
        "camera",
        "object_id",
        "object_type",
        "frame_id",
        "tracklet_id",
    ]
    tracklet_fieldnames = [
        "tracklet_id",
        "pid",
        "camid",
        "scene",
        "camera",
        "object_id",
        "object_type",
        "start_frame",
        "end_frame",
        "num_frames",
        "tracklet_dir",
    ]

    for split in tqdm(args.splits, desc="Writing CSVs", unit="split"):
        image_rows = image_rows_by_split[split]
        tracklet_rows = tracklet_rows_by_split[split]

        write_csv(
            manifest_dir / f"image_{split}.csv",
            image_rows,
            image_fieldnames,
        )
        write_csv(
            manifest_dir / f"tracklet_{split}.csv",
            tracklet_rows,
            tracklet_fieldnames,
        )
        print(f"{split}: wrote {len(image_rows)} image rows")
        print(f"{split}: wrote {len(tracklet_rows)} tracklet rows")

    if "val" in tracklet_rows_by_split:
        query_rows, gallery_rows = build_query_gallery(tracklet_rows_by_split["val"])
        write_csv(
            manifest_dir / "query_tracklets.csv",
            query_rows,
            tracklet_fieldnames,
        )
        write_csv(
            manifest_dir / "gallery_tracklets.csv",
            gallery_rows,
            tracklet_fieldnames,
        )
        print(f"val query tracklets: {len(query_rows)}")
        print(f"val gallery tracklets: {len(gallery_rows)}")

    if args.make_small_manifests:
        image_gallery_csv = manifest_dir / "image_gallery.csv"
        image_query_csv = manifest_dir / "image_query.csv"

        if image_gallery_csv.exists() and image_query_csv.exists():
            summaries = create_small_query_gallery_manifests(
                manifest_dir=manifest_dir,
                fraction=args.small_fraction,
                seed=args.small_seed,
                include_train=args.small_include_train,
            )
            print("Created representative small manifests:")
            for filename, summary in summaries.items():
                print(
                    f"  {filename}: "
                    f"input={summary['input_rows']}, "
                    f"target={summary['target_rows']}, "
                    f"output={summary['output_rows']}, "
                    f"min_required={summary['min_required_rows']}"
                )
        else:
            print(
                "Skipping --make-small-manifests: expected "
                "image_gallery.csv and image_query.csv in manifests/."
            )

    print("Done.")


if __name__ == "__main__":
    main()
