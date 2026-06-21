#!/usr/bin/env python3
"""AICITY 2025 MTMC ReID preprocessing pipeline.

Builds training and validation (query / gallery) crops plus manifests from
MTMC_Tracking_2025.  Only ``train/`` and ``val/`` are processed.


Design notes
------------
- Ground-truth JSON keys are direct video-frame indices (0-based).  A 9 000-
  frame 30 fps video has JSON keys ``"0"``–``"8999"``; seeking to frame N means
  ``cap.set(cv2.CAP_PROP_POS_FRAMES, N)``.
- All object types are preserved (Person, Forklift, NovaCarter, …).
- The global PID map is built from *all* tracklets (train + val combined),
  sorted alphabetically by the string key ``"{scene}_{object_id}"``, which
  reproduces the ordering found in existing ``pid_map.json`` files.
- Camera IDs are per-scene, assigned as the sorted-alphabetical index of each
  camera name within that scene.
- Query/gallery selection follows the spec strictly:
    * Identity must appear in ≥ 2 cameras.
    * Longest tracklet (by annotation-frame count) → query.
    * Gallery = tracklets from cameras *other* than the query camera.
    * Query tracklet never appears in gallery.
- Val crops (query **and** gallery) both live under ``crops/val/``; the CSV
  manifests carry the split distinction.
- Resume support: each decode worker checks whether the output JPEG already
  exists before seeking to that video frame.

Output layout::

    <output-root>/
        crops/
            train/<scene>/<camera>/pid_XXXXXX/tracklet_YYY/FFFFFFFF_objOOOOOO.jpg
            val/  <scene>/<camera>/pid_XXXXXX/tracklet_YYY/FFFFFFFF_objOOOOOO.jpg
        manifests/
            pid_map.json
            image_train.csv        tracklet_train.csv
            image_query.csv        image_gallery.csv
            query_tracklets.csv    gallery_tracklets.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline defaults
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_NUM_SAMPLES: int = 30   # evenly-spaced frames sampled per tracklet
_DEFAULT_MAX_GAP: int = 30       # video-frame gap that forces a tracklet split
                                  # (= ann_stride: split on any missing annotation)
_DEFAULT_MIN_LEN: int = 2        # minimum annotation frames to retain a tracklet
_DEFAULT_ANN_STRIDE: int = 30    # keep only frames where frame_id % stride == 0

_IMAGE_HEADER: tuple[str, ...] = (
    "filepath",
    "pid",
    "camid",
    "scene",
    "camera",
    "object_id",
    "object_type",
    "frame_id",
    "tracklet_id",
)
_TRACKLET_HEADER: tuple[str, ...] = (
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
)

# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

_Bbox = tuple[int, int, int, int]  # (x1, y1, x2, y2)


@dataclass(slots=True)
class Tracklet:
    """Temporally contiguous appearances of one object in one camera."""

    tracklet_id: str
    pid: int  # −1 until _assign_pids() is called
    camid: int  # per-scene sorted camera index
    scene: str
    camera: str
    object_id: int
    object_type: str
    trk_idx: int  # 0-based index within (scene, camera, object_id)
    ann_frames: list[int]  # sorted annotated video-frame indices
    ann_bboxes: dict[int, _Bbox]  # frame_id → (x1, y1, x2, y2)
    sampled_frames: list[int] = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def start_frame(self) -> int:
        return self.ann_frames[0]

    @property
    def end_frame(self) -> int:
        return self.ann_frames[-1]

    @property
    def num_frames(self) -> int:
        """Number of *sampled* (output) frames."""
        return len(self.sampled_frames) if self.sampled_frames else len(self.ann_frames)


@dataclass(slots=True)
class _WorkerTask:
    """Serialisable payload sent to each (scene, camera) decode worker."""

    split: str  # "train" | "val"
    scene: str
    camera: str
    video_path: str  # str so dataclass pickles cleanly
    tracklets: list[Tracklet]
    crops_root: str  # str for same reason


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────


def _crop_path(
    crops_root: Path,
    split: str,
    scene: str,
    camera: str,
    pid: int,
    trk_idx: int,
    frame_id: int,
    object_id: int,
) -> Path:
    return (
        crops_root
        / split
        / scene
        / camera
        / f"pid_{pid:06d}"
        / f"tracklet_{trk_idx:03d}"
        / f"{frame_id:06d}_obj{object_id:06d}.jpg"
    )


def _tracklet_dir(crops_root: Path, split: str, trk: Tracklet) -> Path:
    return (
        crops_root
        / split
        / trk.scene
        / trk.camera
        / f"pid_{trk.pid:06d}"
        / f"tracklet_{trk.trk_idx:03d}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Parse annotations & build tracklets
# ─────────────────────────────────────────────────────────────────────────────

# Raw annotation type per (camera, object_id): list of (frame_id, bbox, obj_type)
_RawAnnotations = dict[str, dict[int, list[tuple[int, _Bbox, str]]]]


def _parse_ground_truth(scene_dir: Path) -> _RawAnnotations:
    """Parse ``ground_truth.json`` for one scene.

    Returns:
        Nested mapping ``camera → object_id → [(frame_id, bbox, object_type)]``.
    """
    with (scene_dir / "ground_truth.json").open() as fh:
        raw: dict[str, list[dict[str, Any]]] = json.load(fh)

    result: _RawAnnotations = defaultdict(lambda: defaultdict(list))

    for frame_key, annotations in raw.items():
        fid = int(frame_key)
        for ann in annotations:
            obj_id: int = ann["object id"]
            obj_type: str = ann["object type"]
            for cam, box in ann.get("2d bounding box visible", {}).items():
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                result[cam][obj_id].append((fid, (x1, y1, x2, y2), obj_type))

    return result


def build_scene_tracklets(
    scene_dir: Path,
    max_gap: int,
    min_len: int,
    ann_stride: int = _DEFAULT_ANN_STRIDE,
) -> list[Tracklet]:
    """Parse one scene directory and return its tracklets (``pid`` still −1).

    Annotations are dense (every video frame).  *ann_stride* sub-samples them
    to only frames where ``frame_id % ann_stride == 0``, which matches the
    output of the reference pipeline (e.g. stride 30 → 1 annotation/second at
    30 fps).

    Tracklets are split whenever consecutive annotation frames are more than
    *max_gap* video frames apart, then filtered to at least *min_len* frames.
    Camera IDs are assigned as sorted-alphabetical indices within the scene.
    """
    scene = scene_dir.name
    raw = _parse_ground_truth(scene_dir)
    sorted_cams: list[str] = sorted(raw.keys())
    cam_id: dict[str, int] = {c: i for i, c in enumerate(sorted_cams)}

    tracklets: list[Tracklet] = []

    for camera in sorted_cams:
        for obj_id in sorted(raw[camera].keys()):
            entries = raw[camera][obj_id]
            if not entries:
                continue

            # Deterministic sort by frame_id, then sub-sample
            entries.sort(key=lambda e: e[0])
            if ann_stride > 1:
                entries = [e for e in entries if e[0] % ann_stride == 0]
            if not entries:
                continue
            obj_type: str = entries[0][2]

            # Split into gap-free segments
            segments: list[list[tuple[int, _Bbox]]] = []
            cur: list[tuple[int, _Bbox]] = [(entries[0][0], entries[0][1])]
            for i in range(1, len(entries)):
                fid, bbox, _ = entries[i]
                if fid - cur[-1][0] > max_gap:
                    segments.append(cur)
                    cur = [(fid, bbox)]
                else:
                    cur.append((fid, bbox))
            segments.append(cur)

            trk_idx = 0
            for seg in segments:
                if len(seg) < min_len:
                    continue
                frames = [f for f, _ in seg]
                bboxes: dict[int, _Bbox] = {f: b for f, b in seg}
                tracklets.append(
                    Tracklet(
                        tracklet_id=(
                            f"{scene}_{camera}"
                            f"_obj{obj_id:06d}"
                            f"_trk{trk_idx:03d}"
                        ),
                        pid=-1,
                        camid=cam_id[camera],
                        scene=scene,
                        camera=camera,
                        object_id=obj_id,
                        object_type=obj_type,
                        trk_idx=trk_idx,
                        ann_frames=frames,
                        ann_bboxes=bboxes,
                    )
                )
                trk_idx += 1

    return tracklets


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Build global PID map
# ─────────────────────────────────────────────────────────────────────────────


def build_pid_map(all_tracklets: list[Tracklet]) -> dict[str, int]:
    """Assign globally unique integer PIDs to all identities.

    Keys use the format ``"{scene}_{object_id}"`` (raw integer, no zero-
    padding), sorted **alphabetically** — this reproduces the ordering found in
    existing ``pid_map.json`` files (e.g. ``"Hospital_000_10"`` sorts before
    ``"Hospital_000_2"`` because ``"1" < "2"``).

    Returns:
        Mapping from identity key to sequential integer PID.
    """
    identities: set[tuple[str, int]] = {
        (t.scene, t.object_id) for t in all_tracklets
    }
    sorted_ids = sorted(identities, key=lambda x: f"{x[0]}_{x[1]}")
    return {
        f"{scene}_{obj_id}": pid
        for pid, (scene, obj_id) in enumerate(sorted_ids)
    }


def _assign_pids(tracklets: list[Tracklet], pid_map: dict[str, int]) -> None:
    """Set ``trk.pid`` for every tracklet in-place."""
    for t in tracklets:
        t.pid = pid_map[f"{t.scene}_{t.object_id}"]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Frame sampling
# ─────────────────────────────────────────────────────────────────────────────


def _sample_frames(ann_frames: list[int], num_samples: int) -> list[int]:
    """Return up to *num_samples* evenly-spaced frames from *ann_frames*.

    Uses ``np.linspace`` as specified.  If ``len(ann_frames) <= num_samples``
    all frames are returned unchanged.
    """
    n = len(ann_frames)
    if n <= num_samples:
        return list(ann_frames)
    indices = np.linspace(0, n - 1, num_samples, dtype=np.int32)
    # Defensive deduplication (linspace can produce duplicate int indices at
    # the extremes for very small n/num_samples ratios).
    seen: set[int] = set()
    result: list[int] = []
    for idx in indices.tolist():
        fid = ann_frames[int(idx)]
        if fid not in seen:
            seen.add(fid)
            result.append(fid)
    return result


def sample_all(tracklets: list[Tracklet], num_samples: int) -> None:
    """Populate ``sampled_frames`` for every tracklet (in-place)."""
    for t in tracklets:
        t.sampled_frames = _sample_frames(t.ann_frames, num_samples)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Query / gallery selection (val only)
# ─────────────────────────────────────────────────────────────────────────────


def select_query_gallery(
    val_tracklets: list[Tracklet],
) -> tuple[list[Tracklet], list[Tracklet]]:
    """Select query and gallery tracklets from the validation split.

    Rules applied per identity (scene, object_id):

    1. Skip identities visible in fewer than 2 distinct cameras.
    2. Longest tracklet by annotation-frame count → query.
       Ties broken deterministically by ``(scene, camera, trk_idx)``.
    3. Gallery = all tracklets from cameras **other** than the query camera.
    4. The query tracklet itself is **never** placed in the gallery.

    Returns:
        ``(query_tracklets, gallery_tracklets)`` — both sorted by
        ``tracklet_id``.
    """
    by_identity: dict[tuple[str, int], list[Tracklet]] = defaultdict(list)
    for t in val_tracklets:
        by_identity[(t.scene, t.object_id)].append(t)

    query_list: list[Tracklet] = []
    gallery_list: list[Tracklet] = []

    for key in sorted(by_identity):
        trks = by_identity[key]

        # Must appear in ≥ 2 distinct cameras
        if len({t.camera for t in trks}) < 2:
            continue

        # Pick query: longest ann_frames; tie-break by (scene, camera, trk_idx)
        sorted_trks = sorted(
            trks,
            key=lambda t: (-len(t.ann_frames), t.scene, t.camera, t.trk_idx),
        )
        query_trk = sorted_trks[0]
        query_cam = query_trk.camera

        query_list.append(query_trk)
        for t in trks:
            if t.camera != query_cam:
                gallery_list.append(t)

    query_list.sort(key=lambda t: t.tracklet_id)
    gallery_list.sort(key=lambda t: t.tracklet_id)
    return query_list, gallery_list


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Video decode worker  (executes in a subprocess)
# ─────────────────────────────────────────────────────────────────────────────


def _decode_worker(task: _WorkerTask) -> list[dict[str, Any]]:
    """Decode required frames from one video file and write crop JPEGs.

    This function runs in a worker process.

    * ``cv2.setNumThreads(1)`` prevents OpenCV from spawning its own threads
      inside each subprocess.
    * Frames whose output JPEG already exists are **skipped** (resume support).
    * The video is opened once; frames are decoded in sorted order to minimise
      seek overhead.

    Returns:
        Per-frame image record dicts (both freshly written and pre-existing).
    """
    cv2.setNumThreads(1)
    crops_root = Path(task.crops_root)

    # ── Build frame → list[(tracklet, bbox, out_path)] for missing files ──
    frame_jobs: dict[
        int, list[tuple[Tracklet, _Bbox, Path]]
    ] = defaultdict(list)

    for trk in task.tracklets:
        for fid in trk.sampled_frames:
            out = _crop_path(
                crops_root,
                task.split,
                trk.scene,
                trk.camera,
                trk.pid,
                trk.trk_idx,
                fid,
                trk.object_id,
            )
            if not out.exists():
                frame_jobs[fid].append((trk, trk.ann_bboxes[fid], out))

    # ── Decode missing frames ─────────────────────────────────────────────
    if frame_jobs:
        cap = cv2.VideoCapture(task.video_path)
        if not cap.isOpened():
            logger.error("Cannot open video: %s", task.video_path)
        else:
            for fid in sorted(frame_jobs.keys()):
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(fid))
                ret, frame = cap.read()
                if not ret:
                    logger.warning(
                        "Unreadable frame %d in %s", fid, task.video_path
                    )
                    continue

                h, w = frame.shape[:2]
                for trk, (x1, y1, x2, y2), out_path in frame_jobs[fid]:
                    # Clamp bbox to frame bounds
                    cx1, cy1 = max(0, x1), max(0, y1)
                    cx2, cy2 = min(w, x2), min(h, y2)
                    if cx2 <= cx1 or cy2 <= cy1:
                        continue
                    crop = frame[cy1:cy2, cx1:cx2]
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(out_path),
                        crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 95],
                    )
        cap.release()

    # ── Collect records (pre-existing + newly created) ────────────────────
    records: list[dict[str, Any]] = []
    for trk in task.tracklets:
        for fid in trk.sampled_frames:
            out = _crop_path(
                crops_root,
                task.split,
                trk.scene,
                trk.camera,
                trk.pid,
                trk.trk_idx,
                fid,
                trk.object_id,
            )
            if out.exists():
                records.append(
                    {
                        "filepath": f"./{out}",
                        "pid": trk.pid,
                        "camid": trk.camid,
                        "scene": trk.scene,
                        "camera": trk.camera,
                        "object_id": trk.object_id,
                        "object_type": trk.object_type,
                        "frame_id": fid,
                        "tracklet_id": trk.tracklet_id,
                    }
                )
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Parallel orchestration
# ─────────────────────────────────────────────────────────────────────────────


def _group_by_camera(
    tracklets: list[Tracklet],
) -> dict[tuple[str, str], list[Tracklet]]:
    result: dict[tuple[str, str], list[Tracklet]] = defaultdict(list)
    for t in tracklets:
        result[(t.scene, t.camera)].append(t)
    return dict(result)


def run_decode(
    scenes_root: Path,
    split: str,
    tracklets: list[Tracklet],
    crops_root: Path,
    max_workers: int,
    desc: str = "",
) -> list[dict[str, Any]]:
    """Dispatch one worker process per (scene, camera) pair.

    Returns:
        Combined list of image record dicts from all workers.
    """
    by_cam = _group_by_camera(tracklets)
    tasks: list[_WorkerTask] = []

    for (scene, camera), trks in sorted(by_cam.items()):
        if not trks:
            continue
        video = scenes_root / scene / "videos" / f"{camera}.mp4"
        if not video.exists():
            logger.warning("Video missing, skipping: %s", video)
            continue
        tasks.append(
            _WorkerTask(
                split=split,
                scene=scene,
                camera=camera,
                video_path=str(video),
                tracklets=trks,
                crops_root=str(crops_root),
            )
        )

    all_records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_decode_worker, t): t for t in tasks}
        with tqdm(
            total=len(futures),
            desc=desc or f"  Decode {split}",
            unit="cam",
        ) as pbar:
            for fut in as_completed(futures):
                task = futures[fut]
                try:
                    all_records.extend(fut.result())
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Worker failed %s/%s: %s",
                        task.scene,
                        task.camera,
                        exc,
                    )
                pbar.update()

    return all_records


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6 — CSV / JSON manifest writers
# ─────────────────────────────────────────────────────────────────────────────


def _write_image_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_IMAGE_HEADER))
        writer.writeheader()
        writer.writerows(records)
    logger.info("  %8d rows  →  %s", len(records), path.name)


def _write_tracklet_csv(
    path: Path,
    tracklets: list[Tracklet],
    crops_root: Path,
    split: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_TRACKLET_HEADER))
        writer.writeheader()
        for t in tracklets:
            tdir = _tracklet_dir(crops_root, split, t)
            writer.writerow(
                {
                    "tracklet_id": t.tracklet_id,
                    "pid": t.pid,
                    "camid": t.camid,
                    "scene": t.scene,
                    "camera": t.camera,
                    "object_id": t.object_id,
                    "object_type": t.object_type,
                    "start_frame": t.start_frame,
                    "end_frame": t.end_frame,
                    "num_frames": t.num_frames,
                    "tracklet_dir": f"./{tdir}",
                }
            )
    logger.info("  %8d trks  →  %s", len(tracklets), path.name)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AICITY 2025 MTMC ReID preprocessing pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("AICITY2025/MTMC_Tracking_2025"),
        help="Path to MTMC_Tracking_2025/",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("AICITY2025/aicity"),
        help="Root for crops/ and manifests/ output",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=_DEFAULT_NUM_SAMPLES,
        metavar="N",
        help="Evenly-spaced frames sampled per tracklet",
    )
    p.add_argument(
        "--max-gap",
        type=int,
        default=_DEFAULT_MAX_GAP,
        metavar="FRAMES",
        help="Video-frame gap that forces a new tracklet",
    )
    p.add_argument(
        "--min-len",
        type=int,
        default=_DEFAULT_MIN_LEN,
        metavar="FRAMES",
        help="Minimum annotation frames to keep a tracklet",
    )
    p.add_argument(
        "--ann-stride",
        type=int,
        default=_DEFAULT_ANN_STRIDE,
        metavar="STRIDE",
        help=(
            "Sub-sample dense per-frame annotations: keep only frames where "
            "frame_id %% stride == 0.  Set to 1 to use every frame."
        ),
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=min(os.cpu_count() or 4, 32),
        help="Maximum parallel decode-worker processes",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val"],
        default=["train", "val"],
        metavar="{train,val}",
        help="Splits to decode and crop (both are always parsed for the PID map)",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()

    dataset_root: Path = args.dataset_root
    crops_root: Path = args.output_root / "crops"
    manifests_dir: Path = args.output_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    process_train: bool = "train" in args.splits
    process_val: bool = "val" in args.splits

    # ── Stage 1: Parse all annotations (both splits always, for PID map) ──
    logger.info("══════════  Stage 1 — Parse annotations  ══════════")
    train_tracklets: list[Tracklet] = []
    val_tracklets: list[Tracklet] = []

    for split_name, container in (
        ("train", train_tracklets),
        ("val", val_tracklets),
    ):
        split_dir = dataset_root / split_name
        if not split_dir.exists():
            logger.warning("Split directory not found, skipping: %s", split_dir)
            continue
        scene_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
        for scene_dir in tqdm(
            scene_dirs, desc=f"  Parse {split_name}", unit="scene"
        ):
            container.extend(
                build_scene_tracklets(
                    scene_dir,
                    args.max_gap,
                    args.min_len,
                    args.ann_stride,
                )
            )

    logger.info(
        "  train: %d tracklets   val: %d tracklets",
        len(train_tracklets),
        len(val_tracklets),
    )

    # ── Stage 2: Build global PID map ─────────────────────────────────────
    logger.info("══════════  Stage 2 — Build PID map  ══════════")
    pid_map = build_pid_map(train_tracklets + val_tracklets)
    _assign_pids(train_tracklets, pid_map)
    _assign_pids(val_tracklets, pid_map)

    pid_map_path = manifests_dir / "pid_map.json"
    pid_map_path.write_text(json.dumps(pid_map, indent=2))
    logger.info(
        "  %d unique identities  →  %s", len(pid_map), pid_map_path.name
    )

    # ── Stage 3: Sample frames ────────────────────────────────────────────
    logger.info(
        "══════════  Stage 3 — Sample frames (n=%d)  ══════════",
        args.num_samples,
    )
    sample_all(train_tracklets, args.num_samples)
    sample_all(val_tracklets, args.num_samples)

    # ── Stage 4: Query / gallery selection ────────────────────────────────
    query_tracklets: list[Tracklet] = []
    gallery_tracklets: list[Tracklet] = []

    if process_val:
        logger.info(
            "══════════  Stage 4 — Select query / gallery  ══════════"
        )
        query_tracklets, gallery_tracklets = select_query_gallery(
            val_tracklets
        )
        logger.info(
            "  query: %d tracklets   gallery: %d tracklets",
            len(query_tracklets),
            len(gallery_tracklets),
        )

    # ── Stage 5: Decode & crop ────────────────────────────────────────────
    logger.info("══════════  Stage 5 — Decode & crop  ══════════")
    train_records: list[dict[str, Any]] = []
    query_records: list[dict[str, Any]] = []
    gallery_records: list[dict[str, Any]] = []

    _rec_sort_key = lambda r: (  # noqa: E731
        r["scene"],
        r["camera"],
        r["pid"],
        r["tracklet_id"],
        r["frame_id"],
    )

    if process_train:
        train_records = run_decode(
            dataset_root / "train",
            "train",
            train_tracklets,
            crops_root,
            args.max_workers,
            desc="  Train decode",
        )
        train_records.sort(key=_rec_sort_key)

    if process_val:
        # Deduplicate tracklets (query ∩ gallery = ∅ by design, but be safe)
        unique_val_trks = list(
            {
                t.tracklet_id: t
                for t in query_tracklets + gallery_tracklets
            }.values()
        )
        val_records = run_decode(
            dataset_root / "val",
            "val",
            unique_val_trks,
            crops_root,
            args.max_workers,
            desc="  Val decode",
        )

        query_tids: frozenset[str] = frozenset(
            t.tracklet_id for t in query_tracklets
        )
        gallery_tids: frozenset[str] = frozenset(
            t.tracklet_id for t in gallery_tracklets
        )
        for rec in val_records:
            tid = rec["tracklet_id"]
            if tid in query_tids:
                query_records.append(rec)
            if tid in gallery_tids:
                gallery_records.append(rec)

        query_records.sort(key=_rec_sort_key)
        gallery_records.sort(key=_rec_sort_key)

    # ── Stage 6: Write manifests ──────────────────────────────────────────
    logger.info("══════════  Stage 6 — Write manifests  ══════════")

    if process_train:
        _write_image_csv(manifests_dir / "image_train.csv", train_records)
        _write_tracklet_csv(
            manifests_dir / "tracklet_train.csv",
            sorted(train_tracklets, key=lambda t: t.tracklet_id),
            crops_root,
            "train",
        )

    if process_val:
        _write_image_csv(manifests_dir / "image_query.csv", query_records)
        _write_image_csv(manifests_dir / "image_gallery.csv", gallery_records)
        _write_tracklet_csv(
            manifests_dir / "query_tracklets.csv",
            query_tracklets,  # already sorted
            crops_root,
            "val",
        )
        _write_tracklet_csv(
            manifests_dir / "gallery_tracklets.csv",
            gallery_tracklets,  # already sorted
            crops_root,
            "val",
        )

    logger.info("══════════  Done  ══════════")


if __name__ == "__main__":
    main()
