import os
import json
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def create_target_multicam_video(
    input_path,
    output_video_path,
    target_object_id,
    target_object_type=None,
    cameras=None,
    start_frame=0,
    end_frame=None,
    fps=10,
    grid_cols=3,
    resize_width=640,
    bbox_color=(0, 255, 0),
    bbox_thickness=3,
    show_progress=True,
):
    videos_path = os.path.join(input_path, "videos")
    gt_path = os.path.join(input_path, "ground_truth.json")

    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    target_index = build_target_bbox_index(
        gt,
        target_object_id=target_object_id,
        target_object_type=target_object_type,
    )

    if cameras is None:
        cameras = sorted(target_index.keys())

    if not cameras:
        raise ValueError("No cameras found for this target.")

    videos = [
        f for f in os.listdir(videos_path)
        if f.lower().endswith(".mp4")
    ]

    video_files = {
        cam: find_video_for_camera(videos_path, videos, cam)
        for cam in cameras
    }

    caps = {}

    video_items = video_files.items()

    if show_progress:
        video_items = tqdm(
            video_items,
            desc="Opening camera videos",
            unit="camera",
        )

    for cam, video_file in video_items:
        if video_file is None:
            print(f"[WARN] Missing video for {cam}")
            continue

        cap = cv2.VideoCapture(video_file)

        if not cap.isOpened():
            print(f"[WARN] Cannot open {video_file}")
            continue

        caps[cam] = cap

    if not caps:
        raise RuntimeError("No camera videos could be opened.")

    if end_frame is None:
        end_frame = max(
            max(frames.keys())
            for frames in target_index.values()
            if frames
        )

    grid_rows = int(np.ceil(len(cameras) / grid_cols))

    sample_h = int(resize_width * 9 / 16)
    grid_w = grid_cols * resize_width
    grid_h = grid_rows * sample_h

    output_dir = os.path.dirname(output_video_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (grid_w, grid_h),
    )

    if not writer.isOpened():
        for cap in caps.values():
            cap.release()
        raise RuntimeError(f"Could not create output video: {output_video_path}")

    frame_range = range(start_frame, end_frame + 1)

    if show_progress:
        frame_range = tqdm(
            frame_range,
            desc=f"Rendering target {target_object_id}",
            unit="frame",
        )

    for frame_id in frame_range:
        if show_progress:
            frame_range.set_postfix({
                "frame": frame_id,
                "cams": len(cameras),
            })

        tiles = []

        for cam in cameras:
            if cam not in caps:
                tile = blank_tile(
                    resize_width,
                    sample_h,
                    cam,
                    "NO VIDEO",
                )
                tiles.append(tile)
                continue

            cap = caps[cam]

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()

            if not ok or frame is None:
                tile = blank_tile(
                    resize_width,
                    sample_h,
                    cam,
                    "NO FRAME",
                )
                tiles.append(tile)
                continue

            bbox = target_index.get(cam, {}).get(frame_id)

            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)

                x1 = max(0, min(frame.shape[1] - 1, x1))
                x2 = max(0, min(frame.shape[1] - 1, x2))
                y1 = max(0, min(frame.shape[0] - 1, y1))
                y2 = max(0, min(frame.shape[0] - 1, y2))

                if x2 > x1 and y2 > y1:
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        bbox_color,
                        bbox_thickness,
                    )

                    label = f"{target_object_type or ''} ID={target_object_id}"

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        bbox_color,
                        2,
                    )

            frame = resize_keep_aspect(
                frame,
                resize_width,
                sample_h,
            )

            cv2.putText(
                frame,
                f"{cam} | frame {frame_id}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            tiles.append(frame)

        while len(tiles) < grid_rows * grid_cols:
            tiles.append(
                blank_tile(
                    resize_width,
                    sample_h,
                    "",
                    "",
                )
            )

        rows = []

        for r in range(grid_rows):
            row = tiles[
                r * grid_cols:(r + 1) * grid_cols
            ]
            rows.append(np.hstack(row))

        grid_frame = np.vstack(rows)
        writer.write(grid_frame)

    writer.release()

    for cap in caps.values():
        cap.release()

    print("Saved video:", output_video_path)


def build_target_bbox_index(
    ground_truth,
    target_object_id,
    target_object_type=None,
):
    index = defaultdict(dict)

    for frame_str, objects in ground_truth.items():
        frame_id = int(frame_str)

        for obj in objects:
            obj_id = int(obj["object id"])
            obj_type = obj["object type"]

            if obj_id != target_object_id:
                continue

            if target_object_type is not None and obj_type != target_object_type:
                continue

            boxes = obj.get("2d bounding box visible", {})

            for cam, bbox in boxes.items():
                index[cam][frame_id] = bbox

    return index


def find_video_for_camera(videos_path, videos, camera_id):
    expected_name = f"{camera_id}.mp4"

    if expected_name in videos:
        return os.path.join(videos_path, expected_name)

    for video in videos:
        if os.path.splitext(video)[0] == camera_id:
            return os.path.join(videos_path, video)

    return None


def resize_keep_aspect(frame, target_w, target_h):
    h, w = frame.shape[:2]

    scale = min(
        target_w / w,
        target_h / h,
    )

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(
        frame,
        (new_w, new_h),
    )

    canvas = np.zeros(
        (target_h, target_w, 3),
        dtype=np.uint8,
    )

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w,
    ] = resized

    return canvas


def blank_tile(width, height, camera_id, text):
    tile = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    cv2.putText(
        tile,
        f"{camera_id} {text}",
        (30, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    return tile


def main():
    create_target_multicam_video(
        input_path="./DataSet/MTMC_Tracking_2025/train/Warehouse_000",
        output_video_path="./DataSet/debug_videos/Warehouse_000_Person_0003_multicam.mp4",
        target_object_id=3,
        target_object_type="Person",
        cameras=["Camera_0000", "Camera_0001", "Camera_0004", "Camera_0006", "Camera_0010", "Camera_0014", "Camera_0022", "Camera_0024"],
        start_frame=0,
        end_frame=4000,
        fps=15,
        grid_cols=3,
        show_progress=True,
    )


if __name__ == "__main__":
    main()