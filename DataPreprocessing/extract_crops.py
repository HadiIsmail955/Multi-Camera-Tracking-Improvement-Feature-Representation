import os
import json
import csv
from collections import defaultdict

import cv2
from tqdm import tqdm


def extract_crops(
    input_path,
    output_path,
    frame_step=5,
    min_width=10,
    min_height=10,
    small_object_padding=False,
    small_object_threshold=40,
    small_object_padding_ratio=0.1,
    bbox_format="xyxy",  
    save_debug_rejections=True,
    remove_overlap=True,
    overlap_threshold=0.3
):
    print("Started extraction process on:", input_path)

    videos_path, videos, _, ground_truth = extract_inputs(input_path)
    camera_to_frames = build_crop_index(ground_truth, frame_step)

    metadata = []
    rejected = []

    crops_path = os.path.join(output_path, "crops")
    os.makedirs(crops_path, exist_ok=True)

    for camera_id, frame_to_records in tqdm(
        sorted(camera_to_frames.items()),
        desc="Processing cameras"
    ):
        data, rejected_data = process_video(
            videos_path=videos_path,
            videos=videos,
            camera_id=camera_id,
            frame_to_records=frame_to_records,
            crops_path=crops_path,
            min_width=min_width,
            min_height=min_height,
            small_object_padding=small_object_padding,
            small_object_threshold=small_object_threshold,
            small_object_padding_ratio=small_object_padding_ratio,
            bbox_format=bbox_format,
            remove_overlap=remove_overlap,
            overlap_threshold=overlap_threshold,
        )
        metadata.extend(data)
        rejected.extend(rejected_data)

    save_metadata(output_path, metadata)

    if save_debug_rejections:
        save_rejected_metadata(output_path, rejected)

    print("Done.")
    print("Saved crops:", len(metadata))
    print("Rejected boxes:", len(rejected))


def extract_inputs(input_path):
    videos_path = os.path.join(input_path, "videos")

    if not os.path.exists(videos_path):
        raise FileNotFoundError(f"Videos path does not exist: {videos_path}")

    videos = [
        f for f in os.listdir(videos_path)
        if f.lower().endswith(".mp4")
    ]

    if not videos:
        raise ValueError(f"No video files found in: {videos_path}")

    ground_truth_path = os.path.join(input_path, "ground_truth.json")

    if not os.path.exists(ground_truth_path):
        raise FileNotFoundError(f"Ground truth file does not exist: {ground_truth_path}")

    with open(ground_truth_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    return videos_path, videos, ground_truth_path, ground_truth


def build_crop_index(ground_truth, frame_step):
    camera_to_frames = defaultdict(lambda: defaultdict(list))

    for frame_str, objects in ground_truth.items():
        try:
            frame_id = int(frame_str)
        except ValueError:
            print(f"[WARN] Invalid frame key skipped: {frame_str}")
            continue

        if frame_id % frame_step != 0:
            continue

        if not isinstance(objects, list):
            print(f"[WARN] Frame {frame_id} has invalid object list")
            continue

        for obj in objects:
            obj_type = obj.get("object type", "unknown")

            try:
                obj_id = int(obj.get("object id"))
            except (TypeError, ValueError):
                print(f"[WARN] Invalid object id in frame {frame_id}: {obj.get('object id')}")
                continue

            global_id = f"{obj_type}_{obj_id:04d}"

            world_x, world_y, world_z = obj.get("3d location", [None, None, None])

            visible_boxes = obj.get("2d bounding box visible", {})
            if not isinstance(visible_boxes, dict):
                continue

            for camera_id, bbox in visible_boxes.items():
                if bbox is None or len(bbox) != 4:
                    continue

                camera_to_frames[str(camera_id)][frame_id].append({
                    "frame": frame_id,
                    "object_type": obj_type,
                    "object_id": obj_id,
                    "global_id": global_id,
                    "camera": str(camera_id),
                    "bbox": bbox,
                    "world_x": world_x,
                    "world_y": world_y,
                    "world_z": world_z,
                })

    return camera_to_frames


def process_video(
    videos_path,
    videos,
    camera_id,
    frame_to_records,
    crops_path,
    min_width,
    min_height,
    small_object_padding=False,
    small_object_threshold=40,
    small_object_padding_ratio=0.1,
    bbox_format="xyxy",
    remove_overlap=True,
    overlap_threshold=0.3,
):
    video_file = find_video_for_camera(videos_path, videos, camera_id)

    if video_file is None:
        print(f"[WARN] No video found for camera {camera_id}")
        return [], []

    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_file}")
        return [], []

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    needed_frames = sorted(frame_to_records.keys())

    if not needed_frames:
        cap.release()
        return [], []

    max_needed_frame = max(needed_frames)

    metadata = []
    rejected = []
    frame_id = 0

    pbar = tqdm(
        total=min(max_needed_frame + 1, total_frames if total_frames > 0 else max_needed_frame + 1),
        desc=f"{camera_id}",
        leave=False,
    )

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        if frame_id > max_needed_frame:
            break

        if frame_id in frame_to_records:
            records = frame_to_records[frame_id]

            valid_frame_boxes = []

            for record_index, record in enumerate(records):
                clipped = convert_and_clip_bbox(
                    record["bbox"],
                    width=width,
                    height=height,
                    bbox_format=bbox_format,
                )

                if clipped is None:
                    rejected.append(
                        make_rejection(record, "invalid_or_outside_bbox", width, height)
                    )
                    continue

                x1, y1, x2, y2 = clipped
                original_w = x2 - x1
                original_h = y2 - y1

                if original_w < min_width or original_h < min_height:
                    rejected.append(
                        make_rejection(record, "too_small", width, height, x1, y1, x2, y2)
                    )
                    continue

                valid_frame_boxes.append({
                    "record_index": record_index,
                    "record": record,
                    "box": (x1, y1, x2, y2),
                    "original_w": original_w,
                    "original_h": original_h,
                })

            for item in valid_frame_boxes:
                record = item["record"]
                record_index = item["record_index"]

                x1, y1, x2, y2 = item["box"]
                original_w = item["original_w"]
                original_h = item["original_h"]

                if remove_overlap:
                    occluded, max_overlap = is_box_occluded(
                        target_item=item,
                        all_items=valid_frame_boxes,
                        overlap_threshold=overlap_threshold,
                    )

                    if occluded:
                        rejected.append(
                            make_rejection(
                                record,
                                f"overlap_occlusion_{max_overlap:.3f}",
                                width,
                                height,
                                x1,
                                y1,
                                x2,
                                y2,
                            )
                        )
                        continue
                else:
                    max_overlap = 0.0

                is_small_object = (
                    original_w < small_object_threshold
                    or original_h < small_object_threshold
                )

                padded = False

                if small_object_padding and is_small_object:
                    x1, y1, x2, y2 = add_adaptive_padding(
                        x1,
                        y1,
                        x2,
                        y2,
                        width,
                        height,
                        padding_ratio=small_object_padding_ratio,
                    )
                    padded = True

                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    rejected.append(
                        make_rejection(
                            record,
                            "empty_crop_after_slicing",
                            width,
                            height,
                            x1,
                            y1,
                            x2,
                            y2,
                        )
                    )
                    continue

                global_id = record["global_id"]

                save_dir = os.path.join(crops_path, global_id, camera_id)
                os.makedirs(save_dir, exist_ok=True)

                crop_name = f"{global_id}_{camera_id}_f{frame_id:06d}_r{record_index:03d}.jpg"
                crop_path = os.path.join(save_dir, crop_name)

                ok_write = cv2.imwrite(crop_path, crop)

                if not ok_write:
                    rejected.append(
                        make_rejection(
                            record,
                            "cv2_write_failed",
                            width,
                            height,
                            x1,
                            y1,
                            x2,
                            y2,
                        )
                    )
                    continue

                metadata.append({
                    "crop_path": crop_path,
                    "frame": frame_id,
                    "object_type": record["object_type"],
                    "object_id": record["object_id"],
                    "global_id": global_id,
                    "camera": camera_id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "bbox_width": x2 - x1,
                    "bbox_height": y2 - y1,
                    "original_bbox_width": original_w,
                    "original_bbox_height": original_h,
                    "padding_applied": padded,
                    "max_overlap_ratio": max_overlap,
                    "raw_bbox": json.dumps(record["bbox"]),
                    "bbox_format_used": bbox_format,
                    "world_x": record["world_x"],
                    "world_y": record["world_y"],
                    "world_z": record["world_z"],
                })

        frame_id += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    return metadata, rejected


def convert_and_clip_bbox(bbox, width, height, bbox_format="xyxy"):
    try:
        a, b, c, d = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None

    if bbox_format not in {"xyxy", "xywh", "auto"}:
        raise ValueError("bbox_format must be one of: 'xyxy', 'xywh', 'auto'")

    if bbox_format == "xywh":
        x1, y1, x2, y2 = a, b, a + c, b + d

    elif bbox_format == "xyxy":
        x1, y1, x2, y2 = a, b, c, d

    else:
        xyxy_valid = c > a and d > b
        xywh_valid = c > 0 and d > 0 and (a + c) > a and (b + d) > b

        if xyxy_valid:
            x1, y1, x2, y2 = a, b, c, d
        elif xywh_valid:
            x1, y1, x2, y2 = a, b, a + c, b + d
        else:
            return None

    x1 = int(round(x1))
    y1 = int(round(y1))
    x2 = int(round(x2))
    y2 = int(round(y2))

    if x2 <= x1 or y2 <= y1:
        return None

    if x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
        return None

    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def add_adaptive_padding(x1, y1, x2, y2, width, height, padding_ratio=0.1):
    box_w = x2 - x1
    box_h = y2 - y1

    pad_x = int(round(box_w * padding_ratio))
    pad_y = int(round(box_h * padding_ratio))

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)

    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)

    return x1, y1, x2, y2


def find_video_for_camera(videos_path, videos, camera_id):
    expected_name = f"{camera_id}.mp4"

    if expected_name in videos:
        return os.path.join(videos_path, expected_name)

    normalized_camera = str(camera_id).lower()

    for video in videos:
        stem = os.path.splitext(video)[0].lower()
        if stem == normalized_camera:
            return os.path.join(videos_path, video)

    for video in videos:
        stem = os.path.splitext(video)[0].lower()
        parts = stem.replace("-", "_").split("_")
        if normalized_camera in parts:
            return os.path.join(videos_path, video)

    return None


def make_rejection(record, reason, image_width, image_height, x1=None, y1=None, x2=None, y2=None):
    return {
        "reason": reason,
        "frame": record.get("frame"),
        "object_type": record.get("object_type"),
        "object_id": record.get("object_id"),
        "global_id": record.get("global_id"),
        "camera": record.get("camera"),
        "raw_bbox": json.dumps(record.get("bbox")),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "image_width": image_width,
        "image_height": image_height,
    }


def save_metadata(output_path, metadata):
    os.makedirs(output_path, exist_ok=True)

    metadata_path = os.path.join(output_path, "metadata.csv")

    fieldnames = [
        "crop_path",
        "frame",
        "object_type",
        "object_id",
        "global_id",
        "camera",
        "x1",
        "y1",
        "x2",
        "y2",
        "bbox_width",
        "bbox_height",
        "original_bbox_width",
        "original_bbox_height",
        "padding_applied",
        "max_overlap_ratio",
        "raw_bbox",
        "bbox_format_used",
        "world_x",
        "world_y",
        "world_z",
    ]

    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metadata)


def save_rejected_metadata(output_path, rejected):
    os.makedirs(output_path, exist_ok=True)

    rejected_path = os.path.join(output_path, "rejected_crops.csv")

    fieldnames = [
        "reason",
        "frame",
        "object_type",
        "object_id",
        "global_id",
        "camera",
        "raw_bbox",
        "x1",
        "y1",
        "x2",
        "y2",
        "image_width",
        "image_height",
    ]

    with open(rejected_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rejected)

def bbox_overlap_ratio(target_box, other_box):
    tx1, ty1, tx2, ty2 = target_box
    ox1, oy1, ox2, oy2 = other_box

    ix1 = max(tx1, ox1)
    iy1 = max(ty1, oy1)
    ix2 = min(tx2, ox2)
    iy2 = min(ty2, oy2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter_area = inter_w * inter_h

    target_area = max(1, (tx2 - tx1) * (ty2 - ty1))

    return inter_area / target_area

def is_box_occluded(target_item, all_items, overlap_threshold=0.3):
    target_box = target_item["box"]
    max_overlap = 0.0

    for other_item in all_items:
        if other_item is target_item:
            continue

        other_box = other_item["box"]
        overlap = bbox_overlap_ratio(target_box, other_box)

        max_overlap = max(max_overlap, overlap)

        if overlap >= overlap_threshold:
            return True, max_overlap

    return False, max_overlap