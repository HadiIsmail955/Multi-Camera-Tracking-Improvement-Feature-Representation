import os
import json
import csv
import random
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

def extract_crops(
    input_path,
    output_path,
    frame_step=5,

    min_width=30,
    min_height=30,

    small_object_padding=True,
    small_object_threshold=50,
    small_object_padding_ratio=0.15,

    bbox_format="xyxy",
    save_debug_rejections=True,

    remove_overlap=False,
    overlap_threshold=0.3,

    generate_occlusion_crops=True,
    occlusion_copies=2,
    occlusion_min_width=50,
    occlusion_min_height=50,
    occlusion_min_area=2500,
    occlusion_min_ratio=0.10,
    occlusion_max_ratio=0.30,
    occlusion_mode="background_patch",
    occlusion_seed=42,
):
    print("Started extraction process on:", input_path)

    validate_occlusion_args(
        generate_occlusion_crops=generate_occlusion_crops,
        occlusion_copies=occlusion_copies,
        occlusion_min_ratio=occlusion_min_ratio,
        occlusion_max_ratio=occlusion_max_ratio,
        occlusion_mode=occlusion_mode,
    )

    videos_path, videos, _, ground_truth = extract_inputs(input_path)
    camera_to_frames = build_crop_index(ground_truth, frame_step)

    clean_metadata = []
    occlusion_metadata = []
    rejected = []

    crops_path = os.path.join(output_path, "crops")
    occlusion_crops_path = os.path.join(output_path, "occlusion_crops")
    os.makedirs(crops_path, exist_ok=True)
    os.makedirs(occlusion_crops_path, exist_ok=True)

    rng = random.Random(occlusion_seed)
    np_rng = np.random.default_rng(occlusion_seed)

    for camera_id, frame_to_records in tqdm(
        sorted(camera_to_frames.items()),
        desc="Processing cameras"
    ):
        clean_data, occlusion_data, rejected_data = process_video(
            videos_path=videos_path,
            videos=videos,
            camera_id=camera_id,
            frame_to_records=frame_to_records,
            crops_path=crops_path,
            occlusion_crops_path=occlusion_crops_path,
            min_width=min_width,
            min_height=min_height,
            small_object_padding=small_object_padding,
            small_object_threshold=small_object_threshold,
            small_object_padding_ratio=small_object_padding_ratio,
            bbox_format=bbox_format,
            remove_overlap=remove_overlap,
            overlap_threshold=overlap_threshold,
            generate_occlusion_crops=generate_occlusion_crops,
            occlusion_copies=occlusion_copies,
            occlusion_min_width=occlusion_min_width,
            occlusion_min_height=occlusion_min_height,
            occlusion_min_area=occlusion_min_area,
            occlusion_min_ratio=occlusion_min_ratio,
            occlusion_max_ratio=occlusion_max_ratio,
            occlusion_mode=occlusion_mode,
            rng=rng,
            np_rng=np_rng,
        )
        clean_metadata.extend(clean_data)
        occlusion_metadata.extend(occlusion_data)
        rejected.extend(rejected_data)

    merged_metadata = clean_metadata + occlusion_metadata

    save_metadata(output_path, clean_metadata, filename="normal_metadata.csv")
    save_metadata(output_path, occlusion_metadata, filename="occlusion_metadata.csv")
    save_metadata(output_path, merged_metadata, filename="metadata.csv")

    if save_debug_rejections:
        save_rejected_metadata(output_path, rejected)

    print("Done.")
    print("Saved normal/original crops:", len(clean_metadata))
    print("Saved occlusion crop images:", len(occlusion_metadata))
    print("Saved normal metadata rows:", len(clean_metadata))
    print("Saved occlusion metadata rows:", len(occlusion_metadata))
    print("Saved combined metadata rows:", len(clean_metadata) + len(occlusion_metadata))
    print("Rejected boxes:", len(rejected))


def validate_occlusion_args(
    generate_occlusion_crops,
    occlusion_copies,
    occlusion_min_ratio,
    occlusion_max_ratio,
    occlusion_mode,
):
    allowed_modes = {"black", "gray", "noise", "blur", "random_patch", "background_patch"}

    if occlusion_mode not in allowed_modes:
        raise ValueError(f"occlusion_mode must be one of: {sorted(allowed_modes)}")

    if occlusion_copies < 0:
        raise ValueError("occlusion_copies must be >= 0")

    if not 0.0 <= occlusion_min_ratio <= 1.0:
        raise ValueError("occlusion_min_ratio must be between 0 and 1")

    if not 0.0 <= occlusion_max_ratio <= 1.0:
        raise ValueError("occlusion_max_ratio must be between 0 and 1")

    if occlusion_min_ratio > occlusion_max_ratio:
        raise ValueError("occlusion_min_ratio cannot be greater than occlusion_max_ratio")

    if generate_occlusion_crops and occlusion_copies == 0:
        print("[WARN] generate_occlusion_crops=True but occlusion_copies=0, so no synthetic copies will be created.")


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
    occlusion_crops_path,
    min_width,
    min_height,
    small_object_padding=True,
    small_object_threshold=50,
    small_object_padding_ratio=0.15,
    bbox_format="xyxy",
    remove_overlap=False,
    overlap_threshold=0.3,
    generate_occlusion_crops=True,
    occlusion_copies=2,
    occlusion_min_width=50,
    occlusion_min_height=50,
    occlusion_min_area=2500,
    occlusion_min_ratio=0.10,
    occlusion_max_ratio=0.30,
    occlusion_mode="background_patch",
    rng=None,
    np_rng=None,
):
    if rng is None:
        rng = random.Random(42)
    if np_rng is None:
        np_rng = np.random.default_rng(42)

    video_file = find_video_for_camera(videos_path, videos, camera_id)

    if video_file is None:
        print(f"[WARN] No video found for camera {camera_id}")
        return [], [], []

    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_file}")
        return [], [], []

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    needed_frames = sorted(frame_to_records.keys())

    if not needed_frames:
        cap.release()
        return [], [], []

    max_needed_frame = max(needed_frames)

    clean_metadata = []
    occlusion_metadata = []
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

            # First pass: validate, optionally pad, and collect boxes.
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

                ox1, oy1, ox2, oy2 = clipped
                original_w = ox2 - ox1
                original_h = oy2 - oy1

                x1, y1, x2, y2 = ox1, oy1, ox2, oy2
                padded = False

                # Important: padding happens BEFORE final min-size rejection.
                is_small_object = (
                    original_w < small_object_threshold
                    or original_h < small_object_threshold
                )

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

                final_w = x2 - x1
                final_h = y2 - y1

                if final_w < min_width or final_h < min_height:
                    rejected.append(
                        make_rejection(record, "too_small_after_optional_padding", width, height, x1, y1, x2, y2)
                    )
                    continue

                valid_frame_boxes.append({
                    "record_index": record_index,
                    "record": record,
                    "box": (x1, y1, x2, y2),
                    "overlap_box": (ox1, oy1, ox2, oy2),
                    "original_w": original_w,
                    "original_h": original_h,
                    "padding_applied": padded,
                })

            for item in valid_frame_boxes:
                record = item["record"]
                record_index = item["record_index"]

                x1, y1, x2, y2 = item["box"]
                original_w = item["original_w"]
                original_h = item["original_h"]
                padded = item["padding_applied"]

                naturally_occluded, max_overlap = is_box_occluded(
                    target_item=item,
                    all_items=valid_frame_boxes,
                    overlap_threshold=overlap_threshold,
                )

                if remove_overlap and naturally_occluded:
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

                base_name = f"{global_id}_{camera_id}_f{frame_id:06d}_r{record_index:03d}"
                crop_name = f"{base_name}.jpg"
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

                clean_metadata.append(
                    make_metadata_row(
                        crop_path=crop_path,
                        record=record,
                        frame_id=frame_id,
                        camera_id=camera_id,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        original_w=original_w,
                        original_h=original_h,
                        padded=padded,
                        max_overlap=max_overlap,
                        bbox_format=bbox_format,
                        augmentation="none",
                        data_subset="normal",
                        source_crop_path=crop_path,
                        natural_occlusion=naturally_occluded,
                        synthetic_occlusion_ratio=0.0,
                        synthetic_occlusion_mode="none",
                    )
                )

                if generate_occlusion_crops and should_generate_occlusion(
                    crop,
                    occlusion_min_width=occlusion_min_width,
                    occlusion_min_height=occlusion_min_height,
                    occlusion_min_area=occlusion_min_area,
                ):
                    for copy_idx in range(1, occlusion_copies + 1):
                        occ_crop, occ_ratio, occ_info = apply_synthetic_occlusion(
                            crop=crop,
                            min_ratio=occlusion_min_ratio,
                            max_ratio=occlusion_max_ratio,
                            mode=occlusion_mode,
                            rng=rng,
                            np_rng=np_rng,
                            frame=frame,
                            crop_box=(x1, y1, x2, y2),
                            all_items=valid_frame_boxes,
                        )

                        occ_save_dir = os.path.join(occlusion_crops_path, global_id, camera_id)
                        os.makedirs(occ_save_dir, exist_ok=True)

                        occ_name = f"{base_name}_occ{copy_idx:02d}.jpg"
                        occ_path = os.path.join(occ_save_dir, occ_name)

                        ok_occ_write = cv2.imwrite(occ_path, occ_crop)

                        if not ok_occ_write:
                            rejected.append(
                                make_rejection(
                                    record,
                                    "cv2_write_failed_synthetic_occlusion",
                                    width,
                                    height,
                                    x1,
                                    y1,
                                    x2,
                                    y2,
                                )
                            )
                            continue

                        occlusion_metadata.append(
                            make_metadata_row(
                                crop_path=occ_path,
                                record=record,
                                frame_id=frame_id,
                                camera_id=camera_id,
                                x1=x1,
                                y1=y1,
                                x2=x2,
                                y2=y2,
                                original_w=original_w,
                                original_h=original_h,
                                padded=padded,
                                max_overlap=max_overlap,
                                bbox_format=bbox_format,
                                augmentation="synthetic_occlusion",
                                data_subset="occlusion",
                                source_crop_path=crop_path,
                                natural_occlusion=naturally_occluded,
                                synthetic_occlusion_ratio=occ_ratio,
                                synthetic_occlusion_mode=occlusion_mode,
                                synthetic_occlusion_x1=occ_info["occ_x1"],
                                synthetic_occlusion_y1=occ_info["occ_y1"],
                                synthetic_occlusion_x2=occ_info["occ_x2"],
                                synthetic_occlusion_y2=occ_info["occ_y2"],
                                occluder_source=occ_info["occluder_source"],
                                background_source_x1=occ_info.get("source_x1"),
                                background_source_y1=occ_info.get("source_y1"),
                                background_source_x2=occ_info.get("source_x2"),
                                background_source_y2=occ_info.get("source_y2"),
                            )
                        )

        frame_id += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    return clean_metadata, occlusion_metadata, rejected


def should_generate_occlusion(
    crop,
    occlusion_min_width=50,
    occlusion_min_height=50,
    occlusion_min_area=2500,
):
    crop_h, crop_w = crop.shape[:2]
    crop_area = crop_w * crop_h

    return (
        crop_w >= occlusion_min_width
        and crop_h >= occlusion_min_height
        and crop_area >= occlusion_min_area
    )


def apply_synthetic_occlusion(
    crop,
    min_ratio=0.10,
    max_ratio=0.30,
    mode="background_patch",
    rng=None,
    np_rng=None,
    frame=None,
    crop_box=None,
    all_items=None,
    background_attempts=80,
):
    if rng is None:
        rng = random.Random()
    if np_rng is None:
        np_rng = np.random.default_rng()

    out = crop.copy()
    h, w = out.shape[:2]

    empty_info = {
        "occ_x1": None,
        "occ_y1": None,
        "occ_x2": None,
        "occ_y2": None,
        "occluder_source": "none",
        "source_x1": None,
        "source_y1": None,
        "source_x2": None,
        "source_y2": None,
    }

    if h <= 1 or w <= 1:
        return out, 0.0, empty_info

    ratio = rng.uniform(min_ratio, max_ratio)
    target_area = max(1, int(round(w * h * ratio)))

    aspect = rng.uniform(0.5, 2.0)
    occ_w = int(round((target_area * aspect) ** 0.5))
    occ_h = int(round((target_area / aspect) ** 0.5))

    occ_w = max(1, min(w, occ_w))
    occ_h = max(1, min(h, occ_h))

    x1 = rng.randint(0, max(0, w - occ_w))
    y1 = rng.randint(0, max(0, h - occ_h))
    x2 = x1 + occ_w
    y2 = y1 + occ_h

    info = {
        "occ_x1": x1,
        "occ_y1": y1,
        "occ_x2": x2,
        "occ_y2": y2,
        "occluder_source": mode,
        "source_x1": None,
        "source_y1": None,
        "source_x2": None,
        "source_y2": None,
    }

    if mode == "black":
        out[y1:y2, x1:x2] = 0

    elif mode == "gray":
        # Per-crop mean color hides information without introducing foreign identity cues.
        mean_color = out.reshape(-1, out.shape[-1]).mean(axis=0)
        out[y1:y2, x1:x2] = mean_color.astype(out.dtype)

    elif mode == "noise":
        noise = np_rng.integers(
            low=0,
            high=256,
            size=(occ_h, occ_w, out.shape[2]),
            dtype=out.dtype,
        )
        out[y1:y2, x1:x2] = noise

    elif mode == "blur":
        kernel = max(3, int(round(min(w, h) * 0.15)))
        if kernel % 2 == 0:
            kernel += 1
        blurred = cv2.GaussianBlur(out, (kernel, kernel), sigmaX=0)
        out[y1:y2, x1:x2] = blurred[y1:y2, x1:x2]

    elif mode == "random_patch":
        sx1 = rng.randint(0, max(0, w - occ_w))
        sy1 = rng.randint(0, max(0, h - occ_h))
        sx2 = sx1 + occ_w
        sy2 = sy1 + occ_h
        out[y1:y2, x1:x2] = out[sy1:sy2, sx1:sx2]
        info["occluder_source"] = "same_crop_patch"
        info["source_x1"] = sx1
        info["source_y1"] = sy1
        info["source_x2"] = sx2
        info["source_y2"] = sy2

    elif mode == "background_patch":
        patch, source_box = sample_background_patch(
            frame=frame,
            patch_w=occ_w,
            patch_h=occ_h,
            crop_box=crop_box,
            all_items=all_items,
            rng=rng,
            attempts=background_attempts,
        )

        if patch is not None:
            out[y1:y2, x1:x2] = patch
            info["occluder_source"] = "background_patch"
            info["source_x1"], info["source_y1"], info["source_x2"], info["source_y2"] = source_box
        else:
            # Safe fallback: hide visual information without adding another person's identity.
            mean_color = out.reshape(-1, out.shape[-1]).mean(axis=0)
            out[y1:y2, x1:x2] = mean_color.astype(out.dtype)
            info["occluder_source"] = "background_patch_fallback_gray"

    else:
        raise ValueError(f"Unsupported occlusion mode: {mode}")

    actual_ratio = (occ_w * occ_h) / max(1, w * h)
    return out, actual_ratio, info


def sample_background_patch(
    frame,
    patch_w,
    patch_h,
    crop_box=None,
    all_items=None,
    rng=None,
    attempts=80,
):
    if frame is None:
        return None, None
    if rng is None:
        rng = random.Random()

    frame_h, frame_w = frame.shape[:2]
    if patch_w <= 0 or patch_h <= 0 or patch_w > frame_w or patch_h > frame_h:
        return None, None

    forbidden_boxes = []

    if all_items is not None:
        for item in all_items:
            box = item.get("overlap_box", item.get("box"))
            if box is not None:
                forbidden_boxes.append(tuple(int(v) for v in box))

    if crop_box is not None:
        forbidden_boxes.append(tuple(int(v) for v in crop_box))

    for _ in range(attempts):
        sx1 = rng.randint(0, max(0, frame_w - patch_w))
        sy1 = rng.randint(0, max(0, frame_h - patch_h))
        sx2 = sx1 + patch_w
        sy2 = sy1 + patch_h
        candidate = (sx1, sy1, sx2, sy2)

        if not any(boxes_intersect(candidate, box) for box in forbidden_boxes):
            patch = frame[sy1:sy2, sx1:sx2].copy()
            if patch.shape[0] == patch_h and patch.shape[1] == patch_w:
                return patch, candidate

    return None, None


def boxes_intersect(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return max(ax1, bx1) < min(ax2, bx2) and max(ay1, by1) < min(ay2, by2)


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


def add_adaptive_padding(x1, y1, x2, y2, width, height, padding_ratio=0.15):
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


def make_metadata_row(
    crop_path,
    record,
    frame_id,
    camera_id,
    x1,
    y1,
    x2,
    y2,
    original_w,
    original_h,
    padded,
    max_overlap,
    bbox_format,
    augmentation="none",
    data_subset="normal",
    source_crop_path="",
    natural_occlusion=False,
    synthetic_occlusion_ratio=0.0,
    synthetic_occlusion_mode="none",
    synthetic_occlusion_x1=None,
    synthetic_occlusion_y1=None,
    synthetic_occlusion_x2=None,
    synthetic_occlusion_y2=None,
    occluder_source="none",
    background_source_x1=None,
    background_source_y1=None,
    background_source_x2=None,
    background_source_y2=None,
):
    return {
        "crop_path": crop_path,
        "data_subset": data_subset,
        "source_crop_path": source_crop_path,
        "frame": frame_id,
        "object_type": record["object_type"],
        "object_id": record["object_id"],
        "global_id": record["global_id"],
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
        "natural_occlusion": natural_occlusion,
        "max_overlap_ratio": max_overlap,
        "augmentation": augmentation,
        "synthetic_occlusion_ratio": synthetic_occlusion_ratio,
        "synthetic_occlusion_mode": synthetic_occlusion_mode,
        "synthetic_occlusion_x1": synthetic_occlusion_x1,
        "synthetic_occlusion_y1": synthetic_occlusion_y1,
        "synthetic_occlusion_x2": synthetic_occlusion_x2,
        "synthetic_occlusion_y2": synthetic_occlusion_y2,
        "occluder_source": occluder_source,
        "background_source_x1": background_source_x1,
        "background_source_y1": background_source_y1,
        "background_source_x2": background_source_x2,
        "background_source_y2": background_source_y2,
        "raw_bbox": json.dumps(record["bbox"]),
        "bbox_format_used": bbox_format,
        "world_x": record["world_x"],
        "world_y": record["world_y"],
        "world_z": record["world_z"],
    }


def save_metadata(output_path, metadata, filename="metadata.csv"):
    os.makedirs(output_path, exist_ok=True)

    metadata_path = os.path.join(output_path, filename)

    fieldnames = [
        "crop_path",
        "data_subset",
        "source_crop_path",
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
        "natural_occlusion",
        "max_overlap_ratio",
        "augmentation",
        "synthetic_occlusion_ratio",
        "synthetic_occlusion_mode",
        "synthetic_occlusion_x1",
        "synthetic_occlusion_y1",
        "synthetic_occlusion_x2",
        "synthetic_occlusion_y2",
        "occluder_source",
        "background_source_x1",
        "background_source_y1",
        "background_source_x2",
        "background_source_y2",
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
    # Use original unpadded boxes for natural occlusion estimation.
    target_box = target_item.get("overlap_box", target_item["box"])
    max_overlap = 0.0

    for other_item in all_items:
        if other_item is target_item:
            continue

        other_box = other_item.get("overlap_box", other_item["box"])
        overlap = bbox_overlap_ratio(target_box, other_box)

        max_overlap = max(max_overlap, overlap)

        if overlap >= overlap_threshold:
            return True, max_overlap

    return False, max_overlap