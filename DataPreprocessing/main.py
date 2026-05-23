import os
import shutil
import traceback
from tqdm import tqdm

from DataPreprocessing.extract_crops import extract_crops


DATASET_ROOT = "./DataSet/MTMC_Tracking_2025"
OUTPUT_ROOT = "./DataSet/MTMC_Tracking_2025_Preprocessed"


def find_scenes(dataset_root):
    scenes = []

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(dataset_root, split)

        if not os.path.isdir(split_dir):
            continue

        for scene_name in sorted(os.listdir(split_dir)):
            scene_input = os.path.join(split_dir, scene_name)

            if not os.path.isdir(scene_input):
                continue

            gt_path = os.path.join(scene_input, "ground_truth.json")
            videos_path = os.path.join(scene_input, "videos")

            if os.path.exists(gt_path) and os.path.isdir(videos_path):
                scenes.append((split, scene_name, scene_input))

    return scenes


def is_scene_done(scene_output):
    done_file = os.path.join(scene_output, "_SUCCESS")
    metadata_file = os.path.join(scene_output, "metadata.csv")
    crops_dir = os.path.join(scene_output, "crops")

    return (
        os.path.exists(done_file)
        and os.path.exists(metadata_file)
        and os.path.isdir(crops_dir)
    )


def mark_success(scene_output):
    done_file = os.path.join(scene_output, "_SUCCESS")

    with open(done_file, "w") as f:
        f.write("done\n")


def process_scene(split, scene_name, scene_input):
    scene_output = os.path.join(OUTPUT_ROOT, split, scene_name)
    temp_output = scene_output + "_IN_PROGRESS"

    if is_scene_done(scene_output):
        print(f"[SKIP] Already processed: {split}/{scene_name}")
        return "skipped"

    if os.path.exists(temp_output):
        print(f"[CLEAN] Removing incomplete output: {temp_output}")
        shutil.rmtree(temp_output)

    os.makedirs(os.path.dirname(scene_output), exist_ok=True)

    try:
        extract_crops(
            input_path=scene_input,
            output_path=temp_output,
            frame_step=5,
            small_object_padding=True,
            remove_overlap=True,
            overlap_threshold=0.3,
        )

        if os.path.exists(scene_output):
            shutil.rmtree(scene_output)

        os.rename(temp_output, scene_output)
        mark_success(scene_output)

        print(f"[DONE] {split}/{scene_name}")
        return "done"

    except Exception as e:
        print(f"[ERROR] Failed scene: {split}/{scene_name}")
        print(e)
        traceback.print_exc()

        if os.path.exists(temp_output):
            print(f"[CLEAN] Removing failed output: {temp_output}")
            shutil.rmtree(temp_output)

        return "failed"


def main():
    scenes = find_scenes(DATASET_ROOT)

    print("Total scenes found:", len(scenes))

    results = {
        "done": 0,
        "skipped": 0,
        "failed": 0,
    }

    for split, scene_name, scene_input in tqdm(scenes, desc="Scenes"):
        status = process_scene(split, scene_name, scene_input)
        results[status] += 1

    print("\nSummary:")
    print(results)


if __name__ == "__main__":
    main()