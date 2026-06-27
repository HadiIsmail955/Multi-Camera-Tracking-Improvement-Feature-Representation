import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize

from .MTMCCSVDataset import MTMCCSVDataset
from ..transformation.ReIDTransform import ReIDTransform

from reid.utils.helper import get_batch_value, safe_int


def build_dataset(args):
    transform = ReIDTransform(
        backbone=args.backbone,
        img_size=args.image_size,
        train=False,
    )

    dataset = MTMCCSVDataset(
        root=args.data_root,
        split=args.split,
        scene_folders=args.scenes,
        transform=transform,
        min_images_per_id=args.min_images_per_id,
        min_normal_images_per_id=args.min_normal_images_per_id,
        object_types=args.object_types,
        base_path=args.base_path,
        debug=args.debug_dataset,
        verify_paths=args.verify_paths,
        scene_aware_ids=args.scene_aware_ids,
        min_cameras_per_id=args.min_cameras_per_id,
        include_occlusion_crops=args.include_occlusion_crops,
        only_occlusion_crops=args.only_occlusion_crops,
        metadata_filename=args.metadata_filename,
    )

    return dataset


@torch.no_grad()
def extract_crop_embeddings(
    model,
    loader,
    device: str,
    embedding_key: str = "bn_embedding",
    use_amp: bool = True,
):
    model.eval()
    amp_enabled = bool(use_amp and device == "cuda")

    all_embeddings = []
    rows = []

    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            outputs = model(images)

        if not isinstance(outputs, dict):
            raise TypeError("Expected model(images) to return a dict containing embeddings.")

        if embedding_key not in outputs:
            raise KeyError(
                f"Embedding key '{embedding_key}' not found. Available keys: {list(outputs.keys())}"
            )

        embeddings = outputs[embedding_key].detach().float()
        embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings_np = embeddings.cpu().numpy().astype(np.float32)

        all_embeddings.append(embeddings_np)

        batch_size = embeddings_np.shape[0]

        for i in range(batch_size):
            global_id = get_batch_value(batch, "global_id", i, default=None)
            label = get_batch_value(batch, "label", i, default=None)
            scene = str(get_batch_value(batch, "scene", i, default="unknown"))

            if global_id is None:
                global_id = label

            identity_key = get_batch_value(batch, "identity_key", i, default=None)
            if identity_key is None or str(identity_key).lower() in {"none", "nan", ""}:
                identity_key = f"{scene}__{global_id}"

            camera = str(
                get_batch_value(
                    batch,
                    "camera",
                    i,
                    default=get_batch_value(batch, "camera_id", i, default="unknown"),
                )
            )

            track_id = get_batch_value(
                batch,
                "track_id",
                i,
                default=get_batch_value(batch, "tracklet_id", i, default=None),
            )

            rows.append(
                {
                    "row_index": len(rows),
                    "label": safe_int(label),
                    "global_id": str(global_id),
                    "identity_key": str(identity_key),
                    "camera": camera,
                    "camera_id": safe_int(get_batch_value(batch, "camera_id", i, default=-1)),
                    "frame": safe_int(get_batch_value(batch, "frame", i, default=-1)),
                    "object_type": str(get_batch_value(batch, "object_type", i, default="unknown")),
                    "scene": scene,
                    "is_occluded": safe_int(get_batch_value(batch, "is_occluded", i, default=0), default=0),
                    "data_subset": str(get_batch_value(batch, "data_subset", i, default="unknown")),
                    "augmentation": str(get_batch_value(batch, "augmentation", i, default="none")),
                    "crop_path": str(get_batch_value(batch, "crop_path", i, default="")),
                    "source_crop_path": str(get_batch_value(batch, "source_crop_path", i, default="")),
                    "track_id": None if track_id is None else str(track_id),
                }
            )

        if batch_idx == 1 or batch_idx % 20 == 0 or batch_idx == len(loader):
            print(f"Extracting embeddings: batch {batch_idx}/{len(loader)}")

    if len(all_embeddings) == 0:
        raise RuntimeError("No embeddings extracted. Check the dataset and loader.")

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    embeddings = normalize(embeddings)

    df = pd.DataFrame(rows)

    true_ids = sorted(df["identity_key"].astype(str).unique())
    id_to_label = {gid: idx for idx, gid in enumerate(true_ids)}
    df["true_label"] = df["identity_key"].astype(str).map(id_to_label).astype(int)

    return embeddings, df
