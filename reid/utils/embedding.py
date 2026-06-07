from typing import Dict
import torch
import torch.nn.functional as F
from ..metrics.similarityMetrics import compute_embedding_retrieval_metrics, compute_embedding_similarity_metrics
from ..utils.helper import log_info, parse_model_output
  

@torch.no_grad()
def validate_embeddings_with_labels(
    model,
    val_loader,
    device: str,
    args,
    logger=None,
):
    extracted = extract_validation_embeddings(
        model=model,
        loader=val_loader,
        device=device,
        embedding_key=args.eval_embedding_key,
        use_amp=args.use_amp,
    )

    embeddings = extracted["embeddings"]
    global_ids = extracted["global_ids"]
    cameras = extracted["cameras"]

    retrieval_metrics = compute_embedding_retrieval_metrics(
        embeddings=embeddings,
        global_ids=global_ids,
        cameras=cameras,
        ranks=(1, 5, 10),
    )

    similarity_metrics = compute_embedding_similarity_metrics(
        embeddings=embeddings,
        global_ids=global_ids,
        cameras=cameras,
        max_pairs=args.max_eval_pairs,
    )

    metrics = {
        **retrieval_metrics,
        **similarity_metrics,
        "num_embeddings": int(embeddings.size(0)),
    }

    log_info(
        logger,
        (
            f"ValEmb | "
            f"N={metrics['num_embeddings']} | "
            f"mAP={metrics['mAP']:.4f} | "
            f"Rank1={metrics['Rank1']:.4f} | "
            f"Rank5={metrics['Rank5']:.4f} | "
            f"Rank10={metrics['Rank10']:.4f} | "
            f"valid_queries={metrics['valid_queries']}"
        ),
    )

    log_info(
        logger,
        (
            f"ValEmb similarity | "
            f"same_id={metrics['same_id_cos_mean']:.4f} | "
            f"same_id_cross_cam={metrics['same_id_cross_camera_cos_mean']:.4f} | "
            f"diff_id={metrics['diff_id_cos_mean']:.4f} | "
            f"separation={metrics['embedding_separation_gap']:.4f} | "
            f"cross_cam_gap={metrics['cross_camera_gap']:.4f}"
        ),
    )

    return metrics

@torch.no_grad()
def extract_validation_embeddings(
    model,
    loader,
    device: str,
    embedding_key: str = "bn_embedding",
    use_amp: bool = True,
):
    model.eval()

    amp_enabled = bool(use_amp and device == "cuda")

    all_embeddings = []
    all_global_ids = []
    all_labels = []
    all_cameras = []
    all_camera_ids = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            outputs = model(images)

        if not isinstance(outputs, dict):
            outputs = parse_model_output(outputs)

        if embedding_key not in outputs:
            raise KeyError(
                f"Embedding key '{embedding_key}' not found. "
                f"Available keys: {list(outputs.keys())}"
            )

        embeddings = outputs[embedding_key].float()
        embeddings = F.normalize(embeddings, p=2, dim=1)

        all_embeddings.append(embeddings.cpu())

        if "identity_key" in batch:
            global_ids = batch["identity_key"]
        elif "global_id" in batch:
            global_ids = batch["global_id"]
        elif "label" in batch:
            global_ids = batch["label"]
        else:
            raise KeyError(
                "Validation batch must contain 'identity_key', 'global_id', or 'label' "
                "for label-aware embedding validation."
            )

        if torch.is_tensor(global_ids):
            all_global_ids.extend([str(x.item()) for x in global_ids.cpu()])
        else:
            all_global_ids.extend([str(x) for x in global_ids])

        if "label" in batch:
            labels = batch["label"]
            if torch.is_tensor(labels):
                all_labels.extend([int(x.item()) for x in labels.cpu()])
            else:
                all_labels.extend([int(x) for x in labels])

        if "camera" in batch:
            cameras = batch["camera"]
            if torch.is_tensor(cameras):
                all_cameras.extend([str(x.item()) for x in cameras.cpu()])
            else:
                all_cameras.extend([str(x) for x in cameras])
        elif "camera_id" in batch:
            cameras = batch["camera_id"]
            if torch.is_tensor(cameras):
                all_cameras.extend([str(x.item()) for x in cameras.cpu()])
            else:
                all_cameras.extend([str(x) for x in cameras])
        else:
            raise KeyError(
                "Validation batch must contain 'camera' or 'camera_id' "
                "for cross-camera embedding validation."
            )

        if "camera_id" in batch:
            camera_ids = batch["camera_id"]
            if torch.is_tensor(camera_ids):
                all_camera_ids.extend([int(x.item()) for x in camera_ids.cpu()])
            else:
                all_camera_ids.extend([int(x) for x in camera_ids])

    embeddings = torch.cat(all_embeddings, dim=0)

    return {
        "embeddings": embeddings,
        "global_ids": all_global_ids,
        "labels": all_labels,
        "cameras": all_cameras,
        "camera_ids": all_camera_ids,
    }