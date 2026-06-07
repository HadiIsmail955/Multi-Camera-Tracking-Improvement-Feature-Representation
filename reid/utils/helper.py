from typing import Dict, Tuple

import torch
import torch.nn.functional as F

def log_info(logger, message: str):
    if logger is not None:
        logger.info(message)
    else:
        print(message, flush=True)

def move_batch_to_device(batch, device: str):
    images = batch["image"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True).long()
    cameras = batch["camera_id"].to(device, non_blocking=True).long()

    if "is_occluded" in batch:
        is_occluded = batch["is_occluded"].to(device, non_blocking=True).long()
    else:
        is_occluded = torch.zeros_like(labels, dtype=torch.long, device=device)

    return images, labels, cameras, is_occluded

def parse_model_output(model_output):
    if isinstance(model_output, dict):
        if "embedding" not in model_output:
            raise KeyError("Model output dict is missing key: 'embedding'")
        if "logits" not in model_output:
            raise KeyError("Model output dict is missing key: 'logits'")
        return model_output

    if isinstance(model_output, (tuple, list)):
        if len(model_output) != 2:
            raise ValueError(
                f"Expected model output tuple/list of length 2, got {len(model_output)}"
            )
        embeddings, logits = model_output
        return {
            "embedding": embeddings,
            "logits": logits,
        }

    raise TypeError(
        f"Unsupported model output type: {type(model_output)}. "
        "Expected dict, tuple, or list."
    )

def tensor_outputs_to_float(outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        key: value.float() if torch.is_tensor(value) else value
        for key, value in outputs.items()
    }