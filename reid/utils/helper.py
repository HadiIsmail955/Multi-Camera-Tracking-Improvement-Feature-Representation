from typing import Dict, Tuple
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

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

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if torch.is_tensor(value):
        return value.detach().cpu().tolist()

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    return value


def save_metrics(metrics: Dict, out_dir: Path):
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(metrics), f, indent=2)

    flat = {}

    def flatten(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                flatten(f"{prefix}_{k}" if prefix else str(k), v)
        elif not isinstance(obj, (list, tuple)):
            flat[prefix] = obj

    flatten("", metrics)
    pd.DataFrame([flat]).to_csv(out_dir / "metrics.csv", index=False)


def get_batch_value(batch, key: str, i: int, default=None):
    if key not in batch:
        return default

    value = batch[key]

    if torch.is_tensor(value):
        x = value[i]

        if x.ndim == 0:
            return x.item()

        return x.detach().cpu().numpy().tolist()

    if isinstance(value, (list, tuple)):
        return value[i]

    return value


def safe_int(value, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def safe_torch_load(path: str, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]

        if new_key.startswith("model."):
            new_key = new_key[len("model.") :]

        cleaned[new_key] = value

    return cleaned
