from typing import Dict

import torch

from reid.model.DINOv2ReID import DINOv2ReID
from ..utils.helper import safe_torch_load, clean_state_dict_keys


def _args_to_dict(args_like):
    if args_like is None:
        return {}

    if isinstance(args_like, dict):
        return args_like

    if hasattr(args_like, "__dict__"):
        return vars(args_like)

    return {}


def get_state_dict_from_checkpoint(ckpt):
    if isinstance(ckpt, dict):
        for key in ["model", "model_state_dict", "state_dict", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]

        if all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt

    raise ValueError(
        "Could not find a model state_dict. Expected key: model, model_state_dict, state_dict, net, or raw state_dict."
    )


def infer_model_args_from_checkpoint(ckpt: Dict, cli_args):
    ckpt_args = _args_to_dict(ckpt.get("args", {}) if isinstance(ckpt, dict) else {})

    def pick(name, default=None):
        cli_value = getattr(cli_args, name, None)

        if cli_value is not None:
            return cli_value

        return ckpt_args.get(name, default)

    return {
        "embedding_dim": int(pick("embedding_dim", 512)),
        "backbone_type": pick("backbone_type", "vit_b"),
        "dropout": float(pick("dropout", 0.1)),
    }


def get_num_classes_from_checkpoint(ckpt: Dict, state_dict: Dict[str, torch.Tensor]) -> int:
    if isinstance(ckpt, dict):
        for key in ["id_to_label", "label_to_id"]:
            if key in ckpt and ckpt[key] is not None:
                return len(ckpt[key])

        args = _args_to_dict(ckpt.get("args", {}))
        if "num_classes" in args:
            return int(args["num_classes"])

    if "classifier.weight" in state_dict:
        return int(state_dict["classifier.weight"].shape[0])

    raise KeyError(
        "Could not infer num_classes from checkpoint. Expected id_to_label, label_to_id, args['num_classes'], or classifier.weight."
    )


def load_model_from_checkpoint(checkpoint_path: str, args, device: str):
    ckpt = safe_torch_load(checkpoint_path, "cpu")
    raw_state = get_state_dict_from_checkpoint(ckpt)
    state = clean_state_dict_keys(raw_state)

    num_classes = get_num_classes_from_checkpoint(ckpt, state)
    model_args = infer_model_args_from_checkpoint(ckpt, args)

    model = DINOv2ReID(
        num_classes=num_classes,
        embedding_dim=model_args["embedding_dim"],
        dino_type=model_args["backbone_type"],
        unfreeze_last_blocks=0,
        dropout=model_args["dropout"],
        freeze_backbone=True,
    )

    current_state = model.state_dict()
    filtered_state = {}
    skipped = []

    for key, value in state.items():
        if key not in current_state:
            skipped.append((key, "missing_in_model"))
            continue

        if tuple(current_state[key].shape) != tuple(value.shape):
            skipped.append(
                (
                    key,
                    f"shape_mismatch checkpoint={tuple(value.shape)} model={tuple(current_state[key].shape)}",
                )
            )
            continue

        filtered_state[key] = value

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)

    print("=" * 80)
    print("CHECKPOINT LOADING")
    print("=" * 80)
    print("checkpoint:", checkpoint_path)
    print("num_classes:", num_classes)
    print("model_args:", model_args)
    print("loaded keys:", len(filtered_state))
    print("missing keys:", len(missing))
    print("unexpected keys:", len(unexpected))
    print("skipped keys:", len(skipped))

    if len(skipped) > 0:
        print("First skipped keys:")
        for item in skipped[:10]:
            print(" ", item)

    print("=" * 80)

    model.to(device)
    model.eval()

    return model, ckpt
