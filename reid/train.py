# step5_training.py

import argparse

import torch
import torch.nn as nn

from reid.config import load_yaml_config
from reid.data.records import load_records
from reid.data.loaders import (
    build_train_loader,
    build_eval_loader,
)
from reid.engine.train import train_model, cosine_with_warmup
from reid.losses import BHTripletLoss, SupConLoss, ArcFaceLoss
from reid.models import build_model


def build_optimizer(model, base_lr: float):
    lora_head_params = [
        param
        for name, param in model.named_parameters()
        if param.requires_grad
        and ("lora" in name or "head" in name or "classifier" in name)
    ]

    backbone_params = [
        param
        for name, param in model.named_parameters()
        if param.requires_grad
        and ("lora" not in name and "head" not in name and "classifier" not in name)
    ]

    return torch.optim.AdamW(
        [
            {
                "params": lora_head_params,
                "lr": base_lr,
            },
            {
                "params": backbone_params,
                "lr": base_lr * 0.01,
            },
        ],
        weight_decay=1e-4,
    )


def main(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = config["model"]
    data_cfg = config["data"]
    sampler_cfg = config["sampler"]
    optim_cfg = config["optim"]
    eval_cfg = config["eval"]
    output_cfg = config["output"]

    print(f"\nTraining  |  backbone={model_cfg['backbone']}  device={device}")

    train_records = load_records(data_cfg["train"])
    query_records = load_records(data_cfg["query"])
    gallery_records = load_records(data_cfg["gallery"])

    train_dataset, train_loader = build_train_loader(
        records=train_records,
        P=sampler_cfg["P"],
        K=sampler_cfg["K"],
        num_workers=data_cfg["num_workers"],
    )

    query_dataset, query_loader = build_eval_loader(
        records=query_records,
        batch_size=eval_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
    )

    gallery_dataset, gallery_loader = build_eval_loader(
        records=gallery_records,
        batch_size=eval_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
    )

    print(
        f"  Train   : {len(train_dataset):,} imgs | "
        f"{train_dataset.num_classes} IDs | "
        f"{len(train_loader)} batches/epoch  "
        f"(P={sampler_cfg['P']}, K={sampler_cfg['K']})"
    )
    print(f"  Query   : {len(query_dataset):,} imgs")
    print(f"  Gallery : {len(gallery_dataset):,} imgs")

    model = build_model(
        backbone=model_cfg["backbone"],
        num_classes=train_dataset.num_classes,
        lora_rank=model_cfg["lora_rank"],
        use_lora=model_cfg.get("use_lora", True),
        osnet_weight_path=model_cfg.get("osnet_weights"),
        dino_feature_mode=model_cfg.get("feature_mode", "cls"),
    ).to(device)

    ce_loss = nn.CrossEntropyLoss(
        label_smoothing=optim_cfg["label_smoothing"],
    )

    triplet_loss = BHTripletLoss(
        margin=optim_cfg["margin"],
    )

    supcon_loss = SupConLoss(
        temperature=optim_cfg.get("supcon_temperature", 0.07),
    )

    arcface_loss = ArcFaceLoss(
        scale=optim_cfg.get("arcface_scale", 30.0),
        margin=optim_cfg.get("arcface_margin", 0.5),
        easy_margin=optim_cfg.get("arcface_easy_margin", False),
    )

    optimizer = build_optimizer(
        model=model,
        base_lr=optim_cfg["lr"],
    )

    scheduler = cosine_with_warmup(
        optimizer=optimizer,
        warmup_epochs=optim_cfg["warmup"],
        total_epochs=optim_cfg["epochs"],
    )

    train_model(
        model=model,
        train_loader=train_loader,
        query_loader=query_loader,
        gallery_loader=gallery_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        ce_loss=ce_loss,
        triplet_loss=triplet_loss,
        supcon_loss=supcon_loss,
        arcface_loss=arcface_loss,
        device=device,
        epochs=optim_cfg["epochs"],
        backbone=model_cfg["backbone"],
        ce_weight=optim_cfg.get("ce_weight", 1.0),
        triplet_weight=optim_cfg["triplet_weight"],
        supcon_weight=optim_cfg.get("supcon_weight", 0.2),
        arcface_weight=optim_cfg.get("arcface_weight", 0.0),
        eval_interval=eval_cfg["interval"],
        checkpoint_dir=output_cfg["checkpoint_dir"],
        log_path=output_cfg["log_path"],
        max_grad_norm=optim_cfg["max_grad_norm"],
        amp=optim_cfg.get("mixed_precision", True),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train ReID model from YAML config")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML training config.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_yaml_config(args.config)
    main(config)
