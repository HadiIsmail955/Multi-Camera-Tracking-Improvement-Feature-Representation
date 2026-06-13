# ReID Training and Inference

This repository provides a YAML-driven ReID pipeline for:

- Training with `osnet`, `dinov2`, or `dinov3` (requires manual download)
- Random Erasing Augmentation
- BNNeck
- LoRA or full fine-tuning (for DINO backbones)
- Mixed precision (AMP)
- Combined CE + BatchHard Triplet + SupCon losses + ArcFace Loss
- Inference and cross-camera matching with optional k-reciprocal re-ranking
- Patch Average Embeddings + CLS Embeddings

## Project layout

- `reid/train.py`: training entrypoint
- `reid/inference.py`: inference / matching entrypoint
- `configs/*.yaml`: training and inference configs
