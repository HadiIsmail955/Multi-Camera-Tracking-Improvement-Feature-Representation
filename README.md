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

## Ablation Study on Market-1501 Dataset

The following table presents the ablation study of our proposed DINOv2 pipeline evaluated on the **Market-1501** dataset, detailing the incremental impact of each component.

| Configuration | Rank-1 | mAP |
| :--- | :---: | :---: |
| DINOv2 (pretrained on LVD-142M) | 2.0% | 0.8% |
| + BatchHard Triplet (Full Fine-Tuning) | 93.0% | 85.4% |
| + BNNeck | 93.7% | 86.2% |
| + Random Erasing | 94.0% | 87.9% |
| + CE Loss | 94.6% | 88.2% |
| + ArcFace | 95.4% | 89.3% |
| + SupCon | 95.2% | 90.2% |
| + CLS + Patch Average | 95.4% | 90.1% |
| + Resolution (384x192) | 96.2% | 91.1% |
| + Re-ranking | 96.0% | 92.9% |
| **Final Model** | **96.0%** | **92.9%** |
| OSNet (Baseline - Pretrained on Market-1501) | 94.3% | 82.6% |

## LoRA Rank Impact
Below is a separate comparison evaluating different rank ($r$) configurations for Low-Rank Adaptation (LoRA) against the full fine-tuning baseline using BatchHard Triplet loss.

| Configuration | Rank-1 | mAP |
| :--- | :---: | :---: |
| BatchHard Triplet (LoRA 8) | 92.5% | 84.3% |
| BatchHard Triplet (LoRA 16) | **93.0%** | 85.0% |
| BatchHard Triplet (LoRA 32) | 91.8% | 83.9% |
| BatchHard Triplet (Full Fine-Tuning) | **93.0%** | **85.4%** |