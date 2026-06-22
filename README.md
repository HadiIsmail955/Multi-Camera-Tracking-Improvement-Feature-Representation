# ReID Training and Inference

This repository provides a YAML-driven ReID pipeline for:

- Training with `osnet` or `dinov2`
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

## MTMC Tracking 2025 Dataset Preprocessing

The preprocessing pipeline converts raw MTMC annotations into tracklets and cropped object images for training and evaluation.

### Processing Steps
- Downsample annotations from **30 FPS to 1 FPS**.
- Split tracklets after long visibility gaps (when an object is absent for more than 1 second) and remove very short fragments (<2 frames). 
- Assign globally consistent object identities across all dataset splits (scene_objectID).
- Sample up to **30 evenly spaced frames** per tracklet.
- Construct **Query/Gallery** evaluation sets using identities observed across multiple cameras.
- Extract object crops in parallel with support for resumable processing.

### Dataset Statistics

| Split | Images | Tracklets |
|---------|---------:|---------:|
| Train | 1,214,776 | 79,622 |
| Query | 3,925 | 131 |
| Gallery | 106,374 | 7,503 |

**Total unique identities:** 994

### Evaluation Protocol

* For validation, only identities visible in at least two cameras are considered. 
* The longest tracklet of an identity is selected as the **Query**, while tracklets from other cameras form the **Gallery**.
* The query tracklet is excluded from the gallery.


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

### LoRA Rank Impact
Below is a separate comparison evaluating different rank ($r$) configurations for Low-Rank Adaptation (LoRA) against the full fine-tuning baseline using BatchHard Triplet loss.

| Configuration | Rank-1 | mAP |
| :--- | :---: | :---: |
| BatchHard Triplet (LoRA 8) | 92.5% | 84.3% |
| BatchHard Triplet (LoRA 16) | 93.0% | 85.0% |
| BatchHard Triplet (LoRA 32) | 91.8% | 83.9% |
| BatchHard Triplet (Full Fine-Tuning) | **93.0%** | **85.4%** |

## Ablation Study on MTMC Tracking 2025 Dataset

The following table presents the ablation study of our proposed DINOv2 pipeline evaluated on the **MTMC Tracking 2025 dataset**, which is part of NVIDIA’s Physical AI Smart Spaces initiative and featured in the **2025 AI City Challenge** benchmark. We compare the performance metrics across both **Image Level** and **Tracklet Level** configurations.

| Configuration | Image Rank-1 | Image mAP | Tracklet Rank-1 | Tracklet mAP |
| :--- | :---: | :---: | :---: | :---: |
| DINOv2 (pretrained on LVD-142M) | 26.1% | 4.3% | 29.8% | 6.9% |
| + BatchHard Triplet (Full Training) | 90.3% | 75.8% | 93.9% | 83.4% |
| + BNNeck | 90.2% | 78.0% | 96.2% | 85.6% |
| + Random Erasing | 90.8% | 78.1% | 94.7% | 85.6% |
| + CE Loss | 91.0% | 77.5% | 93.9% | 84.2% |
| + ArcFace | 92.3% | 81.8% | **98.5%** | 88.7% |
| + SupCon | 92.0% | 82.0% | 97.0% | 88.8% |
| + CLS + Patch Average | 92.2% | 81.9% | 96.2% | 88.8% |
| + Resolution (384x192) | 92.9% | 82.9% | 97.0% | 89.3% |
| + 300 epochs | **92.9%** | **84.7%** | **97.0%** | **90.8%** |
| OSNet (Baseline - Pretrained on Market-1501) | 67.9% | 29.8% | 88.6% | 50.5% |

### LoRA Rank Impact
Below is the isolated hyperparameter tracking for the Low-Rank Adaptation (LoRA) configuration variant using the BatchHard Triplet loss setup.

| Configuration | Image Rank-1 | Image mAP | Tracklet Rank-1 | Tracklet mAP |
| :--- | :---: | :---: | :---: | :---: |
| BatchHard Triplet (LoRA 8) | 88.1% | 70.9% | 95.4% | 81.2% |
| BatchHard Triplet (LoRA 16) | 88.7% | 70.8% | 95.4% | 80.7% |
| BatchHard Triplet (LoRA 32) | 79.1% | 46.2% | 82.0% | 56.2% |
| BatchHard Triplet (Full Training) | **90.3%** | **75.8%** | **93.9%** | **83.4%** |


### Clustering & Pooling Methods
The following table compares different embedding pooling and clustering strategies evaluated on tracklet-level feature representations.

| Method | Rank-1 | mAP |
| :--- | :---: | :---: |
| DBScan | 97.0% | 90.8% |
| Mean Pooling | 97.0% | 90.3% |
| Max Pooling | 45.8% | 42.6% |
| Weighted | 97.0% | 91.1% |
| Medoid | 97.0% | 90.4% |
| GeM Pooling | 97.0% | 91.7% |
| + Re-ranking | **96.2%** | **92.0%**|

### Backbone Comparison (Full Setting)
This table provides a head-to-head architectural comparison between the OSNet baseline and our proposed DINOv2 pipeline under the full experimental setting on the tracklet-level.



| Backbone | Rank-1 | mAP |
| :--- | :---: | :---: |
| OSNet | 93.1% | 77.7% |
| + GeM Pooling | 93.9% | 79.0% |
| + Re-ranking | 97.0% | 80.1% |
| DINOv2 | 97.0% | 90.8% |
| + GeM Pooling | 97.0% | 91.7% |
| + Re-ranking | **96.2%** | **92.0%** |


> **Note on Full Experimental Setting:** Both backbones integrate the complete optimized training pipeline, which includes: Full Training, BatchHard Triplet Loss, BNNeck, Random Erasing, CE Loss, SupCon Loss, ArcFace Loss, CLS + Patch Average token merging, a high-resolution input size of 384x192, and training extended to 300 epochs.