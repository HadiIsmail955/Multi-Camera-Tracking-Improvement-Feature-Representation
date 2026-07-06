# ReID Training and Inference

This repository provides a YAML-driven ReID pipeline for:

- Training with `osnet` or `dinov2`
- Random Erasing Augmentation
- BNNeck
- LoRA or full fine-tuning (for DINO backbones)
- Combined CE + BatchHard Triplet + SupCon losses + ArcFace Loss
- Inference and cross-camera matching with optional k-reciprocal re-ranking
- Clustering and pooling methods
- Patch Average Embeddings + CLS Embeddings

## Project layout

- `reid/train.py`: training entrypoint
- `reid/inference.py`: inference / matching entrypoint
<!-- - `configs/*.yaml`: training and inference configs -->

## Datasets
### AI City / MTMC Tracking 2025
- Primary dataset for this project.
- It contains object crops extracted from multi-camera videos.
- Synthetic dataset includes multiple cameras, scenes, object identities, frames, and object types.

### Market-1501
- A standard person ReID benchmark.
- Market-1501 is used to validate the methods on a well-established ReID dataset
- Real dataset, cross-camera, person identities

> **Note**: The test set of Market-1501 dataset comes with a pre-defined query and gallery splits for benchmarking.
AI City validation query / gallery splits are created following Market-1501 / MSMT protocols (1 query per ID)

## MTMC Tracking 2025 Data Preprocessing
**Motivation:**
- Using all frames from videos of AI City has millions on images.
- The validation set of AI City datasets has no query and gallery available

### Tracklet Generation
- Subsample (1 FPS)
- Split on > 30 frame gaps
- Discard tracklets with < 2 frames

### Identity Assignment
- Generate tracklet with unique tracklet ID
- Generate global PID for objects and connect with tracklet ID

### Frame Sampling
- Keeps tracklet if total frames <= 30
- If > 30, sample 30 evenly spaced frames

For validation dataset,
- Discard identities that do not appear across at least 2 distinct cameras.
- For each identity in validation, take the longest tracklet as query.
- Assign all remaining tracklets of that identity (except those from same camera) to gallery split.

## Evaluation Strategy
### Image-Level Evaluation
- Extract image embeddings of the validation set (query and gallery).
- Compute distance between query and gallery embeddings.
- Compute Rank-1 and mAP.

### Tracklet-Level Evaluation
- Extract image embeddings of the validation set (query and gallery).
- For each tracklet ID, combine the image embeddings to a tracklet embedding using the post-processing methods.
- Compute distance between query and gallery tracklet embeddings.
- Compute Rank-1 and mAP.

### Evaluation Protocol
- Positive: same Person ID + diﬀerent Camera ID
- Junk: same Person ID + same CameraID

## Experimental Results & Ablation Studies

### Ablation Study of DINOv2 pipeline on Market-1501

| Configuration | Rank-1 | mAP |
| :--- | :---: | :---: |
| osnet_ain_x1_0 | 93.4% | 82.2% |
| osnet_x1_0 | 94.3% | 83.6% |
| DINOv2 (pretrained on LVD-142M) | 2.0% | 0.8% |
| + BatchHard Triplet (LoRA 8) | 93.3% | 86.5% |
| + BatchHard Triplet (LoRA 16) | 94.2% | 87.6% |
| + BatchHard Triplet (LoRA 32) | 93.9% | 86.8% |
| + BatchHard Triplet (Full Training) | 95.1% | 88.9% |
| + BNNeck | 94.9% | 89.2% |
| + Random Erasing | 95.0% | 90.3% |
| + CE Loss | 95.7% | 90.4% |
| + ArcFace | 95.9% | 91.7% |
| + SupCon | 96.2% | 92.3% |
| + CLS + Patch Average | 96.5% | 92.8% |
| + Resolution ($384 \times 192$) | **96.6%** | 93.1% |
| + Re-ranking | 96.5% | **94.2%** |


### Ablation Study of DINOv2 on MTMC Tracking 2025 Dataset

The performance metrics is compared across both **Image Level** and **Tracklet Level** configurations.

| Configuration | Image Level Rank-1 | Image Level mAP | Tracklet Level Rank-1 | Tracklet Level mAP |
| :--- | :---: | :---: | :---: | :---: |
| DINOv2 (pretrained on LVD-142M) | 32.0% | 6.3% | 35.9% | 10.2% |
| + BatchHard Triplet (LoRA 8) | 88.1% | 70.9% | 95.4% | 81.2% |
| + BatchHard Triplet (LoRA 16) | 88.7% | 70.8% | 95.4% | 80.7% |
| + BatchHard Triplet (LoRA 32) | 79.1% | 46.2% | 82.0% | 56.2% |
| + BatchHard Triplet (Full Training) | 90.4% | 75.0% | 93.9% | 82.2% |
| + BNNeck | 89.7% | 75.7% | 95.4% | 80.7% |
| + Random Erasing | 89.7% | 76.0% | 93.9% | 81.1% |
| + CE Loss | 92.2% | 79.8% | 93.9% | 85.8% |
| + ArcFace | 92.4% | 81.3% | 97.0% | 87.5% |
| + SupCon | 92.2% | 81.5% | 97.0% | 87.7% |
| + CLS + Patch Average | 91.7% | 81.6% | 97.0% | 87.5% |
| + Resolution ($384 \times 192$) | 93.1% | 82.5% | 97.0% | 88.2% |
| + 300 epochs | **93.7%** | **84.5%** | **97.0%** | **90.0%** |

### Clustering & Pooling Methods on Tracklet-Level Results

| Method | Rank-1 | mAP |
| :--- | :---: | :---: |
| DBSCAN | 97.0% | 90.0% |
| HDBSCAN | 96.2% | 88.6% |
| Mean Pooling | 97.0% | 89.6% |
| Medoid | 96.2% | 89.0% |
| Weighted | 97.0% | 90.4% |
| GeM Pooling | 95.4% | 91.2% |
| + Re-ranking | **97.0%** | **91.6%** |

### Backbones Final Comparison (Full Setting) on MTMC Tracking 2025 Dataset
This table provides a head-to-head architectural comparison between the OSNet baseline (osnet_ain_x1_0) and our proposed DINOv2 pipeline under the full experimental setting on the tracklet-level.



| Backbone | Rank-1 | mAP |
| :--- | :---: | :---: |
| osnet_x1_0 (pretrained on Market-1501) | 88.0% | 50.7% |
| osnet_x1_0 (Full trained + DBSCAN) | **95.4%** | 76.8% |
| osnet_x1_0 (GeM Pooling) | 93.9% | 79.5% |
| osnet_x1_0 (+ Re-ranking) | 93.1% | **80.5%** |
| | | |
| osnet_ain_x1_0 (pretrained on M, MS, C) | 94.0% | 54.3% |
| osnet_ain_x1_0 (Full trained + DBSCAN) | 95.4% | 79.0% |
| osnet_ain_x1_0 (GeM Pooling) | **96.2%** | 81.4% |
| osnet_ain_x1_0 (+ Re-ranking) | 93.1% | **82.3%** |
| | | |
| DINOv2 (pretrained on LVD-142M) | 35.9% | 10.2% |
| DINOv2 (Full trained + DBSCAN) | 97.0% | 90.0% |
| DINOv2 (GeM Pooling) | 95.4% | 91.2% |
| DINOv2 (+ Re-ranking) | **97.0%** | **91.6%** |


> **Note on Full Experimental Setting:** The backbones integrate the complete optimized training pipeline, which includes: Full Training, BNNeck, Random Erasing, CE Loss, BatchHard Triplet Loss, SupCon Loss, ArcFace Loss, CLS + Patch Average, High-resolution (384x192), and training extended to 300 epochs.

## Key Findings
- **Feature Extraction**: Transitioning to **DINOv2** (with high-resolution inputs and BNNeck) and applying advanced metric learning (Triplet, ArcFace, SupCon losses) drastically improves visual representation, driving AI City mAP from a baseline of 54.3% to **91.6%**.
- **Improved Feature Selection**: **GeM Pooling** outperforms standard DBSCAN clustering by successfully suppressing background noise and isolating high-activation, discriminative features.
- **Robust Occlusion Handling**: Integrating Random Erasing forces the model to learn from diverse object parts, maintaining reliable feature extraction even when targets are partially hidden.
- **Near State-of-the-Art Performance**: The fully optimized pipeline achieves **94.2% mAP** on Market-1501 and hits **97.0% Rank-1** on the AI City dataset.