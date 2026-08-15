<p align="center">
  <h1 align="center">Multi-Camera Re-Identification</h1>
  <h3 align="center">A Modern Bag of Tricks for Multi-Camera Feature Representation</h3>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/release/python-3110/"><img src="https://img.shields.io/badge/python-3.11-blue.svg" alt="Python 3.11"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg" alt="PyTorch 2.1+"></a>
  <a href="https://github.com/facebookresearch/dinov2"><img src="https://img.shields.io/badge/Backbone-DINOv2%20ViT--B%2F14-6f42c1.svg" alt="DINOv2"></a>
  <img src="https://img.shields.io/badge/Market--1501%20mAP-94.2%25-brightgreen.svg" alt="Market-1501 mAP">
  <img src="https://img.shields.io/badge/AI%20City%202025%20mAP-91.6%25-brightgreen.svg" alt="AI City 2025 mAP">
</p>

<!-- omit in toc -->
<a name="overview"></a>

## Overview

This repository presents a feature representation framework for multi-camera object and person re-identification (ReID) based on the self-supervised Vision Transformer **DINOv2** (`dinov2_vitb14`). Combining the global class token with mean-pooled patch tokens captures both semantic context and local appearance, while a **BNNeck**, hybrid metric-learning losses (Label-Smoothed CE, Batch-Hard Triplet, Supervised Contrastive, and ArcFace), and **Random Erasing** produce compact embeddings that remain robust to appearance changes and partial occlusions. For tracklet-level aggregation, **Generalized Mean (GeM) pooling** retains all frame-level features without the limitations of density-based clustering, and **$k$-reciprocal re-ranking** further refines retrieval.

```
                            Input Image (384 × 192)
                                       │
                                       ▼
                ┌──────────────────────┬──────────────────────┐
                │          DINOv2 Backbone (ViT-B/14)         │
                │  CLS Token (768-D)   │ Patch Tokens (768-D) │
                └──────────┬───────────┴───────────┬──────────┘
                           └───────────┬───────────┘
                                       │
                                       ▼
                            Concatenation (1536-D)
                                       │
                                       ▼
                                   1D BNNeck
                                       │
                                       ▼
                               L2-Normalization
                                       │
                 ┌─────────────────────┴─────────────────────┐
                 ▼                                           ▼
 ┌───────────────────────────────┐           ┌───────────────────────────────┐
 │      Training Objectives      │           │      Inference & Matching     │
 ├───────────────────────────────┤           ├───────────────────────────────┤
 │ • Label-Smoothed CE           │           │ • GeM / DBSCAN Tracklet Pool  │
 │ • Batch-Hard Triplet Loss     │           │ • Cosine Similarity           │
 │ • Supervised Contrastive      │           │ • k-Reciprocal Re-Ranking     │
 │ • ArcFace Loss                │           │ • Embedding Quality Metrics   │
 └───────────────────────────────┘           └───────────────────────────────┘
```

---

<a name="quick-start"></a>

## Quick Start

Get started with Multi-Camera Re-Identification in a few steps.

### Prerequisites

- Linux OS
- Python `>=3.11, <3.12`
- NVIDIA GPU with CUDA support (e.g., A100, H100)

### Installation & Environment Setup

Clone this repository:

```bash
git clone https://github.com/HadiIsmail955/Multi-Camera-Tracking-Improvement-Feature-Representation.git
git checkout main
cd Multi-Camera-Tracking-Improvement-Feature-Representation
```

This repository uses [`uv`](https://github.com/astral-sh/uv) for dependency management. Create the virtual environment and install dependencies:

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv sync
```

---

<a name="dataset-preprocessing"></a>

## Dataset Preparation & Preprocessing

### 1. Download & Directory Layout
Download `train/` and `val/` splits from Hugging Face: [nvidia/PhysicalAI-SmartSpaces (MTMC_Tracking_2025)](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces/tree/main/MTMC_Tracking_2025/).

```
MTMC_Tracking_2025/
├── train/<scene>/      # videos/ (.mp4), ground_truth.json, calibration.json
└── val/<scene>/        # videos/ (.mp4), ground_truth.json, calibration.json
```

### 2. Run Preprocessing
Extract crops and generate train/query/gallery splits (DukeMTMC-style one-query protocol):

```bash
python -m reid.data.preprocess_aicc \
    --dataset-root /path/to/MTMC_Tracking_2025 \
    --output-root /path/to/aicity_preprocessed \
    --ann-stride 30 \
    --max-gap 30 \
    --min-len 2 \
    --num-samples 30 \
    --max-workers 16
```

### 3. Configure Manifest Paths
Update the dataset paths in your YAML configuration (e.g. `configs/dinov2_aicc.yaml`):

```yaml
data:
  train: /path/to/aicity_preprocessed/manifests/image_train.csv
  query: /path/to/aicity_preprocessed/manifests/image_query.csv
  gallery: /path/to/aicity_preprocessed/manifests/image_gallery.csv
```

---

<a name="training-inference"></a>

## Pipeline Execution

All training and inference workflows are controlled via YAML configurations. For a complete reference of parameters, loss functions, optimization settings, tracklet pooling, and evaluation options, see the [Configuration Reference](configs/README.md).

### 1. Training
Train DINOv2 with hybrid token representation (`cls_patchavg`), 1D BNNeck, and multi-task metric learning losses:
```bash
python reid/train.py --config configs/dinov2_aicc.yaml
```
> **Output:** Best model weights and loss logs are saved to `checkpoints/best_model.pth` and `training_log.csv`.

### 2. Inference & Tracklet Matching
Run cross-camera retrieval with Generalized Mean (GeM) tracklet pooling and $k$-reciprocal re-ranking:
```bash
python reid/inference.py --config configs/dinov2_aicc_infer_tracklet.yaml
```
> **Output:** Evaluation metrics, distance matrices, and top-10 retrieval rankings are saved to `matching_results.csv` and `distance_matrix.pt`.

---

<a name="checkpoints"></a>

## Model Checkpoints

Fine-tuned model checkpoints are available on Google Drive:

| Model | Target Benchmark | Input Size | mAP | Download Link |
|---|---|:---:|:---:|:---:|
| **DINOv2 (ViT-B/14)** | AI City / MTMC Tracking 2025 | $384 \times 192$ | **91.6%** | [Google Drive Checkpoint](https://drive.google.com/file/d/1yPUAox9N0lwK_fKZixIOJKrG_BDpl1-o/view?usp=sharing) |
| **DINOv2 (ViT-B/14)** | Market-1501 | $384 \times 192$ | **94.2%** | [Google Drive Checkpoint](https://drive.google.com/file/d/10FuX0NnQeM1IAVL6FnNFAxQuuRfrcLDt/view?usp=sharing) |

> Place downloaded checkpoint files into `checkpoints/` (e.g. `checkpoints/best_model.pth`).

> **Note:** 94.2% mAP for Market-1501 is achieved with re-ranking, and 91.6% mAP for AI City 2025 is achieved at tracklet-level using GeM pooling and re-ranking.
---

<a name="benchmark-results"></a>

## Key Results

### 1. Tracklet-Level Evaluation (AI City / MTMC Tracking 2025)

| Backbone / Model | Aggregation / Method | Rank-1 (%) | mAP (%) |
|---|---|:---:|:---:|
| **OSNet-AIN Baseline** (Hashempoor et al.) | Pretrained + DBSCAN ($\varepsilon=0.2$) | 93.9% | 54.3% |
| **OSNet-AIN** | Full Fine-Tuned + GeM + Re-ranking | 93.1% | 82.3% |
| **DINOv2 (ViT-B/14)** | **Full Fine-Tuned + GeM + Re-ranking** | **97.0%** | **91.6%** |

### 2. Image-Level Evaluation (AI City & Market-1501)

| Benchmark | Backbone / Model | Rank-1 (%) | mAP (%) |
|---|---|:---:|:---:|
| **AI City 2025** | **DINOv2 (ViT-B/14)** | **93.7%** | **84.5%** |
| **Market-1501** | **DINOv2 (ViT-B/14)** | **96.6%** | **93.1%** |

---

