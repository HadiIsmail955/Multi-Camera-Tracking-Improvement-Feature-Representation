# Multi-Camera Tracking Improvement – Feature Representation

This repository contains the **Feature Representation / Re-Identification (ReID)** part of a multi-camera tracking pipeline.  
The goal is to improve cross-camera identity matching by learning stronger ReID embeddings and by selecting representative tracklet features through clustering.

The project focuses on:

- ReID feature extraction
- Tracklet-level feature aggregation
- Cross-camera identity matching
- Representative feature selection
- DBSCAN / HDBSCAN clustering analysis
- Retrieval evaluation using Rank-k and mAP
- Failure analysis for noise, fragmentation, and identity merges

---

## Project Motivation

Multi-camera tracking requires matching the same object/person across different cameras.  
This is difficult because of:

- camera viewpoint changes
- occlusions
- different lighting conditions
- partial crops
- visually similar identities
- noisy detections and tracklets

This project improves the feature representation stage by training and evaluating a DINOv2-based ReID model and comparing it with OSNet baselines.

---

## Main Contributions

1. **DINOv2-based ReID embedding model**
   - Uses a DINOv2 visual backbone
   - Adds a projection head
   - Uses BNNeck embeddings
   - Trains with classification and metric-learning losses

2. **Tracklet-level ReID evaluation**
   - Crop embeddings are aggregated into tracklet embeddings
   - Evaluation is performed at tracklet level using `global_id_camera` grouping
   - Final aggregation method: `mean_topk`

3. **Baseline comparison**
   - OSNet-x1.0 paper-code baseline
   - OSNet-AIN paper-text baseline
   - DINOv2 proposed model

4. **Clustering-based representative feature selection**
   - DBSCAN
   - HDBSCAN
   - Failure analysis for:
     - noise samples
     - merge errors
     - fragmented identities
     - misclustered samples

5. **Interactive visual analysis**
   - 2D PCA plots
   - 3D PCA/Plotly visualizations
   - cluster diagnosis plots
   - real identity vs predicted cluster visualizations

---

## Repository Structure

```text
project-root/
│
├── DataPreprocessing/
│   ├── extract_crops.py
│   ├── main.py
│   └── target_video_grid.py
│
├── reid/
│   ├── clustering/
│   │   └── clustering.py
│   │
│   ├── dataLoader/
│   │   ├── customData/
│   │   │   ├── MTMCCSVDataset.py
│   │   │   └── val_data.py
│   │   ├── sampler/
│   │   │   ├── camera_aware_pk_sampler.py
│   │   │   └── val_sampler.py
│   │   └── transformation/
│   │
│   ├── losses/
│   │   └── contrastiveLoss.py
│   │
│   ├── metrics/
│   │   ├── health.py
│   │   ├── retrieval.py
│   │   └── similarityMetrics.py
│   │
│   ├── model/
│   │   ├── backbone/
│   │   ├── DINOv2ReID.py
│   │   ├── model_loading.py
│   │   └── paper_osnet_ain.py
│   │
│   ├── train/
│   │   └── train_reid.py
│   │
│   ├── utils/
│   │   ├── aggregation.py
│   │   ├── dataloader.py
│   │   ├── embedding.py
│   │   ├── experiment_logger.py
│   │   ├── helper.py
│   │   ├── loadAndSaveModel.py
│   │   └── optimizerAndScheduler.py
│   │
│   ├── val/
│   │   └── analyze_reid.py
│   │
│   └── visualization/
│       └── visualization.py
│
├── script/
│   └── con_v1_0/
│       ├── full_experiment.py
│       ├── train_experiment.py
│       └── val_experiment.py
│
├── full_experiment_dev_gpu.sh
├── full_experiment_gpu.sh
├── full_val_experiment_dev_gpu.sh
├── full_val_experiment_dev_gpu_hdbsacn.sh
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Pipeline Overview

```text
Raw videos / metadata
        |
        v
Crop extraction
        |
        v
ReID dataset creation
        |
        v
DINOv2 ReID training
        |
        v
Crop-level embedding extraction
        |
        v
Tracklet-level aggregation
        |
        v
Retrieval evaluation
        |
        v
DBSCAN / HDBSCAN clustering
        |
        v
Failure analysis and visualization
```

---

## Data Preprocessing

The preprocessing module prepares image crops from multi-camera video data.

Main files:

| File | Purpose |
|---|---|
| `DataPreprocessing/main.py` | Main preprocessing entry point |
| `DataPreprocessing/extract_crops.py` | Extracts detection crops from frames/videos |
| `DataPreprocessing/target_video_grid.py` | Utility for visualizing or organizing target video views |

Expected preprocessed structure:

```text
DataSet/MTMC_Tracking_2025_Preprocessed/
├── train/
│   └── <scene_name>/
│       ├── metadata.csv
│       └── crops/
└── val/
    └── <scene_name>/
        ├── metadata.csv
        └── crops/
```

---

## Model Architecture

### DINOv2 ReID

The main model is implemented in:

```text
reid/model/DINOv2ReID.py
```

The model contains:

- DINOv2 backbone
- projection head
- BNNeck layer
- identity classifier

General architecture:

```text
Input crop
   |
DINOv2 backbone
   |
Projection head
   |
Embedding
   |
BNNeck
   |
Classifier
```

The model outputs embeddings used for both classification and metric learning.

---

## Paper Baselines

Two OSNet baselines are used for comparison.

| Baseline | Model | Checkpoint |
|---|---|---|
| Paper-code baseline | `osnet_x1_0` | `osnet_ms_m_c.pth.tar` |
| Paper-text baseline | `osnet_ain_x1_0` | `osnet_ain_ms_m_c.pth.tar` |

The OSNet-AIN wrapper is implemented in:

```text
reid/model/paper_osnet_ain.py
```

The valid OSNet-AIN experiment used:

```text
model_name = osnet_ain_x1_0
checkpoint = osnet_ain_ms_m_c.pth.tar
```

---

## Training Losses

The training objective combines classification and metric-learning losses.

Implemented in:

```text
reid/losses/contrastiveLoss.py
```

The main loss components are:

| Loss | Purpose |
|---|---|
| Cross-Entropy with label smoothing | identity classification |
| Batch-hard triplet loss | increase same-ID / different-ID separation |
| Camera-aware supervised contrastive loss | improve cross-camera identity consistency |
| Occlusion consistency loss | improve robustness to occluded crops |

Overall objective:

```text
total_loss =
    id_weight * CE
  + triplet_weight * Triplet
  + contrastive_weight * SupCon
  + occlusion_consistency_weight * OcclusionConsistency
```

---

## Training

Main training script:

```text
script/con_v1_0/train_experiment.py
```

Training backend:

```text
reid/train/train_reid.py
```

Example command:

```bash
python -u -m script.con_v1_0.train_experiment \
  --data_root DataSet/MTMC_Tracking_2025_Preprocessed \
  --output_dir outputs_reid/dinov2_reid \
  --batch_size 128 \
  --epochs 30
```

On SLURM, use:

```bash
sbatch full_experiment_dev_gpu.sh
```

---

## Training Strategy

The final DINOv2 ReID model was trained in two stages for a total of **12 epochs**.

| Stage | Epochs | Backbone Setting | Purpose |
|---|---:|---|---|
| Stage 1 | 10 epochs | DINOv2 backbone frozen | Train the ReID projection head, BNNeck, and classifier while preserving the pretrained DINOv2 representation |
| Stage 2 | 2 epochs | Last two DINOv2 backbone blocks unfrozen | Fine-tune the highest-level visual features for the multi-camera ReID task |

Final training schedule:

```text
Total training epochs = 12
Frozen backbone training = 10 epochs
Fine-tuning with last 2 backbone blocks unfrozen = 2 epochs
```

This strategy keeps the pretrained DINOv2 representation stable during most of training, then allows limited task-specific adaptation near the end.


---

## Validation and Diagnostics

Main validation script:

```text
script/con_v1_0/val_experiment.py
```

The validation pipeline performs:

1. model loading
2. crop embedding extraction
3. tracklet aggregation
4. retrieval evaluation
5. pairwise similarity analysis
6. DBSCAN or HDBSCAN clustering
7. cluster failure analysis
8. 2D and 3D visualization export

---

## DBSCAN Evaluation

Final DINOv2 DBSCAN command:

```bash
CHECKPOINT="./outputs_reid/dinov2_reid_embedding_v2_20260613_211015/checkpoints/last.pt"
DATA_ROOT="DataSet/MTMC_Tracking_2025_Preprocessed"
OUT_DIR="outputs_reid/final_dino_dbscan"

python -u -m script.con_v1_0.val_experiment \
  --checkpoint "$CHECKPOINT" \
  --data_root "$DATA_ROOT" \
  --split val \
  --out_dir "$OUT_DIR" \
  --level tracklet \
  --tracklet_group_mode auto \
  --aggregation mean_topk \
  --embedding_key bn_embedding \
  --identity_col identity_key \
  --cluster_method dbscan \
  --dbscan_eps 0.045 \
  --min_samples 2 \
  --include_occlusion_crops \
  --batch_size 256 \
  --workers 8 \
  --max_pairs 300000 \
  --pair_sampling balanced \
  --reduce_method pca \
  --make_3d_plots \
  --reduce_3d_method pca \
  --max_plot_points 50000
```

---

## HDBSCAN Evaluation

Final DINOv2 HDBSCAN command:

```bash
CHECKPOINT="./outputs_reid/dinov2_reid_embedding_v2_20260613_211015/checkpoints/last.pt"
DATA_ROOT="DataSet/MTMC_Tracking_2025_Preprocessed"
OUT_DIR="outputs_reid/final_dino_hdbscan"

python -u -m script.con_v1_0.val_experiment \
  --checkpoint "$CHECKPOINT" \
  --data_root "$DATA_ROOT" \
  --split val \
  --out_dir "$OUT_DIR" \
  --level tracklet \
  --tracklet_group_mode auto \
  --aggregation mean_topk \
  --embedding_key bn_embedding \
  --identity_col identity_key \
  --cluster_method hdbscan \
  --min_cluster_size 3 \
  --min_samples 2 \
  --include_occlusion_crops \
  --batch_size 256 \
  --workers 8 \
  --max_pairs 300000 \
  --pair_sampling balanced \
  --reduce_method pca \
  --make_3d_plots \
  --reduce_3d_method pca \
  --max_plot_points 50000
```

---

## Evaluation Metrics

### Retrieval Metrics

| Metric | Description |
|---|---|
| Rank-1 | Percentage of queries where the correct identity is the first retrieved result |
| Rank-5 | Correct identity appears in top 5 |
| Rank-10 | Correct identity appears in top 10 |
| Rank-20 | Correct identity appears in top 20 |
| mAP | Mean average precision over all valid queries |

### Similarity Metrics

| Metric | Description |
|---|---|
| Same-ID cosine mean | Mean similarity for same identity pairs |
| Different-ID cosine mean | Mean similarity for different identity pairs |
| Separation gap | Same-ID mean minus different-ID mean |
| Pair ROC-AUC | Binary same/different identity separability |
| Best threshold | Similarity threshold selected from pairwise evaluation |

### Clustering Metrics

| Metric | Description |
|---|---|
| Noise rate | Percentage of samples marked as noise |
| Cluster purity | Dominant identity purity inside predicted clusters |
| Pair F1 | Pairwise clustering F1 |
| Miscluster rate | Percentage of samples assigned to the wrong identity cluster |
| Merge error rate | Percentage of clusters containing multiple identities |
| Fragmentation rate | Percentage of identities split into multiple clusters |
| ARI | Adjusted Rand Index |
| NMI | Normalized Mutual Information |
| Silhouette cosine | Cluster separation score using cosine distance |

---

## Final Results

### Retrieval Performance

| Model | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP |
|---|---:|---:|---:|---:|---:|
| OSNet-x1.0 | 73.32% | 87.85% | 90.95% | 93.73% | 45.66% |
| OSNet-AIN | 77.28% | 89.50% | 92.87% | 94.85% | 52.23% |
| **DINOv2 ReID** | **92.47%** | **97.16%** | **98.88%** | **99.27%** | **85.53%** |

### Embedding Similarity

| Model | Same-ID Cosine | Different-ID Cosine | Separation Gap | Pair ROC-AUC |
|---|---:|---:|---:|---:|
| OSNet-x1.0 | 0.9004 | 0.8021 | 0.0984 | 0.8135 |
| OSNet-AIN | 0.8819 | 0.7455 | 0.1364 | 0.8564 |
| **DINOv2 ReID** | **0.8863** | **0.0222** | **0.8641** | **0.9921** |

---

## DBSCAN Results

| Model | eps | Clusters | Noise Rate | Miscluster Rate | Correct Assigned | Cluster Purity | Pair F1 | Merge Error | Fragmentation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OSNet-x1.0 | 0.015 | 113 | 71.14% | 5.55% | 353 / 1514 | 80.78% | 50.84% | 15.93% | 43.43% |
| OSNet-AIN strict | 0.015 | 77 | 87.32% | **0.26%** | 188 / 1514 | **97.92%** | 76.52% | **5.19%** | 27.42% |
| OSNet-AIN coverage | 0.025 | 147 | 60.57% | 5.35% | 516 / 1514 | 86.43% | 65.45% | 17.01% | 47.27% |
| **DINOv2 final** | **0.045** | 128 | 18.36% | 9.64% | **1090 / 1514** | 88.19% | **82.92%** | 10.16% | **11.45%** |

---

## HDBSCAN Results

| Model | min_cluster_size | Clusters | Noise Rate | Miscluster Rate | Correct Assigned | Cluster Purity | Pair F1 | Merge Error | Fragmentation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **DINOv2** | **3** | 115 | 6.67% | 11.89% | **1233 / 1514** | **87.26%** | **82.14%** | 17.39% | **6.11%** |
| DINOv2 | 5 | 110 | 6.67% | 13.21% | 1213 / 1514 | 85.85% | 78.84% | 18.18% | 6.92% |
| DINOv2 | 10 | 84 | 6.14% | 26.49% | 1020 / 1514 | 71.78% | 67.56% | 34.52% | 3.94% |
| **OSNet-AIN** | **3** | 121 | 41.55% | 10.44% | **727 / 1514** | **82.15%** | **73.29%** | 33.88% | 40.00% |
| OSNet-AIN | 5 | 82 | 43.33% | 14.33% | 641 / 1514 | 74.71% | 62.26% | 36.59% | 17.95% |
| OSNet-AIN | 10 | 2 | 6.01% | 91.88% | 32 / 1514 | 2.25% | 1.54% | 100.00% | 3.85% |

---

## Final Comparison

| Model / Method | Setting | Rank-1 | mAP | Noise Rate | Miscluster Rate | Correct Assigned | Cluster Purity | Pair F1 | Merge Error | Final Use |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| OSNet-x1.0 + DBSCAN | eps=0.015 | 73.32% | 45.66% | 71.14% | 5.55% | 353 / 1514 | 80.78% | 50.84% | 15.93% | Weakest baseline |
| OSNet-AIN + DBSCAN | eps=0.025 | 77.28% | 52.23% | 60.57% | 5.35% | 516 / 1514 | 86.43% | 65.45% | 17.01% | Safer OSNet-AIN coverage |
| OSNet-AIN + HDBSCAN | MCS=3 | 77.28% | 52.23% | 41.55% | 10.44% | 727 / 1514 | 82.15% | 73.29% | 33.88% | Best OSNet-AIN coverage |
| **DINOv2 + DBSCAN** | **eps=0.045** | **92.47%** | **85.53%** | 18.36% | **9.64%** | 1090 / 1514 | **88.19%** | **82.92%** | **10.16%** | **Best safe final** |
| **DINOv2 + HDBSCAN** | **MCS=3** | **92.47%** | **85.53%** | **6.67%** | 11.89% | **1233 / 1514** | 87.26% | 82.14% | 17.39% | **Most predictions** |

---

## Final Decision

The proposed **DINOv2 ReID** model is the strongest model overall.

It achieves:

- Rank-1 = **92.47%**
- mAP = **85.53%**
- Pair ROC-AUC = **0.9921**
- Separation gap = **0.8641**

For safe representative clustering, the final selected setup is:

```text
DINOv2 ReID + DBSCAN eps = 0.045
```

For high-coverage clustering, the best setup is:

```text
DINOv2 ReID + HDBSCAN min_cluster_size = 3
```

HDBSCAN gives more correct assigned predictions, but DBSCAN is safer because it produces fewer identity merge errors.

---

## Output Files

The validation pipeline saves:

```text
metrics.json
metrics.csv
embedding_metadata.csv
embedding_metadata_with_cluster_diagnosis.csv
misclustered_points.csv
noise_points.csv
merge_errors.csv
fragmentation_errors.csv
pair_similarity_sample.csv
query_retrieval_results.csv
rank_curve.csv
rank_curve.png
similarity_histogram.png
embedding_by_cluster.png
embedding_by_real_identity.png
embedding_misclustered_points.png
interactive_by_cluster.html
interactive_by_real_identity.html
interactive_miscluster_diagnosis.html
interactive_3d_by_cluster.html
interactive_3d_by_real_identity.html
interactive_3d_miscluster_diagnosis.html
```

---

## Installation

Create and activate an environment:

```bash
python -m venv env
source env/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies include:

```text
opencv-python
tqdm
torch
torchvision
scikit-learn
matplotlib
pandas
hdbscan
PyYAML
plotly
```

---

## Notes

- The dataset is not included in the repository.
- Checkpoints and generated training outputs are stored outside Git.
