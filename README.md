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



## Datasets Used

The experiments use the preprocessed MTMC dataset stored under:

```text
DataSet/MTMC_Tracking_2025_Preprocessed/
```

The dataset is organized into `train` and `val` splits. Each scene contains a `metadata.csv` file and extracted 2D crop images.



### Training Dataset

The DINOv2 ReID model was trained on the **training split**:

```text
DataSet/MTMC_Tracking_2025_Preprocessed/train/
```

The training data consists of 2D object/person crops extracted from multi-camera video scenes.  
Each crop is linked to metadata containing the scene, camera, frame, object identity, and bounding box information.

Training scenes used for the project:

```text
Warehouse_001
Warehouse_002
Warehouse_003
Warehouse_004
```

A training run used approximately:

| Training property | Value |
|---|---:|
| Crop samples | 547,933 |
| Identity classes | 50 |
| Cameras | 42 |
| Input type | 2D image crops extracted from videos |
| Occlusion crops | Included |
| Training level | Crop-level |
| Evaluation level after training | Tracklet-level |

The model was trained for **12 total epochs**:

| Stage | Epochs | Backbone Setting |
|---|---:|---|
| Stage 1 | 10 | Frozen DINOv2 backbone |
| Stage 2 | 2 | Last two DINOv2 blocks unfrozen |



### Validation Dataset

The final validation experiments were performed on the **validation split**:

```text
DataSet/MTMC_Tracking_2025_Preprocessed/val/
```

The validation set contains four scenes:

| Validation scene | Metadata rows |
|---|---:|
| `Hospital_000` | 133,915 |
| `Lab_000` | 14,112 |
| `Warehouse_015` | 282,863 |
| `Warehouse_016` | 54,105 |

Final validation summary:

| Validation property | Value |
|---|---:|
| Total crop samples | 484,689 |
| Validation identities | 131 |
| Tracklet embeddings after aggregation | 1,514 |
| Embedding dimension | 512 |
| Tracklet grouping | `global_id_camera` |
| Aggregation | `mean_topk` |

The validation pipeline first extracts crop-level embeddings, then aggregates them into tracklet-level embeddings.  
The final reported results are based on these **1,514 tracklet embeddings**.



### Dataset Role in the Pipeline

| Dataset split | Used for | Level |
|---|---|---|
| `train` | Training the DINOv2 ReID model | Crop-level |
| `val` | Retrieval, similarity, DBSCAN/HDBSCAN clustering, and failure analysis | Tracklet-level |

The OSNet baselines were not trained on this dataset. They were used as pretrained ReID baselines and evaluated on the same validation split for comparison.

## Data Preprocessing

The preprocessing stage converts raw multi-camera videos and metadata into ReID-ready crop images.  
This step is important because the ReID model does not train directly on full video frames. Instead, it trains on cropped objects/persons extracted from detections or tracking metadata.

Main preprocessing files:

| File | Purpose |
|---|---|
| `DataPreprocessing/main.py` | Main preprocessing entry point |
| `DataPreprocessing/extract_crops.py` | Extracts object/person crops from video frames using metadata |
| `DataPreprocessing/target_video_grid.py` | Utility for checking or visualizing camera/video layout |

Expected preprocessed dataset structure:

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

The metadata file is used to locate objects in the video frames and associate each crop with:

- scene name
- camera ID
- frame ID
- object/person ID
- bounding box coordinates
- visibility or occlusion information when available

The preprocessing pipeline supports:

- extracting 2D image crops from videos
- filtering invalid or very small crops
- padding small objects when needed
- removing or reducing overlapping detections
- saving crops in a structure suitable for ReID training and validation
- preserving camera and identity information for cross-camera evaluation

For the final ReID experiments, crop embeddings are later aggregated into tracklet-level embeddings using:

```text
tracklet_group_mode = global_id_camera
aggregation = mean_topk
```

This means that multiple crop embeddings belonging to the same identity-camera tracklet are combined into one representative tracklet embedding.



## Occlusion Handling

Occlusion is a major challenge in multi-camera ReID because the same identity may appear partially hidden, truncated, or visually incomplete in some cameras.  
To make the model more robust, the pipeline includes occlusion-aware crop handling and occlusion-based training support.

Occlusion handling is used in three ways:

### 1. Occlusion-aware preprocessing

During preprocessing, the dataset can include both normal crops and occluded crops.  
Occluded samples are kept because they represent realistic tracking conditions in multi-camera scenes.

The preprocessing stage can also apply filtering and padding to improve crop quality:

| Step | Purpose |
|---|---|
| Small-object padding | Adds context around very small crops |
| Overlap filtering | Reduces duplicate or heavily overlapping detections |
| Crop extraction | Saves object/person crops from frame metadata |
| Occlusion crop inclusion | Keeps occluded examples for robustness |

### 2. Occlusion-aware training

During training, occluded crops can be included using:

```text
include_occlusion_crops = True
```

This helps the model learn embeddings that remain stable even when objects are partially visible.

The training objective may also include an occlusion consistency component:

```text
total_loss =
    id_weight * CE
  + triplet_weight * Triplet
  + contrastive_weight * SupCon
  + occlusion_consistency_weight * OcclusionConsistency
```

The goal is to make embeddings from normal and occluded views of the same identity closer in feature space.

### 3. Occlusion-aware evaluation and failure analysis

The validation pipeline reports separate failure statistics for clean and occluded samples, including:

| Metric | Meaning |
|---|---|
| `clean_noise_rate` | Percentage of clean samples marked as noise |
| `occluded_noise_rate` | Percentage of occluded samples marked as noise |
| `clean_miscluster_rate` | Percentage of clean samples assigned to wrong clusters |
| `occluded_miscluster_rate` | Percentage of occluded samples assigned to wrong clusters |

These metrics help identify whether clustering errors are mainly caused by occlusion.

In the final DINOv2 DBSCAN result, clean samples had no misclustered samples, while most wrong assignments came from occluded samples. This shows that occlusion is one of the main remaining failure cases.



## Preprocessing and Occlusion Summary

The full data preparation and occlusion-aware pipeline can be summarized as:

```text
Raw video frames + metadata
        |
        v
Detection / tracking boxes
        |
        v
Crop extraction
        |
        v
Small-object padding and overlap filtering
        |
        v
Normal + occluded crop dataset
        |
        v
DINOv2 ReID training with occlusion-aware samples
        |
        v
Tracklet-level embedding aggregation
        |
        v
Retrieval + clustering + occlusion failure analysis
```

This preprocessing and occlusion handling improves the realism of the ReID evaluation because the model is tested under challenging multi-camera conditions rather than only clean isolated crops.


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



## Training

Main training script:

```text
script/con_v1_0/train_experiment.py
```

Training backend:

```text
reid/train/train_reid.py
```


### Training Data and Schedule

The ReID model was trained on the preprocessed MTMC training split. The training data consists of object/person crops extracted from multi-camera video scenes and organized using the metadata files generated during preprocessing.

Training was performed in two stages:

| Stage | Epochs | Backbone Setting | Description |
|---|---:|---|---|
| Stage 1 | 10 epochs | Frozen DINOv2 backbone | Train the ReID projection head, BNNeck, and classifier while keeping the DINOv2 backbone fixed |
| Stage 2 | 2 epochs | Last two DINOv2 blocks unfrozen | Fine-tune the last two backbone layers together with the ReID head |

Total training length:

```text
10 frozen-backbone epochs + 2 fine-tuning epochs = 12 total epochs
```

This schedule was used to keep the pretrained DINOv2 representation stable during early training, then adapt the final backbone layers to the MTMC ReID task.

### Training Configuration Notes

Important training settings used in the final model:

| Setting | Value |
|---|---|
| Backbone | DINOv2 |
| Backbone stage 1 | Frozen for 10 epochs |
| Backbone stage 2 | Last two blocks unfrozen for 2 epochs |
| Total epochs | 12 |
| Crop type | Normal + occlusion-aware crops |
| Occlusion crops | Included during training |
| Input level | 2D image crops extracted from videos |
| Evaluation level | Tracklet-level embeddings |



Example command:

```bash
python -u -m script.con_v1_0.train_experiment \
  --data_root DataSet/MTMC_Tracking_2025_Preprocessed \
  --output_dir outputs_reid/dinov2_reid \
  --batch_size 128 \
  --epochs 12
```

On SLURM, use:

```bash
sbatch full_experiment_dev_gpu.sh
```



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



## Multiple Queries per Identity

In ReID evaluation, one identity can appear many times across different cameras and tracklets.  
Therefore, the validation is not based on a single image per identity. Instead, each identity can have multiple query examples.

The evaluation works as follows:

```text
Identity A
├── Camera 1 / Tracklet 1  -> query example
├── Camera 2 / Tracklet 2  -> gallery candidate
├── Camera 3 / Tracklet 3  -> gallery candidate
└── Camera 4 / Tracklet 4  -> gallery candidate
```

Each tracklet embedding can be used as a query and compared against the remaining gallery tracklets.  
A correct retrieval happens when another tracklet with the same identity appears in the top-ranked results.

This is important because the same object/person may look different depending on:

- camera viewpoint
- occlusion
- lighting
- crop quality
- distance from the camera
- partial visibility
- background clutter

The final evaluation uses **tracklet-level embeddings**, not only single crop embeddings.  
This makes the evaluation more stable because multiple crop embeddings from the same identity-camera tracklet are aggregated into one representative embedding.

Final evaluation setup:

| Setting | Value |
|---|---|
| Evaluation level | Tracklet-level |
| Tracklet grouping | `global_id_camera` |
| Aggregation method | `mean_topk` |
| Number of tracklet embeddings | 1514 |
| Number of identities | 131 |
| Embedding dimension | 512 |

The purpose of multiple queries per identity is to test whether the model can retrieve the same identity under different camera conditions.



## Inference Explanation

During inference, the ReID model does not directly predict a new identity class.  
Instead, it extracts an embedding vector that represents the visual appearance of the crop or tracklet.

The inference pipeline is:

```text
Input crop or tracklet
        |
        v
DINOv2 ReID model
        |
        v
512-dimensional embedding
        |
        v
L2 normalization
        |
        v
Cosine similarity comparison
        |
        v
Nearest gallery matches
```

For crop-level inference:

```text
image crop -> model -> embedding
```

For tracklet-level inference:

```text
multiple crops from same tracklet
        |
        v
extract embedding for each crop
        |
        v
aggregate embeddings using mean_topk
        |
        v
final tracklet embedding
```

Matching is performed by comparing the query embedding with gallery embeddings using cosine similarity.

```text
higher cosine similarity = more likely same identity
lower cosine similarity  = more likely different identity
```

The model is therefore used as a **feature extractor**.  
The final identity association is decided by retrieval or clustering, not by the classifier head alone.

### What is Compared During Inference?

The comparison is between embeddings:

```text
query tracklet embedding  <->  gallery tracklet embeddings
```

The gallery embeddings are ranked by similarity.  
The highest-ranked gallery tracklets are considered the most likely matches.

This is why Rank-1, Rank-5, Rank-10, Rank-20, and mAP are used as retrieval metrics.



## Occlusion Augmentation

Occlusion is added during training to make the ReID model more robust to real-world partial visibility.  
In multi-camera tracking, an object/person may be partially blocked by another object, another person, equipment, shelves, or scene clutter.

To simulate this, artificial occlusion is applied to some training crops.

The idea is:

```text
original object crop
        +
random patch from another part of the image / scene
        =
occluded object crop
```

Instead of simply using a black rectangle or random noise, the occluder can be created using a crop or patch taken from another part of the image or scene.  
This makes the occlusion more realistic because the blocking region looks like real background or real scene content.

Example process:

```text
1. Take an object/person crop.
2. Select a random patch from another image region or another crop.
3. Resize or place the patch over part of the object crop.
4. Use the modified crop as an occluded training sample.
```

This creates training examples where parts of the object are hidden by realistic visual content.

Occlusion augmentation helps the model learn that:

- the same identity should remain close in embedding space even when partially hidden
- the model should not rely only on one visible body/object part
- the embedding should remain stable under partial occlusion
- background or blocking regions should not dominate the identity representation

The training pipeline can include these occluded crops using:

```text
include_occlusion_crops = True
```

Occlusion is especially important for the MTMC task because cross-camera tracking often contains crowded scenes and imperfect detections.



## Failure Analysis

Failure analysis is used to understand where the ReID and clustering pipeline makes mistakes.  
The goal is not only to report final accuracy, but also to identify the main reasons for failure.

The validation pipeline saves diagnostic files for:

| File | Purpose |
|---|---|
| `misclustered_points.csv` | Samples assigned to the wrong cluster |
| `noise_points.csv` | Samples marked as noise by DBSCAN/HDBSCAN |
| `merge_errors.csv` | Clusters that contain more than one real identity |
| `fragmentation_errors.csv` | Real identities split across multiple clusters |
| `embedding_metadata_with_cluster_diagnosis.csv` | Metadata with cluster and failure labels |
| `interactive_3d_miscluster_diagnosis.html` | Interactive 3D visualization of clustering errors |



### Noise Samples

Noise samples are tracklets that the clustering algorithm does not assign to any cluster.

This can happen when:

- the crop is heavily occluded
- the object is very small
- the tracklet has poor visual quality
- the embedding is far from other same-identity embeddings
- the identity appears only weakly across cameras

A high noise rate means the method is conservative.  
It avoids wrong matches, but it also leaves many samples unassigned.



### Merge Errors

A merge error happens when one predicted cluster contains multiple real identities.

Example:

```text
Predicted Cluster 12
├── Real ID 4
├── Real ID 4
├── Real ID 9
└── Real ID 9
```

This is dangerous for multi-camera tracking because it means different identities are incorrectly linked together.

Merge errors usually happen when:

- two identities look visually similar
- occlusion hides important identity features
- the clustering threshold is too loose
- HDBSCAN groups nearby identities into the same cluster
- embeddings are not separated enough

For the final result, DBSCAN was preferred over HDBSCAN because DBSCAN had fewer merge errors.



### Fragmentation Errors

A fragmentation error happens when one real identity is split into multiple predicted clusters.

Example:

```text
Real ID 7
├── Predicted Cluster 3
├── Predicted Cluster 18
└── Predicted Cluster 29
```

This means the system fails to connect all appearances of the same identity.

Fragmentation can happen because:

- the same identity looks different across cameras
- some crops are occluded
- lighting or viewpoint changes are large
- the clustering threshold is too strict
- some tracklets are marked as noise

Fragmentation is less dangerous than merge errors, but it reduces tracking continuity.



### Misclustered Samples

A misclustered sample is a sample assigned to a cluster where the dominant identity is not its real identity.

This means the sample was assigned, but assigned incorrectly.

Misclustered samples are especially important because they represent wrong identity associations.  
In this project, wrong associations are considered worse than leaving a sample as noise.



### Occlusion Failure Analysis

The pipeline also checks whether failures are mainly caused by clean or occluded samples.

Important metrics include:

| Metric | Meaning |
|---|---|
| `clean_noise_rate` | Clean samples marked as noise |
| `occluded_noise_rate` | Occluded samples marked as noise |
| `clean_miscluster_rate` | Clean samples assigned to wrong clusters |
| `occluded_miscluster_rate` | Occluded samples assigned to wrong clusters |

For the final DINOv2 + DBSCAN result, clean samples had no misclustered samples, while most wrong assignments came from occluded samples.  
This shows that occlusion remains one of the main remaining challenges.



### DBSCAN vs HDBSCAN Failure Behavior

| Method | Strength | Weakness |
|---|---|---|
| DBSCAN | Safer, fewer identity merges | More noise samples |
| HDBSCAN | More assigned predictions, lower noise | More identity merge errors |

Final decision:

```text
DINOv2 + DBSCAN eps = 0.045
```

was selected as the safest final method because it gives the best balance between correct predictions, cluster purity, pair F1, and merge errors.

HDBSCAN is still useful as a high-coverage comparison because it assigns more samples, but it is less safe when avoiding wrong identity merges is the priority.

## Occlusion-Aware Evaluation and Failure Analysis Values

The following values were collected from the committed GitHub validation outputs under the `output/` folder.  
The values come from each run's `metrics.json` file, specifically from the `cluster_failure` block.



### Source Output Files

| Run | GitHub output path |
|---|---|
| DINOv2 + DBSCAN | `output/output_dbscan/output_dino/20260622_161449/metrics.json` |
| DINOv2 + HDBSCAN | `output/output_hdbscan/output_dino/hdbscan_dino_mcs3_20260701_141653/metrics.json` |
| OSNet-x1.0 + DBSCAN | `output/output_dbscan/output_paper/20260622_185216/metrics.json` |
| OSNet-AIN + DBSCAN strict | `output/output_dbscan/output_paper/final_paper_text_osnet_ain_val_tracklet_20260629_001735/metrics.json` |
| OSNet-AIN + HDBSCAN | `output/output_hdbscan/output_paper/hdbscan_osnet_ain_mcs3_20260701_180958/metrics.json` |



### General Failure Analysis Values

| Model / Method | Setting | Noise Samples | Noise Rate | Misclustered Samples | Miscluster Rate | Merge Error Clusters | Merge Error Rate | Fragmented Identities | Fragmentation Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OSNet-x1.0 + DBSCAN | eps=0.015 | 1077 | 71.14% | 84 | 5.55% | 18 | 15.93% | 43 | 43.43% |
| OSNet-AIN + DBSCAN strict | eps=0.015 | 1322 | 87.32% | 4 | 0.26% | 4 | 5.19% | 17 | 27.42% |
| DINOv2 + DBSCAN final | eps=0.045 | 278 | 18.36% | 146 | 9.64% | 13 | 10.16% | 15 | 11.45% |
| DINOv2 + HDBSCAN | MCS=3 | 101 | 6.67% | 180 | 11.89% | 20 | 17.39% | 8 | 6.11% |
| OSNet-AIN + HDBSCAN | MCS=3 | 629 | 41.55% | 158 | 10.44% | 41 | 33.88% | 50 | 40.00% |



### Occlusion-Aware Failure Values

| Model / Method | Setting | Clean Noise Rate | Occluded Noise Rate | Clean Miscluster Rate | Occluded Miscluster Rate |
|---|---:|---:|---:|---:|---:|
| OSNet-x1.0 + DBSCAN | eps=0.015 | 96.05% | 69.82% | 0.00% | 5.84% |
| OSNet-AIN + DBSCAN strict | eps=0.015 | 98.68% | 86.72% | 0.00% | 0.28% |
| DINOv2 + DBSCAN final | eps=0.045 | 73.68% | 15.44% | 0.00% | 10.15% |
| DINOv2 + HDBSCAN | MCS=3 | 27.63% | 5.56% | 3.95% | 12.31% |
| OSNet-AIN + HDBSCAN | MCS=3 | 72.37% | 39.92% | 13.16% | 10.29% |



### Final DINOv2 DBSCAN Failure Analysis

The final selected safe method is:

```text
DINOv2 ReID + DBSCAN eps = 0.045
```

Its failure analysis values are:

| Metric | Value |
|---|---:|
| Noise samples | 278 |
| Noise sample rate | 18.36% |
| Misclustered samples | 146 |
| Misclustered sample rate | 9.64% |
| Merge error clusters | 13 |
| Merge error cluster rate | 10.16% |
| Fragmented identities | 15 |
| Fragmented identity rate | 11.45% |
| Clean noise rate | 73.68% |
| Occluded noise rate | 15.44% |
| Clean miscluster rate | 0.00% |
| Occluded miscluster rate | 10.15% |

Interpretation:

- Clean samples were not misclustered in the final DBSCAN result.
- Most wrong assignments came from occluded samples.
- Occlusion remains one of the main remaining failure cases.
- DBSCAN is safer than HDBSCAN because it produces fewer identity merge errors.
- HDBSCAN gives more assigned predictions, but it increases merge errors.



### DBSCAN vs HDBSCAN Failure Trade-Off

| Final DINOv2 Method | Noise Rate | Miscluster Rate | Merge Error Rate | Fragmentation Rate | Main Advantage | Main Weakness |
|---|---:|---:|---:|---:|---|---|
| DBSCAN eps=0.045 | 18.36% | 9.64% | 10.16% | 11.45% | Safer clustering with fewer merge errors | More noise than HDBSCAN |
| HDBSCAN MCS=3 | 6.67% | 11.89% | 17.39% | 6.11% | More assigned predictions and lower noise | More identity merge errors |

Final decision:

```text
Use DINOv2 + DBSCAN eps=0.045 as the main final result.
Report DINOv2 + HDBSCAN MCS=3 as a high-coverage comparison.
```


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



## DBSCAN Results

| Model | eps | Clusters | Noise Rate | Miscluster Rate | Correct Assigned | Cluster Purity | Pair F1 | Merge Error | Fragmentation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OSNet-x1.0 | 0.015 | 113 | 71.14% | 5.55% | 353 / 1514 | 80.78% | 50.84% | 15.93% | 43.43% |
| OSNet-AIN strict | 0.015 | 77 | 87.32% | **0.26%** | 188 / 1514 | **97.92%** | 76.52% | **5.19%** | 27.42% |
| OSNet-AIN coverage | 0.025 | 147 | 60.57% | 5.35% | 516 / 1514 | 86.43% | 65.45% | 17.01% | 47.27% |
| **DINOv2 final** | **0.045** | 128 | 18.36% | 9.64% | **1090 / 1514** | 88.19% | **82.92%** | 10.16% | **11.45%** |



## HDBSCAN Results

| Model | min_cluster_size | Clusters | Noise Rate | Miscluster Rate | Correct Assigned | Cluster Purity | Pair F1 | Merge Error | Fragmentation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **DINOv2** | **3** | 115 | 6.67% | 11.89% | **1233 / 1514** | **87.26%** | **82.14%** | 17.39% | **6.11%** |
| DINOv2 | 5 | 110 | 6.67% | 13.21% | 1213 / 1514 | 85.85% | 78.84% | 18.18% | 6.92% |
| DINOv2 | 10 | 84 | 6.14% | 26.49% | 1020 / 1514 | 71.78% | 67.56% | 34.52% | 3.94% |
| **OSNet-AIN** | **3** | 121 | 41.55% | 10.44% | **727 / 1514** | **82.15%** | **73.29%** | 33.88% | 40.00% |
| OSNet-AIN | 5 | 82 | 43.33% | 14.33% | 641 / 1514 | 74.71% | 62.26% | 36.59% | 17.95% |
| OSNet-AIN | 10 | 2 | 6.01% | 91.88% | 32 / 1514 | 2.25% | 1.54% | 100.00% | 3.85% |



## Final Comparison

| Model / Method | Setting | Rank-1 | mAP | Noise Rate | Miscluster Rate | Correct Assigned | Cluster Purity | Pair F1 | Merge Error | Final Use |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| OSNet-x1.0 + DBSCAN | eps=0.015 | 73.32% | 45.66% | 71.14% | 5.55% | 353 / 1514 | 80.78% | 50.84% | 15.93% | Weakest baseline |
| OSNet-AIN + DBSCAN | eps=0.025 | 77.28% | 52.23% | 60.57% | 5.35% | 516 / 1514 | 86.43% | 65.45% | 17.01% | Safer OSNet-AIN coverage |
| OSNet-AIN + HDBSCAN | MCS=3 | 77.28% | 52.23% | 41.55% | 10.44% | 727 / 1514 | 82.15% | 73.29% | 33.88% | Best OSNet-AIN coverage |
| **DINOv2 + DBSCAN** | **eps=0.045** | **92.47%** | **85.53%** | 18.36% | **9.64%** | 1090 / 1514 | **88.19%** | **82.92%** | **10.16%** | **Best safe final** |
| **DINOv2 + HDBSCAN** | **MCS=3** | **92.47%** | **85.53%** | **6.67%** | 11.89% | **1233 / 1514** | 87.26% | 82.14% | 17.39% | **Most predictions** |



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



## Notes

- The dataset is not included in the repository.
- Checkpoints and generated training outputs are stored outside Git.
