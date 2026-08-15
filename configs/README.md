# Configuration Reference

This guide provides a concise reference for all configuration options available for training and inference.

---

## Quick Start Commands

Run all commands from the repository root (`Multi-Camera-Tracking-Improvement-Feature-Representation/`):

### Training (Primary: DINOv2 on AI City)
```bash
python reid/train.py --config configs/dinov2_aicc.yaml
```

### Inference & Tracklet Matching (Primary: DINOv2 on AI City)
```bash
python reid/inference.py --config configs/dinov2_aicc_infer_tracklet.yaml
```

---

## Configuration Sections & Parameters

### 1. `model` (Model Architecture)

| Parameter | Type | Options / Default | Description |
|---|---|---|---|
| `backbone` | `str` | `dinov2`, `osnet`, `osnet_ain` | Feature extractor backbone. |
| `model_name` | `str` | `dinov2_vitb14`, `dinov2_vits14`, `dinov2_vitl14` | DINOv2 ViT architecture variant. |
| `feature_mode` | `str` | `cls_patchavg` (1536-D), `cls` (768-D) | Feature representation (CLS + Avg Patch vs. CLS only). |
| `pretrained` | `bool` | `true` / `false` | Whether to load pretrained weights. |
| `use_lora` | `bool` | `false` (full fine-tune), `true` (LoRA) | Enable Low-Rank Adaptation. |
| `lora_rank` | `int` | `8`, `16`, `32` | LoRA rank dimension. |
| `osnet_weights` | `str` \| `null` | File path or `null` | Pretrained weight path for OSNet models. |

### 2. `data` (Data Paths & Tracklet Pooling)

| Parameter | Type | Description |
|---|---|---|
| `train` | `str` | Path to training CSV manifest. *(Training only)* |
| `query` | `str` | Path to query CSV manifest. |
| `gallery` | `str` | Path to gallery CSV manifest. |
| `num_workers` | `int` | DataLoader worker threads (e.g. `4`). |
| `query_embeddings_path` | `str` | Cache path (`.pt`) for extracted query features. *(Inference only)* |
| `gallery_embeddings_path` | `str` | Cache path (`.pt`) for extracted gallery features. *(Inference only)* |
| `tracklet_pool` | `str` \| `null` | Tracklet pooling mode: `gem`, `dbscan`, `hdbscan`, `mean`, `max`, `weighted`, `medoid`, or `null` (image-level). *(Inference only)* |

### 3. `checkpoint` (Inference Model Loading)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | `checkpoints/best_model.pth` | Path to fine-tuned checkpoint file. |
| `use_pretrained` | `bool` | `false` | If `true`, runs evaluation with pretrained weights only. |

### 4. `sampler` (Batch Sampling - Training)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `P` | `int` | `16` | Number of distinct IDs per batch. |
| `K` | `int` | `4` | Number of samples per ID (Total batch size = `P * K`). |

### 5. `optim` (Optimization & Loss Objectives - Training)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `epochs` | `int` | `300` | Total training epochs. |
| `warmup` | `int` | `10` | Linear warmup epochs before cosine decay. |
| `lr` | `float` | `0.001` | Base learning rate |
| `mixed_precision` | `bool` | `true` | Automatic Mixed Precision (AMP FP16). |
| `max_grad_norm` | `float` | `1.0` | Gradient clipping threshold. |
| `ce_weight` | `float` | `1.0` | Cross-Entropy classification loss weight. |
| `label_smoothing` | `float` | `0.1` | Label smoothing parameter $\epsilon$. |
| `triplet_weight` | `float` | `1.0` | Batch-Hard Triplet loss weight. |
| `margin` | `float` | `0.3` | Triplet loss margin. |
| `arcface_weight` | `float` | `1.0` | ArcFace angular margin loss weight. |
| `margin` | `float` | `0.5` | ArcFace angular margin $m$. |
| `supcon_weight` | `float` | `1.0` | Supervised Contrastive loss weight. |

### 6. `eval` (Evaluation Batching)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `interval` | `int` | `50` | Validation frequency in epochs. *(Training only)* |
| `batch_size` | `int` | `64` / `256` / `512` | Batch size for feature extraction and evaluation. |

### 7. `rerank` (k-Reciprocal Re-Ranking - Inference)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `true` / `false` | Enable $k$-reciprocal re-ranking. |
| `k1` | `int` | `20` | Size of reciprocal neighborhood. |
| `k2` | `int` | `6` | Size of local expansion neighborhood. |
| `lambda` | `float` | `0.3` | Interpolation weight between original and Jaccard distances. |

### 8. `output` (Output Paths & Artifacts)

| Parameter | Type | Description |
|---|---|---|
| `checkpoint_dir` | `str` | Directory to save checkpoint files (`best_model.pth`, `last_model.pth`). *(Training)* |
| `log_path` | `str` | CSV training log path. *(Training)* |
| `distance_matrix` | `str` | Output `.pt` path for distance matrix and embedding metrics. *(Inference)
