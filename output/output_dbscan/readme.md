# ReID Validation Results Comparison

## Evaluation Setup

All models were evaluated using the same validation pipeline and the same validation split.

| Setting | Value |
|---|---|
| Split | `val` |
| Level | `tracklet` |
| Tracklet grouping | `global_id_camera` |
| Aggregation | `mean_topk` |
| Embedding key | `bn_embedding` |
| Identity label | `identity_key` |
| Number of tracklets | 1514 |
| Number of identities | 131 |
| Embedding dimension | 512 |
| Clustering method | DBSCAN |
| Distance space | L2-normalized embedding space |

The validation split contains four scenes:

- `Hospital_000`
- `Lab_000`
- `Warehouse_015`
- `Warehouse_016`

---

## Compared Models

| Model | Description | Checkpoint |
|---|---|---|
| DINOv2 ReID | Trained model from this project | `outputs_reid/dinov2_reid_embedding_v2_20260613_211015/checkpoints/last.pt` |
| OSNet-AIN | Paper-text baseline | `osnet_ain_ms_m_c.pth.tar` |
| OSNet-x1.0 | Paper-code baseline from Glance-MCMT implementation | `osnet_ms_m_c.pth.tar` |

---

# 1. Retrieval Performance

| Model | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP |
|---|---:|---:|---:|---:|---:|
| OSNet-x1.0 paper-code | 73.32% | 87.85% | 90.95% | 93.73% | 45.66% |
| OSNet-AIN paper-text | 77.28% | 89.50% | 92.87% | 94.85% | 52.23% |
| **DINOv2 ReID** | **92.47%** | **97.16%** | **98.88%** | **99.27%** | **85.53%** |

**Observation:**  
The proposed DINOv2 ReID model achieves the best retrieval performance. It improves Rank-1 by **15.19 percentage points** over OSNet-AIN and improves mAP by **33.30 percentage points**.

---

# 2. Embedding Similarity Quality

| Model | Same-ID Cosine Mean | Different-ID Cosine Mean | Separation Gap | Pair ROC-AUC | Best Threshold |
|---|---:|---:|---:|---:|---:|
| OSNet-x1.0 paper-code | 0.9004 | 0.8021 | 0.0984 | 0.8135 | 0.9504 |
| OSNet-AIN paper-text | 0.8819 | 0.7455 | 0.1364 | 0.8564 | 0.9262 |
| **DINOv2 ReID** | **0.8863** | **0.0222** | **0.8641** | **0.9921** | 0.8062 |

**Observation:**  
Although OSNet and OSNet-AIN produce high same-identity similarity, they also produce very high different-identity similarity. This means different identities are still close together in embedding space.  
DINOv2 produces a much larger separation gap, showing that it separates identities much more clearly.

---

# 3. Embedding Health

| Model | Effective Rank | Effective Rank Ratio | Participation Ratio | Norm Mean |
|---|---:|---:|---:|---:|
| OSNet-AIN paper-text | 56.19 | 10.98% | 26.23 | 1.000 |
| **DINOv2 ReID** | 26.05 | 5.09% | 13.80 | 1.000 |

**Observation:**  
OSNet-AIN uses more embedding dimensions, but this does not lead to better ReID performance. DINOv2 has lower effective rank but much stronger identity separation and retrieval performance.

---

# 4. DBSCAN Clustering Comparison

## Final/Recommended DBSCAN Settings

| Model | Selected eps | Reason |
|---|---:|---|
| **DINOv2 ReID** | **0.045** | Best overall balance between purity, pair F1, noise, and merge errors |
| **OSNet-AIN paper-text** | **0.018** | Best balanced OSNet-AIN setting |
| OSNet-AIN strict | 0.015 | Very high precision, but too much noise |
| OSNet-x1.0 paper-code | 0.015 | Best clean setting for weaker OSNet baseline |

---

## Main DBSCAN Metrics

| Model / eps | Clusters | Noise Rate | Cluster Purity | Pair Precision | Pair Recall | Pair F1 | Miscluster Rate | Merge Error Rate | Fragmentation Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OSNet-x1.0 / 0.015 | 113 | 71.14% | 80.78% | 40.15% | 69.28% | 50.84% | 5.55% | 15.93% | 43.43% |
| OSNet-AIN / 0.015 | 77 | 87.32% | **97.92%** | **96.70%** | 63.31% | 76.52% | **0.26%** | **5.19%** | 27.42% |
| OSNet-AIN / 0.018 | 98 | 79.72% | 93.16% | — | — | 77.22% | 1.39% | 11.22% | 34.15% |
| OSNet-AIN / 0.025 | 147 | 60.57% | 86.43% | 62.41% | 68.79% | 65.45% | 5.35% | 17.01% | 47.27% |
| **DINOv2 / 0.045** | **128** | **18.36%** | 88.19% | 73.13% | **95.74%** | **82.92%** | 9.64% | 10.16% | **11.45%** |

**Observation:**  
OSNet-AIN at `eps=0.015` gives very clean clusters, but rejects **87.32%** of samples as noise.  
DINOv2 assigns far more samples while still achieving the best overall cluster-pair F1 and the lowest fragmentation rate.

---

# 5. OSNet-AIN DBSCAN eps Grid

| eps | Clusters | Noise Rate | Purity | Pair F1 | Miscluster Rate | Merge Error Rate | Fragmentation Rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.010 | 27 | 95.90% | 100.00% | 92.59% | 0.00% | 0.00% | 8.00% |
| 0.012 | 43 | 93.33% | 100.00% | 86.27% | 0.00% | 0.00% | 19.44% |
| 0.015 | 77 | 87.32% | 97.92% | 76.52% | 0.26% | 5.19% | 27.42% |
| **0.018** | **98** | **79.72%** | **93.16%** | **77.22%** | **1.39%** | **11.22%** | **34.15%** |
| 0.020 | 115 | 73.98% | 90.86% | 73.91% | 2.38% | 15.65% | 41.49% |
| 0.022 | 128 | 69.15% | 90.36% | 71.55% | 2.97% | 15.63% | 45.92% |
| 0.025 | 147 | 60.57% | 86.43% | 65.45% | 5.35% | 17.01% | 47.27% |
| 0.030 | 140 | 50.33% | 78.06% | 50.42% | 10.90% | 16.43% | 43.10% |
| 0.035 | 120 | 43.00% | 64.31% | 15.20% | 20.34% | 13.33% | 40.83% |
| 0.040 | 107 | 36.86% | 57.01% | 10.01% | 27.15% | 14.95% | 33.87% |

**Selected OSNet-AIN eps:** `0.018`

`eps=0.018` is selected as the balanced OSNet-AIN setting because it keeps high purity, has the best balanced pair F1 among practical settings, and has lower noise than stricter settings such as `0.010`, `0.012`, and `0.015`.

---

# 6. DINOv2 Final DBSCAN Metrics

| Metric | Value |
|---|---:|
| DBSCAN eps | 0.045 |
| Number of clusters | 128 |
| Number of true identities | 131 |
| Noise samples | 278 |
| Noise rate | 18.36% |
| Cluster purity | 88.19% |
| Cluster-pair precision | 73.13% |
| Cluster-pair recall | 95.74% |
| Cluster-pair F1 | 82.92% |
| Misclustered samples | 146 |
| Misclustered rate | 9.64% |
| Merge error clusters | 13 |
| Merge error rate | 10.16% |
| Fragmented identities | 15 |
| Fragmentation rate | 11.45% |
| ARI no noise | 82.78% |
| NMI no noise | 97.22% |
| Silhouette cosine | 0.7725 |

---

# 7. OSNet-AIN Strict vs Balanced Setting

| Metric | OSNet-AIN eps=0.015 | OSNet-AIN eps=0.018 |
|---|---:|---:|
| Clusters | 77 | 98 |
| Noise rate | 87.32% | 79.72% |
| Cluster purity | **97.92%** | 93.16% |
| Pair F1 | 76.52% | **77.22%** |
| Miscluster rate | **0.26%** | 1.39% |
| Merge error rate | **5.19%** | 11.22% |
| Fragmentation rate | **27.42%** | 34.15% |

**Interpretation:**  
`eps=0.015` is better for high precision and avoiding wrong merges.  
`eps=0.018` is better as a balanced baseline because it keeps more samples assigned while maintaining high clustering quality.

---

# 8. Final Model Ranking

| Rank | Model | Reason |
|---:|---|---|
| 1 | **DINOv2 ReID** | Best Rank-1, mAP, embedding separation, pair ROC-AUC, and overall clustering balance |
| 2 | OSNet-AIN paper-text | Stronger OSNet baseline, but high different-ID similarity and high noise in clustering |
| 3 | OSNet-x1.0 paper-code | Valid implementation baseline, but weaker than OSNet-AIN and much weaker than DINOv2 |

---

# 9. Final Conclusion

The proposed DINOv2 ReID model substantially outperforms both OSNet baselines on the same validation split and evaluation pipeline.

At tracklet level, DINOv2 achieves:

- Rank-1 = **92.47%**
- mAP = **85.53%**
- Pair ROC-AUC = **99.21%**
- Embedding separation gap = **0.8641**

The strongest OSNet baseline, OSNet-AIN, achieves:

- Rank-1 = **77.28%**
- mAP = **52.23%**
- Pair ROC-AUC = **85.64%**
- Embedding separation gap = **0.1364**

For DBSCAN-based representative feature clustering, DINOv2 with `eps=0.045` provides the best overall trade-off. It keeps most samples assigned, achieves strong cluster-pair F1, and has much lower fragmentation than OSNet-AIN.

Therefore, the final selected model is:

```text
DINOv2 ReID + DBSCAN eps = 0.045