# Final ReID Model Comparison

All models were evaluated on the same validation split using tracklet-level embeddings.  
The evaluation used `global_id_camera` grouping, `mean_topk` aggregation, and DBSCAN clustering.

## Evaluation Setup

| Setting | Value |
|---|---|
| Split | `val` |
| Level | `tracklet` |
| Tracklet grouping | `global_id_camera` |
| Aggregation | `mean_topk` |
| Embedding key | `bn_embedding` |
| Identity column | `identity_key` |
| Number of tracklets | 1514 |
| Number of identities | 131 |
| Embedding dimension | 512 |
| Clustering method | DBSCAN |

---

## Compared Models

| Model | Description | Selected DBSCAN eps |
|---|---|---:|
| **DINOv2 ReID** | Proposed model | **0.045** |
| **OSNet-AIN** | Paper-text baseline | **0.018** |
| **OSNet-x1.0** | Paper-code baseline | **0.015** |

---

## Retrieval Performance

| Model | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP |
|---|---:|---:|---:|---:|---:|
| OSNet-x1.0 | 73.32% | 87.85% | 90.95% | 93.73% | 45.66% |
| OSNet-AIN | 77.28% | 89.50% | 92.87% | 94.85% | 52.23% |
| **DINOv2 ReID** | **92.47%** | **97.16%** | **98.88%** | **99.27%** | **85.53%** |

---

## Embedding Similarity Quality

| Model | Same-ID Cosine | Different-ID Cosine | Separation Gap | Pair ROC-AUC |
|---|---:|---:|---:|---:|
| OSNet-x1.0 | 0.9004 | 0.8021 | 0.0984 | 0.8135 |
| OSNet-AIN | 0.8819 | 0.7455 | 0.1364 | 0.8564 |
| **DINOv2 ReID** | **0.8863** | **0.0222** | **0.8641** | **0.9921** |

---

## DBSCAN Clustering Performance

| Model | eps | Clusters | Noise Rate | Cluster Purity | Pair F1 | Miscluster Rate | Merge Error Rate | Fragmentation Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OSNet-x1.0 | 0.015 | 113 | 71.14% | 80.78% | 50.84% | 5.55% | 15.93% | 43.43% |
| OSNet-AIN | 0.018 | 98 | 79.72% | 93.16% | 77.22% | 1.39% | 11.22% | 34.15% |
| **DINOv2 ReID** | **0.045** | **128** | **18.36%** | 88.19% | **82.92%** | 9.64% | **10.16%** | **11.45%** |

---

## Final Result

The proposed **DINOv2 ReID** model is the best overall model.

It achieves the strongest retrieval performance:

- **Rank-1:** 92.47%
- **mAP:** 85.53%

It also provides the best embedding separation:

- **Embedding separation gap:** 0.8641
- **Pair ROC-AUC:** 0.9921

For clustering, DINOv2 with `eps=0.045` achieves the best overall balance.  
Compared with OSNet-AIN and OSNet-x1.0, it has much lower noise, stronger cluster-pair F1, and lower fragmentation.

Therefore, the final selected model is:

```text
DINOv2 ReID + DBSCAN eps = 0.045