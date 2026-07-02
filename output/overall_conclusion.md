# Overall ReID Conclusion

## Final Comparison of Best Settings

| Model / Method | Setting | Rank-1 | mAP | Noise Rate | Miscluster Rate | Correct Assigned | Cluster Purity | Pair F1 | Merge Error | Final Use |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| OSNet-x1.0 + DBSCAN | eps=0.015 | 73.32% | 45.66% | 71.14% | 5.55% | 353 / 1514 | 80.78% | 50.84% | 15.93% | Weakest baseline |
| OSNet-AIN + DBSCAN | eps=0.025 | 77.28% | 52.23% | 60.57% | 5.35% | 516 / 1514 | 86.43% | 65.45% | 17.01% | Safer OSNet-AIN coverage |
| OSNet-AIN + HDBSCAN | MCS=3 | 77.28% | 52.23% | 41.55% | 10.44% | 727 / 1514 | 82.15% | 73.29% | 33.88% | Best OSNet-AIN coverage |
| **DINOv2 + DBSCAN** | **eps=0.045** | **92.47%** | **85.53%** | 18.36% | **9.64%** | 1090 / 1514 | **88.19%** | **82.92%** | **10.16%** | **Best safe final** |
| **DINOv2 + HDBSCAN** | **MCS=3** | **92.47%** | **85.53%** | **6.67%** | 11.89% | **1233 / 1514** | 87.26% | 82.14% | 17.39% | **Most predictions** |

---

## Final Ranking

| Rank | Model / Method | Reason |
|---:|---|---|
| 1 | **DINOv2 + DBSCAN eps=0.045** | Best final balance: highest mAP, strong cluster-pair F1, good purity, fewer merge errors |
| 2 | **DINOv2 + HDBSCAN MCS=3** | Highest number of correct assigned predictions, but more merge errors |
| 3 | **OSNet-AIN + HDBSCAN MCS=3** | Best OSNet-AIN setting for more predictions |
| 4 | **OSNet-AIN + DBSCAN eps=0.025** | Safer than OSNet-AIN HDBSCAN but fewer predictions |
| 5 | **OSNet-AIN + DBSCAN eps=0.015** | Very clean but too much noise |
| 6 | **OSNet-x1.0 + DBSCAN eps=0.015** | Weakest baseline |

---

## Final Decision

The proposed **DINOv2 ReID** model is the strongest model overall.

It achieves the best retrieval performance:

- Rank-1 = **92.47%**
- mAP = **85.53%**

It also gives the strongest embedding separation:

- Separation gap = **0.8641**
- Pair ROC-AUC = **0.9921**

For clustering, there are two useful final settings:

## Safe Final Clustering

```text
DINOv2 ReID + DBSCAN eps = 0.045
```

This is selected as the main final result because it gives the best balance between:

- cluster purity
- cluster-pair F1
- misclustered samples
- merge errors
- fragmentation

## High-Coverage Clustering

```text
DINOv2 ReID + HDBSCAN min_cluster_size = 3
```

This gives the highest number of correct assigned predictions:

```text
1233 / 1514 correct assigned tracklets
```

However, it has more identity merge errors than DBSCAN.

## Final Statement

DINOv2 clearly outperforms both OSNet baselines.  
OSNet-AIN improves over OSNet-x1.0, but it remains much weaker than DINOv2 in retrieval accuracy, embedding separation, and clustering quality.

Therefore, the final selected method is:

```text
DINOv2 ReID + DBSCAN eps = 0.045
```

HDBSCAN is reported as an additional high-coverage experiment.
