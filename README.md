# AI - DL Lab – ReID Pipeline

This project focuses on improving the **Re-Identification (ReID)** pipeline for multi-camera tracking and cross-camera identity matching.

The goal is to enhance feature representation and representative feature selection.

Team: Hadi Ismail, Rose Francis

---

## 🔍 Project Overview

The pipeline is based on a standard tracking + ReID framework:

- Object detection (precomputed or external module)
- Multi-object tracking
- ReID feature extraction
- Representative feature selection (clustering-based)
- Cross-camera matching
- Evaluation using HOTA and standard ReID metrics

---

## 📦 Project Structure

The project is organized into modular components for clarity and scalability:

```text
project-root/
│
├── reid/                         # Core ReID library (all model logic)
│   ├── data/                     # Dataset loaders, transforms
│   ├── models/                   # Backbones, embedding networks
│   ├── losses/                   # Metric learning losses
│   ├── engine/                   # Train / eval / inference loops
│   ├── clustering/               # Feature clustering 
│   ├── matching/                 # Cross-camera identity matching
│   ├── metrics/                  # HOTA, mAP, CMC
│   ├── utils/                   # Logging, checkpointing, visualization
│   └── consts.py                # Global constants
│
├── configs/                      # YAML/JSON experiment configs
├── scripts/                      # Entry point scripts
├── data/                         # Raw + processed datasets
├── outputs/                      # Generated results
├── tests/                        # Unit tests
├── notebooks/                    # Experiments / visualization
│
├── requirements.txt
├── pyproject.toml
├── README.md
└── .gitignore
```

