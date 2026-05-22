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

reid/
├── data/   # Dataset loading, preprocessing, transforms
├── models/ # Backbone + embedding networks
├── losses/ # Metric learning losses 
├── engine/ # Training, evaluation, inference loops
├── clustering/ # Feature clustering 
├── matching/ # Cross-camera identity matching logic
├── metrics/ # Evaluation metrics
├── utils/   # Logging, checkpoints, visualization
├── consts.py # Global constants

