# Ontology-Grounded-NLP-with-Anomaly-Detection-for-Autonomous-Science-Discovery-on-Mars
This repository contains the full implementation of the paper “Ontology-Grounded NLP with Anomaly Detection for Autonomous Science Discovery on Mars.”

## Overview

This project presents a subsystem-grounded textification framework for interpretable anomaly detection in Mars rover telemetry.
Numerical telemetry data from the Mars Science Laboratory (MSL) are converted into short, subsystem-based log statements (e.g.,
“Power current rises while thermal temperature increases.”)
Each text segment is classified as NORMAL, HIGH_PRIORITY, or ANOMALY using multiple machine learning and NLP models.

## Key Features

Textification Layer: Converts numeric sensor readings into subsystem-level textual summaries.

Multi-Model Evaluation: Includes six modeling paradigms — TF–IDF + SVC, Gradient Boosting, CRF, TextCNN, BERT, and Zero-Shot NLI.

Temporal and Interpretability Focus: Models evaluated for quantitative accuracy and qualitative temporal coherence.

Visualization Suite: Generates confusion matrices, ROC curves, calibration plots, and per-class F1 comparisons.

Fully Reproducible: Independent data splits and random seeds for each model ensure consistent evaluation.

## pipeline 

Raw Telemetry (.npy)
        ↓
Per-file Z-Normalization
        ↓
Windowing (W=80, S=40)
        ↓
Subsystem Mapping + Textification
        ↓
Model Training & Evaluation (TF–IDF, GB, CRF, TextCNN, BERT, NLI)


        ↓
## Structured Anomaly Reports

├── data/                     # Processed data, metrics, and output folders
│   ├── processed/            # Textified MSL dataset
│   ├── figs/                 # Generated plots
│   └── tables/               # Model evaluation tables
├── notebooks/
│   └── MSL_Textification.ipynb   # Full training & evaluation pipeline
├── models/                   # Saved model weights and vectorizers
├── requirements.txt          # Dependencies list
├── README.md
└── LICENSE


## Summary Results

| Model                  | Accuracy | Macro-F1 | Notable Strength                    |
| ---------------------- | -------- | -------- | ----------------------------------- |
| **Gradient Boosting**  | 0.78     | **0.42** | Best quantitative anomaly detection |
| **CRF (seq + policy)** | 0.80     | 0.35     | Strong temporal consistency         |
| **TextCNN**            | 0.75     | 0.41     | Robust contextual detection         |
| **BERT**               | **0.82** | 0.30     | High accuracy, majority bias        |
| **Zero-Shot NLI**      | 0.29     | 0.21     | Semantic baseline                   |
| **TF–IDF + SVC**       | 0.78     | 0.31     | Reliable linear baseline            |



## Future Work

- Enhance minority anomaly detection through focal/contrastive training

- Integrate CRF with transformer-based embeddings

- Deploy the framework for real-time onboard anomaly monitoring
