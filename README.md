# Ontology-Grounded NLP with Anomaly Detection for Autonomous Science Discovery on Mars

This repository contains the full implementation of the paper **“Ontology-Grounded NLP with Anomaly Detection for Autonomous Science Discovery on Mars.”**
It includes data preprocessing, subsystem-grounded textification, six classification models, and the full visualization suite used in the final report.

---

## Overview

This project introduces a **subsystem-grounded textification framework** for interpretable anomaly detection in Mars rover telemetry.
Raw multivariate sensor readings from NASA’s Mars Science Laboratory (MSL) are automatically converted into short, subsystem-aware log statements (e.g., *“Power current rises while thermal temperature increases.”*).

Each textified window is then classified as:

* **NORMAL**
* **HIGH_PRIORITY**
* **ANOMALY**

using six modeling paradigms:

* TF–IDF + LinearSVC
* Gradient Boosting (adaptive SVD TF–IDF)
* Conditional Random Fields (CRF + post-policy)
* TextCNN with focal loss
* Fine-tuned BERT
* Zero-Shot NLI (BART-MNLI)

---

## Key Features

* **Subsystem-Grounded Textification**
  Converts numeric telemetry channels into domain-aware natural-language summaries.

* **Multi-Model Evaluation**
  Six architectures evaluated independently with consistent data splits and random seeds.

* **Temporal Sensitivity**
  CRF + post-inference policy captures sequential continuity in telemetry streams.

* **Interpretability and Diagnostics**
  Confusion matrices, ROC curves, reliability diagrams, and confidence distributions.

* **Fully Reproducible Pipeline**
  Deterministic preprocessing, consistent splits, and version-controlled code.

---

## Dataset

This project uses the publicly available **MSL telemetry dataset** from NASA:
🔗 [https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl)

* Includes multivariate time-series telemetry for MSL subsystems
* Contains pre-annotated anomaly intervals
* After preprocessing: **27 files** and **~73,729 records**

To run this repository, download the MSL data from Kaggle and place it in:

```
data/msl/
```

(Any alternative path can be set in the configuration file.)

---

## How to Run the Code (Opening & Executing the Files)

Below is the minimum set of steps to reproduce the full pipeline.

### 1. **Clone the repository**

```bash
git clone https://github.com/Malbasahi/Ontology-Grounded-NLP-with-Anomaly-Detection-for-Autonomous-Science-Discovery-on-Mars.git
cd Ontology-Grounded-NLP-with-Anomaly-Detection-for-Autonomous-Science-Discovery-on-Mars
```

### 2. **Install dependencies**

```bash
pip install -r requirements.txt
```

### 3. **Download the dataset**

Place the MSL data under:

```
data/msl/
```

### 4. **Open the project structure**

The main components are:

```
/textification/     → numeric → linguistic conversion  
/models/            → implementations of SVC, GBoost, CRF, TextCNN, BERT, NLI  
/utils/             → preprocessing, windowing, plotting  
/results/           → outputs (metrics, confusion matrices, ROC curves)  
```

### 5. **Run textification**

```bash
python textification/run_textification.py
```

This generates the textified dataset under:

```
data/textified/
```

### 6. **Train a model**

Example: Gradient Boosting

```bash
python models/run_gboost.py
```

Example: TextCNN

```bash
python models/run_textcnn.py
```

### 7. **Open and inspect results**

All evaluation outputs will be generated under:

```
results/
results/plots/
results/metrics/
```

To view a specific file:

```bash
xdg-open results/plots/confusion_matrix_gboost.png
```

or on Windows:

```powershell
start results\plots\confusion_matrix_gboost.png
```

---

## Summary Results

| Model                  | Accuracy | Macro-F1 | Notable Strength                    |
| ---------------------- | -------- | -------- | ----------------------------------- |
| **Gradient Boosting**  | 0.78     | **0.42** | Best quantitative anomaly detection |
| **CRF (seq + policy)** | 0.80     | 0.35     | Strong temporal consistency         |
| **TextCNN**            | 0.75     | 0.41     | Robust contextual detection         |
| **BERT**               | **0.82** | 0.30     | High accuracy, majority bias        |
| **Zero-Shot NLI**      | 0.29     | 0.21     | Semantic baseline                   |
| **TF–IDF + SVC**       | 0.78     | 0.31     | Reliable linear baseline            |

---

## 🔧 Future Work

* Improve minority anomaly detection via
  **focal / contrastive learning** and **data augmentation**
* Integrate CRF with transformer-derived embeddings
* Expand subsystem templates with richer temporal patterns
* Evaluate the framework in **real-time/onboard computing environments**
* Explore semi-supervised and domain-adaptive training

---

✔ add examples of generated outputs

Just tell me your preferred level of formality or detail.
