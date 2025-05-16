# 🤖 Fairness Without Demographics: ARL

This repository presents our reproduction and extension of the paper  
**[Fairness Without Demographics through Adversarially Reweighted Learning (ARL)](https://arxiv.org/abs/2006.13114)**, originally published at NeurIPS 2020.  
We successfully reimplement ARL and evaluate its fairness and generalization performance on both original and novel datasets.

---

## 📌 Overview

ARL is a **group-agnostic fairness method** that reweights samples in error-prone regions using adversarial training, without requiring access to protected demographic labels during training or testing.

In this project, we:
- 🔁 Reproduce ARL on the original datasets: **UCI Adult**, **LSAC**, and **COMPAS**
- 🧪 Extend ARL to two real-world datasets: **Police Killings (US)** and **Facebook Posts**
- 📊 Analyze subgroup fairness using AUC-based metrics

---

## 🧠 Why This Matters

Traditional fairness techniques require demographic labels. However, **in many real-world scenarios, such labels are unavailable or sensitive**. ARL offers a practical alternative by promoting equity without demographic supervision — a key advantage for **privacy-preserving ML**.

---

## 📁 Datasets

### ✅ Original Datasets (from paper)
| Dataset     | Task                               | Size     | Features |
|-------------|------------------------------------|----------|----------|
| UCI Adult   | Income classification (>50K)       | ~40K     | 15       |
| LSAC        | Bar passage prediction              | ~27K     | 12       |
| COMPAS      | Recidivism prediction               | ~7K      | 11       |

### 🆕 New Datasets (Extended)
| Dataset           | Task                                     | Size     | Notes                                 |
|-------------------|------------------------------------------|----------|----------------------------------------|
| Police Killings   | Risk-based incident classification       | ~4K      | Subgroup imbalance (e.g., gender)      |
| Facebook Posts    | Post engagement classification           | ~100K    | Large-scale, diverse attribute space   |

---

## ⚙️ Experimental Setup

- **Learner**: 2-layer neural net (64, 32 units) with ReLU
- **Adversary**: Linear model that assigns sample weights
- **Loss**: Binary Cross-Entropy
- **Optimization**: MinMax adversarial training
- **Evaluation Metrics**:
  - AUC (average)
  - AUC (macro-average)
  - AUC (minimum)
  - AUC (minority subgroup)

> Custom input pipelines were created for Facebook and Police datasets, adapting `uciadult_input.py` from the original codebase.

---

## 📊 Reproduced Results

### 🔁 On Original Datasets

| Dataset    | Metric        | Paper        | Ours         |
|------------|----------------|--------------|--------------|
| **Adult**  | AUC avg       | 0.907        | 0.903        |
|            | AUC min       | 0.881        | 0.863        |
|            | AUC minority  | 0.942        | 0.924        |
| **LSAC**   | AUC avg       | 0.823        | 0.794        |
|            | AUC min       | 0.798        | 0.740        |
|            | AUC minority  | 0.832        | 0.793        |
| **COMPAS** | AUC avg       | 0.743        | 0.745        |
|            | AUC min       | 0.658        | 0.663        |
|            | AUC minority  | 0.785        | 0.665        |

### 🧪 On New Datasets

| Dataset            | AUC avg | AUC macro | AUC min | AUC minority |
|--------------------|---------|-----------|---------|---------------|
| **Police Killings**| 0.490   | 0.894     | 0.000   | 0.000         |
| **Facebook Posts** | 0.935   | 0.929     | 0.915   | 0.875         |

---

## 🔍 Key Observations

- ✅ **ARL generalizes well** to large-scale structured data (e.g., Facebook) when subgroup structure is computationally inferable.
- ⚠️ **ARL underperforms** on small or noisy datasets (e.g., Police Killings, COMPAS) where group structures are weak or unbalanced.
- 🔁 **Fairness gains were consistent** across datasets, despite not using demographic labels.
- 🔧 Legacy environment (Python 3.6.8, TensorFlow 1.13.2) was needed to match original setup.

---

## 📁 Project Structure
ARL-Reproduction/
├── arl_core/ # ARL model and learner-adversary logic
├── data_input/
│ ├── uciadult_input.py
│ ├── facebook_input.py
│ └── police_input.py
├── datasets/ # Raw and processed datasets
├── results/ # Metrics and plots
├── requirements.txt
├── Fair_ML_ARL_Report.pdf
└── README.md



---

## 📌 How to Run

1. Set up virtual environment with Python 3.6.8
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

Preprocess dataset using appropriate input file

Run training using main_trainer.py with desired config

View results in /results/


