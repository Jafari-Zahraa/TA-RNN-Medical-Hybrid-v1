# TA-RNN-Medical-Hybrid

**Time-Aware and Knowledge-Driven Interpretable Framework for Longitudinal ICU Mortality Prediction**

---

## 1. Introduction

This repository provides the official implementation of **TA-RNN-Medical-Hybrid**, a time-aware and interpretable deep learning architecture designed for mortality risk prediction from longitudinal electronic health records (EHRs).

The proposed framework integrates external medical knowledge, explicit temporal modeling, sequential representation learning, and dual-level attention mechanisms to ensure both predictive performance and clinical interpretability.

The model is evaluated on the **MIMIC-III (Medical Information Mart for Intensive Care III)** critical care database.

---

## 2. Dataset

Experiments are conducted using the MIMIC-III database, a large-scale, de-identified ICU dataset comprising demographic information, longitudinal clinical observations, diagnosis codes, procedures, and mortality outcomes.

Due to data usage restrictions, the dataset is **not included** in this repository.

To obtain access:

1. Complete required credentialing via PhysioNet.
2. Request access to MIMIC-III.
3. Download the dataset locally.
4. Follow preprocessing instructions in the `data_preprocessing/` directory.

All preprocessing steps are performed at the patient level to prevent information leakage.

---

## 3. Methodological Overview

The TA-RNN-Medical-Hybrid framework consists of the following core components:

### 3.1 Knowledge-Driven Medical Embedding

* ICD diagnosis codes are mapped to SNOMED CT concepts.
* Textual embeddings are generated using a pretrained clinical language model.
* Structural embeddings are derived from the SNOMED relational graph using GraphSAGE.
* Textual and structural representations are concatenated to form enriched medical embeddings.
* The embedding matrix is fixed during training to preserve encoded clinical semantics.

### 3.2 Explicit Temporal Encoding

Irregular visit intervals are modeled using normalized elapsed-time encoding.
A sinusoidal embedding mechanism transforms continuous time differences into dense temporal representations, ensuring time-aware modeling without enforcing regular visit spacing.

### 3.3 Sequential Visit Modeling

* Visit-level representations are processed using a two-layer Bidirectional GRU (BiGRU).
* Multi-head self-attention captures long-range temporal dependencies.
* Residual connections and layer normalization enhance training stability.

### 3.4 Dual-Level Interpretability

The framework incorporates:

* Visit-level attention weights (α)
* Disease-level feature importance scores (β)

This design enables clinically meaningful interpretation at both temporal and diagnostic levels.

### 3.5 Demographic Fusion and Prediction

Static demographic features are concatenated with learned temporal representations.
Final mortality risk predictions are generated via a multilayer perceptron followed by sigmoid activation.

---

## 4. Loss Function and Optimization

Training is performed end-to-end using a weighted binary cross-entropy loss to address class imbalance in ICU mortality prediction.

The positive class weighting parameter δ is tuned empirically, emphasizing recall and sensitivity.

Optimization details:

* Optimizer: Adam
* Learning rate: 0.001
* Early stopping based on validation loss
* Dropout and L2 regularization applied

---

## 5. Evaluation Metrics

Model performance is assessed using:

* Accuracy
* Area Under the ROC Curve (AUC)
* F2-score (β = 2)

The decision threshold is optimized on validation data to maximize the F2-score, prioritizing recall in high-risk ICU scenarios.

---

## 6. Computational Complexity

Let:

* T = number of visits
* K = number of diagnosis codes per visit
* d = embedding dimension
* h = hidden dimension
* H = number of attention heads
* L = number of recurrent layers

The overall computational complexity is:

O(TKd + LTh(d + h) + HT²h)

The model remains computationally feasible for typical longitudinal EHR settings.

---

## 7. Reproducibility

All experiments are conducted with patient-level data splitting to prevent leakage.
Hyperparameters are tuned using a held-out validation subset (20% of training data).
Random seeds are fixed to ensure experimental reproducibility.

---

## 8. Ethical Considerations

This study uses de-identified clinical data.
All experiments must comply with PhysioNet credentialing and data usage agreements.
No identifiable patient information is included in this repository.

---


##9. Contact


For academic correspondence, technical questions, or potential research 
collaborations, please contact the corresponding author at:

jafari.zahra2637@gmail.com

Kindly reference “TA-RNN-Medical-Hybrid” in the subject line.
