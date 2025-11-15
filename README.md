# Parkinsons detector

This project explores early-stage Parkinson’s screening from vocal markers.
The aim is to build a lightweight model that can help flag individuals who may
benefit from a neurological evaluation. The work is motivated by family history, 
personal interest in machine learning architectures 
and the practical need for low-cost, accessible preclinical tools.

---

## Problem

Parkinson’s disease is usually diagnosed only after noticeable motor symptoms
appear. However, measurable irregularities in voice (e.g., pitch stability,
jitter, shimmer, tremor-related modulation) show up much earlier.  
This repository investigates whether these subtle acoustic patterns can be
separated reliably using machine-learning models.

---

## Approach

The task is framed as a binary classification problem:

**Parkinson’s (1) vs. healthy control (0)**

### Pipeline overview
1. **Data preparation**
   - cleaned feature tables (`train_features.csv`, `test_features.csv`)
   - stratified train/test split to preserve class balance

2. **Preprocessing**
   - scaling experiments (Standard, Robust, MinMax, and unscaled variants)
   - optional feature selection using ANOVA F-scores

3. **Modeling**
   - classical ML models: Logistic Regression, SVM (RBF/linear),
     Random Forest, Gradient Boosting, AdaBoost, KNN, Naive Bayes,
     MLP classifier
   - full cross-validation across all experiments

4. **Evaluation**
   - accuracy, precision, recall, F1
   - AUC-ROC as the main comparison metric
   - confusion matrices, sensitivity/specificity
   - overfitting analysis using train vs. cross-validation gaps

5. **Outputs**
   - results stored under results/ (tables, plots, summary reports)
   - confusion matrices and model comparison charts
   - LaTeX tables for documentation or reports

The full experimental pipeline is implemented in **`experiment.py`**.

research_phase

└── ongoing approaches/         # experimental branches (prototypes)

    │
    ├── cnn + classifier/       # CNN on audio-derived features + classifier head
    │
    ├── linguistic + vocalistic/ # simple linguistic markers fused with acoustic features
    │                           
    ├── multimodal/ (speech + writing)  # exploratory fusion: voice + handwriting/drawing metrics
    │                         
    └── testing a custom model
