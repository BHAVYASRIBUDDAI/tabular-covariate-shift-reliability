# Quantifying Model Reliability under Covariate Shift in Tabular Machine Learning

This project studies how classical machine learning models behave under covariate shift
using the UCI Adult Income (Census) dataset. The focus is on **model reliability and calibration**,
rather than just raw accuracy.

Covariate shift occurs when the distribution of input features changes between training and deployment.
Even if overall accuracy remains reasonable, the predicted confidence can be misleading.
This project demonstrates how to detect, mitigate, and correct these reliability issues in tabular ML pipelines.

---

## Features and Techniques

- **Baseline models**: Logistic Regression and Random Forest
- **Covariate shift detection**: Train a classifier to quantify distribution shift
- **Mitigation strategies**:
  - Importance weighting during training
  - Post-hoc calibration using temperature scaling
- **Evaluation**:
  - Accuracy and F1-score under shifted test sets
  - Expected Calibration Error (ECE) and reliability diagrams

---
##Install Dependencies
pip install -r requirements.txt


