# HealthAware ML Dataset & Model Evaluation Results

This folder contains the official dataset and comprehensive machine learning model results for the **HealthAware Diabetes Screening and Awareness Application**.

---

## 📂 Folder Contents

* **`nhanes_diabetes_2021_2023.csv`**: The processed US NHANES dataset containing 6,001 survey participant records used for model training, validation, and testing.
* **`README.md`**: Complete documentation of dataset schema, training structure, hyperparameter configurations, evaluation metrics, feature importances, and confusion matrix analysis.

---

## 📊 1. Dataset Overview & Feature Schema

* **Data Source:** US NHANES (National Health and Nutrition Examination Survey) 2021–2023 cycle.
* **Total Records:** 6,001 participants.
* **Target Variable (`glycemic_status`):**
  * `0: Normal` (2,640 records — 44.0%)
  * `1: Prediabetes` (2,308 records — 38.5%)
  * `2: Diabetes` (1,053 records — 17.5%)

### 11 Non-Invasive Model Features

| # | Feature Name | Data Type / Value Range | Description |
|---|---|---|---|
| 1 | `age_years` | Numeric (18–80) | Age of the participant in years |
| 2 | `sex` | Categorical (`male`, `female`) | Biological sex |
| 3 | `bmi_kg_m2` | Numeric (12.0–80.0) | Body Mass Index in kg/m² |
| 4 | `hypertension_diagnosed` | Binary (`0` = No, `1` = Yes) | Self-reported diagnosis of high blood pressure |
| 5 | `high_cholesterol_diagnosed` | Binary (`0` = No, `1` = Yes) | Self-reported diagnosis of high cholesterol |
| 6 | `stroke` | Binary (`0` = No, `1` = Yes) | History of stroke |
| 7 | `heart_disease_or_attack` | Binary (`0` = No, `1` = Yes) | History of coronary heart disease or heart attack |
| 8 | `meets_activity_guidelines` | Binary (`0` = No, `1` = Yes) | Meets physical activity guidelines (≥150 min/week) |
| 9 | `smoking_status` | Categorical (`never`, `former`, `current`) | Tobacco smoking status |
| 10 | `heavy_alcohol_use` | Binary (`0` = No, `1` = Yes) | Alcohol consumption exceeding recommended limits |
| 11 | `general_health_label` | Categorical (`Excellent`, `Very good`, `Good`, `Fair`, `Poor`) | Self-reported general health status |

---

## 🏗️ 2. Training Structure & Methodology

* **Algorithm:** `CatBoostClassifier` (Gradient Boosted Decision Trees).
* **Dataset Split Strategy (60 : 20 : 20 Stratified Split):**
  * **Training Set:** 3,600 samples (60%) — Used for tree construction.
  * **Validation Set:** 1,200 samples (20%) — Used for early stopping (prevents overfitting).
  * **Test Set:** 1,201 samples (20%) — Held-out set for final performance evaluation.
  * *Stratification:* Preserved 44% Normal, 38.5% Prediabetes, and 17.5% Diabetes proportions across all splits (`random_states=[2026, 2027]`).

### Hyperparameter Configuration & Class Weighting
```python
{
    "iterations": 1000,
    "depth": 5,
    "learning_rate": 0.03,
    "l2_leaf_reg": 8,
    "class_weights": [0.5, 1.2, 2.5],
    "random_seed": 2026,
    "early_stopping_rounds": 100
}
```

* **Why Class Weights `[0.5, 1.2, 2.5]` Were Used:**
  * Unweighted training caused severe class imbalance bias toward **Normal** (44% of dataset), resulting in a 62% false negative rate for Diabetes.
  * Applying `[0.5, 1.2, 2.5]` weights penalizes misclassifying Prediabetes and Diabetes, maximizing screening recall for high-risk individuals.

---

## 📈 3. Model Results & Performance Metrics

### Test Set Metrics (1,201 Held-Out Samples)

| Metric | Score | Explanation |
| :--- | :--- | :--- |
| **Diabetes Recall (`Class 2`)** | **66.35%** | 140 out of 211 Diabetic cases correctly identified |
| **Diabetes Severe False Negative Rate** | **0.95%** | Only 2 out of 211 Diabetic cases misclassified as Normal |
| **Prediabetes Recall (`Class 1`)** | **54.11%** | 250 out of 462 Prediabetic cases identified |
| **Normal Recall (`Class 0`)** | **33.52%** | Shifted to prioritize high-risk detection in screening |
| **Macro F1 Score** | **0.4724** | Unweighted mean F1 across 3 classes |
| **Weighted F1 Score** | **0.4714** | F1 score weighted by class sample sizes |
| **Overall Accuracy** | **47.21%** | Overall 3-class classification accuracy |

### Confusion Matrix (Rows = Actual Class, Columns = Predicted Class)

```
                  Predicted Normal   Predicted Prediabetes   Predicted Diabetes
Actual Normal           177                  278                    73
Actual Prediabetes       48                  250                   164
Actual Diabetes           2                   69                   140  <-- (66.4% Recall)
```

---

## 🔑 4. Feature Importance Ranking

Feature importance values extracted from CatBoost decision tree splits:

| Feature Name | Importance (%) | Impact Summary |
| :--- | :--- | :--- |
| **`age_years`** | **28.07%** | Single largest risk indicator |
| **`bmi_kg_m2`** | **20.71%** | Key continuous obesity measure |
| **`general_health_label`** | **18.43%** | Strong subjective health perception indicator |
| **`hypertension_diagnosed`** | **9.15%** | Vascular comorbidity indicator |
| **`high_cholesterol_diagnosed`** | **8.23%** | Metabolic comorbidity indicator |
| **`smoking_status`** | **6.44%** | Behavioral factor |
| **`sex`** | **4.17%** | Demographic baseline |
| **`meets_activity_guidelines`** | **2.04%** | Exercise indicator |
| **`heavy_alcohol_use`** | **1.51%** | Lifestyle indicator |
| **`heart_disease_or_attack`** | **0.94%** | Cardiovascular disease history |
| **`stroke`** | **0.30%** | Cerebrovascular history |

---

## ⚠️ 5. Research Limitations & Usage Scope

1. **Screening Prototype Only:** This non-invasive survey model is designed for educational health awareness and risk stratification. It is not a clinical diagnostic tool and cannot replace blood glucose testing (HbA1c / Fasting Plasma Glucose).
2. **Geographic Validation:** The model is trained on US NHANES data. Local population validation and calibration should be performed before deploying in other regions (e.g., East Africa / Kenya).
