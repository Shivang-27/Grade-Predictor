# 🎓 Student Grade Predictor

A Random Forest regression model that predicts student total scores based on study habits, attendance, and class participation — trained on 1,000,000 student records.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter)](https://jupyter.org)

---

## Dataset

| Property | Detail |
|---|---|
| **Source** | `student_performance.csv` |
| **Size** | 1,000,000 student records |
| **Target** | Total Score (numeric, 9.4–100) |
| **Mean Score** | 84.3 |
| **Score Std Dev** | 15.4 |

### Correlation with Total Score

| Feature | Correlation |
|---|---|
| `weekly_self_study_hours` | **+0.812** |
| `class_participation` | ~0.001 (negligible) |
| `attendance_percentage` | ~-0.001 (negligible) |

> Weekly self-study hours has a strong positive correlation (0.81) with total score. Attendance and class participation show almost no linear correlation — an unexpected and analytically interesting finding.

---

## Features

| Feature | Description |
|---|---|
| `weekly_self_study_hours` | Hours of self-study per week (0–40) |
| `attendance_percentage` | Class attendance rate (50–100%) |
| `class_participation` | Participation score (0–10) |

Stratified split on weekly study hour bands ensured balanced train/test distribution across all study levels.

---

## Pipeline

A `ColumnTransformer` with a numerical pipeline was used:
- **Numerical** — Median imputation + Standard scaling

---

## Model

```
Algorithm       : Random Forest Regressor
Estimators      : 200 trees
Max Depth       : 10
Min Samples Leaf: 20
Train/Test      : 80% / 20%  (stratified by study hour bands)
```

---

## Results

| Metric | Value |
|---|---|
| **R² (Test)** | **0.716** |
| **MAE** | 6.119 |
| **RMSE** | 8.223 |
| Train R² | 0.720 |

> The near-identical train and test R² (0.720 vs 0.716) confirms the model generalises well with no overfitting. An R² of 0.716 means the model explains ~72% of the variance in student scores — strong performance given only 3 input features.

---

## Feature Importance

| Feature | Importance |
|---|---|
| `weekly_self_study_hours` | **0.9966** |
| `attendance_percentage` | 0.0020 |
| `class_participation` | 0.0015 |

> Weekly self-study hours accounts for **99.7% of the model's predictive power**. This extreme dominance suggests the dataset was designed with this relationship in mind — but it also demonstrates that the model successfully identified and prioritised the most informative signal over noise.
