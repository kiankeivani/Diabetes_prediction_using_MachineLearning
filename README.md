# Diabetes Prediction with Classical Machine Learning

> Data Analysis course | Author: Kian Keivani

A Jupyter Notebook project that predicts whether a person has diabetes from basic clinical and demographic measurements. Four classical classifiers (Decision Tree, K-Nearest Neighbors, Support Vector Machine, and Gaussian Naive Bayes) are trained on the same data and compared with the same evaluation metrics and confusion matrices.

The model is intended as a **decision-support idea for early screening and risk management**, not as a clinical diagnostic tool.

---

## Table of Contents

1. [Project Goal](#project-goal)
2. [Dataset](#dataset)
3. [Workflow](#workflow)
4. [Models](#models)
5. [Evaluation](#evaluation)
6. [Results](#results)
7. [Interpretation](#interpretation)
8. [Limitations and Notes](#limitations-and-notes)
9. [Getting Started](#getting-started)
10. [Repository Structure](#repository-structure)
11. [Tech Stack](#tech-stack)

---

## Project Goal

Build a binary classifier that labels a person as **diabetic (1)** or **non-diabetic (0)** from five measurements, then compare several standard algorithms on held-out test data. In a screening setting, missing a diabetic patient (a false negative) is the costly error, so the evaluation pays particular attention to recall on the diabetic class.

---

## Dataset

- **Samples:** 360
- **Features:** 5 input features and 1 target (6 columns)
- **Missing values:** none
- **Data types:** all numeric (`Age`, `Glucose`, `Insulin`, `DiastolicBP`, `Outcome` are integers; `BMI` is a float)

| Column | Description |
|---|---|
| `Age` | Age of the person |
| `BMI` | Body Mass Index |
| `Glucose` | Blood glucose level |
| `Insulin` | Insulin level |
| `DiastolicBP` | Diastolic blood pressure |
| `Outcome` | Target: `1` = diabetic, `0` = non-diabetic |

**Class balance:** about 72.5% of the samples are diabetic (mean of `Outcome` = 0.725), so the classes are imbalanced.

> **Note:** the dataset was provided as part of a course assignment. Check its terms before redistributing it, and add it to the repository only if allowed.

---

## Workflow

1. **Loading and exploration**
   - Load the CSV with Pandas.
   - Inspect shape, column names, data types, and missing values.
   - Compute descriptive statistics (mean, standard deviation, min, max, quartiles) for every feature.
2. **Preprocessing**
   - Separate the features (`X`) from the target (`y = Outcome`).
   - Split the data into **80% training (288 samples)** and **20% testing (72 samples)** with `random_state=42`.
   - Standardize the features with `StandardScaler`. The scaler is **fit on the training set only** and then applied to the test set, to avoid data leakage.
3. **Modeling and training**
   - Train the four classifiers on the scaled training data.
4. **Evaluation**
   - Predict on the test set and compute Accuracy, Precision, Recall, and F1-score (macro averages).
   - Print a per-class classification report.
   - Plot and interpret a confusion matrix for each model.

---

## Models

| Model | scikit-learn class | Settings |
|---|---|---|
| Decision Tree | `DecisionTreeClassifier` | `criterion="gini"`, `max_depth=4` |
| K-Nearest Neighbors | `KNeighborsClassifier` | `n_neighbors=4` |
| Support Vector Machine | `SVC` | `kernel="rbf"` (default parameters) |
| Naive Bayes | `GaussianNB` | default parameters |

---

## Evaluation

Metrics are computed on the 72 held-out test samples (53 diabetic, 19 non-diabetic):

- **Accuracy:** share of correct predictions
- **Precision (macro):** how reliable the positive predictions of each class are, averaged over the classes
- **Recall (macro):** how many real members of each class are found, averaged over the classes
- **F1-score (macro):** harmonic mean of precision and recall
- **Confusion matrix** per model, with positive = diabetic (1) and negative = non-diabetic (0)

---

## Results

All four models produced the **same predictions** on the test set:

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| Decision Tree | 0.986 | 0.975 | 0.991 | 0.982 |
| K-Nearest Neighbors | 0.986 | 0.975 | 0.991 | 0.982 |
| SVM (RBF) | 0.986 | 0.975 | 0.991 | 0.982 |
| Gaussian Naive Bayes | 0.986 | 0.975 | 0.991 | 0.982 |

**Shared confusion matrix** (positive = diabetic):

| | Predicted diabetic | Predicted non-diabetic |
|---|---|---|
| **Actual diabetic** | TP = 52 | FN = 1 |
| **Actual non-diabetic** | FP = 0 | TN = 19 |

Derived metrics for the diabetic class: recall (sensitivity) ≈ 0.981, precision = 1.00, specificity = 1.00, F1 ≈ 0.99.

---

## Interpretation

- Only **one diabetic patient** was classified as healthy (FN = 1). In screening, false negatives matter most because a missed case can delay treatment, so a low FN count is the main strength of these models.
- There are **no false positives** (FP = 0), so no healthy person was flagged as diabetic.
- Because the four models give identical results, **none can be declared better on numeric performance alone**. The final choice can rest on qualitative criteria: interpretability (Decision Tree), implementation simplicity (Naive Bayes, KNN), or flexibility on more complex data (SVM).

---

## Limitations and Notes

- **Small test set:** 72 samples, so one error changes the metrics noticeably, and the models cannot be reliably separated from each other.
- **Single split:** results come from one train/test split. Stratified splitting and cross-validation would give more robust estimates.
- **Class imbalance:** about 72.5% of samples are diabetic, so accuracy alone can be misleading. Recall and the confusion matrix are the more informative measures.
- **No tuning:** hyperparameters were set by hand (`max_depth=4`, `k=4`) or left at defaults. A grid search could improve and stabilize the results.
- **Very high scores:** near-perfect results on a small, clean dataset suggest the classes are easy to separate here. Performance on real-world clinical data is likely to be lower.
- **Not medical advice:** this is an educational project and must not be used for real diagnosis.

Possible extensions: stratified k-fold cross-validation, hyperparameter tuning, ROC/AUC analysis, feature-importance analysis, and testing on a larger public dataset.


---

## Tech Stack

- **Language:** Python 3 (Jupyter Notebook)
- **Data handling:** Pandas, NumPy
- **Machine learning:** scikit-learn (`DecisionTreeClassifier`, `KNeighborsClassifier`, `SVC`, `GaussianNB`, `StandardScaler`, metrics)
- **Visualization:** Matplotlib, Seaborn
