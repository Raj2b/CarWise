# Used Car Price Band Classifier

## 1. Overview

This project builds a classifier that predicts a **price band** for used car listings based on:

- **Structured specs** (year, manufacturer, model, condition, cylinders, fuel, odometer, title_status, transmission, type)
- **Unstructured text** (the seller’s description)

The goal is not to predict an exact dollar price, but to place each listing into one of **five price ranges** using interpretable AI methods 

---

## 2. Price Bands

We convert the raw `price` into a **discrete label** `price_band`:

- **Band 0**: \$0 – \$8,000  
- **Band 1**: \$8,001 – \$15,000  
- **Band 2**: \$15,001 – \$25,000  
- **Band 3**: \$25,001 – \$40,000  
- **Band 4**: \$40,001 and Up

The model’s job is:  
> Given the listing specs and/or description, **predict which band (0–4)** the car belongs to.

---

## 3. Repository Structure

Main files:

- `main.py`  
  Orchestrates the whole pipeline:
  - loads and cleans `data.csv`
  - makes train/val/test splits
  - builds feature matrices
  - trains and tunes models
  - shows validation results
  - lets you choose a model and evaluates it on the **held-out test set**.

- `data_utils.py`  
  - `load_and_clean_data(...)`: loads `data.csv`, cleans prices, filters outliers, optional subsampling, and creates the `price_band` label.  
  - `make_splits(...)`: splits into **train (60%) / val (20%) / test (20%)** with stratification by band.

- `features.py`  
  - `build_spec_feature_matrix(...)`: turns structured specs into a numeric matrix using:
    - median imputation (numeric)
    - one-hot encoding (categorical)
  - `build_text_feature_matrix(...)`: builds **bag-of-words** features from `description` using `CountVectorizer`.

- `models.py`  
  Implements all models:
  - `tune_decision_tree(...)`: Decision Tree with small hyperparameter search.
  - `train_knn(...)`: k-Nearest Neighbours on spec features.
  - `train_naive_bayes_text(...)`: Multinomial Naive Bayes on text features.
  - `train_decision_tree(...)`: simple fixed-param tree (kept as a baseline/debug model).

- `requirements.txt`  
  Python dependencies (pandas, numpy, scikit-learn, etc.).

---

## 4. Models Used


1. **Decision Tree (spec-only)**  
   - Uses only structured vehicle specs.
   - Hyperparameters tuned on the validation set:
     - `max_depth ∈ {10, 15, 20, None}`
     - `min_samples_leaf ∈ {20, 50, 100}`  
   - We pick the combo with the best **validation macro-F1**.

2. **k-Nearest Neighbours (spec-only)**  
   - Uses only structured specs.
   - For each new car, finds the **25 closest cars** in training and votes on the band.

3. **Naive Bayes (text-only)**  
   - Uses only `description` (bag-of-words).
   - Multinomial Naive Bayes, a standard baseline for text classification.

4. **Decision Tree (fused: spec + text)**  
   - Concatenates spec features and text features into one big matrix.
   - Runs the same hyperparameter search as spec-only.
   - This is usually our **best overall model**.

We evaluate models using:

- **Accuracy** (overall % correct), and  
- **Macro-F1** (average F1 across all 5 bands, treating bands more evenly even if they’re imbalanced).

---

## 5. What the Script Does

### 5.1 Load & Clean Data

- Reads `data.csv`.
- Converts `price` to numeric, drops invalid or missing prices.
- Filters out extreme prices.
- Fills missing `description` fields and removes duplicate rows.
- Optionally subsamples to at most **100,000** rows for speed.
- Creates a `price_band` label using **5 ranges** (0–8k, 8–15k, 15–25k, 25–40k, 40k+).

### 5.2 Train / Validation / Test Split (60 / 20 / 20)

- Uses `make_splits(...)` with **stratified sampling** so each `price_band` is represented.
- Split ratio:
  - 60% training  
  - 20% validation  
  - 20% test  

### 5.3 Build Feature Matrices

- `build_spec_feature_matrix(...)` →  
  Produces `X_train_spec`, `X_val_spec`, `X_test_spec` from structured specs  
  (year, manufacturer, model, condition, cylinders, fuel, odometer, etc.).

- `build_text_feature_matrix(...)` →  
  Produces `X_train_text`, `X_val_text`, `X_test_text` from listing descriptions  
  using a bag-of-words (CountVectorizer) representation.

- Fused matrix:  
  - `X_*_fused = [spec, text]` built using sparse `hstack` to combine both views.

### 5.4 Train & Validate Models

The script trains **4 models** and evaluates them on the validation set:

1. **Decision Tree (spec-only)**
   - Tuned with a small hyperparameter search (`max_depth`, `min_samples_leaf`).
   - Prints validation:
     - Accuracy  
     - Macro-F1  
     - Confusion matrix  

2. **k-Nearest Neighbours (spec-only)**
   - Uses imputed + one-hot-encoded spec features.
   - Prints validation accuracy, macro-F1, and confusion matrix.

3. **Naive Bayes (text-only)**
   - Multinomial Naive Bayes on bag-of-words description features.
   - Prints validation accuracy, macro-F1, and confusion matrix.

4. **Decision Tree (fused spec+text)**
   - Trained on the concatenated spec + text features.
   - Tuned with hyperparameter search.
   - Typically achieves the best validation performance.

### 5.5 Validation Summary

After training, the script prints a short validation summary:

- Validation accuracy for:
  - Decision Tree (spec-only)
  - k-NN (spec-only)
  - Naive Bayes (text-only)
  - Decision Tree (fused spec+text)

This makes it easy to compare which model works best before testing.


