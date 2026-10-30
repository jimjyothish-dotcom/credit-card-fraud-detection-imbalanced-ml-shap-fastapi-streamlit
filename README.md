# End-to-End Credit Card Fraud Detection on Imbalanced Data

A production-style, resume-ready Data Science project that solves **credit card fraud detection** as a highly imbalanced binary classification problem using robust statistics, multiple ML strategies, explainability (SHAP), API serving (FastAPI + Uvicorn), and an interactive dashboard (Streamlit).

## 📋 Project Overview

Fraud detection is a rare-event classification challenge where only a tiny fraction of transactions are fraudulent. This project demonstrates an end-to-end workflow that prioritizes:

- [x] Practical model performance on imbalanced data
- [x] Statistically grounded analysis
- [x] Explainability with SHAP
- [x] Deployment readiness through FastAPI + Streamlit

## 🎯 Problem Statement

Given transaction-level features (`Time`, `Amount`, and anonymized PCA-like features `V1` to `V28`), predict whether a transaction is fraudulent (`Class = 1`) or not (`Class = 0`).

## ⚠️ Why Fraud Detection is Challenging

- **Extreme class imbalance** (fraud << non-fraud)
- **False negatives are costly** (missed fraud = direct financial loss)
- **False positives create review overhead** and customer friction
- **Accuracy alone can be misleading** (model can predict "non-fraud" everywhere and still show high accuracy)

## 📊 Why Imbalance Matters

If a model predicts every transaction as non-fraud, it can still show very high accuracy but detect **no fraud**. This project emphasizes:

| Metric | Why It Matters |
|--------|---------------|
| **Precision** | Of flagged transactions, how many are actually fraud? |
| **Recall** | Of actual fraud cases, how many are caught? |
| **F1-score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Model ranking quality across all thresholds |
| **PR-AUC (Average Precision)** | Performance on the positive (fraud) class |
| **Confusion Matrix** | True/False Positives/Negatives |
| **Threshold tuning** | Balances precision-recall tradeoff based on business needs |

## 🛠️ Tech Stack

- **Python**
- **Pandas, NumPy, SciPy**
- **Scikit-learn** (Logistic Regression, Random Forest, etc.)
- **imbalanced-learn** (`SMOTE`, `SMOTEENN`)
- **XGBoost** (with HistGradientBoosting fallback)
- **Matplotlib, Seaborn**
- **SHAP**
- **FastAPI + Uvicorn**
- **Streamlit**
- **Joblib, Pydantic**
- **Pytest**
- **DagsHub + MLflow** (experiment tracking)
- **Docker**

## 📁 Folder Structure

```
credit-card-fraud-detection/
+-- README.md
+-- requirements.txt
+-- .gitignore
+-- Dockerfile
+-- docker-compose.yml
+-- app.py
+-- data/
|   +-- raw/
|   +-- processed/
|   +-- sample/
+-- notebooks/
|   +-- 01_data_understanding.ipynb
|   +-- 02_eda_statistics.ipynb
|   +-- 03_modeling_imbalanced_data.ipynb
|   +-- 04_explainability_shap.ipynb
+-- src/
|   +-- __init__.py
|   +-- config.py
|   +-- logger.py
|   +-- exception.py
|   +-- utils.py
|   +-- data_ingestion.py
|   +-- data_validation.py
|   +-- preprocessing.py
|   +-- feature_engineering.py
|   +-- statistical_analysis.py
|   +-- model_training.py
|   +-- model_evaluation.py
|   +-- explainability.py
|   +-- prediction_pipeline.py
+-- backend/
|   +-- fastapi_app.py
+-- dashboard/
|   +-- streamlit_app.py
+-- artifacts/
|   +-- models/
|   +-- metrics/
|   +-- plots/
|   +-- shap/
+-- tests/
    +-- test_data.py
    +-- test_model.py
    +-- test_api.py
```

## 📦 Dataset Setup

Use the **Kaggle Credit Card Fraud Detection** dataset.

Place the CSV at:

```
data/raw/creditcard.csv
```

**Expected schema:**
- Features: `Time`, `V1` through `V28`, `Amount`
- Target: `Class`

## 🌿 Environment Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
python -m pip install --upgrade pip
pip install -r requirements.txt -c constraints.txt
```

**Notes:**
- `requirements.txt` keeps dependency names flexible.
- `constraints.txt` pins a tested set for reproducible local setup.
- If you intentionally want latest packages, run `pip install -r requirements.txt` without constraints.

## 🚀 Run Training Pipeline

This single command runs ingestion, validation, statistics, EDA plots, model comparison, threshold tuning, and SHAP artifact generation:

```bash
python app.py train --data-path data/raw/creditcard.csv
```

**Optional quick-run mode:**

```bash
python app.py train --data-path data/raw/creditcard.csv --sample-size 120000 --disable-smoteenn
```

**DagsHub MLflow tracking** (enabled by default):

```bash
set DAGSHUB_REPO_OWNER=<USER>
set DAGSHUB_REPO_NAME=<REPO_NAME>
set MLFLOW_EXPERIMENT_NAME=credit-card-fraud-detection
python app.py train --data-path data/raw/creditcard.csv
```

**Disable MLflow for a single run:**

```bash
python app.py train --data-path data/raw/creditcard.csv --disable-mlflow
```

**Generated outputs include:**
- `artifacts/models/best_model.joblib`
- `artifacts/models/model_metadata.json`
- `artifacts/metrics/model_comparison.csv`
- `artifacts/metrics/best_model_metrics.json`
- `artifacts/metrics/statistics_summary.json`
- Plots and SHAP images under `artifacts/plots/` and `artifacts/shap/`

## ⚡ Run API (FastAPI)

```bash
uvicorn backend.fastapi_app:app --host 0.0.0.0 --port 8000 --reload
```

**Endpoints:**
- `GET /` — Root endpoint
- `GET /health` — Health check
- `POST /predict` — Single transaction prediction
- `POST /predict_batch` — Batch prediction

## 🐚 Run Dashboard (Streamlit)

```bash
streamlit run dashboard/streamlit_app.py
```

**Dashboard sections:**
- [x] Project overview
- [x] Class imbalance summary
- [x] EDA plots
- [x] Model performance table and key metrics
- [x] Confusion matrix / ROC / PR visuals
- [x] Sample transaction prediction form
- [x] SHAP explainability visuals

## 📈 Statistical Analysis Included

Implemented in notebooks and `src/statistical_analysis.py`:

- mean, median, std, variance
- quartiles and IQR
- skewness and kurtosis
- class ratio and imbalance ratio
- IQR-based outlier detection
- Correlation with target
- Fraud vs non-fraud comparison
- Hypothesis-style tests (`t-test`, `Mann-Whitney U`)
- Interpretation of false positives/false negatives business impact

## 🤖 Modeling Included

- Dummy baseline
- Logistic Regression (with `class_weight='balanced'`)
- Random Forest (`class_weight='balanced_subsample'`)
- XGBoost (if installed) or HistGradientBoosting fallback
- SMOTE + Logistic Regression
- SMOTEENN + Random Forest (conditionally enabled for manageable dataset sizes)

## 📊 Evaluation Strategy

**Main focus metrics:**
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrix

**Threshold tuning** is done via precision-recall curve (F1 optimization with recall floor).

## 📝 Explainability

SHAP outputs include:

- Global importance (bar plot)
- Global summary (beeswarm plot)
- Local waterfall explanations for high-risk transactions

Artifacts are saved to `artifacts/shap/`.

## 🧪 Testing

Run:

```bash
pytest -q
```

**Tests cover:**
- Ingestion/validation sanity
- Prediction output schema sanity
- API endpoint response sanity

## 🐋 Docker

Build and run API + dashboard:

```bash
docker compose up --build
```

- **API:** `http://localhost:8000`
- **Dashboard:** `http://localhost:8501`

## 💼 Resume-Ready Summary

This project demonstrates:

- Practical statistics for fraud analytics
- Imbalanced classification strategy design
- Model benchmarking and threshold tuning
- Explainable AI with SHAP
- Production-style modular code in Python
- Deployment skills via FastAPI, Streamlit, and Docker

## 🔮 Future Improvements

- cost-sensitive learning with explicit fraud-loss matrix
- drift monitoring and periodic model retraining
- feature store and orchestration (Airflow/Prefect)
- model registry and CI/CD deployment pipeline
- real-time inference with message queue integration

---

*Built with Claude Code. 🤖 Generated with [Claude Code](https://claude.com/claude-code)*