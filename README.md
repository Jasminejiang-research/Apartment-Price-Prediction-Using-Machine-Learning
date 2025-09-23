# Apartment Price Prediction Using Machine Learning

> An end-to-end apartment price prediction pipeline with standardized regression workflows, an analysis of why traditional feature engineering often fails, and an ensemble that substantially outperforms an XGBoost baseline.

**Status:** `WIP` · **Language:** Python 3.x  · **License:** `MIT`

---

## Badges

```md
![CI](https://img.shields.io/github/actions/workflow/status/Jasmine_research/Apartment-Price-Prediction-Using-Machine-Learning/ci.yml?style=flat-square)
![Python](https://img.shields.io/badge/python-3.8%2B-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![ML](https://img.shields.io/badge/Machine%20Learning-Regression-orange?style=flat-square)
```

---

## Table of Contents

* [Project Overview](#project-overview)
* [Repository Layout](#repository-layout)
* [Quick Start](#quick-start)
* [Data](#data)
* [Notebooks](#notebooks)
* [Code (src/)](#code-src)
* [Training & Inference](#training--inference)
* [Experiments & Results](#experiments--results)
* [Contributing](#contributing)
* [License](#license)
* [Contact & Acknowledgements](#contact--acknowledgements)
* [Checklist](#checklist)

---

## Project Overview

> This repo implements a **reproducible machine learning pipeline** for apartment price prediction, covering:
**data loading → preprocessing → 5-fold cross-validation → baseline & ensemble modeling → model interpretation and ablation analysis**.

#### Project Focus

This project integrates **interpretable machine learning** to investigate a key question:

> Why does **traditional feature engineering** often fail to improve model performance in the presence of **complex data structures** and **large sample sizes**?

I systematically evaluate this through two optimization experiments:

* **Feature engineering** based on EDA-driven hypotheses and time feature construction.
* **Stacked ensembling**, combining linear and tree-based models to leverage their complementary strengths.

#### Key Insight

In this dataset, **manual feature engineering yields limited gains**, while **model ensembling with a linear meta-learner** (on top of tree and linear base models) delivers a **more robust accuracy boost**, confirming that ensemble strategies are often more effective than hand-crafted transformations in real-world, high-dimensional settings.


**Intended audience:** researchers, ML engineers, Data Scientist

---

## Repository Layout

```
├── data/
│   ├── apartments_train.csv/       #Raw data   
    ├── README.md
├── notebooks/        # Jupyter notebooks for EDA and experiments
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_Baseline_Model.ipynb
│   ├── 03_Model_Optimization.ipynb
├── src/              # Source code (package and scripts)
│   ├── __init__.py
│   ├── data_display.py
│   ├── data_processing.py
│   ├── pipeline_design.py
│   └── plot.py
├── models/                  # trained artifacts (gitignore large files)
│   └── xgb_baseline.joblib  # example baseline
├── requirements.txt  
└── README.md  
```

---

## Quick Start

### 1) Create env & install deps

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt

```
### 2) Preprocess data (example)

```bash
python -m src.data_processing \
  --input data/raw \
  --output data/processed
```

### 3) Train (example)

```bash
python -m src.pipeline_design train \
  --config configs/train_xgb.yaml \
  --output models/xgb_exp1 \
  --seed 42
```
### 4) Inference (example)

```bash
python -m src.pipeline_design predict \
  --model models/xgb_exp1/best.joblib \
  --input data/processed/test.csv \
  --output outputs/predictions.csv
```

---

## Data

**Data Processing Methods:**

* Numeric & time: KNN imputation
* Categorical: add an explicit missing level
* Boolean: impute most frequent value
* Encoding: one-hot (or hashing if needed)
* CV: 5-fold, all preprocessing fitted within fold to avoid leakage

---

## Notebooks
* `01_exploration.ipynb` — EDA, distributions, correlations
* `01_Data_Exploration.ipynb`: distributions, correlations, time features
* `02_Baseline_Model.ipynb`: standardized preprocessing + linear baselines vs. XGBoost
* `03_Model_Optimization.ipynb`: feature engineering (log-target, time recast, pruning) and stacking

---

## Code (src/)

`src/data_processing.py`
* Role: load raw → clean → impute → type-safe encoding → write to processed.
* Suggested API: load_raw(path) -> DataFrame, clean(df) -> DataFrame, encode(df) -> (X, y), save_processed(out_dir).

`src/pipeline_design.py`
* Role: fold-aware pipeline stitching preprocessing and models; provides train/predict CLI.
* Models: linear baselines (Ridge/ElasticNet/Lasso), XGBoost; optional stacking (meta-learner: Lasso).

`src/plot.py / src/data_display.py`
* Role: visualization (error distribution, feature importance/SHAP, learning curves, etc.).


### `src/data_processing.py`

* Purpose: load raw data, clean, create features, and write processed files
* Example public functions: `load_raw(path) -> DataFrame`, `clean(df) -> DataFrame`, `save_processed(df, out_dir)`

### `src/models.py`

* Purpose: define models, training loop, evaluation, and inference utilities
* Example public functions: `train(config)`, `evaluate(model, data)`, `predict(model, inputs)`

If the project expands, break `src/` into subpackages (`src/data/`, `src/models/`, `src/utils/`) and add API docs under `docs/`.

---

## Training & Inference

* training:

```bash
python -m src.pipeline_design train \
  --config configs/train_xgb.yaml \
  --output models/xgb_exp1 \
  --seed 42

```

* Inference::

```bash
python -m src.pipeline_design predict \
  --model models/xgb_exp1/best.joblib \
  --input data/processed/test.csv \
  --output outputs/predictions.csv

```

---

# Experiments & Results

## Baseline (standard preprocessing + 5-fold CV)
| Metric | Value |
|--------|-------|
| Best single model | XGBoost |
| CV mean RMSE | 94,229.97 |
| Test RMSE | 94,052.25 |
| Generalization performance | ≈ −0.19% vs. CV mean (strong generalization) |
| Linear baselines (Ridge/ElasticNet/Lasso) RMSE | ≈ 95,505 |
| XGBoost advantage vs linear | ≈ −1.34% RMSE gap |

## Optimization 1: Feature Engineering + Explainability
| Aspect | Details |
|--------|---------|
| Pipeline steps | - Prune very low Spearman-correlation features<br>- Log-target training with exp(ŷ) back-transform<br>- Time recast examples:<br>  - house_age_months = 2024-06 − year_built<br>  - months_since_ref = src_month − 2024-06 |
| Test RMSE | 93,998.18 |
| Change vs. baseline | −54.07 (−0.06%) → not significant |
| Key takeaway | With nonlinearity + large n, simple linear-correlation pruning and basic time recasts add little to accuracy, though they help interpretability (see SHAP notes) |

## Optimization 2: Stacking Ensemble
| Aspect | Details |
|--------|---------|
| Method | - Keep all features & baseline preprocessing<br>- Train diverse base learners (linear + trees)<br>- Use Lasso meta-learner on OOF predictions |
| Test RMSE | 92,887.82 |
| vs. baseline | −1,164.43 (−1.24%) |
| vs. Opt-1 | −1,110.36 (−1.18%) |
| Conclusion | Stacking exploits complementary inductive biases (linear + tree) and yields consistent, reproducible improvements |

---

## Testing & Continuous Integration

* Run tests locally:

```bash
pytest tests/
```

---

## Contributing

1. Fork the repo
2. Create a branch: feat/<name> or fix/<issue>
3. Add tests and docs
4. Open a PR linking issues and describing settings/metric changes


---

## License

`MIT`

---

## Contact & Acknowledgements

* **Author**: Jasmine Jiang
* **Email**: Jasminejiang57@.gmail.com
* **Acknowledgement**: 
  * the open-source community, the dataset providers, and library authors
  * Belkin et al. (2019). Reconciling modern machine-learning practice and the classical bias-variance trade-off. PNAS, 116(32), 15849-15854.

---

## Checklist

* [ ] `data\apartments_train.csv` contains no sensitive or large binary files
* [ ] data/raw/ is free of sensitive data/large binaries; proper .gitignore rules
* [ ] requirements.txt has pinned versions (or provide environment.yml)
* [ ] data/README.md documents source/schema/process
* [ ] Notebooks execute end-to-end following the documented order
* [ ] Experiments record seeds/configs/env info
* [ ] CI runs lint + tests on PRs and main
