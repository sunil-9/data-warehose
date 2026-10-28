# Explainable Cardiovascular Disease Classification

NIT6160 individual project: routine clinical and lifestyle indicators, reproducible evaluation, and a path toward a research prototype.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.audit
python -m src.train
jupyter lab notebooks/01_exploration.ipynb
```

On Windows use `.venv\Scripts\activate`. Run commands from the repository root. Dataset `cardio_train.csv` is already present and uses a semicolon separator.

## Data and scope

Source: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

The checked-in file has 70,000 records and 13 columns: identifier, 11 predictors and binary `cardio` outcome. Age is recorded in days. Cholesterol and glucose are ordinal categories, not measured concentrations. There is **no triglyceride (TG) variable**. Confirm code meanings against the original data card before interpreting them. Country, recruitment design and label timing need stronger documentation; do not call this an Indian or Nepali patient dataset.

This task classifies recorded cardiovascular disease presence. It does not predict a future heart attack or provide a validated clinical risk percentage. A personal experience of high TG motivates the topic but is not measured by this model.

## Workflow

- `src/audit.py`: schema, missingness, duplicate profiles, conflicting labels, distributions and quality flags.
- `notebooks/01_exploration.ipynb`: descriptive analysis and explanatory plots.
- `src/train.py`: identical group-separated train/validation/test partitions for a dummy baseline plus Logistic Regression, Random Forest and HistGradientBoosting.
- `reports/`: reproducible audit and baseline evaluation outputs.
- `docs/project_plan.md`: research questions, assignment alignment and future validation.

Exact predictor profiles are grouped before splitting, so identical profiles cannot cross partitions (even if their labels conflict). This is not a patient-level split: patient identity is not documented. The initial workflow flags suspicious measurements but retains them; it does not silently invent clinical exclusion thresholds. Imputation and scaling are fitted inside training pipelines. The untouched test set is evaluated once after validation-based model selection. Do not repeatedly choose features or tune settings using the test results.

The baseline uses fixed parameters, not hyperparameter search. Before the final assignment, add training-only cross-validation and tuning, feature importance, a reduced-feature experiment, calibration, subgroup analysis and three recent papers. This repository is a starting point, not a completed assignment or deployable medical device.

## Collaborating

```bash
git fetch origin
git switch project/cardiovascular-baseline
git pull --ff-only origin project/cardiovascular-baseline
```

Commit and push your edits before requesting a review. Keep changes on a branch, inspect `git status` before pulling, and never force-push shared history. Dataset redistribution and commercial reuse terms must be verified before distributing a product. No patient data or Kaggle credentials should be added.
