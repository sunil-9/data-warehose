"""Fixed-parameter baseline. No tuning; test data is not used for selection."""
import json
import platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
from .data import ROOT, FEATURES, load_data

SEED = 42

def metrics(y, probabilities):
    predicted = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, predicted),
        "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted, zero_division=0),
        "f1": f1_score(y, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
        "average_precision": average_precision_score(y, probabilities),
        "brier": brier_score_loss(y, probabilities),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }

def main():
    df = load_data()
    X, y = df[FEATURES], df.cardio
    groups = pd.util.hash_pandas_object(X, index=False)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    development, test = next(splitter.split(X, y, groups))
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
    train_local, valid_local = next(inner.split(X.iloc[development], y.iloc[development], groups.iloc[development]))
    train, valid = development[train_local], development[valid_local]
    for a, b in [(train, valid), (train, test), (valid, test)]:
        assert set(groups.iloc[a]).isdisjoint(set(groups.iloc[b]))
    models = {
        "dummy": make_pipeline(SimpleImputer(strategy="median"), DummyClassifier(strategy="prior")),
        "logistic_regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)),
        "random_forest": make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=150, min_samples_leaf=10, max_depth=14, n_jobs=2, random_state=SEED)),
        "hist_gradient_boosting": make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15, early_stopping=False, random_state=SEED)),
    }
    validation = {}
    for name, model in models.items():
        model.fit(X.iloc[train], y.iloc[train])
        validation[name] = metrics(y.iloc[valid], model.predict_proba(X.iloc[valid])[:, 1])
        print(name, validation[name], flush=True)
    selected = max((n for n in models if n != "dummy"), key=lambda n: validation[n]["roc_auc"])
    # Refit selected model on development data, then evaluate held-out test once.
    best = models[selected]
    best.fit(X.iloc[development], y.iloc[development])
    test_result = metrics(y.iloc[test], best.predict_proba(X.iloc[test])[:, 1])
    result = {
        "seed": SEED, "split_sizes": {"train": len(train), "validation": len(valid), "test": len(test)},
        "split": "Stratified group split by exact predictor profile; not verified patient identity",
        "quality_policy": "All rows retained for initial baseline; investigate flagged values before final modelling",
        "selection_metric": "validation ROC-AUC", "threshold": 0.5,
        "validation": validation, "selected_model": selected, "selected_model_test": test_result,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "limitations": "Fixed parameters; no CV tuning, external validation, calibration fitting or clinical deployment claim.",
    }
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "baseline_metrics.json").write_text(json.dumps(result, indent=2))
    # Final assignment also needs test metrics for the other models: add them only after freezing the experiment plan.
    print("Selected model:", selected, "Test:", test_result)

if __name__ == "__main__":
    main()
