from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]

def load_data(path=None):
    df = pd.read_csv(path or ROOT / "cardio_train.csv", sep=";")
    expected = {"id", "cardio", *FEATURES}
    if set(df.columns) != expected:
        raise ValueError(f"Unexpected schema: {list(df.columns)}")
    if df.cardio.isna().any() or not set(df.cardio.unique()) <= {0, 1}:
        raise ValueError("Target must contain only 0 and 1")
    if any(not pd.api.types.is_numeric_dtype(df[c]) for c in FEATURES):
        raise ValueError("All predictors must be numeric; inspect input parsing")
    return df

def quality_flags(df):
    return pd.DataFrame({
        "nonpositive_height_or_weight": (df.height <= 0) | (df.weight <= 0),
        "nonpositive_blood_pressure": (df.ap_hi <= 0) | (df.ap_lo <= 0),
        "systolic_not_above_diastolic": df.ap_hi <= df.ap_lo,
        "invalid_gender_code": ~df.gender.isin([1, 2]),
        "invalid_cholesterol_or_glucose_code": ~df.cholesterol.isin([1, 2, 3]) | ~df.gluc.isin([1, 2, 3]),
        "invalid_binary_predictor": ~df[["smoke", "alco", "active"]].isin([0, 1]).all(axis=1),
    })
