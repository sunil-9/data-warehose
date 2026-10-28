import json
from .data import ROOT, FEATURES, load_data, quality_flags

def main():
    df = load_data()
    profiles = df.groupby(FEATURES, dropna=False).cardio.nunique()
    flags = quality_flags(df)
    result = {
        "rows": len(df), "columns": len(df.columns), "predictors": len(FEATURES),
        "missing_by_column": df.isna().sum().to_dict(),
        "class_counts": df.cardio.value_counts().sort_index().to_dict(),
        "duplicate_ids": int(df.id.duplicated().sum()),
        "duplicate_predictor_profiles": int(df.duplicated(FEATURES).sum()),
        "duplicate_records_excluding_id": int(df.drop(columns="id").duplicated().sum()),
        "profiles_with_conflicting_targets": int((profiles > 1).sum()),
        "quality_flag_counts": flags.sum().to_dict(),
        "rows_with_any_flag": int(flags.any(axis=1).sum()),
        "note": "Flags are review prompts, not automatic clinical exclusions. No rows removed.",
    }
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "data_audit.json").write_text(json.dumps(result, indent=2))
    df.describe().T.assign(median=df.median(), variance=df.var()).to_csv(out / "descriptive_statistics.csv")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
