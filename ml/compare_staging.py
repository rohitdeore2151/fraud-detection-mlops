
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "processed" / "splits"
CANDIDATE_DIR = ROOT / "models" / "candidates"
STAGING_DIR = ROOT / "models" / "staging"

FEATURE_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
]
TARGET = "isFraud"
THRESHOLD = 0.95
VAL_ROWS = 100000


def load_model(model_path):
    model = XGBClassifier()
    model.load_model(str(model_path))
    return model


def evaluate_model(name, model_path, preprocessor_path, X_val, y_val):
    model = load_model(model_path)
    preprocessor = joblib.load(preprocessor_path)

    X_ready = preprocessor.transform(X_val)
    expected_features = model.get_booster().num_features()
    if X_ready.shape[1] != expected_features:
        raise ValueError(
            f"{name} preprocessor produced {X_ready.shape[1]} features, "
            f"but the model expects {expected_features}."
        )

    print(f"  Transformed feature shape: {X_ready.shape}")
    print(f"  Model expects features: {expected_features}")
    probabilities = model.predict_proba(X_ready)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    return {
        "model": name,
        "average_precision": float(
            average_precision_score(y_val, probabilities)
        ),
        "precision": float(
            precision_score(y_val, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_val, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_val, predictions, zero_division=0)
        ),
    }


def main():
    val_path = SPLIT_DIR / "val.csv"

    if not val_path.exists():
        raise FileNotFoundError(f"Validation data not found: {val_path}")

    print(f"Loading the first {VAL_ROWS:,} validation rows...")
    val = pd.read_csv(val_path, nrows=VAL_ROWS)

    missing = set(FEATURE_COLUMNS + [TARGET]) - set(val.columns)
    if missing:
        raise ValueError(f"Missing validation columns: {sorted(missing)}")

    X_val = val[FEATURE_COLUMNS].copy()
    y_val = val[TARGET].astype(int)

    models = [
        (
            "Existing candidate",
            CANDIDATE_DIR / "xgboost_decision_time.json",
            CANDIDATE_DIR / "decision_time_preprocessor.joblib",
        ),
        (
            "Staging candidate",
            STAGING_DIR / "xgboost_staging.json",
            STAGING_DIR / "staging_preprocessor.joblib",
        ),
    ]

    results = []

    for name, model_path, preprocessor_path in models:
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not preprocessor_path.exists():
            raise FileNotFoundError(
                f"Preprocessor not found: {preprocessor_path}"
            )

        print(f"Evaluating {name}...")
        results.append(
            evaluate_model(
                name,
                model_path,
                preprocessor_path,
                X_val,
                y_val,
            )
        )

    results_df = pd.DataFrame(results)
    print(f"\nComparison at threshold {THRESHOLD}:")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    output_path = STAGING_DIR / "comparison_metrics.json"
    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    print(f"\nComparison saved to: {output_path}")
    print("Neither model was modified.")
    print("The held-out test set was not used.")


if __name__ == "__main__":
    main()
