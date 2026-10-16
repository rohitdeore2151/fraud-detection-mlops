from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
MODEL_DIR = PROJECT_ROOT / "models" / "candidates"

TARGET = "isFraud"
NUMERIC_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
]
CATEGORICAL_COLUMNS = ["type"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

MODEL_PATH = MODEL_DIR / "xgboost_decision_time.json"
PREPROCESSOR_PATH = MODEL_DIR / "decision_time_preprocessor.joblib"


def main():
    print("Loading saved model and preprocessor...")
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    print("Loading validation sample...")
    val = pd.read_csv(SPLIT_DIR / "val.csv", nrows=100_000)

    X_val = val[FEATURE_COLUMNS].copy()
    y_val = val[TARGET].astype(int)

    print("Preparing validation features...")
    X_val_ready = preprocessor.transform(X_val)

    print("Calculating fraud probabilities...")
    probabilities = model.predict_proba(X_val_ready)[:, 1]

    print(
        "\nValidation average precision:",
        round(average_precision_score(y_val, probabilities), 6),
    )

    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    results = []

    print("\nTHRESHOLD COMPARISON")
    print("-" * 92)
    print(
        f"{'Threshold':>9} {'Recall':>9} {'Precision':>11} "
        f"{'TP':>6} {'FP':>7} {'FN':>6} {'TN':>8}"
    )
    print("-" * 92)

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_val, predictions, labels=[0, 1]
        ).ravel()

        recall = recall_score(
            y_val, predictions, zero_division=0
        )
        precision = precision_score(
            y_val, predictions, zero_division=0
        )

        row = {
            "threshold": threshold,
            "recall": recall,
            "precision": precision,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
        }
        results.append(row)

        print(
            f"{threshold:>9.2f} {recall:>9.4f} {precision:>11.4f} "
            f"{tp:>6} {fp:>7} {fn:>6} {tn:>8}"
        )

    report_path = MODEL_DIR / "threshold_comparison.csv"
    pd.DataFrame(results).to_csv(report_path, index=False)

    print("\nReport saved to:", report_path)
    print("\nThreshold tuning inspection completed.")
    print("No model or threshold was changed.")


if __name__ == "__main__":
    main()