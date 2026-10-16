
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "candidates" / "xgboost_decision_time.json"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "candidates" / "decision_time_preprocessor.joblib"
VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "val.csv"
OUTPUT_PATH = PROJECT_ROOT / "models" / "evaluation" / "validation_thresholds.json"


def main():
    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation file not found: {VALIDATION_PATH}"
        )

    from xgboost import XGBClassifier

    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    validation = pd.read_csv(VALIDATION_PATH)

    target_column = "isFraud"
    if target_column not in validation.columns:
        raise ValueError(
            f"Expected target column '{target_column}' in validation data."
        )

    features = validation[
        ["step", "type", "amount", "oldbalanceOrg", "oldbalanceDest"]
    ]
    y_true = validation[target_column].astype(int)

    transformed = preprocessor.transform(features)
    probabilities = model.predict_proba(transformed)[:, 1]

    thresholds = np.arange(0.90, 1.00, 0.01)
    results = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        results.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": float(precision_score(y_true, predictions, zero_division=0)),
                "recall": float(recall_score(y_true, predictions, zero_division=0)),
                "f1": float(f1_score(y_true, predictions, zero_division=0)),
            }
        )

    output = {
        "validation_rows": int(len(validation)),
        "fraud_rows": int(y_true.sum()),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "threshold_results": results,
        "note": "Threshold selection uses validation data only, not the test set.",
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Validation rows: {len(validation):,}")
    print(f"Fraud rows: {int(y_true.sum()):,}")
    print(f"Average precision: {output['average_precision']:.4f}")
    print(f"Saved threshold results to: {OUTPUT_PATH}")
    print("\nThreshold  Precision  Recall  F1")
    for row in results:
        print(
            f"{row['threshold']:>9.2f}  "
            f"{row['precision']:>9.4f}  "
            f"{row['recall']:>6.4f}  "
            f"{row['f1']:>6.4f}"
        )


if __name__ == "__main__":
    main()
