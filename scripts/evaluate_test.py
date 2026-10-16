
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
    confusion_matrix,
    classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "candidates"
    / "xgboost_decision_time.json"
)
PREPROCESSOR_PATH = (
    PROJECT_ROOT
    / "models"
    / "candidates"
    / "decision_time_preprocessor.joblib"
)
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "test.csv"
OUTPUT_PATH = PROJECT_ROOT / "models" / "evaluation" / "final_test_metrics.json"

THRESHOLD = 0.95


def main():
    from xgboost import XGBClassifier

    for path in (MODEL_PATH, PREPROCESSOR_PATH, TEST_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    test = pd.read_csv(TEST_PATH)
    target_column = "isFraud"
    feature_columns = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest",
    ]

    y_true = test[target_column].astype(int)
    X = test[feature_columns]

    X_transformed = preprocessor.transform(X)
    probabilities = model.predict_proba(X_transformed)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    metrics = {
        "test_rows": int(len(test)),
        "fraud_rows": int(y_true.sum()),
        "threshold": THRESHOLD,
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "classification_report": classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=["Not Fraud", "Fraud"],
            output_dict=True,
            zero_division=0,
        ),
        "note": (
            "Final evaluation on the held-out test set. "
            "Threshold selected using validation data only."
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(f"Test rows: {metrics['test_rows']:,}")
    print(f"Actual fraud rows: {metrics['fraud_rows']:,}")
    print(f"Threshold: {THRESHOLD:.2f}")
    print(f"Average Precision: {metrics['average_precision']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 score: {metrics['f1']:.4f}")
    print("\nConfusion matrix:")
    print(f"True negatives:  {tn:,}")
    print(f"False positives: {fp:,}")
    print(f"False negatives: {fn:,}")
    print(f"True positives:  {tp:,}")
    print(f"\nSaved final metrics to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
