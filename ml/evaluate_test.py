
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = ROOT / "data" / "processed" / "splits" / "test.csv"
MODEL_PATH = ROOT / "models" / "candidates" / "xgboost_decision_time.json"
PREPROCESSOR_PATH = (
    ROOT / "models" / "candidates" / "decision_time_preprocessor.joblib"
)
OUTPUT_PATH = ROOT / "models" / "candidates" / "test_evaluation.json"

# Keep this threshold fixed for the first test evaluation.
# It is a comparison baseline, not a finalized production threshold.
THRESHOLD = 0.50

ID_COLUMNS = ["nameOrig", "nameDest"]
TARGET = "isFraud"


def main():
    start_time = time.time()

    print("Loading the untouched test set...")
    test_df = pd.read_csv(TEST_PATH)

    y_test = test_df[TARGET].astype(int)
    X_test = test_df.drop(columns=[TARGET] + ID_COLUMNS)

    print(f"Test rows: {len(test_df):,}")
    print(f"Fraud cases: {int(y_test.sum()):,}")
    print(f"Fraud rate: {y_test.mean():.4%}")

    print("\nLoading the saved model and preprocessor...")
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    from xgboost import XGBClassifier

    model = XGBClassifier()
    model.load_model(MODEL_PATH)

    print("Transforming test features...")
    X_test_transformed = preprocessor.transform(X_test)

    print("Calculating fraud probabilities...")
    probabilities = model.predict_proba(X_test_transformed)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    results = {
        "model": "xgboost_decision_time",
        "evaluation_dataset": "untouched_test_set",
        "test_rows": int(len(y_test)),
        "fraud_cases": int(y_test.sum()),
        "fraud_rate": float(y_test.mean()),
        "threshold": THRESHOLD,
        "average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
        "precision": float(precision_score(
            y_test, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_test, predictions, zero_division=0
        )),
        "f1_score": float(f1_score(
            y_test, predictions, zero_division=0
        )),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "production_threshold_finalized": False,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\n===== UNTOUCHED TEST SET RESULTS =====")
    print(f"Threshold:          {THRESHOLD:.2f}")
    print(f"Average precision:  {results['average_precision']:.6f}")
    print(f"Precision:          {results['precision']:.4f}")
    print(f"Recall:             {results['recall']:.4f}")
    print(f"F1 score:           {results['f1_score']:.4f}")
    print(f"True positives:     {tp:,}")
    print(f"False positives:    {fp:,}")
    print(f"False negatives:    {fn:,}")
    print(f"True negatives:     {tn:,}")
    print(f"\nResults saved to:\n{OUTPUT_PATH}")
    print("\nReminder: 0.50 is a comparison threshold, not a production decision.")


if __name__ == "__main__":
    main()
