from pathlib import Path
import time
import pandas as pd
import numpy as np

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "models" / "candidates"

TARGET = "isFraud"
DROP_COLUMNS = ["nameOrig", "nameDest"]

CATEGORICAL_COLUMNS = ["type"]

ALL_NUMERIC_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
    "amount_to_origin_balance_ratio",
    "amount_to_destination_balance_ratio",
    "origin_balance_error",
    "destination_balance_error",
]

BALANCE_ERROR_COLUMNS = [
    "origin_balance_error",
    "destination_balance_error",
]


def build_preprocessor(numeric_columns):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        sparse_threshold=1.0,
    )


def evaluate_model(name, train, val, numeric_columns):
    print(f"\n{'=' * 60}")
    print(f"Model: {name}")
    print(f"{'=' * 60}")

    X_train = train.drop(columns=[TARGET] + DROP_COLUMNS)
    y_train = train[TARGET]

    X_val = val.drop(columns=[TARGET] + DROP_COLUMNS)
    y_val = val[TARGET]

    # Remove the selected columns from the model inputs.
    columns_to_keep = numeric_columns + CATEGORICAL_COLUMNS
    X_train = X_train[columns_to_keep]
    X_val = X_val[columns_to_keep]

    preprocessor = build_preprocessor(numeric_columns)

    print("Preparing features...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)

    fraud_count = int(y_train.sum())
    legit_count = len(y_train) - fraud_count
    scale_pos_weight = legit_count / max(fraud_count, 1)

    model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=4,
        tree_method="hist",
    )

    print("Training model...")
    start_time = time.time()
    model.fit(X_train_transformed, y_train)
    training_seconds = time.time() - start_time

    probabilities = model.predict_proba(X_val_transformed)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    matrix = confusion_matrix(y_val, predictions, labels=[0, 1])
    true_negatives, false_positives, false_negatives, true_positives = (
        matrix.ravel()
    )

    results = {
        "model": name,
        "average_precision": average_precision_score(y_val, probabilities),
        "fraud_recall": recall_score(y_val, predictions, zero_division=0),
        "fraud_precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "true_positives": int(true_positives),
        "training_seconds": training_seconds,
    }

    print(f"Average precision: {results['average_precision']:.6f}")
    print(f"Fraud recall:      {results['fraud_recall']:.6f}")
    print(f"Fraud precision:   {results['fraud_precision']:.6f}")
    print(f"False positives:   {results['false_positives']}")
    print(f"False negatives:   {results['false_negatives']}")
    print(f"Training time:     {training_seconds:.1f} seconds")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / f"{name}.json"
    model.save_model(model_path)
    print(f"Model saved to: {model_path}")

    return results


def main():
    print("Loading the same training and validation samples...")
    train = pd.read_csv(SPLIT_DIR / "train.csv", nrows=200_000)
    val = pd.read_csv(SPLIT_DIR / "val.csv", nrows=100_000)

    print("Training rows:", len(train))
    print("Validation rows:", len(val))
    print("Training fraud cases:", int(train[TARGET].sum()))
    print("Validation fraud cases:", int(val[TARGET].sum()))

    # Model A: all engineered features.
    all_features = evaluate_model(
        "xgboost_all_features",
        train,
        val,
        ALL_NUMERIC_COLUMNS,
    )

    # Model B: remove both balance-error features.
    numeric_without_errors = [
        column
        for column in ALL_NUMERIC_COLUMNS
        if column not in BALANCE_ERROR_COLUMNS
    ]

    without_errors = evaluate_model(
        "xgboost_without_balance_errors",
        train,
        val,
        numeric_without_errors,
    )

    comparison = pd.DataFrame([all_features, without_errors])

    print("\n\nFEATURE COMPARISON")
    print(
        comparison[
            [
                "model",
                "average_precision",
                "fraud_recall",
                "fraud_precision",
                "false_positives",
                "false_negatives",
                "true_positives",
                "training_seconds",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )

    report_path = OUTPUT_DIR / "feature_comparison.csv"
    comparison.to_csv(report_path, index=False)
    print(f"\nComparison report saved to: {report_path}")
    print("\nFeature comparison completed.")


if __name__ == "__main__":
    main()