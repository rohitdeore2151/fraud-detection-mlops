
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
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
NUMERIC_COLUMNS = [
    "step", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest", "isFlaggedFraud",
    "amount_to_origin_balance_ratio",
    "amount_to_destination_balance_ratio",
    "origin_balance_error", "destination_balance_error",
]


def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
        ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
    ], sparse_threshold=1.0)


def main():
    print("Loading training sample...")
    train = pd.read_csv(SPLIT_DIR / "train.csv", nrows=200_000)

    print("Loading validation sample...")
    val = pd.read_csv(SPLIT_DIR / "val.csv", nrows=100_000)

    X_train = train.drop(columns=[TARGET] + DROP_COLUMNS)
    y_train = train[TARGET]
    X_val = val.drop(columns=[TARGET] + DROP_COLUMNS)
    y_val = val[TARGET]

    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)

    fraud_count = int(y_train.sum())
    legit_count = len(y_train) - fraud_count
    scale_pos_weight = legit_count / max(fraud_count, 1)

    print("Training rows:", len(X_train))
    print("Fraud cases in training sample:", fraud_count)
    print("scale_pos_weight:", round(scale_pos_weight, 2))

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

    print("Training baseline XGBoost model...")
    model.fit(X_train_transformed, y_train)

    probabilities = model.predict_proba(X_val_transformed)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print("\nValidation average precision:",
          round(average_precision_score(y_val, probabilities), 6))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_val, predictions))
    print("\nClassification report:")
    print(classification_report(y_val, predictions, zero_division=0))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(OUTPUT_DIR / "xgboost_baseline.json")
    print("\nBaseline model saved to:", OUTPUT_DIR / "xgboost_baseline.json")


if __name__ == "__main__":
    main()
