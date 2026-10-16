from pathlib import Path
import json
import os
import time

import joblib
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "processed" / "splits"
STAGING_DIR = ROOT / "models" / "staging"
TRACKING_DB = ROOT / "mlflow.db"

TRAIN_PATH = SPLIT_DIR / "train.csv"
VAL_PATH = SPLIT_DIR / "val.csv"

TARGET = "isFraud"
FEATURE_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
]

THRESHOLD = 0.95
TRAIN_ROWS = int(os.getenv("TRAIN_ROWS", "200000"))
VAL_ROWS = int(os.getenv("VAL_ROWS", "100000"))

def main():
    # Check that the required data files exist.
    for path in (TRAIN_PATH, VAL_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Required dataset file not found: {path}"
            )

    if TRAIN_ROWS < 1 or VAL_ROWS < 1:
        raise ValueError("TRAIN_ROWS and VAL_ROWS must be positive.")

    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading up to {TRAIN_ROWS:,} training rows...")
    train = pd.read_csv(TRAIN_PATH, nrows=TRAIN_ROWS)

    print(f"Loading up to {VAL_ROWS:,} validation rows...")
    val = pd.read_csv(VAL_PATH, nrows=VAL_ROWS)

    for frame_name, frame in (
        ("training", train),
        ("validation", val),
    ):
        missing = set(FEATURE_COLUMNS + [TARGET]) - set(frame.columns)
        if missing:
            raise ValueError(
                f"Missing columns in {frame_name} data: {sorted(missing)}"
            )

    X_train = train[FEATURE_COLUMNS].copy()
    y_train = train[TARGET].astype(int)

    X_val = val[FEATURE_COLUMNS].copy()
    y_val = val[TARGET].astype(int)

    if y_train.nunique() < 2:
        raise ValueError("Training data must contain both target classes.")

    if y_val.nunique() < 2:
        raise ValueError("Validation data must contain both target classes.")

    print(f"Training rows: {len(train):,}")
    print(f"Training fraud cases: {int(y_train.sum()):,}")
    print(f"Validation rows: {len(val):,}")
    print(f"Validation fraud cases: {int(y_val.sum()):,}")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                ["step", "amount", "oldbalanceOrg", "oldbalanceDest"],
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                ["type"],
            ),
        ],
        sparse_threshold=1.0,
    )

    print("Preparing features...")
    X_train_ready = preprocessor.fit_transform(X_train)
    X_val_ready = preprocessor.transform(X_val)

    fraud_count = int(y_train.sum())
    legitimate_count = len(y_train) - fraud_count
    scale_pos_weight = legitimate_count / max(fraud_count, 1)

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

    print("Training staging candidate...")
    start_time = time.time()
    model.fit(X_train_ready, y_train)
    training_seconds = time.time() - start_time

    probabilities = model.predict_proba(X_val_ready)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    metrics = {
        "validation_rows": int(len(val)),
        "validation_fraud_rows": int(y_val.sum()),
        "threshold": THRESHOLD,
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
        "training_seconds": round(training_seconds, 2),
    }

    print("Validation results at threshold 0.95:")
    print(f"Average precision: {metrics['average_precision']:.4f}")
    print(f"Precision:         {metrics['precision']:.4f}")
    print(f"Recall:            {metrics['recall']:.4f}")
    print(f"F1 score:          {metrics['f1']:.4f}")

    model_path = STAGING_DIR / "xgboost_staging.json"
    preprocessor_path = STAGING_DIR / "staging_preprocessor.joblib"
    metrics_path = STAGING_DIR / "validation_metrics.json"

    model.save_model(str(model_path))
    joblib.dump(preprocessor, preprocessor_path)
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB.as_posix()}")
    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run(run_name="xgboost-staging-candidate"):
        mlflow.log_params(
            {
                "model_type": "XGBoost",
                "feature_set": "decision_time",
                "threshold": THRESHOLD,
                "training_rows": len(train),
                "validation_rows": len(val),
                "n_estimators": 150,
                "max_depth": 6,
                "learning_rate": 0.1,
                "scale_pos_weight": scale_pos_weight,
                "random_state": 42,
            }
        )

        mlflow.log_metrics(
            {
                "validation_average_precision": metrics["average_precision"],
                "validation_precision": metrics["precision"],
                "validation_recall": metrics["recall"],
                "validation_f1": metrics["f1"],
                "training_seconds": training_seconds,
            }
        )

        mlflow.log_artifact(str(model_path), artifact_path="staging_model")
        mlflow.log_artifact(
            str(preprocessor_path), artifact_path="staging_model"
        )
        mlflow.log_artifact(str(metrics_path), artifact_path="validation")

        print("MLflow run recorded.")
        print("Run ID:", mlflow.active_run().info.run_id)

    print(f"\nStaging model saved to: {model_path}")
    print(f"Preprocessor saved to: {preprocessor_path}")
    print(f"Metrics saved to: {metrics_path}")
    print("Existing candidate model was not changed.")
    print("The held-out test set was not used.")


if __name__ == "__main__":
    main()