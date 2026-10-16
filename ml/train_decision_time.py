from pathlib import Path
import time
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "models" / "candidates"

TARGET = "isFraud"
DROP_COLUMNS = ["nameOrig", "nameDest"]

NUMERIC_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
]
CATEGORICAL_COLUMNS = ["type"]


def main():
    print("Loading training sample...")
    train = pd.read_csv(SPLIT_DIR / "train.csv", nrows=200_000)

    print("Loading validation sample...")
    val = pd.read_csv(SPLIT_DIR / "val.csv", nrows=100_000)

    columns = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

    X_train = train[columns].copy()
    y_train = train[TARGET]

    X_val = val[columns].copy()
    y_val = val[TARGET]

    preprocessor = ColumnTransformer([
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]),
            NUMERIC_COLUMNS,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            CATEGORICAL_COLUMNS,
        ),
    ], sparse_threshold=1.0)

    print("Preparing features...")
    X_train_ready = preprocessor.fit_transform(X_train)
    X_val_ready = preprocessor.transform(X_val)

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

    print("Training decision-time candidate...")
    start = time.time()
    model.fit(X_train_ready, y_train)
    elapsed = time.time() - start

    probabilities = model.predict_proba(X_val_ready)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ).ravel()

    print("\nDECISION-TIME CANDIDATE RESULTS")
    print("Training fraud cases:", fraud_count)
    print("Validation fraud cases:", int(y_val.sum()))
    print("Average precision:", round(
        average_precision_score(y_val, probabilities), 6
    ))
    print("Fraud recall:", round(
        recall_score(y_val, predictions, zero_division=0), 6
    ))
    print("Fraud precision:", round(
        precision_score(y_val, predictions, zero_division=0), 6
    ))
    print("True positives:", int(tp))
    print("False negatives:", int(fn))
    print("False positives:", int(fp))
    print("True negatives:", int(tn))
    print("Training time (seconds):", round(elapsed, 2))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = OUTPUT_DIR / "xgboost_decision_time.json"
    model.save_model(model_path)

    # Save preprocessing together with the model so they can later
    # be loaded as a matched pair.
    import joblib
    preprocessor_path = OUTPUT_DIR / "decision_time_preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)

    print("\nModel saved to:", model_path)
    print("Preprocessor saved to:", preprocessor_path)
    print("\nDecision-time candidate training completed.")


if __name__ == "__main__":
    main()