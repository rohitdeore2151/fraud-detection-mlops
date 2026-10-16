
from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "train.csv"
OUTPUT_DIR = PROJECT_ROOT / "models" / "preprocessing"
OUTPUT_PATH = OUTPUT_DIR / "preprocessor_test.joblib"

TARGET = "isFraud"
DROP_COLUMNS = ["nameOrig", "nameDest"]
CATEGORICAL_COLUMNS = ["type"]

NUMERIC_COLUMNS = [
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


def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        sparse_threshold=1.0,
    )


def main():
    print("Loading a 100,000-row training sample...")

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_PATH}")

    sample = pd.read_csv(TRAIN_PATH, nrows=100_000)
    X_sample = sample.drop(columns=[TARGET] + DROP_COLUMNS)

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X_sample)

    print("Input shape:", X_sample.shape)
    print("Transformed shape:", transformed.shape)
    print("Sparse output:", hasattr(transformed, "toarray"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, OUTPUT_PATH)

    print("Test preprocessor saved to:", OUTPUT_PATH)
    print("Sparse preprocessing test completed.")


if __name__ == "__main__":
    main()
