
from pathlib import Path
import json
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "models" / "preprocessing"

TARGET = "isFraud"

# Account names are identifiers, not useful direct numeric model inputs.
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
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
        ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
    ])


def main():
    print("=" * 60)
    print("CHECKING DATA FOR PREPROCESSING")
    print("=" * 60)

    train_path = SPLIT_DIR / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    # Read only the header first; avoid loading the full 600+ MB file
    # just to check the expected columns.
    columns = pd.read_csv(train_path, nrows=0).columns.tolist()

    required = set(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [TARGET])
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    missing_drop = [c for c in DROP_COLUMNS if c not in columns]
    if missing_drop:
        raise ValueError(f"Expected identifier columns missing: {missing_drop}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the intended feature configuration for reference.
    config = {
        "target": TARGET,
        "dropped_columns": DROP_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "note": (
            "This file records the preprocessing configuration. "
            "The fitted preprocessor will be saved after training."
        ),
    }

    config_path = OUTPUT_DIR / "preprocessing_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    print("Training columns checked successfully.")
    print(f"Numeric features: {len(NUMERIC_COLUMNS)}")
    print(f"Categorical features: {len(CATEGORICAL_COLUMNS)}")
    print(f"Identifier columns to drop: {DROP_COLUMNS}")
    print(f"Target column: {TARGET}")
    print(f"Configuration saved to: {config_path}")
    print("\nPreprocessing configuration check completed.")


if __name__ == "__main__":
    main()
