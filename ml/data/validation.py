from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# ============================================================
# EXPECTED DATASET STRUCTURE
# ============================================================

EXPECTED_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]


# ============================================================
# FIND RAW CSV
# ============================================================

def find_raw_csv():
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found in: {RAW_DATA_DIR}"
        )

    if len(csv_files) > 1:
        print("Warning: More than one CSV file found.")
        print(csv_files)

    return csv_files[0]


# ============================================================
# VALIDATE COLUMNS
# ============================================================

def validate_columns(df):
    actual_columns = df.columns.tolist()

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in actual_columns
    ]

    unexpected_columns = [
        column
        for column in actual_columns
        if column not in EXPECTED_COLUMNS
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    if unexpected_columns:
        print(
            f"Warning: Unexpected columns: {unexpected_columns}"
        )

    print("Column validation: PASSED")


# ============================================================
# VALIDATE DATASET
# ============================================================

def validate_dataset():

    print("=" * 60)
    print("RAW DATA VALIDATION")
    print("=" * 60)

    csv_path = find_raw_csv()

    print(f"Dataset: {csv_path}")

    # Read dataset
    df = pd.read_csv(csv_path)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # Check empty dataset
    if df.empty:
        raise ValueError("Dataset is empty.")

    print("Dataset not empty: PASSED")

    # Check columns
    validate_columns(df)

    # Check target columns
    if "isFraud" not in df.columns:
        raise ValueError("Target column 'isFraud' not found.")

    if "isFlaggedFraud" not in df.columns:
        raise ValueError(
            "Column 'isFlaggedFraud' not found."
        )

    print("Target columns: PASSED")

    # Check missing values
    missing_values = df.isnull().sum()

    total_missing = missing_values.sum()

    print(f"Total missing values: {total_missing:,}")

    if total_missing == 0:
        print("Missing-value check: PASSED")
    else:
        print("Missing-value check: WARNING")
        print(
            missing_values[missing_values > 0]
        )

    # Check target values
    fraud_values = sorted(df["isFraud"].dropna().unique())

    print(f"isFraud unique values: {fraud_values}")

    if set(fraud_values).issubset({0, 1}):
        print("isFraud values: PASSED")
    else:
        print("isFraud values: WARNING")

    # Check transaction types
    print(
        f"Transaction types: "
        f"{df['type'].dropna().unique().tolist()}"
    )

    print()
    print("=" * 60)
    print("RAW DATA VALIDATION COMPLETED")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    validate_dataset()