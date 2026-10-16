from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PS_20174392719_1491204439457_log.csv"
)

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

PROCESSED_DATA_PATH = (
    PROCESSED_DATA_DIR
    / "paysim_cleaned.csv"
)


# ============================================================
# EXPECTED COLUMNS
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
# CLEAN DATA
# ============================================================

def clean_data():

    print("=" * 60)
    print("DATA CLEANING")
    print("=" * 60)

    print(f"Reading raw dataset:")
    print(RAW_DATA_PATH)

    df = pd.read_csv(RAW_DATA_PATH)

    original_rows = len(df)

    print(f"\nOriginal rows: {original_rows:,}")

    # --------------------------------------------------------
    # 1. Validate columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("Required columns: PASSED")

    # --------------------------------------------------------
    # 2. Remove exact duplicate rows
    # --------------------------------------------------------

    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        df = df.drop_duplicates()

    print(
        f"Duplicate rows removed: "
        f"{duplicate_count:,}"
    )

    # --------------------------------------------------------
    # 3. Remove rows with missing target
    # --------------------------------------------------------

    before_target_cleaning = len(df)

    df = df.dropna(
        subset=["isFraud"]
    )

    target_rows_removed = (
        before_target_cleaning - len(df)
    )

    print(
        f"Rows removed due to missing "
        f"isFraud: {target_rows_removed:,}"
    )

    # --------------------------------------------------------
    # 4. Keep valid target values
    # --------------------------------------------------------

    valid_target_values = {0, 1}

    invalid_target_mask = ~df[
        "isFraud"
    ].isin(valid_target_values)

    invalid_target_count = (
        invalid_target_mask.sum()
    )

    if invalid_target_count > 0:
        df = df.loc[
            ~invalid_target_mask
        ]

    print(
        f"Rows removed due to invalid "
        f"isFraud: {invalid_target_count:,}"
    )

    # --------------------------------------------------------
    # 5. Reset index
    # --------------------------------------------------------

    df = df.reset_index(drop=True)

    # --------------------------------------------------------
    # 6. Save cleaned dataset
    # --------------------------------------------------------

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )

    # --------------------------------------------------------
    # 7. Final summary
    # --------------------------------------------------------

    final_rows = len(df)

    rows_removed = (
        original_rows - final_rows
    )

    print()
    print(f"Final rows: {final_rows:,}")
    print(f"Total rows removed: {rows_removed:,}")

    print()
    print("Fraud distribution:")

    print(
        df["isFraud"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("=" * 60)
    print("DATA CLEANING COMPLETED")
    print("=" * 60)

    print(
        f"Cleaned dataset saved to:"
    )
    print(PROCESSED_DATA_PATH)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    clean_data()