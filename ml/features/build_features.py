
from pathlib import Path
import pandas as pd

from feature_engineering import create_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "paysim_cleaned.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "paysim_features.csv"
)


def main():
    print("=" * 60)
    print("BUILDING FEATURE-ENGINEERED DATASET")
    print("=" * 60)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {INPUT_PATH}"
        )

    print(f"Reading: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    print("Original shape:", df.shape)

    featured_df = create_features(df)

    print("Feature-engineered shape:", featured_df.shape)

    # Verify essential properties
    assert len(featured_df) == len(df)
    assert "isFraud" in featured_df.columns

    new_features = [
        "amount_to_origin_balance_ratio",
        "amount_to_destination_balance_ratio",
        "origin_balance_error",
        "destination_balance_error",
    ]

    assert all(
        feature in featured_df.columns
        for feature in new_features
    )

    # Save without modifying the cleaned source dataset
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    featured_df.to_csv(OUTPUT_PATH, index=False)

    print("\nFeature engineering completed successfully.")
    print("Rows:", len(featured_df))
    print("Columns:", len(featured_df.columns))
    print("Output saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
