from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "train.csv"
)

df = pd.read_csv(DATA_PATH, nrows=1000)

feature_groups = {
    "Transaction information": [
        "step",
        "type",
        "amount",
    ],
    "Sender balances": [
        "oldbalanceOrg",
        "newbalanceOrig",
    ],
    "Recipient balances": [
        "oldbalanceDest",
        "newbalanceDest",
    ],
    "Existing system flag": [
        "isFlaggedFraud",
    ],
    "Engineered features": [
        "amount_to_origin_balance_ratio",
        "amount_to_destination_balance_ratio",
        "origin_balance_error",
        "destination_balance_error",
    ],
}

print("FEATURE AVAILABILITY REVIEW")
print("=" * 55)

for group, columns in feature_groups.items():
    print(f"\n{group}")
    for column in columns:
        print(f"  {column}: {'present' if column in df.columns else 'MISSING'}")

print("\nImportant:")
print("Presence in the dataset does not prove availability")
print("at the moment a real-time fraud decision is made.")

print("\nFeature availability check completed.")
