from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "train.csv"
VAL_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "val.csv"

print("Loading training sample...")
train = pd.read_csv(TRAIN_PATH, nrows=100_000)

print("Loading validation sample...")
val = pd.read_csv(VAL_PATH, nrows=100_000)

features = [
    "origin_balance_error",
    "destination_balance_error",
    "amount_to_origin_balance_ratio",
    "amount_to_destination_balance_ratio",
]

for df in [train, val]:
    missing = [column for column in features if column not in df.columns]
    if missing:
        raise ValueError(f"Missing expected features: {missing}")

print("\nTRAINING SAMPLE: feature averages by target")
print(
    train.groupby("isFraud")[features]
    .mean()
    .T
    .round(4)
)

print("\nVALIDATION SAMPLE: feature averages by target")
print(
    val.groupby("isFraud")[features]
    .mean()
    .T
    .round(4)
)

print("\nFeature-pattern inspection completed.")
