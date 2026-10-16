
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VAL_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "val.csv"

print("Loading validation sample...")
df = pd.read_csv(VAL_PATH, nrows=100_000)

print("\nFraud distribution:")
print(df["isFraud"].value_counts())

print("\nFraud rate by transaction type:")
print(
    df.groupby("type", observed=True)["isFraud"]
    .agg(["count", "sum", "mean"])
    .rename(columns={"sum": "fraud_count", "mean": "fraud_rate"})
    .sort_values("fraud_rate", ascending=False)
)

print("\nExisting fraud flag versus actual fraud:")
print(pd.crosstab(df["isFlaggedFraud"], df["isFraud"]))

print("\nBalance error statistics by target:")
for column in ["origin_balance_error", "destination_balance_error"]:
    print(f"\n{column}")
    print(df.groupby("isFraud")[column].describe())

print("\nValidation inspection completed.")
