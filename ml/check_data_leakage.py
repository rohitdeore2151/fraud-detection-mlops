
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

print("Loading transaction identifiers...")

train_ids = pd.read_csv(
    SPLIT_DIR / "train.csv",
    usecols=["nameOrig", "nameDest"],
    nrows=200_000,
)

val_ids = pd.read_csv(
    SPLIT_DIR / "val.csv",
    usecols=["nameOrig", "nameDest"],
    nrows=100_000,
)

train_pairs = set(
    zip(train_ids["nameOrig"], train_ids["nameDest"])
)
val_pairs = set(
    zip(val_ids["nameOrig"], val_ids["nameDest"])
)

overlap = train_pairs.intersection(val_pairs)

print("Training sample rows:", len(train_ids))
print("Validation sample rows:", len(val_ids))
print("Unique transaction pairs in training:", len(train_pairs))
print("Unique transaction pairs in validation:", len(val_pairs))
print("Overlapping transaction pairs:", len(overlap))

if overlap:
    print("WARNING: Some account pairs appear in both samples.")
else:
    print("No overlapping account pairs found in these samples.")
