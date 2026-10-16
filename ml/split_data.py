
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "paysim_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

TARGET = "isFraud"
RANDOM_STATE = 42


def main():
    print("=" * 60)
    print("SPLITTING DATASET")
    print("=" * 60)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_PATH}")

    print("Reading feature dataset...")
    df = pd.read_csv(INPUT_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' is missing.")

    # Shuffle legitimate and fraudulent rows separately.
    rng = np.random.default_rng(RANDOM_STATE)
    fraud = df[df[TARGET] == 1].sample(
        frac=1, random_state=RANDOM_STATE
    )
    legitimate = df[df[TARGET] == 0].sample(
        frac=1, random_state=RANDOM_STATE
    )

    def split_group(group):
        n = len(group)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        return (
            group.iloc[:train_end],
            group.iloc[train_end:val_end],
            group.iloc[val_end:],
        )

    fraud_train, fraud_val, fraud_test = split_group(fraud)
    legit_train, legit_val, legit_test = split_group(legitimate)

    def combine_and_shuffle(fraud_part, legit_part):
        result = pd.concat(
            [fraud_part, legit_part], ignore_index=True
        )
        return result.sample(
            frac=1, random_state=RANDOM_STATE
        ).reset_index(drop=True)

    train = combine_and_shuffle(fraud_train, legit_train)
    val = combine_and_shuffle(fraud_val, legit_val)
    test = combine_and_shuffle(fraud_test, legit_test)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, data in [
        ("train", train),
        ("val", val),
        ("test", test),
    ]:
        data.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
        fraud_count = int(data[TARGET].sum())

        print(
            f"{name.capitalize()}: {len(data):,} rows | "
            f"{fraud_count:,} fraud cases | "
            f"{data[TARGET].mean() * 100:.4f}% fraud"
        )

    print("\nData splitting completed successfully.")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
