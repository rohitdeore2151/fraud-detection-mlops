
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

required_files = [
    ROOT / "models" / "candidates" / "xgboost_decision_time.json",
    ROOT / "models" / "candidates" / "decision_time_preprocessor.joblib",
    ROOT / "models" / "candidates" / "threshold_comparison.csv",
    ROOT / "models" / "candidates" / "threshold_experiment_summary.json",
    ROOT / "models" / "candidates" / "test_evaluation.json",
]

print("===== EXPERIMENT ARTIFACT CHECK =====")

all_found = True

for path in required_files:
    exists = path.is_file()
    print(f"[{'OK' if exists else 'MISSING'}] {path.relative_to(ROOT)}")
    if not exists:
        all_found = False

if all_found:
    print("\n===== THRESHOLD EXPERIMENT =====")
    with open(required_files[3], encoding="utf-8") as file:
        threshold_data = json.load(file)

    print("Experiment:", threshold_data["experiment"])
    print("Validation rows:", threshold_data["validation_rows"])
    print("Validation fraud cases:", threshold_data["validation_fraud_cases"])
    print("Baseline threshold:", threshold_data["baseline_threshold"])
    print("Production threshold finalized:",
          threshold_data["decision"].endswith("defined." ) is False)

    print("\n===== TEST EVALUATION =====")
    with open(required_files[4], encoding="utf-8") as file:
        test_data = json.load(file)

    for key in [
        "evaluation_dataset",
        "test_rows",
        "fraud_cases",
        "threshold",
        "average_precision",
        "precision",
        "recall",
        "f1_score",
        "true_positives",
        "false_positives",
        "false_negatives",
        "true_negatives",
    ]:
        print(f"{key}: {test_data[key]}")

    threshold_csv = pd.read_csv(required_files[2])
    print("\nThreshold comparison rows:", len(threshold_csv))

if not all_found:
    print("\nSome artifacts are missing. Do not proceed until we investigate.")
else:
    print("\nArtifact existence check completed.")
