from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = (
    PROJECT_ROOT
    / "models"
    / "candidates"
    / "threshold_comparison.csv"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "models"
    / "candidates"
    / "threshold_experiment_summary.json"
)

results = pd.read_csv(REPORT_PATH)

summary = {
    "experiment": "Decision-time candidate threshold comparison",
    "validation_rows": 100000,
    "validation_fraud_cases": 122,
    "average_precision": 0.700889,
    "baseline_threshold": 0.50,
    "threshold_results": results.to_dict(orient="records"),
    "decision": (
        "Keep 0.50 as the comparison baseline. "
        "Do not finalize a production threshold until alert-volume "
        "limits and the costs of missed fraud and false alerts "
        "are defined."
    ),
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4)

print("Threshold experiment summary saved to:")
print(OUTPUT_PATH)
print("\nThresholds documented:", len(results))
print("Production threshold finalized: No")
