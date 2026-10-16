from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    PROJECT_ROOT
    / "models"
    / "preprocessing"
    / "feature_plan.json"
)

feature_plan = {
    "target": "isFraud",
    "decision_time_assumption": (
        "Predict fraud before the transaction is completed, "
        "using information available at decision time."
    ),
    "candidate_features": [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest",
    ],
    "excluded_pending_verification": [
        "newbalanceOrig",
        "newbalanceDest",
        "isFlaggedFraud",
        "amount_to_origin_balance_ratio",
        "amount_to_destination_balance_ratio",
        "origin_balance_error",
        "destination_balance_error",
    ],
    "warning": (
        "This is a provisional feature plan. Confirm the real "
        "payment-system data timing before production deployment."
    ),
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(feature_plan, file, indent=4)

print("Feature plan saved to:", OUTPUT_PATH)
print(json.dumps(feature_plan, indent=4))
