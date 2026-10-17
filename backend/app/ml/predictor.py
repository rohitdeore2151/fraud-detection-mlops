from functools import lru_cache

import pandas as pd

from backend.app.ml.model_loader import load_model, load_preprocessor

PROVISIONAL_THRESHOLD = 0.95


@lru_cache(maxsize=1)
def _load_artifacts():
    return load_model(), load_preprocessor()


def predict_fraud(transaction: dict) -> dict:
    """Return a fraud probability and prediction for one transaction."""
    model, preprocessor = _load_artifacts()
    input_data = pd.DataFrame([transaction])

    transformed_data = preprocessor.transform(input_data)
    probability = float(model.predict_proba(transformed_data)[0, 1])

    # Provisional validation threshold; production criteria are not defined.
    threshold = PROVISIONAL_THRESHOLD
    prediction = int(probability >= threshold)

    return {
        "fraud_probability": round(probability, 6),
        "is_fraud": prediction,
        "threshold": threshold,
    }


def predict_fraud_batch(transactions: list[dict]) -> list[dict]:
    """Return predictions for a batch using the loaded model pipeline."""
    if not transactions:
        return []

    model, preprocessor = _load_artifacts()
    columns = ["step", "type", "amount", "oldbalanceOrg", "oldbalanceDest"]
    input_data = pd.DataFrame(transactions, columns=columns)
    transformed_data = preprocessor.transform(input_data)
    probabilities = model.predict_proba(transformed_data)[:, 1]

    threshold = PROVISIONAL_THRESHOLD
    return [
        {
            "fraud_probability": round(float(probability), 6),
            "is_fraud": int(probability >= threshold),
            "threshold": threshold,
        }
        for probability in probabilities
    ]