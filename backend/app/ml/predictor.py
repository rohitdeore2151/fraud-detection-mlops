import pandas as pd

from backend.app.ml.model_loader import load_model, load_preprocessor


# Load the saved artifacts once when this module is imported.
model = load_model()
preprocessor = load_preprocessor()


def predict_fraud(transaction: dict) -> dict:
    """Return a fraud probability and prediction for one transaction."""
    input_data = pd.DataFrame([transaction])

    transformed_data = preprocessor.transform(input_data)
    probability = float(model.predict_proba(transformed_data)[0, 1])

    # Provisional threshold; we can tune this using validation data later.
    threshold = 0.95  # selected using validation F1 score
    prediction = int(probability >= threshold)

    return {
        "fraud_probability": round(probability, 6),
        "is_fraud": prediction,
        "threshold": threshold,
    }