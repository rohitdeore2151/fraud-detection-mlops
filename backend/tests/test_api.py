import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)
VALID_PREDICTION = {
    "fraud_probability": 0.25,
    "is_fraud": 0,
    "threshold": 0.95,
}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_valid_transaction_prediction(monkeypatch):
    from backend.app.api.routes import prediction

    monkeypatch.setattr(
        prediction,
        "predict_fraud",
        lambda transaction: VALID_PREDICTION,
    )
    payload = {
        "step": 1,
        "amount": 1000.0,
        "oldbalanceOrg": 5000.0,
        "oldbalanceDest": 2000.0,
        "type": "TRANSFER",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result == VALID_PREDICTION


def test_prediction_reports_unavailable_model_artifacts(monkeypatch):
    from backend.app.ml import predictor

    def missing_model():
        raise FileNotFoundError("model artifacts are missing")

    predictor._load_artifacts.cache_clear()
    monkeypatch.setattr(predictor, "load_model", missing_model)
    payload = {
        "step": 1,
        "amount": 1000.0,
        "oldbalanceOrg": 5000.0,
        "oldbalanceDest": 2000.0,
        "type": "TRANSFER",
    }

    try:
        response = client.post("/predict", json=payload)
    finally:
        predictor._load_artifacts.cache_clear()

    assert response.status_code == 503
    assert response.json()["detail"] == "Model artifacts are unavailable."


def test_predictor_loads_artifacts_on_demand_and_predicts(monkeypatch):
    from backend.app.ml import predictor

    class FakePreprocessor:
        def transform(self, data):
            assert list(data.columns) == [
                "step",
                "type",
                "amount",
                "oldbalanceOrg",
                "oldbalanceDest",
            ]
            return [[1, 0, 0, 0, 0, 0, 0, 0, 0]]

    class FakeModel:
        def predict_proba(self, transformed):
            assert transformed == [[1, 0, 0, 0, 0, 0, 0, 0, 0]]
            return np.array([[0.1, 0.9]])

    predictor._load_artifacts.cache_clear()
    monkeypatch.setattr(predictor, "load_model", FakeModel)
    monkeypatch.setattr(
        predictor,
        "load_preprocessor",
        FakePreprocessor,
    )

    try:
        result = predictor.predict_fraud(
            {
                "step": 1,
                "type": "TRANSFER",
                "amount": 1000.0,
                "oldbalanceOrg": 5000.0,
                "oldbalanceDest": 2000.0,
            }
        )
    finally:
        predictor._load_artifacts.cache_clear()

    assert result == {
        "fraud_probability": 0.9,
        "is_fraud": 0,
        "threshold": 0.95,
    }


def test_negative_amount_is_rejected():
    payload = {
        "step": 1,
        "amount": -100.0,
        "oldbalanceOrg": 5000.0,
        "oldbalanceDest": 2000.0,
        "type": "TRANSFER",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422