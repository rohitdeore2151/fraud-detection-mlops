import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)
VALID_PREDICTION = {
    "fraud_probability": 0.25,
    "is_fraud": 0,
    "threshold": 0.95,
}


@pytest.fixture(autouse=True)
def isolated_history_database(monkeypatch, tmp_path):
    from backend.app.database import history

    monkeypatch.setattr(
        history,
        "HISTORY_DB_PATH",
        tmp_path / "transaction_history.sqlite3",
    )


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
    history_response = client.get("/history")
    assert history_response.status_code == 200
    saved = history_response.json()["items"]
    assert len(saved) == 1
    assert saved[0]["id"]
    assert saved[0]["timestamp"]
    assert saved[0]["step"] == payload["step"]
    assert saved[0]["type"] == payload["type"]
    assert saved[0]["amount"] == payload["amount"]
    assert saved[0]["oldbalanceOrg"] == payload["oldbalanceOrg"]
    assert saved[0]["oldbalanceDest"] == payload["oldbalanceDest"]
    assert saved[0]["is_fraud"] == result["is_fraud"]
    assert saved[0]["fraud_probability"] == result["fraud_probability"]
    assert saved[0]["threshold"] == result["threshold"]


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
    assert client.get("/history").json()["total"] == 0


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


def test_batch_prediction_scores_valid_rows_and_reports_invalid_rows(monkeypatch):
    from backend.app.api.routes import batch

    monkeypatch.setattr(
        batch,
        "predict_fraud_batch",
        lambda transactions: [
            {"fraud_probability": 0.12, "is_fraud": 0, "threshold": 0.95}
            for _ in transactions
        ],
    )
    csv_data = (
        "step,type,amount,oldbalanceOrg,oldbalanceDest,optional\n"
        "1,TRANSFER,100,200,0,ignored\n"
        "2,CASH_OUT,-1,0,0,ignored\n"
        "3,PAYMENT,100,200,0\n"
    )

    response = client.post(
        "/predict/batch",
        content=csv_data,
        headers={"Content-Type": "text/csv"},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["total_rows"] == 3
    assert result["valid_rows"] == 1
    assert result["predictions"] == [
        {
            "row_number": 2,
            "step": 1,
            "type": "TRANSFER",
            "amount": 100.0,
            "oldbalanceOrg": 200.0,
            "oldbalanceDest": 0.0,
            **VALID_PREDICTION
            | {"fraud_probability": 0.12},
        }
    ]
    assert [row["row_number"] for row in result["invalid_rows"]] == [3, 4]
    history = client.get("/history").json()
    assert history["total"] == 1
    assert history["items"][0]["step"] == 1
    assert history["items"][0]["fraud_probability"] == 0.12


def test_history_filters_searches_sorts_and_paginates(monkeypatch):
    from backend.app.api.routes import prediction

    monkeypatch.setattr(
        prediction,
        "predict_fraud",
        lambda transaction: {
            "fraud_probability": 0.99 if transaction["type"] == "TRANSFER" else 0.12,
            "is_fraud": int(transaction["type"] == "TRANSFER"),
            "threshold": 0.95,
        },
    )
    for transaction_type, amount in (("TRANSFER", 100), ("PAYMENT", 200), ("CASH_OUT", 300)):
        response = client.post(
            "/predict",
            json={
                "step": amount,
                "type": transaction_type,
                "amount": amount,
                "oldbalanceOrg": 500,
                "oldbalanceDest": 0,
            },
        )
        assert response.status_code == 200

    fraud = client.get("/history", params={"fraud_status": "fraud"})
    assert fraud.status_code == 200
    assert fraud.json()["total"] == 1
    assert fraud.json()["items"][0]["type"] == "TRANSFER"

    search = client.get("/history", params={"search": "PAYMENT"})
    assert search.json()["total"] == 1
    assert search.json()["items"][0]["amount"] == 200

    newest = client.get("/history", params={"sort": "newest", "limit": 2})
    oldest = client.get("/history", params={"sort": "oldest", "limit": 2})
    assert newest.json()["total"] == 3
    assert newest.json()["items"][0]["timestamp"] >= newest.json()["items"][1]["timestamp"]
    assert oldest.json()["items"][0]["timestamp"] <= oldest.json()["items"][1]["timestamp"]
    assert len(newest.json()["items"]) == 2
    assert newest.json()["offset"] == 0
    assert client.get("/history", params={"fraud_status": "invalid"}).status_code == 422


def test_history_records_survive_separate_database_connections():
    from backend.app.database.history import get_history, save_prediction

    transaction = {
        "step": 3,
        "type": "PAYMENT",
        "amount": 45.0,
        "oldbalanceOrg": 50.0,
        "oldbalanceDest": 8.0,
    }
    save_prediction(transaction, VALID_PREDICTION)

    result = get_history()

    assert result["total"] == 1
    assert result["items"][0]["type"] == "PAYMENT"


def test_reading_empty_history_does_not_create_or_change_database():
    from backend.app.database import history

    response = client.get("/history")

    assert response.status_code == 200
    assert response.json()["items"] == []
    assert response.json()["total"] == 0
    assert not history.HISTORY_DB_PATH.exists()


def test_history_accepts_missing_optional_probability():
    from backend.app.database.history import save_prediction

    record = save_prediction(
        {
            "step": 4,
            "type": "DEBIT",
            "amount": 12.0,
            "oldbalanceOrg": 20.0,
            "oldbalanceDest": 5.0,
        },
        {"is_fraud": 0, "threshold": 0.95},
    )

    assert record["fraud_probability"] is None
    assert client.get("/history").json()["items"][0]["fraud_probability"] is None


def test_batch_prediction_requires_model_feature_columns():
    response = client.post(
        "/predict/batch",
        content="step,type,amount\n1,TRANSFER,100\n",
        headers={"Content-Type": "text/csv"},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["missing_columns"] == [
        "oldbalanceOrg",
        "oldbalanceDest",
    ]


def test_batch_prediction_reports_unavailable_artifacts(monkeypatch):
    from backend.app.api.routes import batch

    def missing_model(_transactions):
        assert len(_transactions) == 1
        raise FileNotFoundError("model artifacts are missing")

    monkeypatch.setattr(batch, "predict_fraud_batch", missing_model)
    response = client.post(
        "/predict/batch",
        content=(
            "step,type,amount,oldbalanceOrg,oldbalanceDest\n"
            "1,TRANSFER,100,200,0\n"
        ),
        headers={"Content-Type": "text/csv"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Model artifacts are unavailable."


def test_model_status_reports_artifact_presence():
    from backend.app.ml.model_loader import MODEL_PATH, PREPROCESSOR_PATH

    response = client.get("/model/status")

    assert response.status_code == 200
    status = response.json()
    assert status["model_artifacts_available"] is (
        MODEL_PATH.is_file() and PREPROCESSOR_PATH.is_file()
    )
    assert status["artifacts"]["model"]["available"] is MODEL_PATH.is_file()
    assert status["artifacts"]["preprocessor"]["available"] is PREPROCESSOR_PATH.is_file()
    assert status["capabilities"]["drift_monitoring"] == "not_implemented"
    assert status["model_version"] is None


def test_model_performance_returns_recorded_reports():
    response = client.get("/model/performance")

    assert response.status_code == 200
    performance = response.json()
    assert performance["threshold_status"] == "provisional"
    assert performance["test_evaluation"]["test_rows"] == 954394
    assert performance["validation_thresholds"]["validation_rows"] == 954393


def test_predictor_batch_preserves_feature_order_and_provisional_threshold(monkeypatch):
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
            assert data.iloc[0].tolist() == [
                1,
                "TRANSFER",
                1000.0,
                5000.0,
                2000.0,
            ]
            return [[1], [2]]

    class FakeModel:
        def predict_proba(self, transformed):
            assert transformed == [[1], [2]]
            return np.array([[0.8, 0.2], [0.01, 0.99]])

    predictor._load_artifacts.cache_clear()
    monkeypatch.setattr(predictor, "load_model", FakeModel)
    monkeypatch.setattr(predictor, "load_preprocessor", FakePreprocessor)

    try:
        result = predictor.predict_fraud_batch(
            [
                {
                    "step": 1,
                    "type": "TRANSFER",
                    "amount": 1000.0,
                    "oldbalanceOrg": 5000.0,
                    "oldbalanceDest": 2000.0,
                },
                {
                    "step": 2,
                    "type": "CASH_OUT",
                    "amount": 300.0,
                    "oldbalanceOrg": 300.0,
                    "oldbalanceDest": 0.0,
                },
            ]
        )
    finally:
        predictor._load_artifacts.cache_clear()

    assert result == [
        {"fraud_probability": 0.2, "is_fraud": 0, "threshold": 0.95},
        {"fraud_probability": 0.99, "is_fraud": 1, "threshold": 0.95},
    ]