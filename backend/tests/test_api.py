from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_valid_transaction_prediction():
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
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["is_fraud"] in (0, 1)
    assert result["threshold"] == 0.5


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