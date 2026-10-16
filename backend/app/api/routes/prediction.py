from fastapi import APIRouter

from backend.app.ml.predictor import predict_fraud
from backend.app.schemas.prediction import TransactionPredictionRequest, TransactionPredictionResponse

router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=TransactionPredictionResponse)
def predict(transaction: TransactionPredictionRequest):
    return predict_fraud(transaction.model_dump())
