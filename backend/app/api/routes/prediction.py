from fastapi import APIRouter, HTTPException

from backend.app.ml.predictor import predict_fraud
from backend.app.schemas.prediction import TransactionPredictionRequest, TransactionPredictionResponse

router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=TransactionPredictionResponse)
def predict(transaction: TransactionPredictionRequest):
    try:
        return predict_fraud(transaction.model_dump())
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts are unavailable.",
        ) from error
