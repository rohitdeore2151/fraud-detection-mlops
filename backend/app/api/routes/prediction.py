import logging
import sqlite3

from fastapi import APIRouter, HTTPException

from backend.app.database.history import save_prediction
from backend.app.ml.predictor import predict_fraud
from backend.app.schemas.prediction import TransactionPredictionRequest, TransactionPredictionResponse

router = APIRouter(tags=["Prediction"])
logger = logging.getLogger(__name__)


@router.post("/predict", response_model=TransactionPredictionResponse)
def predict(transaction: TransactionPredictionRequest):
    try:
        transaction_data = transaction.model_dump()
        prediction = TransactionPredictionResponse.model_validate(
            predict_fraud(transaction_data)
        )
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts are unavailable.",
        ) from error

    try:
        save_prediction(transaction_data, prediction.model_dump())
    except sqlite3.Error as error:
        logger.exception("Could not persist a successful transaction prediction")
        raise HTTPException(
            status_code=500,
            detail="Prediction succeeded, but its history record could not be saved.",
        ) from error

    return prediction
