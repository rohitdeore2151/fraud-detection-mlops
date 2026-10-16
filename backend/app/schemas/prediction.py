from pydantic import BaseModel, Field


class TransactionPredictionRequest(BaseModel):
    step: int = Field(..., ge=1)
    amount: float = Field(..., gt=0)
    oldbalanceOrg: float = Field(..., ge=0)
    oldbalanceDest: float = Field(..., ge=0)
    type: str = Field(..., min_length=1)


class TransactionPredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: int
    threshold: float
