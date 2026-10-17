
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.routes.batch import router as batch_router
from backend.app.api.routes.model import router as model_router
from backend.app.api.routes.prediction import router as prediction_router
from backend.app.api.routes.transactions import router as transactions_router

default_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
configured_origins = os.getenv("BACKEND_CORS_ORIGINS")
allowed_origins = (
    [origin.strip() for origin in configured_origins.split(",") if origin.strip()]
    if configured_origins is not None
    else default_origins
)

app = FastAPI(
    title="Fraud Detection API",
    description="API for the fraud detection MLOps project",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction_router)
app.include_router(batch_router)
app.include_router(model_router)
app.include_router(transactions_router)


@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is running",
        "status": "healthy",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}
