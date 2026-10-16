
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.routes.prediction import router as prediction_router

app = FastAPI(
    title="Fraud Detection API",
    description="API for the fraud detection MLOps project",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction_router)


@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is running",
        "status": "healthy",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}
