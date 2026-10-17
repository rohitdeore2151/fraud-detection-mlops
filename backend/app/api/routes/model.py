import json

from fastapi import APIRouter, HTTPException

from backend.app.ml.model_loader import MODEL_PATH, PREPROCESSOR_PATH, PROJECT_ROOT

router = APIRouter(tags=["Model"])

TEST_REPORT_PATH = PROJECT_ROOT / "models" / "evaluation" / "final_test_metrics.json"
VALIDATION_REPORT_PATH = (
    PROJECT_ROOT / "models" / "evaluation" / "validation_thresholds.json"
)


@router.get("/model/status")
def model_status():
    model_exists = MODEL_PATH.is_file()
    preprocessor_exists = PREPROCESSOR_PATH.is_file()
    return {
        "model_artifacts_available": model_exists and preprocessor_exists,
        "artifacts": {
            "model": {"name": MODEL_PATH.name, "available": model_exists},
            "preprocessor": {
                "name": PREPROCESSOR_PATH.name,
                "available": preprocessor_exists,
            },
        },
        "model_version": None,
        "training_information": None,
        "capabilities": {
            "prediction": "implemented",
            "batch_prediction": "implemented",
            "persistent_transaction_history": "implemented",
            "drift_monitoring": "not_implemented",
            "automated_retraining": "not_implemented",
            "production_deployment": "not_implemented",
        },
    }


@router.get("/model/performance")
def model_performance():
    if not TEST_REPORT_PATH.is_file() or not VALIDATION_REPORT_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="Model evaluation reports are unavailable.",
        )

    with TEST_REPORT_PATH.open(encoding="utf-8") as report_file:
        test_report = json.load(report_file)
    with VALIDATION_REPORT_PATH.open(encoding="utf-8") as report_file:
        validation_report = json.load(report_file)

    return {
        "test_evaluation": test_report,
        "validation_thresholds": validation_report,
        "threshold_status": "provisional",
    }