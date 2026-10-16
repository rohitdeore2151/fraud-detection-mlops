from pathlib import Path

import joblib
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "models" / "candidates"

MODEL_PATH = MODEL_DIR / "xgboost_decision_time.json"
PREPROCESSOR_PATH = MODEL_DIR / "decision_time_preprocessor.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))
    return model


def load_preprocessor():
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor file not found: {PREPROCESSOR_PATH}"
        )

    return joblib.load(PREPROCESSOR_PATH)