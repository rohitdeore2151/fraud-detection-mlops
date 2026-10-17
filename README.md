# Fraud Detection MLOps

An end-to-end machine learning project for detecting fraudulent financial transactions using the PaySim dataset. The project combines model training and evaluation with a FastAPI prediction service and a React dashboard.

## Project Overview

Fraudulent transactions are rare, making fraud detection an imbalanced classification problem. This project trains an XGBoost model to estimate fraud probability and currently classifies transactions using a provisional decision threshold.

## Tech Stack

- **Python** — data processing and machine learning
- **Pandas** — dataset preparation
- **Scikit-learn** — preprocessing and evaluation metrics
- **XGBoost** — fraud classification model
- **MLflow** — experiment tracking
- **FastAPI** — prediction API
- **React + Vite** — interactive dashboard
- **Pytest** — API testing

## Dataset

The project uses the PaySim synthetic financial transaction dataset.

- Target column: `isFraud`
- Decision-time features: `step`, `type`, `amount`, `oldbalanceOrg`, `oldbalanceDest`
- The data is split into training, validation, and test sets.
- The fraud class is highly imbalanced.

The raw dataset is not included in this repository by default. Follow the dataset's original source and terms to obtain it.

## Model Evaluation

The API currently uses **0.95 as a provisional threshold**. In the available full-validation sweep, 0.95 had the highest F1 among the tested thresholds from 0.90 to 0.99. A separate 100,000-row validation comparison records 0.50 as a baseline and explicitly says not to finalize a production threshold until alert-volume limits and the costs of missed fraud and false alerts are defined. Those operational criteria are not available, so 0.95 is not a finalized production threshold. The held-out test metrics below are a recorded evaluation only; the test set was not used to choose the threshold.

| Metric | Test result |
|---|---:|
| Average Precision | 0.7904 |
| Precision | 81.54% |
| Recall | 69.56% |
| F1 score | 0.7508 |
| True positives | 857 |
| False positives | 194 |
| False negatives | 375 |
| True negatives | 952,968 |

At the reported 0.95 threshold, the test evaluation detected 857 of 1,232 fraudulent transactions. This is a historical evaluation, not evidence that 0.95 is operationally optimal. These results reflect the current candidate model and evaluation setup, not a guarantee of real-world fraud detection performance.

## API

The FastAPI service exposes a prediction endpoint:

`POST /predict`

Example request:

```json
{
  "step": 10,
  "type": "TRANSFER",
  "amount": 5000,
  "oldbalanceOrg": 10000,
  "oldbalanceDest": 2000
}
```

The response contains `fraud_probability`, `is_fraud`, and `threshold`. For example:

```json
{
  "fraud_probability": 0.12,
  "is_fraud": 0,
  "threshold": 0.95
}
```

The example response values are illustrative, not model results. The threshold is provisional. If the candidate model or preprocessor files are missing, the endpoint returns HTTP 503 with `{"detail":"Model artifacts are unavailable."}`.

The portfolio dashboard also uses these read-only endpoints:

- `GET /health` — API process health.
- `GET /history` — locally persisted prediction records with search, fraud-status filtering, timestamp sorting, and pagination.
- `GET /model/status` — model/preprocessor file availability and implemented capability status; this does not load the model.
- `GET /model/performance` — checked-in held-out test metrics and validation-only threshold comparisons.
- `POST /predict/batch` — CSV data upload for server-side validation and batch scoring.

Batch CSV files must include `step`, `type`, `amount`, `oldbalanceOrg`, and `oldbalanceDest`. The API reports invalid rows and scores valid rows using the same lazy-loaded model and preprocessor as single predictions. Uploads are limited to 5 MB and 10,000 data rows. It does not load the large PaySim dataset into the browser.

## Run Locally

### 1. Create and activate a virtual environment

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```cmd
pip install -r requirements-lock.txt
```

### 3. Start the backend

From the project root:

```cmd
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

API documentation: http://127.0.0.1:8000/docs

### 4. Start the frontend

Open a second terminal:

```cmd
cd frontend
npm ci
npm run dev
```

Open the local URL printed by Vite, typically http://localhost:5173/.

The frontend API URL defaults to `http://127.0.0.1:8000`. To use another backend URL, copy `frontend\.env.example` to `frontend\.env`, change `VITE_API_BASE_URL`, and restart Vite. Configure backend CORS origins with `BACKEND_CORS_ORIGINS` (comma-separated); `.env.example` shows the default local origins. For example, in PowerShell set `$env:BACKEND_CORS_ORIGINS` before starting Uvicorn, or pass an environment file with `--env-file .env`.

The dashboard and model performance views are sourced from `models\evaluation\final_test_metrics.json` and `models\evaluation\validation_thresholds.json`. Test results are reported separately from threshold selection, which uses validation data only. The 0.95 threshold remains provisional.

Batch scoring is implemented for uploaded CSVs. Each successful single or batch prediction is persisted to a local SQLite database at `backend\data\transaction_history.sqlite3`. The database and schema are created automatically on the first successful prediction, survive backend restarts, and are excluded from Git because they contain local prediction history. `GET /history` provides read-only search, fraud-status filtering, timestamp sorting, and pagination. Only successful predictions are recorded; invalid batch rows and failed predictions are not.

Continuous traffic metrics, drift monitoring, automated retraining, model version/training metadata, and production deployment are not implemented; the monitoring page reports those limitations explicitly.

The raw dataset and model artifacts are not committed. The backend can start without model artifacts, but `POST /predict` returns HTTP 503 until both `models/candidates/xgboost_decision_time.json` and `models/candidates/decision_time_preprocessor.joblib` are present locally.

## Tests

From the project root, with the virtual environment activated:

```cmd
python -m pytest backend/tests tests/unit -v
```

## Limitations

- The model was trained on a subset of the available training and validation data.
- Fraud probabilities and the provisional threshold depend on the current training setup; production use requires agreed alert-volume and error-cost criteria.
- False negatives remain: some fraudulent transactions are not detected.
- This is a portfolio project and is not validated for production financial decisions.

## Future Improvements

- Automate data and model pipelines.
- Add data and model drift monitoring.
- Expand CI coverage and improve training reproducibility.
- Finalize threshold selection using agreed business costs and alert-volume limits.
- Add model versioning and deployment monitoring.

## Disclaimer

This project is for educational and portfolio purposes. It should not be used as the sole basis for real financial decisions.