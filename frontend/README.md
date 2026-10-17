# FraudGuard frontend

FraudGuard is a React and Vite portfolio dashboard for the project's FastAPI candidate-model service.

## Run locally

From this directory:

```cmd
npm ci
npm run dev
```

The frontend reads the API base URL from `VITE_API_BASE_URL`. Copy `.env.example` to `.env` to use a non-default backend URL, then restart Vite:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
```

The backend must allow the frontend origin through `BACKEND_CORS_ORIGINS`.

## Features and data boundaries

- Dashboard and model performance pages use the checked-in test evaluation and validation-threshold reports.
- Transaction analysis calls the existing single-transaction prediction endpoint.
- Batch analysis uploads CSV data to the backend, which validates rows and scores valid rows through the trained preprocessing/model pipeline. Uploads are limited to 5 MB and 10,000 data rows.
- Successful single and batch predictions are persisted in `backend\data\transaction_history.sqlite3`. The history page supports search, fraud-status filtering, timestamp sorting, and pagination.
- MLOps monitoring checks API health and model artifact file availability on demand. It does not claim to provide drift, retraining, or deployment monitoring.
- The 0.95 prediction threshold remains provisional; held-out test metrics are not used to select it.
