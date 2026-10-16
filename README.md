# Fraud Detection MLOps

An end-to-end machine learning project for detecting fraudulent financial transactions using the PaySim dataset. The project combines model training and evaluation with a FastAPI prediction service and a React dashboard.

## Project Overview

Fraudulent transactions are rare, making fraud detection an imbalanced classification problem. This project trains an XGBoost model to estimate fraud probability and classifies transactions using a configurable decision threshold.

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

The decision threshold was selected using validation data, then evaluated on the held-out test set.

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

The selected threshold is **0.95**. On the test set, the model detected 857 of 1,232 fraudulent transactions. These results reflect the current candidate model and evaluation setup, not a guarantee of real-world fraud detection performance.

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

The API returns a fraud probability, classification, and decision threshold. The request values are illustrative.

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
npm install
npm run dev
```

Open the local URL printed by Vite, typically http://localhost:5173/.

Ensure the trained model and preprocessing artifacts are present before starting the backend.

## Tests

From the project root, with the virtual environment activated:

```cmd
python -m pytest backend/tests -v
```

## Limitations

- The model was trained on a subset of the available training and validation data.
- Fraud probabilities and the selected threshold depend on the current training setup.
- False negatives remain: some fraudulent transactions are not detected.
- This is a portfolio project and is not validated for production financial decisions.

## Future Improvements

- Automate data and model pipelines.
- Add data and model drift monitoring.
- Add CI checks and reproducible training.
- Improve threshold selection using business costs.
- Add model versioning and deployment monitoring.

## Disclaimer

This project is for educational and portfolio purposes. It should not be used as the sole basis for real financial decisions.