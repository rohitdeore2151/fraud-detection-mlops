import csv
import io
import logging
import sqlite3

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from backend.app.database.history import save_predictions
from backend.app.ml.predictor import predict_fraud_batch
from backend.app.schemas.prediction import TransactionPredictionRequest

router = APIRouter(tags=["Batch prediction"])
logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_BATCH_ROWS = 10_000
REQUIRED_COLUMNS = (
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
)


@router.post("/predict/batch")
async def predict_batch(request: Request):
    content = bytearray()
    async for chunk in request.stream():
        content.extend(chunk)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"CSV upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
            )

    try:
        csv_text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise HTTPException(
            status_code=400,
            detail="The CSV file must use UTF-8 encoding.",
        ) from error

    reader = csv.reader(io.StringIO(csv_text, newline=""), strict=True)
    try:
        headers = next(reader, None)
        if not headers:
            raise HTTPException(status_code=422, detail="The CSV file is empty.")
        headers = [header.strip() for header in headers]
        if len(headers) != len(set(headers)):
            raise HTTPException(
                status_code=422,
                detail="The CSV file contains duplicate column names.",
            )
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in headers]
        if missing_columns:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "The CSV is missing required feature columns.",
                    "missing_columns": missing_columns,
                    "required_columns": list(REQUIRED_COLUMNS),
                },
            )

        valid_rows = []
        invalid_rows = []
        total_rows = 0
        for row_number, values in enumerate(reader, start=2):
            if not values:
                continue
            total_rows += 1
            if total_rows > MAX_BATCH_ROWS:
                raise HTTPException(
                    status_code=413,
                    detail=f"CSV uploads are limited to {MAX_BATCH_ROWS:,} data rows.",
                )
            if len(values) != len(headers):
                invalid_rows.append(
                    {
                        "row_number": row_number,
                        "errors": ["The number of values does not match the CSV header."],
                    }
                )
                continue

            transaction = {
                column: values[headers.index(column)].strip()
                for column in REQUIRED_COLUMNS
            }
            try:
                validated = TransactionPredictionRequest.model_validate(transaction)
            except ValidationError as error:
                invalid_rows.append(
                    {
                        "row_number": row_number,
                        "errors": [
                            f"{'.'.join(str(part) for part in issue['loc'])}: {issue['msg']}"
                            for issue in error.errors()
                        ],
                    }
                )
                continue
            valid_rows.append((row_number, validated.model_dump()))
    except csv.Error as error:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse the CSV file: {error}",
        ) from error

    try:
        predictions = await run_in_threadpool(
            predict_fraud_batch,
            [row for _, row in valid_rows],
        )
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts are unavailable.",
        ) from error

    results = [
        {"row_number": row_number, **row, **prediction}
        for (row_number, row), prediction in zip(valid_rows, predictions)
    ]
    try:
        save_predictions(
            [
                (transaction, prediction)
                for (_, transaction), prediction in zip(valid_rows, predictions)
            ]
        )
    except sqlite3.Error as error:
        logger.exception("Could not persist successful batch predictions")
        raise HTTPException(
            status_code=500,
            detail="Batch prediction succeeded, but its history records could not be saved.",
        ) from error

    return {
        "total_rows": total_rows,
        "valid_rows": len(results),
        "invalid_rows": invalid_rows,
        "predictions": results,
    }
