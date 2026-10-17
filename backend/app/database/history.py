from contextlib import closing
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

HISTORY_DB_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "transaction_history.sqlite3"
)
HISTORY_COLUMNS = (
    "id",
    "timestamp",
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "is_fraud",
    "fraud_probability",
    "threshold",
)
SEARCHABLE_COLUMNS = ("id", "type", "step", "amount", "oldbalanceOrg", "oldbalanceDest")


def _connect(*, initialize: bool = False) -> sqlite3.Connection:
    HISTORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(HISTORY_DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    if initialize:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS transaction_history (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                step INTEGER NOT NULL,
                type TEXT NOT NULL,
                amount REAL NOT NULL,
                oldbalanceOrg REAL NOT NULL,
                oldbalanceDest REAL NOT NULL,
                is_fraud INTEGER NOT NULL CHECK (is_fraud IN (0, 1)),
                fraud_probability REAL,
                threshold REAL NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_transaction_history_timestamp
            ON transaction_history (timestamp)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_transaction_history_is_fraud
            ON transaction_history (is_fraud)
            """
        )
    return connection


def save_prediction(transaction: dict, prediction: dict) -> dict:
    """Persist one successful prediction and return the stored record."""
    record = {
        "id": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **{column: transaction[column] for column in HISTORY_COLUMNS[2:7]},
        "is_fraud": prediction["is_fraud"],
        "fraud_probability": prediction.get("fraud_probability"),
        "threshold": prediction["threshold"],
    }
    placeholders = ", ".join("?" for _ in HISTORY_COLUMNS)
    columns = ", ".join(HISTORY_COLUMNS)
    values = tuple(record[column] for column in HISTORY_COLUMNS)

    with closing(_connect(initialize=True)) as connection, connection:
        connection.execute(
            f"INSERT INTO transaction_history ({columns}) VALUES ({placeholders})",
            values,
        )
    return record


def save_predictions(records: list[tuple[dict, dict]]) -> None:
    """Persist successful batch predictions together in one database transaction."""
    if not records:
        return

    now = datetime.now(timezone.utc).isoformat()
    values = [
        (
            str(uuid4()),
            now,
            transaction["step"],
            transaction["type"],
            transaction["amount"],
            transaction["oldbalanceOrg"],
            transaction["oldbalanceDest"],
            prediction["is_fraud"],
            prediction.get("fraud_probability"),
            prediction["threshold"],
        )
        for transaction, prediction in records
    ]
    placeholders = ", ".join("?" for _ in HISTORY_COLUMNS)
    columns = ", ".join(HISTORY_COLUMNS)
    with closing(_connect(initialize=True)) as connection, connection:
        connection.executemany(
            f"INSERT INTO transaction_history ({columns}) VALUES ({placeholders})",
            values,
        )


def get_history(
    *,
    search: str | None = None,
    fraud_status: str = "all",
    sort: str = "newest",
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """Read a page of saved predictions with matching filters."""
    if not HISTORY_DB_PATH.is_file():
        return {"items": [], "total": 0, "limit": limit, "offset": offset}

    conditions = []
    parameters: list[str | int] = []

    if fraud_status == "fraud":
        conditions.append("is_fraud = ?")
        parameters.append(1)
    elif fraud_status == "not_fraud":
        conditions.append("is_fraud = ?")
        parameters.append(0)

    if search:
        escaped_search = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        matching = " OR ".join(
            f"CAST({column} AS TEXT) LIKE ? ESCAPE '\\'" for column in SEARCHABLE_COLUMNS
        )
        conditions.append(f"({matching})")
        parameters.extend([f"%{escaped_search}%"] * len(SEARCHABLE_COLUMNS))

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    order = "ASC" if sort == "oldest" else "DESC"

    with closing(_connect()) as connection, connection:
        total = connection.execute(
            f"SELECT COUNT(*) FROM transaction_history {where_clause}",
            parameters,
        ).fetchone()[0]
        rows = connection.execute(
            f"""
            SELECT {", ".join(HISTORY_COLUMNS)}
            FROM transaction_history
            {where_clause}
            ORDER BY timestamp {order}, id {order}
            LIMIT ? OFFSET ?
            """,
            [*parameters, limit, offset],
        ).fetchall()

    return {
        "items": [dict(row) for row in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }
