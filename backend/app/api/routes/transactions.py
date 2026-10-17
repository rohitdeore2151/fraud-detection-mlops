from fastapi import APIRouter, Query

from backend.app.database.history import get_history

router = APIRouter(tags=["Transaction history"])


@router.get("/history")
def transaction_history(
    search: str | None = Query(default=None, max_length=100),
    fraud_status: str = Query(default="all", pattern="^(all|fraud|not_fraud)$"),
    sort: str = Query(default="newest", pattern="^(newest|oldest)$"),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    return get_history(
        search=search,
        fraud_status=fraud_status,
        sort=sort,
        limit=limit,
        offset=offset,
    )