"""History route (build-plan section 8.1).

``GET /history?ticker=AAPL&limit=500`` returns past predictions joined to their
realized outcomes for one ticker, oldest first, for the dashboard's accuracy chart.
"""

from __future__ import annotations

from flask import Blueprint, Response, current_app, jsonify, request

from volatility_mlops.api.errors import ApiError
from volatility_mlops.api.service import prediction_history

bp = Blueprint("history", __name__)


@bp.get("/history")
def history() -> tuple[Response, int]:
    """Return the predicted-vs-realized series for a single ticker."""
    ticker = request.args.get("ticker", "").strip().upper()
    if not ticker:
        raise ApiError(400, "Query parameter 'ticker' is required.")

    raw_limit = request.args.get("limit", "500")
    try:
        limit = int(raw_limit)
    except ValueError as exc:
        raise ApiError(400, "'limit' must be an integer.") from exc

    engine = current_app.config["ENGINE"]
    records = prediction_history(engine, ticker, limit)

    return jsonify({"ticker": ticker, "count": len(records), "history": records}), 200
