"""Prediction route (build-plan section 8.1).

``POST /predict`` with ``{"tickers": [...], "as_of_date": "YYYY-MM-DD"?}`` returns the
forecast annualized volatility per ticker, each stamped with the serving model version.
"""

from __future__ import annotations

from flask import Blueprint, Response, current_app, jsonify, request

from volatility_mlops.api.errors import ApiError
from volatility_mlops.api.service import parse_as_of_date, predict_volatility

bp = Blueprint("predict", __name__)


@bp.post("/predict")
def predict() -> tuple[Response, int]:
    """Forecast volatility for the requested tickers as of an optional date."""
    body = request.get_json(silent=True) or {}
    tickers = body.get("tickers")
    if (
        not isinstance(tickers, list)
        or not tickers
        or not all(isinstance(t, str) and t.strip() for t in tickers)
    ):
        raise ApiError(400, "Request body must include a non-empty 'tickers' list of strings.")

    tickers = [t.strip().upper() for t in tickers]
    as_of = parse_as_of_date(body.get("as_of_date"))

    engine = current_app.config["ENGINE"]
    holder = current_app.config["MODEL_HOLDER"]
    predictions = predict_volatility(engine, holder, tickers, as_of)

    return jsonify(
        {
            "as_of_date": as_of.isoformat() if as_of else None,
            "model": holder.status(),
            "predictions": predictions,
        }
    ), 200
