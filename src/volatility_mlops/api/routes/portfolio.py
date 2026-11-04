"""Portfolio risk route (build-plan section 8.1 / Day 18).

``POST /portfolio/risk`` with ``{"holdings": [{"ticker","weight"}, ...], "as_of_date"?}``
returns the portfolio's forecast annualized volatility, the weighted-average volatility
for comparison, the diversification ratio, and each holding's share of total risk.
"""

from __future__ import annotations

from typing import Any

from flask import Blueprint, Response, current_app, jsonify, request

from volatility_mlops.api.errors import ApiError
from volatility_mlops.api.portfolio import compute_portfolio_risk
from volatility_mlops.api.service import parse_as_of_date

bp = Blueprint("portfolio", __name__)


def _parse_holdings(raw: Any) -> list[dict[str, Any]]:
    """Validate the request's holdings into ``[{ticker, weight}]``, or raise a 400."""
    if not isinstance(raw, list) or not raw:
        raise ApiError(400, "Request body must include a non-empty 'holdings' list.")
    holdings: list[dict[str, Any]] = []
    for h in raw:
        if not isinstance(h, dict) or "ticker" not in h or "weight" not in h:
            raise ApiError(400, "Each holding must be an object with 'ticker' and 'weight'.")
        ticker, weight = h["ticker"], h["weight"]
        if not isinstance(ticker, str) or not ticker.strip():
            raise ApiError(400, "Holding 'ticker' must be a non-empty string.")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ApiError(400, "Holding 'weight' must be a number.")
        holdings.append({"ticker": ticker.strip().upper(), "weight": float(weight)})
    return holdings


@bp.post("/portfolio/risk")
def portfolio_risk() -> tuple[Response, int]:
    """Forecast portfolio volatility with covariance-based aggregation."""
    body = request.get_json(silent=True) or {}
    holdings = _parse_holdings(body.get("holdings"))
    as_of = parse_as_of_date(body.get("as_of_date"))

    engine = current_app.config["ENGINE"]
    holder = current_app.config["MODEL_HOLDER"]
    result = compute_portfolio_risk(engine, holder, holdings, as_of)

    return jsonify(result), 200
