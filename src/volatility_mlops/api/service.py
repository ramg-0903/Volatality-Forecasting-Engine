"""Service layer for the serving API: DB reads + prediction (build-plan section 8).

Keeps all SQL and model-calling logic out of the HTTP routes so it is unit/integration
testable on its own. Predictions are served from *precomputed* feature rows in the
``features`` table (section 8.2: compute on demand only when necessary); the API does
not recompute features. Client-facing problems (unknown ticker, no features for the
requested date) are raised as :class:`ApiError` so they become structured responses.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

from volatility_mlops.api.errors import ApiError
from volatility_mlops.api.model_holder import ModelHolder
from volatility_mlops.features.contract import load_feature_contract

# Cap on /history rows so one noisy ticker cannot return an unbounded payload.
MAX_HISTORY_ROWS = 2000


def known_tickers(engine: Engine) -> set[str]:
    """Return the set of tickers in the universe (``dim_ticker``)."""
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT ticker FROM dim_ticker")).scalars().all()
    return set(rows)


def latest_feature_rows(
    engine: Engine,
    tickers: list[str],
    feature_cols: list[str],
    as_of_date: date | None = None,
) -> pd.DataFrame:
    """Load the most recent feature row per ticker, as of ``as_of_date`` if given.

    Args:
        engine: Application database engine.
        tickers: Tickers to fetch (already validated against the universe).
        feature_cols: Feature columns the model expects, in contract order.
        as_of_date: Upper bound (inclusive) on ``trade_date``. ``None`` takes each
            ticker's latest available row.

    Returns:
        One row per ticker found, with ``ticker``, ``trade_date``, and the feature
        columns. Tickers with no matching row are simply absent.
    """
    cols = ", ".join(["ticker", "trade_date", *feature_cols])
    clause = "WHERE ticker IN :tickers"
    params: dict[str, object] = {"tickers": tuple(tickers)}
    if as_of_date is not None:
        clause += " AND trade_date <= :as_of"
        params["as_of"] = as_of_date
    # DISTINCT ON (ticker) with ORDER BY ticker, trade_date DESC -> newest row per ticker.
    query = text(
        f"SELECT DISTINCT ON (ticker) {cols} FROM features "
        f"{clause} ORDER BY ticker, trade_date DESC"
    ).bindparams(bindparam("tickers", expanding=True))
    # pandas-stubs types `params` narrowly and rejects the expanding tuple bind.
    df = pd.read_sql(query, engine, params=params, parse_dates=["trade_date"])  # type: ignore[arg-type]
    df["trade_date"] = df["trade_date"].dt.date
    return df


def predict_volatility(
    engine: Engine,
    holder: ModelHolder,
    tickers: list[str],
    as_of_date: date | None = None,
) -> list[dict[str, object]]:
    """Forecast annualized volatility for each ticker, stamped with the model version.

    Args:
        engine: Application database engine.
        holder: The loaded Production model holder.
        tickers: Requested tickers (upper-cased by the route).
        as_of_date: Optional as-of date; defaults to each ticker's latest features.

    Returns:
        One dict per ticker, in request order, with the forecast and model stamp.

    Raises:
        ApiError: 503 if no model is loaded, 400 for unknown tickers, 422 if a known
            ticker has no feature row at the requested date.
    """
    if holder.model is None:
        raise ApiError(503, "No Production model is loaded.")

    valid = known_tickers(engine)
    unknown = sorted({t for t in tickers if t not in valid})
    if unknown:
        raise ApiError(400, "Unknown ticker(s).", {"unknown_tickers": unknown})

    feature_cols = load_feature_contract().feature_names()
    rows = latest_feature_rows(engine, tickers, feature_cols, as_of_date)

    missing = sorted(set(tickers) - set(rows["ticker"]))
    if missing:
        raise ApiError(
            422,
            "No feature rows available for ticker(s) at the requested date.",
            {
                "missing_tickers": missing,
                "as_of_date": as_of_date.isoformat() if as_of_date else None,
            },
        )

    # The pyfunc wrapper selects its own feature columns and returns vol *levels*.
    preds = np.asarray(holder.model.predict(rows[feature_cols]), dtype=float)
    rows = rows.assign(predicted_vol=preds)

    info = holder.info
    results = [
        {
            "ticker": r["ticker"],
            "as_of_date": r["trade_date"].isoformat(),
            "predicted_vol": round(float(r["predicted_vol"]), 6),
            "model_name": info.get("name"),
            "model_version": info.get("version"),
            "model_stage": info.get("stage"),
        }
        for r in rows.to_dict(orient="records")
    ]
    order = {t: i for i, t in enumerate(tickers)}
    results.sort(key=lambda d: order[str(d["ticker"])])
    return results


def _num(value: Any) -> float | None:
    """Coerce a possibly-``None``/Decimal/NaN DB value to a JSON float or ``None``."""
    if value is None or pd.isna(value):
        return None
    return float(value)


def prediction_history(engine: Engine, ticker: str, limit: int = 500) -> list[dict[str, object]]:
    """Return past predictions joined to realized outcomes for a ticker, by date.

    Early in the system's life (before the Day-23 delayed-label flow runs) the outcome
    columns are mostly ``null`` — that is expected, the label matures 21 trading days
    after each prediction.

    Args:
        engine: Application database engine.
        ticker: Ticker to fetch (upper-cased by the route).
        limit: Maximum rows, clamped to ``MAX_HISTORY_ROWS``.

    Returns:
        Oldest-first list of ``{as_of_date, predicted_vol, realized_vol, ...}`` records.

    Raises:
        ApiError: 400 if the ticker is not in the universe.
    """
    if ticker not in known_tickers(engine):
        raise ApiError(400, "Unknown ticker.", {"ticker": ticker})

    limit = max(1, min(limit, MAX_HISTORY_ROWS))
    query = text(
        """
        SELECT p.as_of_date, p.predicted_vol, p.model_version, p.horizon_days,
               p.resolves_on, o.realized_vol, o.abs_error, o.qlike
        FROM predictions p
        LEFT JOIN prediction_outcomes o USING (prediction_id)
        WHERE p.ticker = :ticker
        ORDER BY p.as_of_date
        LIMIT :limit
        """
    )
    df = pd.read_sql(
        query,
        engine,
        params={"ticker": ticker, "limit": limit},  # type: ignore[arg-type]
        parse_dates=["as_of_date", "resolves_on"],
    )
    return [
        {
            "as_of_date": r["as_of_date"].date().isoformat() if pd.notna(r["as_of_date"]) else None,
            "predicted_vol": _num(r["predicted_vol"]),
            "model_version": r["model_version"],
            "horizon_days": int(r["horizon_days"]),
            "resolves_on": r["resolves_on"].date().isoformat()
            if pd.notna(r["resolves_on"])
            else None,
            "realized_vol": _num(r["realized_vol"]),
            "abs_error": _num(r["abs_error"]),
            "qlike": _num(r["qlike"]),
        }
        for r in df.to_dict(orient="records")
    ]
