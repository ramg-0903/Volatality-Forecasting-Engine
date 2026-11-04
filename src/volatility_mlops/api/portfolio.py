"""Portfolio risk aggregation (build-plan section 8.3 / Day 18).

Turns per-name volatility *forecasts* into a portfolio-level risk number that accounts
for diversification. The construction follows section 8.3 exactly:

* **Volatilities come from the model** — the diagonal of the covariance matrix is the
  challenger's forecast annualized vol for each holding (reuses ``predict_volatility``).
* **Correlations come from history** — the off-diagonal structure is estimated from a
  trailing window of daily log returns (adjusted close), so it reflects how the names
  have actually co-moved recently.

Covariance is assembled as ``Sigma = D R D`` (D = diag(forecast vols), R = correlation
matrix), portfolio volatility is ``sqrt(wᵀ Sigma w)``, and each holding's share of
total risk is the Euler decomposition ``w_i (Sigma w)_i / sigma_p`` (these sum to
``sigma_p`` exactly). All volatilities are annualized levels, so the portfolio number
is annualized too. This is a risk calculation only — never a return or P&L (rule 7).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

from volatility_mlops.api.errors import ApiError
from volatility_mlops.api.model_holder import ModelHolder
from volatility_mlops.api.service import predict_volatility

# repo_root/src/volatility_mlops/api/portfolio.py -> parents[3] == repo root
SERVING_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "serving.yml"


@dataclass(frozen=True)
class ServingConfig:
    """Serving-layer parameters for portfolio risk."""

    correlation_window_days: int
    min_observations: int
    weight_sum_tolerance: float


def load_serving_config(path: Path = SERVING_CONFIG_PATH) -> ServingConfig:
    """Load serving parameters from ``config/serving.yml``."""
    data = yaml.safe_load(path.read_text())
    return ServingConfig(
        correlation_window_days=data["correlation_window_days"],
        min_observations=data["min_observations"],
        weight_sum_tolerance=float(data["weight_sum_tolerance"]),
    )


# --------------------------------------------------------------------------------------
# Pure linear algebra (no I/O) — the math that the unit tests pin down.
# --------------------------------------------------------------------------------------
def covariance_from_vols_and_corr(vols: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """Assemble the covariance matrix ``Sigma = D R D`` from vols and a correlation matrix.

    Args:
        vols: Length-n vector of (annualized) volatilities.
        corr: n x n correlation matrix.

    Returns:
        The n x n covariance matrix.
    """
    d = np.diag(vols)
    return d @ corr @ d


def portfolio_risk(vols: np.ndarray, weights: np.ndarray, corr: np.ndarray) -> dict[str, Any]:
    """Compute portfolio volatility and per-holding risk contributions.

    Args:
        vols: Length-n forecast volatilities, aligned with ``weights``/``corr``.
        weights: Length-n portfolio weights (sum to 1).
        corr: n x n correlation matrix, same ordering.

    Returns:
        ``portfolio_volatility`` (float), ``weighted_avg_volatility`` (float), and
        ``risk_contributions`` (length-n array summing to the portfolio volatility).
    """
    cov = covariance_from_vols_and_corr(vols, corr)
    variance = float(weights @ cov @ weights)
    sigma = float(np.sqrt(max(variance, 0.0)))
    weighted_avg = float(weights @ vols)
    # Euler/marginal risk contribution: w_i * (Sigma w)_i / sigma_p; these sum to sigma_p.
    contributions = weights * (cov @ weights) / sigma if sigma > 0.0 else np.zeros_like(weights)
    return {
        "portfolio_volatility": sigma,
        "weighted_avg_volatility": weighted_avg,
        "risk_contributions": contributions,
    }


# --------------------------------------------------------------------------------------
# Data access + orchestration
# --------------------------------------------------------------------------------------
def load_return_matrix(
    engine: Engine, tickers: list[str], as_of_date: date | None, window: int
) -> pd.DataFrame:
    """Load the trailing window of daily log returns as a dates x tickers frame.

    Log returns are computed from adjusted close (never unadjusted). Returns the most
    recent ``window`` daily-return rows; tickers with no price rows are simply absent
    as columns.
    """
    clause = "WHERE ticker IN :tickers"
    params: dict[str, object] = {"tickers": tuple(tickers)}
    if as_of_date is not None:
        clause += " AND trade_date <= :as_of"
        params["as_of"] = as_of_date
    query = text(
        f"SELECT ticker, trade_date, adj_close FROM raw_ohlcv {clause} ORDER BY trade_date"
    ).bindparams(bindparam("tickers", expanding=True))
    # pandas-stubs types `params` narrowly and rejects the expanding tuple bind.
    df = pd.read_sql(query, engine, params=params, parse_dates=["trade_date"])  # type: ignore[arg-type]
    if df.empty:
        return pd.DataFrame()
    df["adj_close"] = df["adj_close"].astype(float)
    wide = df.pivot_table(
        index="trade_date", columns="ticker", values="adj_close", aggfunc="first"
    ).sort_index()
    log_returns = pd.DataFrame(np.log(wide / wide.shift(1)), index=wide.index, columns=wide.columns)
    return log_returns.tail(window)


def compute_portfolio_risk(
    engine: Engine,
    holder: ModelHolder,
    holdings: list[dict[str, Any]],
    as_of_date: date | None = None,
) -> dict[str, Any]:
    """Compute portfolio volatility, diversification, and risk contributions.

    Args:
        engine: Application database engine.
        holder: The loaded Production model holder.
        holdings: List of ``{"ticker": str, "weight": float}`` (upper-cased by the route).
        as_of_date: Optional as-of date; defaults to each name's latest data.

    Returns:
        The response dict: portfolio and weighted-average volatility, diversification
        ratio, correlation window, observation count, and per-holding detail sorted by
        risk contribution (largest first), each stamped nowhere but under the top-level
        model block.

    Raises:
        ApiError: 400 for bad weights/duplicates, 422 for insufficient return history.
            (Unknown tickers / missing features surface via ``predict_volatility``.)
    """
    config = load_serving_config()
    tickers = [h["ticker"] for h in holdings]
    weights_list = [h["weight"] for h in holdings]

    if len(set(tickers)) != len(tickers):
        raise ApiError(400, "Duplicate tickers in holdings.")
    negatives = sorted({t for t, w in zip(tickers, weights_list, strict=True) if w < 0})
    if negatives:
        raise ApiError(
            400, "Weights must be non-negative (long-only).", {"negative_weight_tickers": negatives}
        )
    total = float(sum(weights_list))
    if abs(total - 1.0) > config.weight_sum_tolerance:
        raise ApiError(400, "Weights must sum to 1.", {"weight_sum": total})

    # Forecast vols from the model (also validates unknown tickers / missing features).
    preds = predict_volatility(engine, holder, tickers, as_of_date)
    vol_by_ticker = {str(p["ticker"]): float(p["predicted_vol"]) for p in preds}  # type: ignore[arg-type]

    # Correlation from a trailing window of history.
    returns = load_return_matrix(engine, tickers, as_of_date, config.correlation_window_days)
    missing = sorted(set(tickers) - set(returns.columns))
    if missing:
        raise ApiError(
            422, "No price history available for ticker(s).", {"missing_tickers": missing}
        )
    aligned = returns[tickers].dropna()  # common rows only -> a valid (PSD) correlation matrix
    if len(aligned) < config.min_observations:
        raise ApiError(
            422,
            "Insufficient return history for a correlation estimate.",
            {"observations": int(len(aligned)), "required": config.min_observations},
        )

    corr = aligned.corr().to_numpy()
    vols = np.array([vol_by_ticker[t] for t in tickers], dtype=float)
    weights = np.array(weights_list, dtype=float)

    risk = portfolio_risk(vols, weights, corr)
    sigma = float(risk["portfolio_volatility"])
    weighted_avg = float(risk["weighted_avg_volatility"])
    contributions = risk["risk_contributions"]

    holdings_out = [
        {
            "ticker": t,
            "weight": round(float(w), 6),
            "volatility": round(float(v), 6),
            "risk_contribution": round(float(c), 6),
            "risk_contribution_pct": round(float(c) / sigma, 6) if sigma > 0 else None,
        }
        for t, w, v, c in zip(tickers, weights, vols, contributions, strict=True)
    ]
    holdings_out.sort(key=lambda h: h["risk_contribution"] or 0.0, reverse=True)

    return {
        "as_of_date": as_of_date.isoformat() if as_of_date else None,
        "model": holder.status(),
        "portfolio_volatility": round(sigma, 6),
        "weighted_avg_volatility": round(weighted_avg, 6),
        "diversification_ratio": round(weighted_avg / sigma, 6) if sigma > 0 else None,
        "correlation_window_days": config.correlation_window_days,
        "n_observations": int(len(aligned)),
        "holdings": holdings_out,
    }
