"""Backward-looking feature computations (build-plan section 5.2).

Every feature for date *t* uses only information available at the close of *t*
(leakage rule 3). Realized-vol features use the sum-of-squared-returns convention
(matching the target), are annualized with sqrt(252), and are taken in logs unless
noted. All functions are pure (operate on frames/series, no I/O).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
EWMA_LAMBDA = 0.94  # RiskMetrics decay
MAX_FFILL = 5  # forward-fill macro across at most a week of missing obs
_LN2 = float(np.log(2.0))


def _log_returns(adj_close: pd.Series) -> pd.Series:
    """Daily log returns of an adjusted-close series (first value is NaN)."""
    return pd.Series(np.log(adj_close / adj_close.shift(1)), index=adj_close.index)


def _safe_log(series: pd.Series) -> pd.Series:
    """Natural log with non-positive inputs mapped to NaN (never -inf)."""
    logged = pd.Series(np.log(series.where(series > 0)), index=series.index)
    return logged.replace([np.inf, -np.inf], np.nan)


def _realized_vol(squared_returns: pd.Series, window: int) -> pd.Series:
    """Annualized realized volatility (level) over a trailing window."""
    return (squared_returns.rolling(window).sum() * (TRADING_DAYS / window)).pow(0.5)


def compute_ticker_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute per-ticker features from one ticker's OHLCV.

    Args:
        prices: Frame indexed by ``trade_date`` with ``adj_close``, ``high``,
            ``low``, ``volume`` columns for a single ticker.

    Returns:
        A frame indexed by ``trade_date`` with the ticker-level features plus a
        helper ``rv_21_level`` (used later for the variance-risk-premium and
        index-ratio features).
    """
    p = prices.sort_index()
    log_ret = _log_returns(p["adj_close"])
    squared = log_ret.pow(2)

    out = pd.DataFrame(index=p.index)
    rv = {w: _realized_vol(squared, w) for w in (1, 5, 21, 63, 126)}
    for w in (1, 5, 21, 63, 126):
        out[f"log_rv_{w}"] = _safe_log(rv[w])

    ewma_var = squared.ewm(alpha=1 - EWMA_LAMBDA, adjust=False).mean()
    out["log_ewma_vol"] = _safe_log((ewma_var * TRADING_DAYS).pow(0.5))

    out["rv_ratio_5_21"] = rv[5] / rv[21]
    out["rv_ratio_21_126"] = rv[21] / rv[126]

    hl = pd.Series(np.log(p["high"] / p["low"]), index=p.index)
    parkinson_daily_var = hl.pow(2) / (4 * _LN2)
    parkinson_vol = (parkinson_daily_var.rolling(21).mean() * TRADING_DAYS).pow(0.5)
    out["log_parkinson_21"] = _safe_log(parkinson_vol)

    out["ret_1"] = log_ret
    out["ret_5"] = log_ret.rolling(5).sum()
    out["ret_21"] = log_ret.rolling(21).sum()
    out["abs_ret_1"] = log_ret.abs()

    downside = log_ret.clip(upper=0.0)
    downside_vol = (downside.pow(2).rolling(21).sum() * (TRADING_DAYS / 21)).pow(0.5)
    out["log_downside_vol_21"] = _safe_log(downside_vol)

    volume = p["volume"].astype(float)
    vol_mean = volume.rolling(21).mean()
    vol_std = volume.rolling(21).std()
    out["volume_z_21"] = ((volume - vol_mean) / vol_std).replace([np.inf, -np.inf], np.nan)

    # Expanding-window (never full-sample) so no future information leaks (rule 4).
    out["entity_mean_log_rv"] = out["log_rv_21"].expanding().mean()

    out["rv_21_level"] = rv[21]
    return out


def compute_market_features(
    vix: pd.Series, dgs10: pd.Series, dgs3mo: pd.Series, spy_adj_close: pd.Series
) -> pd.DataFrame:
    """Compute market-wide features shared across all tickers.

    Macro series are aligned to the SPY trading calendar and forward-filled across
    small gaps (market holidays), per the section 5.4 preprocessing policy.

    Args:
        vix: VIX close indexed by observation date.
        dgs10: 10-year Treasury yield indexed by observation date.
        dgs3mo: 3-month Treasury yield indexed by observation date.
        spy_adj_close: SPY adjusted close indexed by trade date (the calendar).

    Returns:
        A frame indexed by trade date with the market/index/macro features plus a
        helper ``vix_level`` and ``index_rv_21_level``.
    """
    spy = spy_adj_close.sort_index()
    calendar = spy.index

    def align(series: pd.Series) -> pd.Series:
        return series.reindex(calendar).ffill(limit=MAX_FFILL)

    vix_aligned = align(vix)
    out = pd.DataFrame(index=calendar)
    out["vix_level"] = vix_aligned
    out["log_vix"] = _safe_log(vix_aligned)
    out["vix_chg_5"] = vix_aligned.diff(5)
    out["term_spread"] = align(dgs10) - align(dgs3mo)

    spy_squared = _log_returns(spy).pow(2)
    index_rv_21 = _realized_vol(spy_squared, 21)
    out["index_rv_21_level"] = index_rv_21
    out["log_index_rv_21"] = _safe_log(index_rv_21)
    return out
