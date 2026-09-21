"""The data quality checks (build-plan section 4.5).

Each check is a pure function over a normalized long OHLCV frame (columns
``ticker, trade_date, open, high, low, close, adj_close, volume`` with
``trade_date`` as python ``date``) and returns a :class:`CheckResult`. No I/O
here, so every check is unit-tested against small deterministic fixtures.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

import numpy as np
import pandas as pd

from volatility_mlops.quality.base import CheckResult

_TABLE = "raw_ohlcv"
_PRICE_COLS = ["open", "high", "low", "close", "adj_close"]
_OHLC = ["open", "high", "low", "close"]


def _log_returns(df: pd.DataFrame) -> pd.Series:
    """Per-ticker daily log returns of ``adj_close``, aligned to ``df``'s index."""
    ordered = df.sort_values(["ticker", "trade_date"])
    rets = ordered.groupby("ticker")["adj_close"].transform(lambda s: np.log(s / s.shift(1)))
    return rets.reindex(df.index)


def check_completeness(
    df: pd.DataFrame, calendar_ticker: str, expected_tickers: Iterable[str]
) -> CheckResult:
    """Every active ticker has a row for every expected trading day.

    Expected trading days are the dates on which ``calendar_ticker`` traded.

    Args:
        df: OHLCV frame.
        calendar_ticker: Symbol defining the trading calendar (the market proxy).
        expected_tickers: Symbols that must cover the calendar.

    Returns:
        A failing result whose observed value is the total count of missing rows.
    """
    calendar = set(df.loc[df["ticker"] == calendar_ticker, "trade_date"])
    if not calendar:
        return CheckResult(
            "completeness",
            _TABLE,
            passed=False,
            detail=f"no rows for calendar ticker {calendar_ticker}",
        )
    missing = 0
    worst: tuple[str, int] | None = None
    for ticker in expected_tickers:
        ticker_dates = set(df.loc[df["ticker"] == ticker, "trade_date"])
        gap = len(calendar - ticker_dates)
        missing += gap
        if gap and (worst is None or gap > worst[1]):
            worst = (ticker, gap)
    detail = "" if worst is None else f"worst: {worst[0]} missing {worst[1]} days"
    return CheckResult(
        "completeness", _TABLE, passed=missing == 0, observed_value=float(missing), detail=detail
    )


def check_non_null_critical(df: pd.DataFrame) -> CheckResult:
    """``adj_close`` and ``volume`` are never null."""
    nulls = int(df[["adj_close", "volume"]].isna().to_numpy().sum())
    return CheckResult("non_null_critical", _TABLE, passed=nulls == 0, observed_value=float(nulls))


def check_positivity(df: pd.DataFrame) -> CheckResult:
    """All prices are strictly positive and volume is non-negative."""
    nonpos_prices = int((df[_PRICE_COLS] <= 0).to_numpy().sum())
    neg_volume = int((df["volume"] < 0).sum())
    violations = nonpos_prices + neg_volume
    return CheckResult(
        "positivity", _TABLE, passed=violations == 0, observed_value=float(violations)
    )


def check_ohlc_consistency(df: pd.DataFrame) -> CheckResult:
    """Low is the minimum and high is the maximum of the day's price fields."""
    row_min = df[_OHLC].min(axis=1)
    row_max = df[_OHLC].max(axis=1)
    violation = (df["low"] > row_min) | (df["high"] < row_max)
    count = int(violation.sum())
    return CheckResult("ohlc_consistency", _TABLE, passed=count == 0, observed_value=float(count))


def check_duplicate_keys(df: pd.DataFrame) -> CheckResult:
    """No repeated ``(ticker, trade_date)`` natural key."""
    dupes = int(df.duplicated(subset=["ticker", "trade_date"]).sum())
    return CheckResult("duplicate_keys", _TABLE, passed=dupes == 0, observed_value=float(dupes))


def check_return_sanity(df: pd.DataFrame, max_abs_log_return: float) -> CheckResult:
    """No absolute daily log return exceeds the threshold (catches unhandled splits)."""
    rets = _log_returns(df).abs()
    flagged = int((rets > max_abs_log_return).sum())
    observed = float(rets.max()) if rets.notna().any() else 0.0
    return CheckResult(
        "return_sanity",
        _TABLE,
        passed=flagged == 0,
        observed_value=observed,
        threshold=max_abs_log_return,
        detail=f"{flagged} returns over threshold",
    )


def check_staleness(df: pd.DataFrame, max_lag_days: int, today: date) -> CheckResult:
    """The latest trade date is within ``max_lag_days`` calendar days of ``today``."""
    latest = df["trade_date"].max()
    lag = (today - latest).days
    return CheckResult(
        "staleness",
        _TABLE,
        passed=lag <= max_lag_days,
        observed_value=float(lag),
        threshold=float(max_lag_days),
        detail=f"latest trade date {latest}, {lag} days old",
    )


def check_volatility_floor(df: pd.DataFrame, window: int) -> CheckResult:
    """Realized volatility is never exactly zero over a full window (frozen feed)."""
    ordered = df.sort_values(["ticker", "trade_date"])
    rets = ordered.groupby("ticker")["adj_close"].transform(lambda s: np.log(s / s.shift(1)))
    rolling_std = rets.groupby(ordered["ticker"]).transform(lambda s: s.rolling(window).std())
    frozen = int((rolling_std == 0).sum())
    return CheckResult(
        "volatility_floor",
        _TABLE,
        passed=frozen == 0,
        observed_value=float(frozen),
        threshold=float(window),
        detail=f"{frozen} zero-volatility windows",
    )
