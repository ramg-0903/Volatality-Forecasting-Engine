"""Target construction: the forward realized-volatility label (build-plan section 5.1).

For ticker *i* on date *t* the label is the annualized realized volatility of the
next ``horizon`` trading days (returns *t+1 .. t+horizon*):

    target(t) = sqrt( sum_{k=1..h} r_{t+k}^2  *  trading_days / h )

computed from log returns of ``adj_close``. It is stored as an annualized *level*;
the model trains on its log and exponentiates at prediction time.

Leakage guarantees (build-plan section 4.6):
- The label for *t* uses only returns strictly after *t* (rules 1-2).
- The label is unobservable until the close of *t+horizon*; that date is returned
  as ``target_available_on`` for the monitoring join (Day 23).
- The last ``horizon`` rows have a null label (the future has not happened yet).

Pure functions only; no database access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 21
TRADING_DAYS = 252
TARGET_COLUMN = "target_fwd_rv_21"


def forward_realized_vol(
    adj_close: pd.Series, horizon: int = HORIZON, trading_days: int = TRADING_DAYS
) -> pd.DataFrame:
    """Annualized forward realized volatility for one ticker.

    Args:
        adj_close: Adjusted-close prices for a single ticker, indexed by trade
            date. Sorted internally.
        horizon: Forward window in trading days (the forecast horizon).
        trading_days: Trading days per year, for annualization.

    Returns:
        A frame indexed by trade date with columns ``target_fwd_rv_21`` (the
        annualized level; null for the last ``horizon`` rows) and
        ``target_available_on`` (the date the label first becomes observable,
        i.e. the trade date ``horizon`` steps ahead; null for the last rows).
    """
    prices = adj_close.sort_index()
    log_ret = pd.Series(np.log(prices / prices.shift(1)), index=prices.index)
    squared = log_ret.pow(2)

    # rolling(h).sum() is labelled at the END of its window; shifting by -horizon
    # re-labels it at t so the window covers the FUTURE days t+1 .. t+horizon.
    fwd_sq_sum = squared.rolling(horizon).sum().shift(-horizon)
    target = np.sqrt(fwd_sq_sum * (trading_days / horizon))

    dates = pd.Series(prices.index, index=prices.index)
    available_on = dates.shift(-horizon)

    return pd.DataFrame({TARGET_COLUMN: target, "target_available_on": available_on})


def build_targets(
    prices: pd.DataFrame, horizon: int = HORIZON, trading_days: int = TRADING_DAYS
) -> pd.DataFrame:
    """Compute the forward realized-vol label for every ticker in a long frame.

    Args:
        prices: Long OHLCV frame with at least ``ticker``, ``trade_date``, and
            ``adj_close`` columns.
        horizon: Forward window in trading days.
        trading_days: Trading days per year.

    Returns:
        A long frame with ``ticker``, ``trade_date``, ``target_fwd_rv_21``, and
        ``target_available_on``. Each ticker is handled independently, so no
        label ever spans two tickers.
    """
    frames: list[pd.DataFrame] = []
    for ticker, group in prices.groupby("ticker"):
        series = group.set_index("trade_date")["adj_close"]
        labelled = forward_realized_vol(series, horizon, trading_days).reset_index()
        labelled.insert(0, "ticker", ticker)
        frames.append(labelled)
    return pd.concat(frames, ignore_index=True)
