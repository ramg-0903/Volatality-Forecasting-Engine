"""Unit tests for target construction (build-plan Day 8 done-when).

The key test builds a synthetic series whose realized volatility has a known
closed form and confirms the target matches it exactly.
"""

import numpy as np
import pandas as pd

from volatility_mlops.features.target import (
    TARGET_COLUMN,
    build_targets,
    forward_realized_vol,
)

TRADING_DAYS = 252
HORIZON = 21


def test_target_matches_closed_form_on_constant_vol_series():
    """A constant daily log return c gives annualized vol |c|*sqrt(252)."""
    target_vol = 0.20
    c = target_vol / np.sqrt(TRADING_DAYS)  # engineer the closed-form answer
    dates = pd.bdate_range("2020-01-01", periods=60)
    prices = pd.Series(np.exp(c * np.arange(60)), index=dates)  # log returns all == c

    result = forward_realized_vol(prices, horizon=HORIZON, trading_days=TRADING_DAYS)

    valid = result[TARGET_COLUMN].dropna()
    assert np.allclose(valid, target_vol)  # sqrt(sum(c^2)*252/21) == |c|*sqrt(252)


def test_last_horizon_rows_have_null_label():
    dates = pd.bdate_range("2020-01-01", periods=60)
    prices = pd.Series(np.linspace(100, 130, 60), index=dates)

    result = forward_realized_vol(prices, horizon=HORIZON)

    assert result[TARGET_COLUMN].tail(HORIZON).isna().all()  # future not yet observable
    assert result[TARGET_COLUMN].iloc[:-HORIZON].notna().all()  # everything earlier is labelled


def test_available_on_is_the_horizon_th_future_trading_day():
    dates = pd.bdate_range("2020-01-01", periods=60)
    prices = pd.Series(np.linspace(100, 130, 60), index=dates)

    result = forward_realized_vol(prices, horizon=HORIZON)

    # The label for the first date becomes observable at the close of t+21.
    assert result["target_available_on"].iloc[0] == dates[HORIZON]
    assert result["target_available_on"].tail(HORIZON).isna().all()


def test_build_targets_handles_each_ticker_independently():
    dates = pd.bdate_range("2020-01-01", periods=40)
    prices = pd.concat(
        [
            pd.DataFrame(
                {"ticker": "AAA", "trade_date": dates, "adj_close": np.linspace(100, 120, 40)}
            ),
            pd.DataFrame(
                {"ticker": "BBB", "trade_date": dates, "adj_close": np.linspace(50, 40, 40)}
            ),
        ],
        ignore_index=True,
    )

    out = build_targets(prices, horizon=HORIZON)

    assert set(out["ticker"]) == {"AAA", "BBB"}
    # Each ticker's last 21 rows are null; no label bleeds across tickers.
    for ticker in ("AAA", "BBB"):
        sub = out[out["ticker"] == ticker].sort_values("trade_date")
        assert sub[TARGET_COLUMN].tail(HORIZON).isna().all()
        assert sub[TARGET_COLUMN].iloc[:-HORIZON].notna().all()
