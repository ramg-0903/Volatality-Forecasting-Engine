"""Unit tests for feature computation (no DB, no network)."""

import numpy as np
import pandas as pd

from volatility_mlops.features.compute import (
    TRADING_DAYS,
    _log_returns,
    compute_market_features,
    compute_ticker_features,
)


def _synthetic_prices(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n).date
    rets = rng.normal(0, 0.01, n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"adj_close": close, "high": high, "low": low, "volume": volume}, index=pd.Index(dates)
    )


def test_return_features_match_hand_computation():
    prices = _synthetic_prices()
    out = compute_ticker_features(prices)
    log_ret = _log_returns(prices["adj_close"])
    i = 150

    assert out["ret_1"].iloc[i] == log_ret.iloc[i]
    assert np.isclose(out["ret_5"].iloc[i], log_ret.iloc[i - 4 : i + 1].sum())
    assert np.isclose(out["abs_ret_1"].iloc[i], abs(log_ret.iloc[i]))


def test_log_rv_1_matches_closed_form():
    prices = _synthetic_prices()
    out = compute_ticker_features(prices)
    log_ret = _log_returns(prices["adj_close"])
    i = 150
    # rv_1 = |r| * sqrt(252); feature is its log
    expected = np.log(abs(log_ret.iloc[i]) * np.sqrt(TRADING_DAYS))
    assert np.isclose(out["log_rv_1"].iloc[i], expected)


def test_entity_encoding_is_expanding_mean():
    prices = _synthetic_prices()
    out = compute_ticker_features(prices)
    expected = out["log_rv_21"].expanding().mean()
    pd.testing.assert_series_equal(out["entity_mean_log_rv"], expected, check_names=False)


def test_features_are_backward_looking():
    """Changing a future price must not change any earlier row's features."""
    prices = _synthetic_prices()
    base = compute_ticker_features(prices)

    tampered = prices.copy()
    tampered.iloc[-1, tampered.columns.get_loc("adj_close")] *= 1.5  # change only the last day
    after = compute_ticker_features(tampered)

    # Everything strictly before the tampered row is identical.
    pd.testing.assert_frame_equal(base.iloc[:-1], after.iloc[:-1])


def test_market_features_match_hand_computation():
    dates = pd.bdate_range("2020-01-01", periods=60).date
    idx = pd.Index(dates)
    vix = pd.Series(np.linspace(15, 25, 60), index=idx)
    dgs10 = pd.Series(np.linspace(2.0, 3.0, 60), index=idx)
    dgs3mo = pd.Series(np.linspace(1.0, 1.5, 60), index=idx)
    spy = pd.Series(400 * np.exp(np.cumsum(np.full(60, 0.001))), index=idx)

    out = compute_market_features(vix, dgs10, dgs3mo, spy)

    assert np.isclose(out["log_vix"].iloc[30], np.log(vix.iloc[30]))
    assert np.isclose(out["term_spread"].iloc[30], dgs10.iloc[30] - dgs3mo.iloc[30])
    assert np.isclose(out["vix_chg_5"].iloc[30], vix.iloc[30] - vix.iloc[25])
