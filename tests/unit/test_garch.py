"""Unit test for the GARCH baseline (no DB; small synthetic series)."""

import numpy as np
import pandas as pd

from volatility_mlops.models.garch import garch


def _ticker_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n).date
    ret = rng.normal(0, 0.01, n)
    return pd.DataFrame(
        {
            "ticker": "AAA",
            "trade_date": dates,
            "ret_1": ret,
            "log_rv_21": np.log(np.full(n, 0.2)),  # persistence fallback value
        }
    )


def test_garch_returns_positive_forecasts_of_correct_length():
    df = _ticker_frame(330, seed=1)
    train, test = df.iloc[:300], df.iloc[300:]

    preds = garch(train, test)

    assert preds.shape == (len(test),)
    assert np.all(np.isfinite(preds))
    assert np.all(preds > 0)  # volatility is strictly positive


def test_garch_falls_back_to_persistence_without_enough_history():
    df = _ticker_frame(60, seed=2)  # < _MIN_TRAIN
    train, test = df.iloc[:40], df.iloc[40:]

    preds = garch(train, test)
    # Fallback is exp(log_rv_21) == 0.2 for every row.
    assert np.allclose(preds, 0.2)
