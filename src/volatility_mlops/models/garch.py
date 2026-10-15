"""GARCH(1,1) econometric baseline (build-plan section 6.4 / Day 13).

For each ticker, a zero-mean GARCH(1,1) is fit on the training returns only; its
parameters are then fixed and the conditional-variance recursion is run over the
test returns (which are known at prediction time) to produce, at each test date, a
21-day-ahead variance forecast. That forecast is annualized to a volatility level,
matching the target's convention.

Fitting on train-only params (not the full series) keeps the test period out of
parameter estimation (leakage rule 3). Returns are scaled to percent for the
optimizer's numerical stability, then un-scaled. Same ``(train, test) -> np.ndarray``
signature as the other models, so it plugs into the shared evaluation loop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model

HORIZON = 21
TRADING_DAYS = 252
_MIN_TRAIN = 250  # need enough history to estimate GARCH; else fall back to persistence


def garch(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Per-ticker rolling GARCH(1,1) forecast of forward 21-day annualized vol."""
    preds = pd.Series(index=test.index, dtype=float)
    for ticker, test_rows in test.groupby("ticker"):
        train_rows = train[train["ticker"] == ticker].sort_values("trade_date")
        test_rows = test_rows.sort_values("trade_date")
        fallback = np.exp(test_rows["log_rv_21"].to_numpy())

        if len(train_rows) < _MIN_TRAIN:
            preds.loc[test_rows.index] = fallback
            continue

        train_ret = train_rows["ret_1"].to_numpy() * 100.0
        full_ret = np.concatenate([train_ret, test_rows["ret_1"].to_numpy() * 100.0])
        try:
            fitted = arch_model(train_ret, mean="Zero", vol="GARCH", p=1, q=1).fit(disp="off")
            fixed = arch_model(full_ret, mean="Zero", vol="GARCH", p=1, q=1).fix(fitted.params)
            forecast = fixed.forecast(horizon=HORIZON, start=len(train_ret), reindex=False)
            var_21 = forecast.variance.to_numpy().sum(axis=1)  # percent^2 over the horizon
            preds.loc[test_rows.index] = np.sqrt(var_21 * (TRADING_DAYS / HORIZON)) / 100.0
        except Exception:  # noqa: BLE001 -- a non-converging fit falls back to persistence
            preds.loc[test_rows.index] = fallback
    return preds.to_numpy()


GARCH_MODELS = {"garch": garch}
