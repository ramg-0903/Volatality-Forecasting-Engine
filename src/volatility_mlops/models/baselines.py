"""Baseline volatility forecasters (build-plan section 6.4).

Each baseline is a plain function with the shared signature
``(train, test) -> np.ndarray`` returning predicted annualized vol *levels* for the
test rows. Persistence and EWMA are analytic (they ignore ``train``); HAR-RV is
fit per fold on the training rows only (leakage rule 3).

All models predict the forward 21-day realized-vol level. HAR-RV is fit in logs
(the target is right-skewed) and exponentiated back to a level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

TARGET = "target_fwd_rv_21"
_HAR_COLUMNS = ["log_rv_1", "log_rv_5", "log_rv_21"]  # daily / weekly / monthly RV ladder


def persistence(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Forecast next-month vol as the current trailing 21-day realized vol."""
    del train  # analytic: no fitting
    return np.exp(test["log_rv_21"].to_numpy())


def ewma(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Forecast next-month vol as the current EWMA volatility."""
    del train  # analytic: no fitting
    return np.exp(test["log_ewma_vol"].to_numpy())


def har_rv(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """HAR-RV: OLS of log target on the daily/weekly/monthly log-RV ladder.

    Fit on the training fold only, then predict and exponentiate to a level.
    """
    x_train = train[_HAR_COLUMNS].to_numpy()
    y_train = np.log(train[TARGET].to_numpy())
    model = LinearRegression().fit(x_train, y_train)
    return np.exp(model.predict(test[_HAR_COLUMNS].to_numpy()))


#: The baseline ladder, keyed by the name used in MLflow and comparison tables.
BASELINE_MODELS = {
    "persistence": persistence,
    "ewma": ewma,
    "har_rv": har_rv,
}
