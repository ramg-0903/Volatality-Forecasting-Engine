"""The gradient-boosting challenger and its preprocessing pipeline (section 5.3/5.4).

A single pooled model across all tickers (with the entity-encoding feature so it can
still express per-name differences). Preprocessing lives inside a scikit-learn
Pipeline that is fit per fold on training rows only (leakage rule 3): median
imputation, then a HistGradientBoostingRegressor. Trees are scale- and
outlier-robust, so no scaling or winsorizing is needed (those matter for the linear
baselines, not here). The model is trained on the log target and exponentiated back
to a level at prediction time.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from volatility_mlops.models.baselines import TARGET

ModelFn = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


def build_pipeline(params: dict[str, object]) -> Pipeline:
    """Build the impute -> HistGradientBoosting pipeline with the given params."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(random_state=0, early_stopping=False, **params),
            ),
        ]
    )


def make_challenger(feature_cols: list[str], params: dict[str, object]) -> ModelFn:
    """Return a ``(train, test) -> preds`` challenger for the given hyperparameters.

    The returned function fits the pipeline on the training fold (log target) and
    predicts annualized vol levels on the test fold.
    """

    def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        pipeline = build_pipeline(params)
        pipeline.fit(train[feature_cols], np.log(train[TARGET].to_numpy()))
        return np.exp(pipeline.predict(test[feature_cols]))

    return fit_predict
