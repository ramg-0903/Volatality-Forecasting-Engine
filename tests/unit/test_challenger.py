"""Unit test for the challenger pipeline (no DB; small synthetic data)."""

import numpy as np
import pandas as pd

from volatility_mlops.models.baselines import TARGET
from volatility_mlops.models.pipeline import make_challenger


def test_make_challenger_fits_and_predicts_positive_levels():
    rng = np.random.default_rng(0)
    n = 200
    feature_cols = ["f1", "f2", "f3"]
    features = {c: rng.normal(0, 1, n) for c in feature_cols}
    log_target = 0.2 * features["f1"] - 0.1 * features["f2"] + rng.normal(0, 0.05, n)
    df = pd.DataFrame({**features, TARGET: np.exp(log_target)})
    train, test = df.iloc[:150], df.iloc[150:]

    fn = make_challenger(feature_cols, {"max_iter": 50, "learning_rate": 0.1})
    preds = fn(train, test)

    assert preds.shape == (len(test),)
    assert np.all(np.isfinite(preds))
    assert np.all(preds > 0)  # exponentiated log predictions are strictly positive
