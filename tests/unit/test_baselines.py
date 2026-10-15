"""Unit tests for baseline models and the walk-forward evaluation loop (no DB)."""

import numpy as np
import pandas as pd

from volatility_mlops.models.baselines import TARGET, ewma, har_rv, persistence
from volatility_mlops.models.evaluate import evaluate_model
from volatility_mlops.validation.splitter import Fold


def test_persistence_returns_exp_of_log_rv_21():
    test = pd.DataFrame({"log_rv_21": np.log([0.2, 0.3])})
    assert np.allclose(persistence(pd.DataFrame(), test), [0.2, 0.3])


def test_ewma_returns_exp_of_log_ewma_vol():
    test = pd.DataFrame({"log_ewma_vol": np.log([0.15, 0.25])})
    assert np.allclose(ewma(pd.DataFrame(), test), [0.15, 0.25])


def test_har_rv_recovers_a_known_linear_map():
    rng = np.random.default_rng(0)
    n = 60
    f = {c: rng.normal(-1.5, 0.3, n) for c in ("log_rv_1", "log_rv_5", "log_rv_21")}
    # log target is an exact linear function of the ladder -> OLS recovers it.
    log_target = 0.1 + 0.5 * f["log_rv_1"] + 0.3 * f["log_rv_5"] + 0.2 * f["log_rv_21"]
    df = pd.DataFrame({**f, TARGET: np.exp(log_target)})
    train, test = df.iloc[:50], df.iloc[50:]

    preds = har_rv(train, test)
    assert np.allclose(preds, test[TARGET].to_numpy())


def test_evaluate_model_returns_one_metric_per_fold():
    n = 20
    panel = pd.DataFrame(
        {
            "log_rv_21": np.log(np.full(n, 0.2)),
            TARGET: np.full(n, 0.2),
        }
    )
    folds = [
        Fold(0, np.arange(0, 8), np.arange(8, 12), None, None, None),  # type: ignore[arg-type]
        Fold(1, np.arange(0, 12), np.arange(12, 16), None, None, None),  # type: ignore[arg-type]
    ]
    results = evaluate_model(panel, folds, persistence)
    assert len(results) == 2
    assert all(np.isfinite(m.qlike) for m in results)
    assert np.isclose(results[0].mae, 0.0)  # persistence == target here
