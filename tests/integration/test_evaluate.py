"""Integration test for the DB-backed evaluation path (no MLflow logging).

Marked ``integration``. Loads the real labelled panel and evaluates each baseline
walk-forward, asserting sane metrics. MLflow logging is exercised manually via
`make baselines`, not here, to avoid polluting the tracking server.
"""

import numpy as np
import pytest

from volatility_mlops.db.engine import get_engine
from volatility_mlops.features.contract import load_feature_contract
from volatility_mlops.models.baselines import BASELINE_MODELS
from volatility_mlops.models.evaluate import evaluate_model, load_labeled_panel
from volatility_mlops.validation.splitter import load_validation_config, walk_forward_folds

pytestmark = pytest.mark.integration


def test_baselines_evaluate_over_real_folds():
    engine = get_engine()
    contract = load_feature_contract()
    config = load_validation_config()

    panel = load_labeled_panel(engine, contract.feature_names())
    assert len(panel) > 0

    folds = walk_forward_folds(panel["trade_date"].to_numpy(), config)
    assert len(folds) == config.n_splits

    for fn in BASELINE_MODELS.values():
        fold_metrics = evaluate_model(panel, folds, fn)
        assert len(fold_metrics) == config.n_splits
        for m in fold_metrics:
            assert np.isfinite(m.qlike) and m.qlike >= 0.0
            assert np.isfinite(m.mae) and m.mae > 0.0
            assert m.n > 0
