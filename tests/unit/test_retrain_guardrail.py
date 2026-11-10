"""Unit tests for the weekly-retrain guardrails and helpers (build-plan Day 20).

Pure: no database, no MLflow, no network. Covers the row-count guardrail (silent
data-loss abort), champion-config parsing from MLflow run params, the golden batch,
and the runtime leakage assertion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volatility_mlops.flows import weekly
from volatility_mlops.flows.weekly import (
    RetrainAborted,
    assert_no_leakage,
    baseline_row_count,
    champion_params,
    check_row_count_guardrail,
    load_golden_batch,
)


def test_guardrail_passes_without_a_baseline():
    # First-ever retrain (no Production model yet): nothing to regress against.
    check_row_count_guardrail(current_rows=10, baseline_rows=None, tolerance=0.02)


def test_guardrail_passes_within_tolerance():
    # 990 vs 1000 baseline is a 1% drop, inside the 2% tolerance.
    check_row_count_guardrail(current_rows=990, baseline_rows=1000, tolerance=0.02)


def test_guardrail_passes_when_panel_grows():
    check_row_count_guardrail(current_rows=1200, baseline_rows=1000, tolerance=0.02)


def test_guardrail_aborts_on_a_large_drop():
    # 950 vs 1000 is a 5% drop, past the 2% floor -> abort.
    with pytest.raises(RetrainAborted, match="silently lost data"):
        check_row_count_guardrail(current_rows=950, baseline_rows=1000, tolerance=0.02)


def test_baseline_row_count_parses_and_defaults():
    assert baseline_row_count({"n_train_rows": "4521"}) == 4521
    assert baseline_row_count({"n_train_rows": "4521.0"}) == 4521
    assert baseline_row_count({}) is None


def test_champion_params_coerces_run_param_strings():
    run_params = {
        "learning_rate": "0.1",
        "max_iter": "300",
        "max_leaf_nodes": "15.0",  # MLflow may stringify an int with a decimal
        "min_samples_leaf": "100",
        "l2_regularization": "1.0",
        "feature_version": "v1",  # ignored: not a hyperparameter
    }
    cfg = champion_params(run_params)
    assert cfg == {
        "learning_rate": 0.1,
        "max_iter": 300,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 100,
        "l2_regularization": 1.0,
    }
    assert isinstance(cfg["max_iter"], int)
    assert isinstance(cfg["learning_rate"], float)


def test_champion_params_falls_back_to_config_defaults():
    # Empty run params -> every hyperparameter comes from config/model.yml.
    cfg = champion_params({})
    for key in ("learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf"):
        assert key in cfg


def test_golden_batch_has_finite_feature_columns():
    from volatility_mlops.features.contract import load_feature_contract

    batch = load_golden_batch()
    expected = load_feature_contract().feature_names()
    assert list(batch.columns) == expected  # exact feature contract, in order
    assert np.isfinite(batch.to_numpy()).all()  # smoke batch must be clean
    assert len(batch) > 0


def _synthetic_panel(n_dates: int = 700, tickers: int = 3) -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=n_dates).date
    rows = [{"trade_date": d, "ticker": f"T{t}"} for d in dates for t in range(tickers)]
    return pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)


def test_assert_no_leakage_passes_on_valid_folds():
    # The real splitter always produces purged/embargoed folds -> no raise.
    assert_no_leakage(_synthetic_panel())


def test_assert_no_leakage_catches_a_leaky_fold(monkeypatch):
    # Inject a fold whose train window overlaps its test window; the guard must fire.
    class _LeakyFold:
        index = 0
        train_idx = np.array([0, 1, 2, 5])
        test_idx = np.array([4, 5, 6])

    monkeypatch.setattr(weekly, "walk_forward_folds", lambda *a, **k: [_LeakyFold()])
    with pytest.raises(RetrainAborted, match="leakage in fold"):
        assert_no_leakage(_synthetic_panel())
