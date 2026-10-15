"""Unit tests for the walk-forward splitter (build-plan Day 10 done-when).

The leakage test is the non-negotiable one and must never be weakened: in every
fold, the max training index is below the min test index, with a purge+embargo gap.
"""

import pandas as pd
import pytest

from volatility_mlops.validation.splitter import (
    WalkForwardConfig,
    load_validation_config,
    walk_forward_folds,
)

CONFIG = WalkForwardConfig(n_splits=3, test_size=40, purge=21, embargo=5, min_train_size=50)


def _panel(n_dates: int = 400, n_tickers: int = 3) -> pd.DataFrame:
    """A pooled panel sorted by trade_date (dates repeat across tickers)."""
    dates = pd.bdate_range("2018-01-01", periods=n_dates).date
    rows = [{"ticker": f"T{t}", "trade_date": d} for d in dates for t in range(n_tickers)]
    return pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)


def test_no_leakage_max_train_index_below_min_test_index():
    panel = _panel()
    folds = walk_forward_folds(panel["trade_date"].to_numpy(), CONFIG)

    assert len(folds) == CONFIG.n_splits
    for fold in folds:
        # Panel is sorted by date, so positional order == time order.
        assert fold.train_idx.max() < fold.test_idx.min()
        assert fold.train_end < fold.test_start


def test_purge_and_embargo_gap_exceeds_horizon():
    panel = _panel()
    folds = walk_forward_folds(panel["trade_date"].to_numpy(), CONFIG)

    unique = sorted(panel["trade_date"].unique())
    pos = {d: i for i, d in enumerate(unique)}
    for fold in folds:
        # Unique trading dates strictly between train_end and test_start.
        strictly_between = pos[fold.test_start] - pos[fold.train_end] - 1
        assert strictly_between == CONFIG.purge + CONFIG.embargo
        assert strictly_between > CONFIG.purge  # the label horizon cannot reach the test window


def test_test_windows_are_disjoint_and_train_expands():
    panel = _panel()
    folds = walk_forward_folds(panel["trade_date"].to_numpy(), CONFIG)

    seen: set[int] = set()
    for fold in folds:
        test_positions = set(fold.test_idx.tolist())
        assert seen.isdisjoint(test_positions)  # no test row reused across folds
        seen |= test_positions
    train_sizes = [f.train_idx.size for f in folds]
    assert train_sizes == sorted(train_sizes)  # expanding training window


def test_raises_when_sample_too_short():
    panel = _panel(n_dates=120)  # not enough for 3 x 40 test + gap + min_train
    with pytest.raises(ValueError, match="too few"):
        walk_forward_folds(panel["trade_date"].to_numpy(), CONFIG)


def test_config_loads_from_yaml():
    config = load_validation_config()
    assert config.purge == 21
    assert config.embargo == 5
    assert config.n_splits >= 1
