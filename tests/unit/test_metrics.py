"""Unit tests for the metrics module (build-plan Day 11 done-when).

Each metric is checked against a hand-computed value on a tiny example.
"""

import numpy as np

from volatility_mlops.metrics.metrics import (
    compute_metrics,
    mae,
    mincer_zarnowitz,
    qlike,
    rmse,
)


def test_mae_and_rmse_match_by_hand():
    y_true = [0.20, 0.30]
    y_pred = [0.25, 0.25]
    # errors are +/-0.05 each
    assert np.isclose(mae(y_true, y_pred), 0.05)
    assert np.isclose(rmse(y_true, y_pred), 0.05)


def test_qlike_is_zero_for_a_perfect_forecast():
    y = [0.15, 0.20, 0.25]
    assert np.isclose(qlike(y, y), 0.0)


def test_qlike_matches_closed_form():
    # Single pair: v_true=0.04 (0.2^2), v_pred=0.01 (0.1^2), ratio=4.
    # QLIKE = 4 - ln(4) - 1 = 3 - ln(4).
    expected = 3.0 - np.log(4.0)
    assert np.isclose(qlike([0.2], [0.1]), expected)


def test_mincer_zarnowitz_recovers_a_known_line():
    y_pred = np.array([0.1, 0.2, 0.3, 0.4])
    y_true = 2.0 * y_pred  # exact line through origin, slope 2
    slope, r2 = mincer_zarnowitz(y_true, y_pred)
    assert np.isclose(slope, 2.0)
    assert np.isclose(r2, 1.0)


def test_perfect_forecast_is_well_calibrated():
    y = [0.12, 0.18, 0.22, 0.30]
    m = compute_metrics(y, y)
    assert np.isclose(m.mae, 0.0)
    assert np.isclose(m.qlike, 0.0)
    assert np.isclose(m.mz_slope, 1.0)
    assert np.isclose(m.mz_r2, 1.0)
    assert m.n == 4


def test_non_finite_and_nonpositive_pairs_are_dropped():
    y_true = [0.2, np.nan, 0.3, 0.0]
    y_pred = [0.25, 0.25, np.inf, 0.1]
    # Only the first pair survives the finite/positive filter.
    assert compute_metrics(y_true, y_pred).n == 1
