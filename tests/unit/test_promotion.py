"""Unit tests for the promotion gate (build-plan Day 16 done-when).

Covers every decision branch: all-pass -> promote, and each criterion failing in
isolation -> reject with the right reason.
"""

from volatility_mlops.metrics.metrics import Metrics
from volatility_mlops.models.promotion import (
    GateConfig,
    evaluate_gate,
    load_promotion_config,
    run_smoke_test,
)

CONFIG = GateConfig(
    primary_margin=0.02,
    secondary_margin=0.01,
    stability_window=4,
    max_losses=1,
    calibration_low=0.8,
    calibration_high=1.2,
)
COLS = ["a", "b"]


def _folds(qlikes, maes, slopes) -> list[Metrics]:
    return [
        Metrics(mae=m, rmse=m, qlike=q, mz_slope=s, mz_r2=0.3, n=100)
        for q, m, s in zip(qlikes, maes, slopes, strict=True)
    ]


# A champion and a clearly-better challenger that should pass every criterion.
CHAMPION = _folds([0.30, 0.30, 0.30, 0.30], [0.08] * 4, [1.0] * 4)
CHALLENGER = _folds([0.28, 0.28, 0.28, 0.28], [0.079] * 4, [1.0] * 4)


def _gate(champion=CHAMPION, challenger=CHALLENGER, cols=COLS, smoke=True, health=True):
    return evaluate_gate(champion, challenger, cols, COLS, smoke, health, CONFIG)


def test_all_criteria_pass_promotes():
    d = _gate()
    assert d.promote
    assert d.reason == "all 7 criteria passed"
    assert all(d.criteria.values())


def test_primary_metric_failure_rejects():
    challenger = _folds([0.299] * 4, [0.079] * 4, [1.0] * 4)  # only 0.3% better, < 2%
    d = _gate(challenger=challenger)
    assert not d.promote and not d.criteria["primary"] and "primary" in d.reason


def test_secondary_regression_rejects():
    challenger = _folds([0.28] * 4, [0.10] * 4, [1.0] * 4)  # MAE 25% worse
    d = _gate(challenger=challenger)
    assert not d.promote and not d.criteria["secondary"] and "secondary" in d.reason


def test_stability_failure_rejects():
    # Beats on the last-2 mean (primary passes) but loses on 2 of the last 4 folds.
    challenger = _folds([0.35, 0.35, 0.26, 0.26], [0.079] * 4, [1.0] * 4)
    d = _gate(challenger=challenger)
    assert not d.promote and not d.criteria["stability"] and "stability" in d.reason


def test_calibration_failure_rejects():
    challenger = _folds([0.28] * 4, [0.079] * 4, [1.5] * 4)  # slope outside [0.8, 1.2]
    d = _gate(challenger=challenger)
    assert not d.promote and not d.criteria["calibration"] and "calibration" in d.reason


def test_schema_failure_rejects():
    d = _gate(cols=["a", "b", "c"])  # mismatches the contract columns
    assert not d.promote and not d.criteria["schema"] and "schema" in d.reason


def test_smoke_failure_rejects():
    d = _gate(smoke=False)
    assert not d.promote and not d.criteria["smoke"] and "smoke" in d.reason


def test_data_health_failure_rejects():
    d = _gate(health=False)
    assert not d.promote and not d.criteria["data_health"] and "data_health" in d.reason


def test_config_loads_from_yaml():
    cfg = load_promotion_config()
    assert cfg.primary_margin == 0.02
    assert cfg.max_losses == 1


class _Const:
    def __init__(self, values):
        self.values = values

    def predict(self, _):
        return self.values


class _Raiser:
    def predict(self, _):
        raise RuntimeError("boom")


def test_run_smoke_test():
    assert run_smoke_test(_Const([0.1, 0.2]), None) is True
    assert run_smoke_test(_Const([-0.1, 0.2]), None) is False  # non-positive
    assert run_smoke_test(_Const([float("nan")]), None) is False  # non-finite
    assert run_smoke_test(_Raiser(), None) is False  # raises
