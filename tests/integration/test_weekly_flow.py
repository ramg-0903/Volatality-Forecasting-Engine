"""Integration tests for the Prefect weekly retrain flow (build-plan Day 20).

Marked ``integration``: needs Postgres (features) and a Production model, and runs the
real flow under an ephemeral Prefect backend. The heavy, non-deterministic steps
(hyperparameter search, per-fold evaluation, MLflow registration/promotion) are
monkeypatched so the tests are fast and leave the registry untouched; what they verify
is the flow WIRING: the gate decision drives promotion, and every run writes an
auditable ``promotion_log`` row with its reason.

The promotion gate's own seven-criteria logic is covered separately in the unit tests.
"""

from __future__ import annotations

import pytest
from prefect.testing.utilities import prefect_test_harness
from sqlalchemy import text

from volatility_mlops.db.engine import get_engine
from volatility_mlops.flows import weekly
from volatility_mlops.models.promotion import PromotionDecision
from volatility_mlops.models.registry import get_production_info

pytestmark = pytest.mark.integration

_FIXED_CFG = {
    "learning_rate": 0.05,
    "max_iter": 100,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 50,
    "l2_regularization": 0.0,
}


@pytest.fixture(scope="module", autouse=True)
def _harness():
    with prefect_test_harness():
        yield


@pytest.fixture(scope="module")
def engine():
    return get_engine()


@pytest.fixture
def _require_model():
    if not get_production_info():
        pytest.skip("no Production model registered; run `make register` first")


def _promotion_rows(engine, decision: str) -> int:
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT count(*) FROM promotion_log WHERE decision = :d"), {"d": decision}
        ).scalar_one()


def _latest_reason(engine) -> str:
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT reason FROM promotion_log ORDER BY id DESC LIMIT 1")
        ).scalar_one()


def _stub_heavy_steps(monkeypatch, decision: PromotionDecision):
    """Replace the slow/stateful steps; keep guardrails + promotion_log writes real."""
    monkeypatch.setattr(weekly, "build_features", lambda engine: 0)  # no rebuild
    monkeypatch.setattr(weekly, "search_challenger", lambda: (dict(_FIXED_CFG), 0.1))
    monkeypatch.setattr(weekly, "evaluate_champion_challenger", lambda *a, **k: ([], []))
    monkeypatch.setattr(weekly, "fit_challenger_pipeline", lambda *a, **k: object())
    monkeypatch.setattr(weekly, "register_pipeline", lambda *a, **k: 99999)  # fake Staging ver
    monkeypatch.setattr(weekly, "_smoke_test_staging", lambda name: True)

    calls: list[tuple] = []

    def _spy_apply(dec, name, version):
        calls.append((dec.promote, name, version))
        return int(version) if dec.promote else None

    monkeypatch.setattr(weekly, "apply_decision", _spy_apply)
    monkeypatch.setattr(weekly, "evaluate_gate", lambda **k: decision)
    return calls


def test_rejection_is_logged_and_nothing_is_promoted(engine, monkeypatch, _require_model):
    decision = PromotionDecision(
        promote=False,
        reason="challenger did not beat champion QLIKE (test)",
        criteria={"primary": False},
        comparison={"champion_qlike": 0.40, "challenger_qlike": 0.42},
    )
    calls = _stub_heavy_steps(monkeypatch, decision)

    before = _promotion_rows(engine, "rejected")
    summary = weekly.weekly_retrain()

    assert summary["decision"] == "rejected"
    assert summary["promoted_version"] is None
    assert _promotion_rows(engine, "rejected") == before + 1  # one audit row written
    assert "did not beat" in _latest_reason(engine)
    assert calls == [(False, "volatility-challenger", "99999")]  # apply_decision saw a no-go


def test_promotion_is_logged_and_applied(engine, monkeypatch, _require_model):
    decision = PromotionDecision(
        promote=True,
        reason="challenger improved QLIKE by 4% with stable calibration (test)",
        criteria={"primary": True},
        comparison={"champion_qlike": 0.42, "challenger_qlike": 0.40},
    )
    calls = _stub_heavy_steps(monkeypatch, decision)

    before = _promotion_rows(engine, "promoted")
    summary = weekly.weekly_retrain()

    assert summary["decision"] == "promoted"
    assert summary["promoted_version"] == 99999
    assert _promotion_rows(engine, "promoted") == before + 1
    assert calls == [(True, "volatility-challenger", "99999")]  # apply_decision asked to promote


def test_row_count_guardrail_aborts_before_training(engine, monkeypatch, _require_model):
    # Pretend the live model trained on far more rows than the current panel holds, so
    # the guardrail trips. Nothing past the gate should run and no log row is written.
    monkeypatch.setattr(weekly, "build_features", lambda engine: 0)
    monkeypatch.setattr(
        weekly, "_production_run_params", lambda name: {"n_train_rows": "100000000"}
    )
    # If training were reached, this would make the failure obvious:
    monkeypatch.setattr(
        weekly,
        "search_challenger",
        lambda: pytest.fail("reached training despite guardrail abort"),
    )

    before_rej = _promotion_rows(engine, "rejected")
    before_prom = _promotion_rows(engine, "promoted")
    with pytest.raises(weekly.RetrainAborted, match="aborting retrain"):
        weekly.weekly_retrain()
    assert _promotion_rows(engine, "rejected") == before_rej  # nothing logged on abort
    assert _promotion_rows(engine, "promoted") == before_prom
