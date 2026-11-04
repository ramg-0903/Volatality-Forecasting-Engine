"""Integration tests for the Prefect daily flow (build-plan Day 19 done-when).

Marked ``integration``: needs Postgres and a Production model. Ingestion is monkeypatched
to no-ops so no test touches a live data provider (project rule). Runs the real flow under
an ephemeral Prefect backend (``prefect_test_harness``). The key guarantee is that a
quality-gate failure halts the flow *before* any prediction is written.
"""

from __future__ import annotations

import pytest
from prefect.testing.utilities import prefect_test_harness
from sqlalchemy import text

from volatility_mlops.db.engine import get_engine
from volatility_mlops.flows import daily
from volatility_mlops.models.registry import get_production_info
from volatility_mlops.quality.runner import QualityGateError

pytestmark = pytest.mark.integration


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


def _prediction_count(engine) -> int:
    with engine.connect() as conn:
        return conn.execute(text("SELECT count(*) FROM predictions")).scalar_one()


def _stub_ingestion(monkeypatch):
    # No network: ingestion becomes a no-op (the DB already holds history).
    monkeypatch.setattr(daily, "ingest_prices", lambda start, end: 0)
    monkeypatch.setattr(daily, "ingest_macro_series", lambda start, end: 0)


def test_quality_failure_halts_before_predictions(engine, monkeypatch):
    _stub_ingestion(monkeypatch)

    def _boom(start, end, run_date):
        raise QualityGateError("injected failure")

    monkeypatch.setattr(daily, "quality_gate", _boom)
    # also stub the downstream steps so, if the gate somehow didn't halt, we'd notice
    monkeypatch.setattr(daily, "rebuild_features", lambda: pytest.fail("ran after gate failure"))

    before = _prediction_count(engine)
    with pytest.raises(QualityGateError):
        daily.daily_inference()
    assert _prediction_count(engine) == before  # nothing written past the gate


def test_happy_path_writes_predictions(engine, monkeypatch, _require_model):
    _stub_ingestion(monkeypatch)
    # Stub the quality gate too: it is clock-sensitive (staleness compares the frozen
    # dev data against the real "today") and is covered on its own in test_quality_runner.
    # This test verifies the flow WIRING -> features rebuild + predictions written.
    monkeypatch.setattr(daily, "quality_gate", lambda start, end, run_date: None)
    n = daily.daily_inference()
    assert n > 0
