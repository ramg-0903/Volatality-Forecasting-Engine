"""Integration test for the quality runner against the real Postgres stack.

Marked ``integration``. Runs the gate over the real (clean) data, confirms every
result is persisted, and that no error-severity check fails. Uses a sentinel
``run_date`` so the persisted rows can be cleaned up.
"""

from datetime import date

import pytest
from sqlalchemy import text

from volatility_mlops.db.engine import get_engine
from volatility_mlops.quality.base import Severity
from volatility_mlops.quality.runner import run_quality_checks

pytestmark = pytest.mark.integration

SENTINEL = date(1990, 1, 1)  # marker run_date so test rows are easy to remove


@pytest.fixture
def engine():
    return get_engine()


@pytest.fixture(autouse=True)
def _clean_results(engine):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM data_quality_results WHERE run_date = :d"), {"d": SENTINEL})
    yield
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM data_quality_results WHERE run_date = :d"), {"d": SENTINEL})


def test_run_persists_results_and_real_data_passes_error_gates(engine):
    results = run_quality_checks(engine, run_date=SENTINEL, raise_on_error=False)

    assert len(results) == 8

    with engine.connect() as conn:
        persisted = conn.execute(
            text("SELECT count(*) FROM data_quality_results WHERE run_date = :d"),
            {"d": SENTINEL},
        ).scalar_one()
    assert persisted == 8

    # Real data must clear every ERROR-severity gate (warn-level flags are allowed).
    errors = [r for r in results if r.severity is Severity.ERROR and not r.passed]
    assert errors == [], f"unexpected error-severity failures: {errors}"
