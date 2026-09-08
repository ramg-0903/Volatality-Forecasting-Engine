"""Integration test: migrations apply to Postgres and are re-runnable.

Requires the running stack (``make up``). Skipped by default in unit-only runs:
select with ``-m integration``. This test is destructive to schema state, so it
targets the local application database only.
"""

import pytest
from sqlalchemy import inspect

from volatility_mlops.db.engine import get_engine
from volatility_mlops.db.migrate import apply_migrations

pytestmark = pytest.mark.integration

EXPECTED_TABLES = {
    "dim_ticker",
    "raw_ohlcv",
    "raw_macro",
    "features",
    "predictions",
    "prediction_outcomes",
    "ingestion_runs",
    "data_quality_results",
    "drift_reports",
    "promotion_log",
    "schema_migrations",
}


def test_migrations_apply_and_are_rerunnable():
    engine = get_engine()

    # First application (no-op if a previous run already applied them).
    apply_migrations(engine)

    tables = set(inspect(engine).get_table_names())
    assert tables >= EXPECTED_TABLES

    # Re-running must apply nothing: the idempotency guarantee.
    assert apply_migrations(engine) == []
