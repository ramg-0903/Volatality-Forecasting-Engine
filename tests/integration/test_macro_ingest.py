"""Integration tests for macro ingestion against the real Postgres stack.

Marked ``integration`` (deselect with ``-m 'not integration'``). Uses a fake,
fixture-backed provider so no test ever touches FRED.
"""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import text

from volatility_mlops.db.engine import get_engine
from volatility_mlops.ingestion.base import MacroDataProvider
from volatility_mlops.ingestion.macro import ingest_macro

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample_macro.csv"
FIXTURE_SERIES = ["FAKEIDX", "FAKEYLD"]


class FixtureMacroProvider(MacroDataProvider):
    """Returns the CSV fixture, ignoring the date range (deterministic)."""

    name = "fixture-macro"

    def fetch_macro(self, series_ids, start, end):  # noqa: ANN001, ANN201, ARG002
        frame = pd.read_csv(FIXTURE)
        return frame[frame["series_id"].isin(series_ids)]


@pytest.fixture
def engine():
    return get_engine()


@pytest.fixture(autouse=True)
def _clean_fixture_rows(engine):
    """Start and end each test with no rows for the fake series.

    Teardown keeps the shared dev database free of fixture macro rows.
    """
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM raw_macro WHERE series_id = ANY(:ids)"),
            {"ids": FIXTURE_SERIES},
        )
    yield
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM raw_macro WHERE series_id = ANY(:ids)"),
            {"ids": FIXTURE_SERIES},
        )


def _count_rows(engine) -> int:
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT count(*) FROM raw_macro WHERE series_id = ANY(:ids)"),
            {"ids": FIXTURE_SERIES},
        ).scalar_one()


def test_ingest_writes_rows_and_records_run(engine):
    result = ingest_macro(
        engine, FixtureMacroProvider(), FIXTURE_SERIES, date(2024, 1, 1), date(2024, 2, 1)
    )

    assert result.status == "success"
    assert result.rows_written == 5  # one missing "." row dropped, one duplicate collapsed
    assert _count_rows(engine) == 5

    with engine.connect() as conn:
        run = conn.execute(
            text("SELECT source, status, rows_written FROM ingestion_runs WHERE run_id = :id"),
            {"id": result.run_id},
        ).one()
    assert run.source == "fixture-macro"
    assert run.status == "success"
    assert run.rows_written == 5


def test_ingest_is_idempotent(engine):
    provider = FixtureMacroProvider()
    ingest_macro(engine, provider, FIXTURE_SERIES, date(2024, 1, 1), date(2024, 2, 1))
    first = _count_rows(engine)
    ingest_macro(engine, provider, FIXTURE_SERIES, date(2024, 1, 1), date(2024, 2, 1))
    second = _count_rows(engine)

    assert first == second == 5  # re-running leaves raw_macro unchanged
