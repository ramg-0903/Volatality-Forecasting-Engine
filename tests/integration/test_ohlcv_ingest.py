"""Integration tests for OHLCV ingestion against the real Postgres stack.

Marked ``integration`` (deselect with ``-m 'not integration'``). Uses a fake,
fixture-backed provider so no test ever touches a live data vendor.
"""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import text

from volatility_mlops.db.engine import get_engine
from volatility_mlops.ingestion.base import MarketDataProvider
from volatility_mlops.ingestion.ohlcv import ingest_ohlcv
from volatility_mlops.ingestion.universe import TickerSpec, upsert_tickers

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample_ohlcv.csv"
FIXTURE_TICKERS = ["FAKEA", "FAKEB"]


class FixtureProvider(MarketDataProvider):
    """Returns the CSV fixture, ignoring the date range (deterministic)."""

    name = "fixture"

    def fetch_ohlcv(self, tickers, start, end):  # noqa: ANN001, ANN201, ARG002
        frame = pd.read_csv(FIXTURE)
        return frame[frame["ticker"].isin(tickers)]


@pytest.fixture
def engine():
    return get_engine()


@pytest.fixture(autouse=True)
def _clean_fixture_rows(engine):
    """Ensure the fixture tickers exist and start with no price rows.

    Cleans up on teardown too, so the shared dev database is never left with
    fake tickers that would pollute later quality checks.
    """
    upsert_tickers(
        engine,
        [TickerSpec(t, f"Fake {t}", "Test") for t in FIXTURE_TICKERS],
    )
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM raw_ohlcv WHERE ticker = ANY(:tickers)"),
            {"tickers": FIXTURE_TICKERS},
        )
    yield
    with engine.begin() as conn:
        # raw_ohlcv first: it references dim_ticker.
        conn.execute(
            text("DELETE FROM raw_ohlcv WHERE ticker = ANY(:tickers)"),
            {"tickers": FIXTURE_TICKERS},
        )
        conn.execute(
            text("DELETE FROM dim_ticker WHERE ticker = ANY(:tickers)"),
            {"tickers": FIXTURE_TICKERS},
        )


def _count_rows(engine) -> int:
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT count(*) FROM raw_ohlcv WHERE ticker = ANY(:tickers)"),
            {"tickers": FIXTURE_TICKERS},
        ).scalar_one()


def test_ingest_writes_rows_and_records_run(engine):
    result = ingest_ohlcv(
        engine, FixtureProvider(), FIXTURE_TICKERS, date(2024, 1, 1), date(2024, 2, 1)
    )

    assert result.status == "success"
    assert result.rows_written == 6
    assert _count_rows(engine) == 6

    with engine.connect() as conn:
        run = conn.execute(
            text("SELECT source, status, rows_written FROM ingestion_runs WHERE run_id = :id"),
            {"id": result.run_id},
        ).one()
    assert run.source == "fixture"
    assert run.status == "success"
    assert run.rows_written == 6


def test_ingest_is_idempotent(engine):
    provider = FixtureProvider()
    ingest_ohlcv(engine, provider, FIXTURE_TICKERS, date(2024, 1, 1), date(2024, 2, 1))
    first = _count_rows(engine)
    ingest_ohlcv(engine, provider, FIXTURE_TICKERS, date(2024, 1, 1), date(2024, 2, 1))
    second = _count_rows(engine)

    assert first == second == 6  # re-running leaves raw_ohlcv unchanged
