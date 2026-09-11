"""Full historical backfill for OHLCV and macro, in chunked batches.

Loads from the universe's ``history_start`` through an end date (default today).
Prices are fetched in year-sized chunks with progress logging (build-plan section
4.4) because a single 15-year, 30-ticker request is slow and failure-prone; the
small macro series are fetched in one pass. Everything reuses the idempotent
upsert paths, so a re-run -- or a resume after a failed chunk -- leaves the
database unchanged.

Run as a module::

    python -m volatility_mlops.ingestion.backfill
    python -m volatility_mlops.ingestion.backfill --start 2015-01-01 --end 2020-12-31
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date, timedelta

from sqlalchemy.engine import Engine

from volatility_mlops.db.engine import get_engine
from volatility_mlops.ingestion.macro import FredCsvProvider, ingest_macro, load_macro_series
from volatility_mlops.ingestion.ohlcv import ingest_ohlcv
from volatility_mlops.ingestion.providers import YFinanceProvider
from volatility_mlops.ingestion.universe import load_universe, seed_dim_ticker


def date_chunks(start: date, end: date, years: int = 1) -> list[tuple[date, date]]:
    """Split ``[start, end]`` into consecutive inclusive chunks of ``years``.

    Chunks tile the range with no gaps or overlaps: each chunk's start is the day
    after the previous chunk's end, and the final chunk is clamped to ``end``.

    Args:
        start: First date, inclusive.
        end: Last date, inclusive.
        years: Chunk width in years.

    Returns:
        A list of ``(chunk_start, chunk_end)`` inclusive ranges. Empty if
        ``start > end``.
    """
    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        try:
            boundary = cursor.replace(year=cursor.year + years)
        except ValueError:  # cursor is Feb 29; step to Feb 28 of the target year
            boundary = cursor.replace(year=cursor.year + years, day=28)
        chunk_end = min(end, boundary - timedelta(days=1))
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def backfill_ohlcv(engine: Engine, tickers: Sequence[str], start: date, end: date) -> int:
    """Backfill OHLCV for ``tickers`` over ``[start, end]`` in yearly chunks.

    Args:
        engine: Application-database engine.
        tickers: Symbols to backfill.
        start: First trade date, inclusive.
        end: Last trade date, inclusive.

    Returns:
        Total rows upserted across all chunks.
    """
    provider = YFinanceProvider()
    chunks = date_chunks(start, end)
    total = 0
    for i, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        result = ingest_ohlcv(engine, provider, tickers, chunk_start, chunk_end)
        total += result.rows_written
        print(
            f"[ohlcv] {chunk_start}..{chunk_end}: {result.rows_written} rows "
            f"(chunk {i}/{len(chunks)})"
        )
    return total


def backfill_macro(engine: Engine, start: date, end: date) -> int:
    """Backfill the configured macro series over ``[start, end]`` in one pass.

    Args:
        engine: Application-database engine.
        start: First observation date, inclusive.
        end: Last observation date, inclusive.

    Returns:
        Total macro rows upserted.
    """
    series_ids = load_macro_series()
    result = ingest_macro(engine, FredCsvProvider(), series_ids, start, end)
    print(f"[macro] {start}..{end}: {result.rows_written} rows for {len(series_ids)} series")
    return result.rows_written


def run_backfill(start: date | None = None, end: date | None = None) -> None:
    """Seed the universe, then backfill all OHLCV and macro history.

    Args:
        start: First date; defaults to the universe ``history_start``.
        end: Last date; defaults to today.
    """
    engine = get_engine()
    universe = load_universe()
    seed_dim_ticker(engine, universe)

    start = start or universe.history_start
    end = end or date.today()

    print(f"Backfilling {len(universe.all_tickers())} tickers + macro [{start} .. {end}]")
    ohlcv_rows = backfill_ohlcv(engine, universe.all_tickers(), start, end)
    macro_rows = backfill_macro(engine, start, end)
    print(f"Done. {ohlcv_rows} OHLCV rows, {macro_rows} macro rows.")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the full backfill for the requested (or full) date range."""
    parser = argparse.ArgumentParser(description="Backfill full OHLCV and macro history.")
    parser.add_argument(
        "--start", type=date.fromisoformat, help="YYYY-MM-DD; default history_start"
    )
    parser.add_argument("--end", type=date.fromisoformat, help="YYYY-MM-DD; default today")
    args = parser.parse_args(argv)
    run_backfill(args.start, args.end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
