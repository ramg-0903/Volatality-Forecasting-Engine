"""OHLCV ingestion: fetch, idempotent upsert, and run auditing.

Ties a provider to the database. Every run is bracketed by an ``ingestion_runs``
row (status ``running`` -> ``success``/``failed``), and prices are written with
an ``ON CONFLICT`` upsert on the ``(ticker, trade_date)`` natural key, so
re-running any date range leaves ``raw_ohlcv`` unchanged (build-plan rule 5).

Run as a module::

    python -m volatility_mlops.ingestion.ohlcv --start 2024-01-01 --end 2024-03-01
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from volatility_mlops.db.engine import get_engine
from volatility_mlops.ingestion.base import MarketDataProvider, normalize_ohlcv
from volatility_mlops.ingestion.providers import YFinanceProvider
from volatility_mlops.ingestion.runlog import IngestResult, finish_run, start_run
from volatility_mlops.ingestion.universe import load_universe, seed_dim_ticker

_UPSERT_OHLCV = text(
    """
    INSERT INTO raw_ohlcv
        (ticker, trade_date, open, high, low, close, adj_close, volume, source)
    VALUES
        (:ticker, :trade_date, :open, :high, :low, :close, :adj_close, :volume, :source)
    ON CONFLICT (ticker, trade_date) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        adj_close = EXCLUDED.adj_close,
        volume = EXCLUDED.volume,
        source = EXCLUDED.source,
        ingested_at = now()
    """
)


def _build_records(frame: pd.DataFrame, source: str) -> list[dict[str, object]]:
    """Turn a normalized frame into upsert parameter dicts (NaN -> None)."""
    clean = frame.where(frame.notna(), None)
    records: list[dict[str, object]] = []
    for row in clean.to_dict("records"):
        volume = row["volume"]
        records.append(
            {
                "ticker": row["ticker"],
                "trade_date": row["trade_date"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "adj_close": row["adj_close"],
                "volume": int(volume) if volume is not None else None,
                "source": source,
            }
        )
    return records


def upsert_ohlcv(engine: Engine, frame: pd.DataFrame, source: str) -> int:
    """Upsert normalized OHLCV rows into ``raw_ohlcv``.

    Args:
        engine: Application-database engine.
        frame: A frame already conforming to the canonical OHLCV contract.
        source: Provider name, stored on every row for provenance.

    Returns:
        The number of rows upserted.
    """
    records = _build_records(frame, source)
    if not records:
        return 0
    with engine.begin() as conn:
        conn.execute(_UPSERT_OHLCV, records)
    return len(records)


def ingest_ohlcv(
    engine: Engine,
    provider: MarketDataProvider,
    tickers: Sequence[str],
    start: date,
    end: date,
) -> IngestResult:
    """Fetch prices for ``tickers`` over ``[start, end]`` and upsert them.

    The run is recorded in ``ingestion_runs`` regardless of outcome. Tickers must
    already exist in ``dim_ticker`` (the foreign key is enforced); the CLI seeds
    the universe first.

    Args:
        engine: Application-database engine.
        provider: Market-data source.
        tickers: Symbols to ingest.
        start: First trade date, inclusive.
        end: Last trade date, inclusive.

    Returns:
        The run id, rows written, and terminal status.

    Raises:
        Exception: Re-raised after the run is marked ``failed``.
    """
    run_id = start_run(engine, provider.name, start, end)
    try:
        frame = normalize_ohlcv(provider.fetch_ohlcv(list(tickers), start, end))
        rows = upsert_ohlcv(engine, frame, source=provider.name)
    except Exception as exc:  # noqa: BLE001 -- record the failure, then re-raise
        finish_run(engine, run_id, "failed", error=str(exc))
        raise
    finish_run(engine, run_id, "success", rows_written=rows)
    return IngestResult(run_id=run_id, rows_written=rows, status="success")


def main(argv: Sequence[str] | None = None) -> int:
    """Seed the universe, then ingest OHLCV for the requested date range."""
    parser = argparse.ArgumentParser(description="Ingest daily OHLCV bars.")
    parser.add_argument("--start", required=True, type=date.fromisoformat, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, type=date.fromisoformat, help="YYYY-MM-DD")
    parser.add_argument(
        "--tickers",
        nargs="*",
        help="Symbols to ingest; defaults to the whole universe.",
    )
    args = parser.parse_args(argv)

    engine = get_engine()
    universe = load_universe()
    seed_dim_ticker(engine, universe)

    tickers = args.tickers or universe.all_tickers()
    result = ingest_ohlcv(engine, YFinanceProvider(), tickers, args.start, args.end)

    print(
        f"run {result.run_id}: {result.status}, "
        f"{result.rows_written} rows upserted for {len(tickers)} tickers "
        f"[{args.start} .. {args.end}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
