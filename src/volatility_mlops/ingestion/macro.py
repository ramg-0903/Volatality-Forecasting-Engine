"""Macro ingestion from FRED: fetch, idempotent upsert, and run auditing.

Pulls the series in ``config/macro.yml`` (VIX and Treasury yields, build-plan
section 4.1) from FRED's public ``fredgraph.csv`` endpoint -- no API key -- and
upserts them into ``raw_macro`` on the ``(series_id, obs_date)`` natural key, so
re-running any range leaves the table unchanged (build-plan rule 5). Every run is
recorded in ``ingestion_runs`` via the shared run-audit helpers.

Run as a module (defaults to the full configured history through today)::

    python -m volatility_mlops.ingestion.macro
    python -m volatility_mlops.ingestion.macro --start 2020-01-01 --end 2020-06-01
"""

from __future__ import annotations

import argparse
import io
import urllib.parse
import urllib.request
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd
import yaml
from sqlalchemy import text
from sqlalchemy.engine import Engine

from volatility_mlops.db.engine import get_engine
from volatility_mlops.ingestion.base import MACRO_COLUMNS, MacroDataProvider, normalize_macro
from volatility_mlops.ingestion.runlog import IngestResult, finish_run, start_run
from volatility_mlops.ingestion.universe import load_universe

# repo_root/src/volatility_mlops/ingestion/macro.py -> parents[3] == repo root
MACRO_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "macro.yml"

_UPSERT_MACRO = text(
    """
    INSERT INTO raw_macro (series_id, obs_date, value)
    VALUES (:series_id, :obs_date, :value)
    ON CONFLICT (series_id, obs_date) DO UPDATE SET
        value = EXCLUDED.value,
        ingested_at = now()
    """
)


def load_macro_series(path: Path = MACRO_CONFIG_PATH) -> list[str]:
    """Return the FRED series ids to ingest, from ``macro.yml``.

    Args:
        path: Path to the macro YAML file.

    Returns:
        The FRED series codes (e.g. ``["VIXCLS", "DGS10", "DGS3MO"]``).
    """
    data = yaml.safe_load(path.read_text())
    return list(data["series"].keys())


class FredCsvProvider(MacroDataProvider):
    """Fetches FRED series from the public ``fredgraph.csv`` endpoint (no key)."""

    name = "fred"
    base_url = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    def fetch_macro(self, series_ids: list[str], start: date, end: date) -> pd.DataFrame:
        """Fetch each series over ``[start, end]`` and return the long frame."""
        frames = [self._fetch_one(series_id, start, end) for series_id in series_ids]
        if not frames:
            return pd.DataFrame(columns=MACRO_COLUMNS)
        return normalize_macro(pd.concat(frames, ignore_index=True))

    def _fetch_one(self, series_id: str, start: date, end: date) -> pd.DataFrame:
        """Download one series as CSV and shape it to the canonical columns."""
        query = urllib.parse.urlencode(
            {"id": series_id, "cosd": start.isoformat(), "coed": end.isoformat()}
        )
        url = f"{self.base_url}?{query}"
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 -- fixed https host
            raw = response.read().decode("utf-8")
        frame = pd.read_csv(io.StringIO(raw))
        # FRED returns two columns: a date column then the value column. Take them
        # by position so a header rename on FRED's side (DATE -> observation_date)
        # does not break parsing.
        date_col, value_col = frame.columns[0], frame.columns[1]
        return pd.DataFrame(
            {"series_id": series_id, "obs_date": frame[date_col], "value": frame[value_col]}
        )


def upsert_macro(engine: Engine, frame: pd.DataFrame) -> int:
    """Upsert normalized macro rows into ``raw_macro``.

    Args:
        engine: Application-database engine.
        frame: A frame already conforming to the canonical macro contract.

    Returns:
        The number of rows upserted.
    """
    clean = frame.where(frame.notna(), None)
    records = [
        {"series_id": row["series_id"], "obs_date": row["obs_date"], "value": row["value"]}
        for row in clean.to_dict("records")
    ]
    if not records:
        return 0
    with engine.begin() as conn:
        conn.execute(_UPSERT_MACRO, records)
    return len(records)


def ingest_macro(
    engine: Engine,
    provider: MacroDataProvider,
    series_ids: Sequence[str],
    start: date,
    end: date,
) -> IngestResult:
    """Fetch macro series over ``[start, end]`` and upsert them.

    The run is recorded in ``ingestion_runs`` regardless of outcome.

    Args:
        engine: Application-database engine.
        provider: Macro data source.
        series_ids: FRED series codes to ingest.
        start: First observation date, inclusive.
        end: Last observation date, inclusive.

    Returns:
        The run id, rows written, and terminal status.

    Raises:
        Exception: Re-raised after the run is marked ``failed``.
    """
    run_id = start_run(engine, provider.name, start, end)
    try:
        frame = normalize_macro(provider.fetch_macro(list(series_ids), start, end))
        rows = upsert_macro(engine, frame)
    except Exception as exc:  # noqa: BLE001 -- record the failure, then re-raise
        finish_run(engine, run_id, "failed", error=str(exc))
        raise
    finish_run(engine, run_id, "success", rows_written=rows)
    return IngestResult(run_id=run_id, rows_written=rows, status="success")


def main(argv: Sequence[str] | None = None) -> int:
    """Ingest the configured macro series for the requested (or full) range."""
    parser = argparse.ArgumentParser(description="Ingest FRED macro series.")
    parser.add_argument(
        "--start", type=date.fromisoformat, help="YYYY-MM-DD; default history_start"
    )
    parser.add_argument("--end", type=date.fromisoformat, help="YYYY-MM-DD; default today")
    args = parser.parse_args(argv)

    start = args.start or load_universe().history_start
    end = args.end or date.today()
    series_ids = load_macro_series()

    engine = get_engine()
    result = ingest_macro(engine, FredCsvProvider(), series_ids, start, end)

    print(
        f"run {result.run_id}: {result.status}, "
        f"{result.rows_written} rows upserted for {len(series_ids)} series "
        f"[{start} .. {end}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
