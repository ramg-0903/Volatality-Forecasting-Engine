"""Shared ingestion run-audit helpers.

Every ingestion (OHLCV or macro) is bracketed by an ``ingestion_runs`` row:
inserted as ``running``, then marked ``success`` or ``failed`` with a row count
or an error message. Centralised here so all ingestion paths audit identically
(build-plan section 4.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from sqlalchemy import text
from sqlalchemy.engine import Engine


@dataclass(frozen=True)
class IngestResult:
    """Outcome of one ingestion run."""

    run_id: int
    rows_written: int
    status: str


def start_run(engine: Engine, source: str, start: date, end: date) -> int:
    """Insert a ``running`` ``ingestion_runs`` row and return its id.

    Args:
        engine: Application-database engine.
        source: Provider name recorded on the run.
        start: First date of the requested range, inclusive.
        end: Last date of the requested range, inclusive.

    Returns:
        The new ``run_id``.
    """
    with engine.begin() as conn:
        row = conn.execute(
            text(
                """
                INSERT INTO ingestion_runs (source, date_start, date_end, status)
                VALUES (:source, :date_start, :date_end, 'running')
                RETURNING run_id
                """
            ),
            {"source": source, "date_start": start, "date_end": end},
        ).one()
    return int(row.run_id)


def finish_run(
    engine: Engine,
    run_id: int,
    status: str,
    rows_written: int | None = None,
    error: str | None = None,
) -> None:
    """Mark an ``ingestion_runs`` row terminal (``success`` or ``failed``).

    Args:
        engine: Application-database engine.
        run_id: The run to finalise.
        status: Terminal status, ``success`` or ``failed``.
        rows_written: Rows upserted on success; ``None`` on failure.
        error: Error message on failure; ``None`` on success.
    """
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE ingestion_runs
                   SET status = :status,
                       rows_written = :rows_written,
                       error_message = :error,
                       finished_at = now()
                 WHERE run_id = :run_id
                """
            ),
            {"status": status, "rows_written": rows_written, "error": error, "run_id": run_id},
        )
