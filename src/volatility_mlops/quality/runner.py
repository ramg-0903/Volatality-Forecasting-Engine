"""Data quality runner: evaluate the checks, persist results, gate the pipeline.

Loads a date window from ``raw_ohlcv``, runs every check in ``config/quality.yml``,
writes each outcome to ``data_quality_results``, and raises
:class:`QualityGateError` if any ``error``-severity check fails -- the "abort the
flow rather than silently poison the model" behaviour of build-plan section 4.5.

Run as a module (defaults to the full history)::

    python -m volatility_mlops.quality.runner
    python -m volatility_mlops.quality.runner --start 2020-01-01 --end 2020-12-31
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import yaml
from sqlalchemy import text
from sqlalchemy.engine import Engine

from volatility_mlops.db.engine import get_engine
from volatility_mlops.quality import checks as C
from volatility_mlops.quality.base import CheckResult, Severity

# repo_root/src/volatility_mlops/quality/runner.py -> parents[3] == repo root
QUALITY_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "quality.yml"


class QualityGateError(RuntimeError):
    """Raised when one or more error-severity checks fail."""


def load_quality_config(path: Path = QUALITY_CONFIG_PATH) -> dict:
    """Load the quality gate configuration from ``quality.yml``."""
    return yaml.safe_load(path.read_text())


def evaluate(
    df: pd.DataFrame, expected_tickers: Sequence[str], config: dict, today: date
) -> list[CheckResult]:
    """Run every check and attach each one's configured severity.

    Args:
        df: Normalized OHLCV frame (``trade_date`` as python ``date``).
        expected_tickers: Active tickers that must cover the calendar.
        config: Parsed ``quality.yml``.
        today: Reference date for the staleness check.

    Returns:
        One :class:`CheckResult` per check, severity attached from config.
    """
    checks_cfg = config["checks"]
    results = [
        C.check_completeness(df, config["calendar_ticker"], expected_tickers),
        C.check_non_null_critical(df),
        C.check_positivity(df),
        C.check_ohlc_consistency(df),
        C.check_duplicate_keys(df),
        C.check_return_sanity(df, checks_cfg["return_sanity"]["max_abs_log_return"]),
        C.check_staleness(df, checks_cfg["staleness"]["max_lag_days"], today),
        C.check_volatility_floor(df, checks_cfg["volatility_floor"]["window"]),
    ]
    return [replace(r, severity=Severity(checks_cfg[r.check_name]["severity"])) for r in results]


def gate_failures(results: Sequence[CheckResult]) -> list[CheckResult]:
    """Return the error-severity checks that failed (the ones that gate)."""
    return [r for r in results if r.severity is Severity.ERROR and not r.passed]


def raise_if_failed(results: Sequence[CheckResult]) -> None:
    """Raise :class:`QualityGateError` if any error-severity check failed."""
    failures = gate_failures(results)
    if failures:
        summary = "; ".join(f"{r.check_name} (observed={r.observed_value})" for r in failures)
        raise QualityGateError(f"{len(failures)} error-severity check(s) failed: {summary}")


_INSERT_RESULT = text(
    """
    INSERT INTO data_quality_results
        (check_name, target_table, run_date, passed, observed_value, threshold)
    VALUES (:check_name, :target_table, :run_date, :passed, :observed_value, :threshold)
    """
)


def persist(engine: Engine, results: Sequence[CheckResult], run_date: date) -> None:
    """Append each check result to ``data_quality_results``."""
    rows = [
        {
            "check_name": r.check_name,
            "target_table": r.target_table,
            "run_date": run_date,
            "passed": r.passed,
            "observed_value": r.observed_value,
            "threshold": r.threshold,
        }
        for r in results
    ]
    with engine.begin() as conn:
        conn.execute(_INSERT_RESULT, rows)


def _load_window(engine: Engine, start: date | None, end: date | None) -> pd.DataFrame:
    """Load OHLCV over an optional ``[start, end]`` window, ``trade_date`` as date."""
    clauses, params = [], {}
    if start is not None:
        clauses.append("trade_date >= :start")
        params["start"] = start
    if end is not None:
        clauses.append("trade_date <= :end")
        params["end"] = end
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    query = text(
        "SELECT ticker, trade_date, open, high, low, close, adj_close, volume "
        f"FROM raw_ohlcv {where} ORDER BY ticker, trade_date"
    )
    df = pd.read_sql(query, engine, params=params, parse_dates=["trade_date"])
    df["trade_date"] = df["trade_date"].dt.date
    return df


def _active_tickers(engine: Engine) -> list[str]:
    """Return the active tickers from ``dim_ticker``."""
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT ticker FROM dim_ticker WHERE active"))
        return [row.ticker for row in rows]


def run_quality_checks(
    engine: Engine,
    start: date | None = None,
    end: date | None = None,
    run_date: date | None = None,
    raise_on_error: bool = True,
) -> list[CheckResult]:
    """Load a window, run all checks, persist results, and optionally gate.

    Args:
        engine: Application-database engine.
        start: Window start (inclusive); full history if omitted.
        end: Window end (inclusive); full history if omitted.
        run_date: Date stamped on the persisted results; today if omitted.
        raise_on_error: Raise :class:`QualityGateError` on error-severity failure.

    Returns:
        Every check's result.

    Raises:
        QualityGateError: If an error-severity check fails and ``raise_on_error``.
    """
    run_date = run_date or date.today()
    config = load_quality_config()
    df = _load_window(engine, start, end)
    expected = _active_tickers(engine)
    results = evaluate(df, expected, config, today=date.today())
    persist(engine, results, run_date)
    if raise_on_error:
        raise_if_failed(results)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    """Run the quality gate over the requested (or full) window and print results."""
    parser = argparse.ArgumentParser(description="Run the data quality gate.")
    parser.add_argument("--start", type=date.fromisoformat, help="YYYY-MM-DD; default all")
    parser.add_argument("--end", type=date.fromisoformat, help="YYYY-MM-DD; default all")
    args = parser.parse_args(argv)

    engine = get_engine()
    results = run_quality_checks(engine, args.start, args.end, raise_on_error=False)

    for r in results:
        status = "PASS" if r.passed else f"FAIL[{r.severity.value}]"
        print(f"  {status:<11} {r.check_name:<20} observed={r.observed_value} {r.detail}")

    failures = gate_failures(results)
    print(f"\n{len(results)} checks run, {len(failures)} error-severity failure(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
