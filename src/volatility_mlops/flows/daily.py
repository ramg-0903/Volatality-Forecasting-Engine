"""Prefect daily inference flow (build-plan section 9.1 / Day 19).

Chains the daily inference path into one scheduled pipeline::

    ingest prices -> ingest macro -> quality gate -> rebuild features
        -> load Production model -> write predictions

Policies (section 9.1):

* The two **ingestion** tasks retry (3x, backoff); everything downstream fails fast.
* The **quality gate** raises on an error-severity failure, which halts the flow
  *before* any prediction is written -- never predict on data you do not trust.
* All writes are idempotent upserts, so re-running a date overwrites rather than
  appends. The ingestion window overlaps the previous few business days to pick up
  late provider corrections.

The resolve-matured-labels, rolling-metrics, drift, and alert steps (the rest of
section 9.1) are the monitoring days' work (22-24) and are intentionally not here yet.

``write_predictions`` carries strict validation: it refuses to write without a model
version stamp, rejects any non-finite or non-positive forecast, and checks the
horizon maturity date is in the future -- so a corrupt batch fails loudly instead of
landing in the table.

Run once::

    python -m volatility_mlops.flows.daily            # as of today
    python -m volatility_mlops.flows.daily --date 2026-09-09
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from prefect import flow, task
from sqlalchemy import text
from sqlalchemy.engine import Engine

from volatility_mlops.db.engine import get_engine
from volatility_mlops.features.build import build_features
from volatility_mlops.features.contract import load_feature_contract
from volatility_mlops.ingestion.macro import FredCsvProvider, ingest_macro, load_macro_series
from volatility_mlops.ingestion.ohlcv import ingest_ohlcv
from volatility_mlops.ingestion.providers import YFinanceProvider
from volatility_mlops.ingestion.universe import load_universe, seed_dim_ticker
from volatility_mlops.models.registry import get_production_info, load_production_model
from volatility_mlops.quality.runner import run_quality_checks

# repo_root/src/volatility_mlops/flows/daily.py -> parents[3] == repo root
FLOWS_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "flows.yml"


@dataclass(frozen=True)
class FlowsConfig:
    """Daily-flow parameters from ``config/flows.yml``."""

    ingest_lookback_days: int
    prediction_horizon_days: int
    schedule_cron: str


def load_flows_config(path: Path = FLOWS_CONFIG_PATH) -> FlowsConfig:
    """Load orchestration parameters from ``flows.yml``."""
    data = yaml.safe_load(path.read_text())
    return FlowsConfig(
        ingest_lookback_days=int(data["ingest_lookback_days"]),
        prediction_horizon_days=int(data["prediction_horizon_days"]),
        schedule_cron=str(data["schedule_cron"]),
    )


def _busday_offset(d: date, n: int) -> date:
    """Shift ``d`` by ``n`` business days (Mon-Fri). Approximates trading days."""
    shifted = np.busday_offset(np.datetime64(d, "D"), n, roll="forward")
    return shifted.astype("datetime64[D]").item()


# --------------------------------------------------------------------------------------
# Batch inference writer (the one substantial new piece; kept a plain, testable fn).
# --------------------------------------------------------------------------------------
_UPSERT_PREDICTIONS = text(
    """
    INSERT INTO predictions
        (ticker, as_of_date, horizon_days, predicted_vol,
         model_name, model_version, model_stage, resolves_on)
    VALUES
        (:ticker, :as_of, :horizon, :vol, :name, :version, :stage, :resolves)
    ON CONFLICT (ticker, as_of_date, model_version) DO UPDATE SET
        predicted_vol = EXCLUDED.predicted_vol,
        model_stage   = EXCLUDED.model_stage,
        resolves_on   = EXCLUDED.resolves_on,
        horizon_days  = EXCLUDED.horizon_days,
        created_at    = now()
    """
)


def write_predictions(
    engine: Engine,
    model: Any,
    info: dict[str, Any],
    as_of_date: date | None = None,
    horizon_days: int | None = None,
) -> int:
    """Forecast each ticker's latest feature row and upsert the predictions.

    Strict validation: refuses to write without a model name+version stamp, rejects a
    forecast that is non-finite or non-positive (vols must be > 0), requires every
    maturity date to be in the future, and forbids duplicate keys in the batch.

    Args:
        engine: Application-database engine.
        model: A loaded model with ``predict`` returning annualized vol *levels*.
        info: Registry metadata; must contain non-empty ``name`` and ``version``.
        as_of_date: Upper bound on the feature date used per ticker; latest if omitted.
        horizon_days: Forecast horizon in business days; from config if omitted.

    Returns:
        The number of prediction rows upserted.

    Raises:
        ValueError: On a missing stamp, a corrupt forecast, or a malformed batch.
    """
    name, version, stage = info.get("name"), info.get("version"), info.get("stage")
    if not name or not version:
        raise ValueError("Refusing to write predictions without a model name/version stamp.")
    horizon = (
        horizon_days if horizon_days is not None else load_flows_config().prediction_horizon_days
    )

    feature_cols = load_feature_contract().feature_names()
    select_cols = ", ".join(["ticker", "trade_date", *feature_cols])
    clause = "WHERE trade_date <= :as_of" if as_of_date is not None else ""
    params = {"as_of": as_of_date} if as_of_date is not None else {}
    query = text(
        f"SELECT DISTINCT ON (ticker) {select_cols} FROM features {clause} "
        f"ORDER BY ticker, trade_date DESC"
    )
    df = pd.read_sql(query, engine, params=params or None, parse_dates=["trade_date"])
    if df.empty:
        return 0
    df["trade_date"] = df["trade_date"].dt.date

    preds = np.asarray(model.predict(df[feature_cols]), dtype=float)
    # --- strict validation: a corrupt forecast must never reach the table ---
    if preds.shape[0] != len(df):
        raise ValueError(f"forecast count {preds.shape[0]} != {len(df)} feature rows")
    good = np.isfinite(preds) & (preds > 0.0)
    if not good.all():
        bad = df.loc[~good, "ticker"].tolist()
        raise ValueError(f"model produced non-finite or non-positive volatility for: {bad}")

    df = df.assign(
        predicted_vol=preds,
        resolves_on=[_busday_offset(d, horizon) for d in df["trade_date"]],
    )
    if df.duplicated(subset=["ticker", "trade_date"]).any():
        raise ValueError("duplicate (ticker, as_of_date) in the prediction batch")
    if not (df["resolves_on"] > df["trade_date"]).all():
        raise ValueError("resolves_on must be strictly after the prediction date")

    records = [
        {
            "ticker": r["ticker"],
            "as_of": r["trade_date"],
            "horizon": horizon,
            "vol": float(r["predicted_vol"]),
            "name": name,
            "version": str(version),
            "stage": stage,
            "resolves": r["resolves_on"],
        }
        for r in df[["ticker", "trade_date", "predicted_vol", "resolves_on"]].to_dict("records")
    ]
    with engine.begin() as conn:
        conn.execute(_UPSERT_PREDICTIONS, records)
    return len(records)


# --------------------------------------------------------------------------------------
# Prefect tasks (thin wrappers; build their own engine so inputs stay serializable).
# --------------------------------------------------------------------------------------
@task(retries=3, retry_delay_seconds=[5, 15, 45])
def ingest_prices(start: date, end: date) -> int:
    """Seed the universe and ingest OHLCV over the overlap window (retried)."""
    engine = get_engine()
    universe = load_universe()
    seed_dim_ticker(engine, universe)
    return ingest_ohlcv(engine, YFinanceProvider(), universe.all_tickers(), start, end).rows_written


@task(retries=3, retry_delay_seconds=[5, 15, 45])
def ingest_macro_series(start: date, end: date) -> int:
    """Ingest the configured FRED macro series over the overlap window (retried)."""
    engine = get_engine()
    return ingest_macro(engine, FredCsvProvider(), load_macro_series(), start, end).rows_written


@task
def quality_gate(start: date, end: date, run_date: date) -> None:
    """Run the data quality gate; raises QualityGateError on error-severity failure."""
    run_quality_checks(get_engine(), start=start, end=end, run_date=run_date, raise_on_error=True)


@task
def rebuild_features() -> int:
    """Rebuild the features table from raw data (idempotent)."""
    return build_features(get_engine())


@task
def write_predictions_step(as_of_date: date | None) -> int:
    """Load the Production model by stage and write its forecasts."""
    info = get_production_info()
    if not info:
        raise RuntimeError("No Production model in the registry; cannot write predictions.")
    model = load_production_model()
    return write_predictions(get_engine(), model, info, as_of_date)


@flow(name="daily-inference")
def daily_inference(as_of_date: date | None = None, lookback_days: int | None = None) -> int:
    """Run the daily inference path end to end; returns the number of predictions written."""
    config = load_flows_config()
    lookback = lookback_days if lookback_days is not None else config.ingest_lookback_days
    end = as_of_date or date.today()
    start = _busday_offset(end, -lookback)

    ingest_prices(start, end)
    ingest_macro_series(start, end)
    quality_gate(start, end, end)  # raises -> halts before features/predictions
    rebuild_features()
    return write_predictions_step(as_of_date)


def serve() -> None:  # pragma: no cover - needs a running Prefect worker (Day 26)
    """Register the flow on its cron schedule (requires a Prefect worker process)."""
    daily_inference.serve(name="daily-inference", cron=load_flows_config().schedule_cron)


def main(argv: list[str] | None = None) -> int:
    """Run the daily flow once for the given (or current) date."""
    parser = argparse.ArgumentParser(description="Run the daily inference flow once.")
    parser.add_argument(
        "--date", type=date.fromisoformat, help="as-of date YYYY-MM-DD; default today"
    )
    args = parser.parse_args(argv)
    n = daily_inference(as_of_date=args.date)
    print(f"daily-inference wrote {n} predictions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
