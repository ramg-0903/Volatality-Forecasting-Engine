"""Prefect weekly retrain flow (build-plan section 9.2 / Day 20).

Trains a fresh challenger, scores it against the reigning champion on identical
walk-forward folds, runs it through the Day-16 promotion gate, and promotes it to
``Production`` **only if every criterion passes**. Otherwise the incumbent stays and
the rejection is written to ``promotion_log`` with its reason. Every run -- accept or
reject -- leaves an auditable row.

Pipeline (section 9.2)::

    rebuild features -> leakage assertions -> row-count guardrail
        -> evaluate champion vs challenger (same folds)
        -> register challenger to Staging -> promotion gate (+ smoke test)
        -> promote or reject -> write promotion_log -> notify

Policies:

* **Guardrail first.** If the labelled panel shrinks by more than the configured
  tolerance versus the live model's training size, abort before touching the registry
  -- silent data loss is the most common cause of a bad automated retrain.
* **Conditional promotion.** The challenger is registered to ``Staging`` and only
  transitioned to ``Production`` when ``evaluate_gate`` returns ``promote``. A rejected
  challenger remains in ``Staging`` as an inspectable artifact.
* **Champion = configuration, re-fit per fold.** The gate compares the incumbent's
  *configuration* (its logged hyperparameters) against a freshly searched challenger,
  both re-fit walk-forward on the current panel -- never the already-fitted Production
  artifact, which cannot be scored walk-forward without leakage.

The monitoring steps (resolve matured labels, rolling metrics, drift, alerting) are
Days 22-24 and are intentionally not here yet; ``_notify`` is a seam for Day 24.

Run once::

    python -m volatility_mlops.flows.weekly            # as of today
    python -m volatility_mlops.flows.weekly --date 2026-09-09
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from prefect import flow, task
from sqlalchemy import text
from sqlalchemy.engine import Engine

from volatility_mlops.db.engine import get_engine
from volatility_mlops.features.build import build_features
from volatility_mlops.features.contract import load_feature_contract
from volatility_mlops.flows.daily import load_flows_config
from volatility_mlops.metrics.metrics import Metrics
from volatility_mlops.models.baselines import TARGET
from volatility_mlops.models.challenger import search_challenger
from volatility_mlops.models.evaluate import evaluate_model, load_labeled_panel
from volatility_mlops.models.pipeline import build_pipeline, make_challenger
from volatility_mlops.models.promotion import (
    PromotionDecision,
    apply_decision,
    evaluate_gate,
    load_promotion_config,
    log_promotion,
    run_smoke_test,
)
from volatility_mlops.models.registry import load_model_config, register_pipeline
from volatility_mlops.models.tracking import configure_mlflow
from volatility_mlops.validation.splitter import load_validation_config, walk_forward_folds

# repo_root/src/volatility_mlops/flows/weekly.py -> parents[3] == repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_BATCH_PATH = _REPO_ROOT / "tests" / "fixtures" / "golden_batch.csv"

# HistGradientBoosting hyperparameters and their types, for coercing MLflow run
# params (which come back as strings) into the champion configuration.
_HGB_PARAM_TYPES: dict[str, type] = {
    "learning_rate": float,
    "max_iter": int,
    "max_leaf_nodes": int,
    "min_samples_leaf": int,
    "l2_regularization": float,
}


class RetrainAborted(RuntimeError):
    """Raised when a guardrail refuses to let the retrain proceed."""


# --------------------------------------------------------------------------------------
# Plain, testable helpers (no Prefect) -- the substantial new logic for Day 20.
# --------------------------------------------------------------------------------------
def _production_run_params(model_name: str) -> dict[str, str]:
    """Return the MLflow run params of the current Production version, or ``{}``.

    Reads the params logged when the live model was registered (its hyperparameters
    and ``n_train_rows``). Returns an empty dict when there is no Production model or
    its run cannot be read.
    """
    configure_mlflow()
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    versions = client.get_latest_versions(model_name, stages=["Production"])
    if not versions or not versions[0].run_id:
        return {}
    try:
        return dict(client.get_run(versions[0].run_id).data.params)
    except Exception:  # noqa: BLE001 -- a missing/unreadable run means "no baseline"
        return {}


def champion_params(run_params: dict[str, str]) -> dict[str, object]:
    """Extract the champion's HGB hyperparameters from its logged run params.

    Falls back to ``config/model.yml`` for any key the run did not record, so the
    champion configuration is always complete and type-correct.
    """
    defaults = load_model_config().best_params
    params: dict[str, object] = {}
    for key, caster in _HGB_PARAM_TYPES.items():
        if key in run_params:
            # MLflow stores params as strings (an int may arrive as "31" or "31.0").
            params[key] = int(float(run_params[key])) if caster is int else float(run_params[key])
        elif key in defaults:
            params[key] = defaults[key]
    return params


def baseline_row_count(run_params: dict[str, str]) -> int | None:
    """Return the ``n_train_rows`` the live model was trained on, or ``None``."""
    raw = run_params.get("n_train_rows")
    return int(float(raw)) if raw is not None else None


def check_row_count_guardrail(
    current_rows: int, baseline_rows: int | None, tolerance: float
) -> None:
    """Abort when the labelled panel shrank past tolerance versus the live model.

    A missing baseline (first-ever retrain, or no Production model) passes: there is
    nothing to regress against yet.

    Raises:
        RetrainAborted: If ``current_rows < baseline_rows * (1 - tolerance)``.
    """
    if baseline_rows is None:
        return
    floor = baseline_rows * (1.0 - tolerance)
    if current_rows < floor:
        raise RetrainAborted(
            f"labelled panel shrank to {current_rows} rows, below the "
            f"{floor:.0f}-row floor ({baseline_rows} baseline, {tolerance:.0%} tolerance); "
            f"aborting retrain to avoid training on silently lost data."
        )


def assert_no_leakage(panel: pd.DataFrame) -> None:
    """Re-assert the walk-forward guarantee on the current panel's folds.

    For every fold, the maximum training row index must sit strictly below the
    minimum test row index after purge and embargo. The splitter guarantees this and
    is unit-tested; this is a runtime safety net on live data.

    Raises:
        RetrainAborted: If any fold violates the ordering.
    """
    folds = walk_forward_folds(panel["trade_date"].to_numpy(), load_validation_config())
    for fold in folds:
        if fold.train_idx.max() >= fold.test_idx.min():
            raise RetrainAborted(
                f"leakage in fold {fold.index}: max train index "
                f"{fold.train_idx.max()} >= min test index {fold.test_idx.min()}"
            )


def evaluate_champion_challenger(
    panel: pd.DataFrame,
    feature_cols: list[str],
    champion_cfg: dict[str, object],
    challenger_cfg: dict[str, object],
) -> tuple[list[Metrics], list[Metrics]]:
    """Score both configurations walk-forward on the same folds; return per-fold metrics."""
    folds = walk_forward_folds(panel["trade_date"].to_numpy(), load_validation_config())
    champion_metrics = evaluate_model(panel, folds, make_challenger(feature_cols, champion_cfg))
    challenger_metrics = evaluate_model(panel, folds, make_challenger(feature_cols, challenger_cfg))
    return champion_metrics, challenger_metrics


def load_golden_batch() -> pd.DataFrame:
    """Load the committed golden feature batch for the smoke test (deterministic)."""
    return pd.read_csv(GOLDEN_BATCH_PATH)


_DQ_FAILURES_IN_WINDOW = text(
    """
    SELECT count(*) FROM data_quality_results
    WHERE passed = false AND run_date BETWEEN :start AND :end
    """
)


def data_health_ok(engine: Engine, start: date, end: date) -> bool:
    """True when no data-quality check failed within the training window."""
    with engine.connect() as conn:
        failures = conn.execute(_DQ_FAILURES_IN_WINDOW, {"start": start, "end": end}).scalar_one()
    return int(failures) == 0


def fit_challenger_pipeline(
    panel: pd.DataFrame, feature_cols: list[str], params: dict[str, object]
) -> Any:
    """Fit the challenger pipeline on the whole labelled panel (log target)."""
    pipeline = build_pipeline(params)
    pipeline.fit(panel[feature_cols], np.log(panel[TARGET].to_numpy()))
    return pipeline


# --------------------------------------------------------------------------------------
# Prefect tasks (thin wrappers; build their own engine so inputs stay serializable).
# --------------------------------------------------------------------------------------
@task
def rebuild_features_step() -> int:
    """Full feature rebuild from raw data (idempotent); returns rows upserted."""
    return build_features(get_engine())


@task
def guardrails_step() -> dict[str, Any]:
    """Leakage assertions + row-count guardrail. Raises to halt before any training."""
    engine = get_engine()
    config = load_flows_config()
    feature_cols = load_feature_contract().feature_names()
    panel = load_labeled_panel(engine, feature_cols)

    assert_no_leakage(panel)

    model_name = load_model_config().registered_model_name
    run_params = _production_run_params(model_name)
    check_row_count_guardrail(
        len(panel), baseline_row_count(run_params), config.row_count_drop_tolerance
    )
    return {"n_rows": len(panel), "champion_run_params": run_params}


@task
def retrain_and_gate_step(guardrail_info: dict[str, Any]) -> dict[str, Any]:
    """Search a challenger, register it to Staging, run the gate, promote or reject."""
    engine = get_engine()
    contract = load_feature_contract()
    feature_cols = contract.feature_names()
    gate_cfg = load_promotion_config()
    model_name = load_model_config().registered_model_name

    panel = load_labeled_panel(engine, feature_cols)
    run_params = guardrail_info.get("champion_run_params", {})
    champ_cfg = champion_params(run_params)

    # Freshly search the challenger on current data; evaluate both on the same folds.
    challenger_cfg, _ = search_challenger()
    champ_metrics, chall_metrics = evaluate_champion_challenger(
        panel, feature_cols, champ_cfg, challenger_cfg
    )

    # Register the candidate to Staging (does not disturb the live Production model).
    params = {
        **challenger_cfg,
        "feature_version": contract.version,
        "n_train_rows": len(panel),
    }
    pipeline = fit_challenger_pipeline(panel, feature_cols, challenger_cfg)
    challenger_version = register_pipeline(
        pipeline, feature_cols, panel.head(5), model_name, params, stage="Staging"
    )

    # Smoke test the *registered Staging artifact* on the golden batch (round-trips too).
    smoke_ok = _smoke_test_staging(model_name)

    span_start = panel["trade_date"].min()
    span_end = panel["trade_date"].max()
    decision = evaluate_gate(
        champion_metrics=champ_metrics,
        challenger_metrics=chall_metrics,
        challenger_columns=feature_cols,
        contract_columns=contract.feature_names(),
        smoke_ok=smoke_ok,
        data_health_ok=data_health_ok(engine, span_start, span_end),
        config=gate_cfg,
    )

    champion_version = _production_version(model_name)
    promoted = apply_decision(decision, model_name, str(challenger_version))
    log_promotion(engine, decision, str(challenger_version), champion_version)
    _notify(decision, challenger_version)

    return {
        "decision": "promoted" if decision.promote else "rejected",
        "reason": decision.reason,
        "challenger_version": challenger_version,
        "champion_version": champion_version,
        "promoted_version": promoted,
        "criteria": decision.criteria,
    }


def _smoke_test_staging(model_name: str) -> bool:
    """Load the just-registered Staging model by stage and smoke-test it."""
    configure_mlflow()
    import mlflow.pyfunc

    try:
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/Staging")
    except Exception:  # noqa: BLE001 -- a model that will not load fails the smoke test
        return False
    return run_smoke_test(model, load_golden_batch())


def _production_version(model_name: str) -> str | None:
    """Return the current Production version string, or ``None``."""
    configure_mlflow()
    from mlflow.tracking import MlflowClient

    versions = MlflowClient().get_latest_versions(model_name, stages=["Production"])
    return versions[0].version if versions else None


def _notify(decision: PromotionDecision, challenger_version: int) -> None:  # pragma: no cover
    """Seam for Day-24 alerting; for now, a single structured log line."""
    verb = "promoted" if decision.promote else "rejected"
    print(f"weekly-retrain: challenger v{challenger_version} {verb} -- {decision.reason}")


@flow(name="weekly-retrain")
def weekly_retrain(as_of_date: date | None = None) -> dict[str, Any]:  # noqa: ARG001
    """Run the weekly retrain + promotion gate end to end; returns the decision summary."""
    rebuild_features_step()
    guardrail_info = guardrails_step()
    return retrain_and_gate_step(guardrail_info)


def serve() -> None:  # pragma: no cover - needs a running Prefect worker (Day 26)
    """Register the flow on its cron schedule (requires a Prefect worker process)."""
    weekly_retrain.serve(name="weekly-retrain", cron=load_flows_config().retrain_cron)


def main(argv: list[str] | None = None) -> int:
    """Run the weekly retrain flow once for the given (or current) date."""
    parser = argparse.ArgumentParser(description="Run the weekly retrain flow once.")
    parser.add_argument(
        "--date", type=date.fromisoformat, help="as-of date YYYY-MM-DD; default today"
    )
    args = parser.parse_args(argv)
    summary = weekly_retrain(as_of_date=args.date)
    print(
        f"weekly-retrain: {summary['decision']} "
        f"(challenger v{summary['challenger_version']}, champion v{summary['champion_version']})\n"
        f"  reason: {summary['reason']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
