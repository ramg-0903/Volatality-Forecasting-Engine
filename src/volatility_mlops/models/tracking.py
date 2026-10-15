"""MLflow experiment-tracking helpers (build-plan section 7.3).

Configures the tracking URI from the environment and logs a walk-forward run as a
parent run (aggregated mean/std metrics) with one nested child run per fold. Every
run is tagged with the git commit SHA so results tie back to code.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence

import mlflow
from dotenv import load_dotenv

from volatility_mlops.metrics.metrics import Metrics

_METRIC_FIELDS = ("mae", "rmse", "qlike", "mz_slope", "mz_r2")


def configure_mlflow() -> None:
    """Point MLflow at the tracking server from ``MLFLOW_TRACKING_URI``."""
    load_dotenv()
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)


def git_sha() -> str:
    """Return the short git commit SHA, or ``"unknown"`` if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001 -- provenance tag is best-effort
        return "unknown"


def mean_metrics(fold_metrics: Sequence[Metrics]) -> dict[str, float]:
    """Mean of each metric across folds."""
    return {
        f: float(sum(getattr(m, f) for m in fold_metrics) / len(fold_metrics))
        for f in _METRIC_FIELDS
    }


def _std(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return float((sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5)


def log_walk_forward(
    model_name: str,
    fold_metrics: Sequence[Metrics],
    params: dict[str, object],
    experiment: str,
) -> None:
    """Log a model's walk-forward evaluation to MLflow.

    Args:
        model_name: Name of the model (used as the parent run name).
        fold_metrics: Per-fold metrics.
        params: Parameters to log (splitter config, feature version, ...).
        experiment: MLflow experiment name.
    """
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=model_name):
        mlflow.set_tag("git_sha", git_sha())
        mlflow.log_params(params)
        for i, m in enumerate(fold_metrics):
            with mlflow.start_run(run_name=f"{model_name}-fold{i}", nested=True):
                mlflow.log_metrics({f: getattr(m, f) for f in _METRIC_FIELDS})
                mlflow.log_metric("n_test", m.n)
        means = mean_metrics(fold_metrics)
        mlflow.log_metrics({f"mean_{k}": v for k, v in means.items()})
        mlflow.log_metric("std_qlike", _std([m.qlike for m in fold_metrics]))
