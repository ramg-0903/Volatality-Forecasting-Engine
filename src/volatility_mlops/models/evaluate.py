"""Walk-forward evaluation loop shared by baselines, GARCH, and the challenger.

Loads the labelled feature panel, builds walk-forward folds, evaluates each model
across every fold, logs to MLflow, and returns a comparison table. Models are
plain ``(train, test) -> np.ndarray`` callables (see ``baselines.py``).

Run as a module to evaluate the baselines::

    python -m volatility_mlops.models.evaluate
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from volatility_mlops.db.engine import get_engine
from volatility_mlops.features.contract import load_feature_contract
from volatility_mlops.metrics.metrics import Metrics, compute_metrics
from volatility_mlops.models.baselines import BASELINE_MODELS, TARGET
from volatility_mlops.models.tracking import configure_mlflow, log_walk_forward, mean_metrics
from volatility_mlops.validation.splitter import (
    Fold,
    WalkForwardConfig,
    load_validation_config,
    walk_forward_folds,
)

ModelFn = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


def load_labeled_panel(engine: Engine, feature_cols: list[str]) -> pd.DataFrame:
    """Load labelled feature rows (target not null), sorted by trade date then ticker."""
    cols = ["ticker", "trade_date", *feature_cols, TARGET]
    query = text(
        f"SELECT {', '.join(cols)} FROM features "
        f"WHERE {TARGET} IS NOT NULL ORDER BY trade_date, ticker"
    )
    df = pd.read_sql(query, engine, parse_dates=["trade_date"])
    df["trade_date"] = df["trade_date"].dt.date
    return df


def evaluate_model(panel: pd.DataFrame, folds: list[Fold], model_fn: ModelFn) -> list[Metrics]:
    """Evaluate one model across all folds; returns per-fold metrics."""
    results: list[Metrics] = []
    for fold in folds:
        train = panel.iloc[fold.train_idx]
        test = panel.iloc[fold.test_idx]
        preds = model_fn(train, test)
        results.append(compute_metrics(test[TARGET].to_numpy(), preds))
    return results


def run_evaluation(
    engine: Engine,
    models: Mapping[str, ModelFn],
    experiment: str,
    config: WalkForwardConfig | None = None,
) -> pd.DataFrame:
    """Evaluate several models walk-forward, log to MLflow, return a comparison table.

    Args:
        engine: Application-database engine.
        models: Named ``(train, test) -> preds`` callables.
        experiment: MLflow experiment name.
        config: Walk-forward config; loaded from disk when omitted.

    Returns:
        A frame with one row per model of mean-across-fold metrics.
    """
    configure_mlflow()
    contract = load_feature_contract()
    config = config or load_validation_config()

    panel = load_labeled_panel(engine, contract.feature_names())
    folds = walk_forward_folds(panel["trade_date"].to_numpy(), config)
    params: dict[str, object] = {
        "n_splits": config.n_splits,
        "test_size": config.test_size,
        "purge": config.purge,
        "embargo": config.embargo,
        "feature_version": contract.version,
        "n_folds": len(folds),
        "n_rows": len(panel),
    }

    rows: list[dict[str, object]] = []
    for name, fn in models.items():
        fold_metrics = evaluate_model(panel, folds, fn)
        log_walk_forward(name, fold_metrics, params, experiment)
        rows.append({"model": name, **mean_metrics(fold_metrics)})
    return pd.DataFrame(rows).sort_values("qlike").reset_index(drop=True)


def main() -> int:
    """Evaluate the baseline ladder and print the comparison table."""
    engine = get_engine()
    table = run_evaluation(engine, BASELINE_MODELS, experiment="volatility-baselines")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
