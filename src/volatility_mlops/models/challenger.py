"""Challenger training, time-aware search, and the final comparison (Day 14).

Runs a small hyperparameter search for the HistGradientBoosting challenger using
walk-forward folds (never random CV), selects the configuration with the best mean
QLIKE, evaluates it alongside every baseline and GARCH, writes an honest comparison
write-up, and logs a predicted-vs-actual diagnostic to MLflow.

Run as a module::

    python -m volatility_mlops.models.challenger
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed for artifact plots
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from volatility_mlops.db.engine import get_engine  # noqa: E402
from volatility_mlops.features.contract import load_feature_contract  # noqa: E402
from volatility_mlops.metrics.metrics import compute_metrics  # noqa: E402
from volatility_mlops.models.baselines import BASELINE_MODELS, TARGET  # noqa: E402
from volatility_mlops.models.evaluate import (  # noqa: E402
    load_labeled_panel,
    run_evaluation,
)
from volatility_mlops.models.garch import GARCH_MODELS  # noqa: E402
from volatility_mlops.models.pipeline import make_challenger  # noqa: E402
from volatility_mlops.models.tracking import configure_mlflow, git_sha  # noqa: E402
from volatility_mlops.validation.splitter import (  # noqa: E402
    load_validation_config,
    walk_forward_folds,
)

OUTPUT_PATH = Path(__file__).resolve().parents[3] / "assets" / "model_comparison.md"

# Small, deliberately coarse grid searched over walk-forward folds by mean QLIKE.
PARAM_GRID: list[dict[str, object]] = [
    {
        "learning_rate": 0.05,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 50,
        "l2_regularization": 0.0,
    },
    {
        "learning_rate": 0.05,
        "max_iter": 500,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 100,
        "l2_regularization": 1.0,
    },
    {
        "learning_rate": 0.10,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 100,
        "l2_regularization": 1.0,
    },
    {
        "learning_rate": 0.02,
        "max_iter": 800,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 200,
        "l2_regularization": 1.0,
    },
]


def search_challenger() -> tuple[dict[str, object], float]:
    """Walk-forward search over PARAM_GRID; return the best params and its mean QLIKE."""
    configure_mlflow()
    engine = get_engine()
    contract = load_feature_contract()
    config = load_validation_config()
    feature_cols = contract.feature_names()

    panel = load_labeled_panel(engine, feature_cols)
    folds = walk_forward_folds(panel["trade_date"].to_numpy(), config)

    best_params: dict[str, object] = {}
    best_qlike = float("inf")
    for i, params in enumerate(PARAM_GRID):
        fn = make_challenger(feature_cols, params)
        qlikes = [
            compute_metrics(
                panel.iloc[f.test_idx][TARGET].to_numpy(),
                fn(panel.iloc[f.train_idx], panel.iloc[f.test_idx]),
            ).qlike
            for f in folds
        ]
        mean_qlike = float(np.mean(qlikes))
        mlflow.set_experiment("volatility-challenger-search")
        with mlflow.start_run(run_name=f"hgb_config_{i}"):
            mlflow.set_tag("git_sha", git_sha())
            mlflow.log_params(params)
            mlflow.log_metric("mean_qlike", mean_qlike)
        if mean_qlike < best_qlike:
            best_qlike, best_params = mean_qlike, params
    return best_params, best_qlike


def _log_prediction_scatter(best_params: dict[str, object]) -> None:
    """Log a predicted-vs-actual scatter for the challenger on the last fold."""
    engine = get_engine()
    contract = load_feature_contract()
    config = load_validation_config()
    feature_cols = contract.feature_names()
    panel = load_labeled_panel(engine, feature_cols)
    folds = walk_forward_folds(panel["trade_date"].to_numpy(), config)
    last = folds[-1]
    fn = make_challenger(feature_cols, best_params)
    y_true = panel.iloc[last.test_idx][TARGET].to_numpy()
    y_pred = fn(panel.iloc[last.train_idx], panel.iloc[last.test_idx])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=5, alpha=0.3)
    lim = [0, float(max(y_true.max(), y_pred.max()))]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set(xlabel="realized vol", ylabel="predicted vol", title="Challenger: predicted vs actual")

    mlflow.set_experiment("volatility-challenger")
    with mlflow.start_run(run_name="challenger_diagnostics"):
        mlflow.set_tag("git_sha", git_sha())
        mlflow.log_params(best_params)
        mlflow.log_figure(fig, "predicted_vs_actual.png")
    plt.close(fig)


def _write_comparison(table: pd.DataFrame, best_params: dict[str, object]) -> None:
    """Write the honest comparison write-up to assets/model_comparison.md."""
    row = {r["model"]: r for _, r in table.iterrows()}
    challenger_q = row["challenger"]["qlike"]

    def improvement(name: str) -> str:
        base = row[name]["qlike"]
        return f"{(base - challenger_q) / base * 100:+.1f}%"

    lines = [
        "# Model comparison",
        "",
        f"Walk-forward evaluation (git `{git_sha()}`), mean across folds, sorted by QLIKE.",
        "",
        "| model | MAE | RMSE | QLIKE | MZ slope | MZ R2 |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"| {r['model']} | {r['mae']:.4f} | {r['rmse']:.4f} | {r['qlike']:.4f} | "
            f"{r['mz_slope']:.3f} | {r['mz_r2']:.3f} |"
        )
    lines += [
        "",
        f"**Best challenger hyperparameters:** `{best_params}`",
        "",
        "## Honest read",
        "",
        f"- Challenger QLIKE vs HAR-RV: **{improvement('har_rv')}** "
        f"(negative means worse than the baseline).",
        f"- Challenger QLIKE vs GARCH: **{improvement('garch')}**.",
        "- QLIKE is the primary metric; MZ slope near 1 indicates good calibration.",
        "- Baselines are strong by design; the operational loop (weekly retrain, "
        "promotion gate, monitoring) is the project's real contribution.",
        "",
    ]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines))


def main() -> int:
    """Search, evaluate against all baselines, write the comparison, log diagnostics."""
    best_params, best_qlike = search_challenger()
    print(f"Best challenger config (mean QLIKE {best_qlike:.4f}): {best_params}")

    engine = get_engine()
    contract = load_feature_contract()
    feature_cols = contract.feature_names()
    models = {
        **BASELINE_MODELS,
        **GARCH_MODELS,
        "challenger": make_challenger(feature_cols, best_params),
    }
    table = run_evaluation(engine, models, experiment="volatility-final")
    _write_comparison(table, best_params)
    _log_prediction_scatter(best_params)

    print(table.to_string(index=False))
    print(f"\nWrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
