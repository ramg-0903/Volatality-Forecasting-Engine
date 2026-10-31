"""The promotion gate (build-plan section 7.5 / Day 16).

An explicit, logged, reversible rule that decides whether a freshly trained
challenger may replace the live champion. ALL seven criteria must pass to promote;
any failure leaves the champion in place and records the reason in ``promotion_log``.

``evaluate_gate`` is a pure decision function over supplied per-fold metrics and a
few booleans, so every branch is unit-testable. Computing the champion/challenger
metrics and running the gate on a schedule is the Day-20 retrain flow's job; this
module owns the decision, the logging, and the registry transition.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import text
from sqlalchemy.engine import Engine

from volatility_mlops.metrics.metrics import Metrics
from volatility_mlops.models.tracking import configure_mlflow

# repo_root/src/volatility_mlops/models/promotion.py -> parents[3] == repo root
PROMOTION_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "promotion.yml"


@dataclass(frozen=True)
class GateConfig:
    """Promotion gate thresholds."""

    primary_margin: float
    secondary_margin: float
    stability_window: int
    max_losses: int
    calibration_low: float
    calibration_high: float


@dataclass(frozen=True)
class PromotionDecision:
    """Outcome of the gate: whether to promote, why, and the supporting numbers."""

    promote: bool
    reason: str
    criteria: dict[str, bool]
    comparison: dict[str, float]


def load_promotion_config(path: Path = PROMOTION_CONFIG_PATH) -> GateConfig:
    """Load gate thresholds from ``promotion.yml``."""
    import yaml

    data = yaml.safe_load(path.read_text())
    return GateConfig(
        primary_margin=data["primary_margin"],
        secondary_margin=data["secondary_margin"],
        stability_window=data["stability_window"],
        max_losses=data["max_losses"],
        calibration_low=data["calibration_low"],
        calibration_high=data["calibration_high"],
    )


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def evaluate_gate(
    champion_metrics: Sequence[Metrics],
    challenger_metrics: Sequence[Metrics],
    challenger_columns: Sequence[str],
    contract_columns: Sequence[str],
    smoke_ok: bool,
    data_health_ok: bool,
    config: GateConfig,
) -> PromotionDecision:
    """Evaluate the seven promotion criteria (pure, no I/O).

    Args:
        champion_metrics: Per-fold metrics for the current Production model.
        challenger_metrics: Per-fold metrics for the candidate, same fold order.
        challenger_columns: Feature columns the challenger expects.
        contract_columns: The current feature contract's columns.
        smoke_ok: Whether the challenger passed the golden-batch smoke test.
        data_health_ok: Whether the training window had no unresolved DQ failures.
        config: Gate thresholds.

    Returns:
        A :class:`PromotionDecision` (promote flag, reason, per-criterion results,
        and the comparison numbers).
    """
    champ_q = [m.qlike for m in champion_metrics]
    chall_q = [m.qlike for m in challenger_metrics]

    # 1. Primary: challenger QLIKE beats champion by >= margin on the last 2 folds.
    champ_last2 = _mean(champ_q[-2:])
    chall_last2 = _mean(chall_q[-2:])
    primary_improvement = (champ_last2 - chall_last2) / champ_last2
    primary = primary_improvement >= config.primary_margin

    # 2. No secondary regression: MAE worsens by at most `secondary_margin`.
    champ_mae = _mean([m.mae for m in champion_metrics])
    chall_mae = _mean([m.mae for m in challenger_metrics])
    secondary_worsening = (chall_mae - champ_mae) / champ_mae
    secondary = secondary_worsening <= config.secondary_margin

    # 3. Stability: challenger loses on at most `max_losses` of the last N folds.
    window = config.stability_window
    losses = sum(c > p for c, p in zip(chall_q[-window:], champ_q[-window:], strict=False))
    stability = losses <= config.max_losses

    # 4. Calibration: challenger mean MZ slope within the band around 1.
    chall_slope = _mean([m.mz_slope for m in challenger_metrics])
    calibration = config.calibration_low <= chall_slope <= config.calibration_high

    # 5. Schema conformance: exact column match with the feature contract.
    schema = list(challenger_columns) == list(contract_columns)

    criteria = {
        "primary": bool(primary),
        "secondary": bool(secondary),
        "stability": bool(stability),
        "calibration": bool(calibration),
        "schema": bool(schema),
        "smoke": bool(smoke_ok),
        "data_health": bool(data_health_ok),
    }
    promote = all(criteria.values())
    failed = [name for name, ok in criteria.items() if not ok]
    reason = "all 7 criteria passed" if promote else f"rejected: {', '.join(failed)} failed"

    comparison = {
        "champion_qlike_last2": champ_last2,
        "challenger_qlike_last2": chall_last2,
        "primary_improvement": primary_improvement,
        "champion_mae": champ_mae,
        "challenger_mae": chall_mae,
        "secondary_worsening": secondary_worsening,
        "stability_losses": float(losses),
        "challenger_slope": chall_slope,
    }
    return PromotionDecision(
        promote=promote, reason=reason, criteria=criteria, comparison=comparison
    )


def run_smoke_test(model: Any, golden_batch: Any) -> bool:  # noqa: ANN401
    """Return True if the model yields finite, strictly-positive predictions."""
    try:
        preds = np.asarray(model.predict(golden_batch), dtype=float)
    except Exception:  # noqa: BLE001 -- any failure is a failed smoke test
        return False
    return bool(preds.size > 0 and np.all(np.isfinite(preds)) and np.all(preds > 0))


_INSERT_PROMOTION = text(
    """
    INSERT INTO promotion_log
        (challenger_version, champion_version, metric_comparison, decision, reason)
    VALUES (:challenger, :champion, CAST(:comparison AS jsonb), :decision, :reason)
    """
)


def log_promotion(
    engine: Engine,
    decision: PromotionDecision,
    challenger_version: str,
    champion_version: str | None,
) -> None:
    """Append the gate's decision and its reason to ``promotion_log``."""
    payload = {**decision.comparison, "criteria": decision.criteria}
    with engine.begin() as conn:
        conn.execute(
            _INSERT_PROMOTION,
            {
                "challenger": challenger_version,
                "champion": champion_version,
                "comparison": json.dumps(payload),
                "decision": "promoted" if decision.promote else "rejected",
                "reason": decision.reason,
            },
        )


def apply_decision(
    decision: PromotionDecision, model_name: str, challenger_version: str
) -> int | None:
    """Promote the challenger to Production on a positive decision; else do nothing.

    Returns the promoted version number, or ``None`` if the challenger was rejected.
    """
    if not decision.promote:
        return None
    from mlflow.tracking import MlflowClient

    configure_mlflow()
    client = MlflowClient()
    client.transition_model_version_stage(
        model_name, challenger_version, "Production", archive_existing_versions=True
    )
    return int(challenger_version)
