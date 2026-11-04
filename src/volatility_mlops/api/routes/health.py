"""Liveness and model-refresh routes (build-plan section 8.1).

``/health`` reports the live model resolved by stage — the endpoint that proves the
promotion gate is wired end to end. ``/admin/refresh`` reloads the Production model
so a promotion takes effect without restarting the container (section 8.2).
"""

from __future__ import annotations

from flask import Blueprint, Response, current_app, jsonify

from volatility_mlops.api.model_holder import ModelHolder

bp = Blueprint("ops", __name__)


def _holder() -> ModelHolder:
    holder: ModelHolder = current_app.config["MODEL_HOLDER"]
    return holder


@bp.get("/health")
def health() -> tuple[Response, int]:
    """Report service liveness and the loaded model's name/version/stage/timestamp."""
    status = _holder().status()
    code = 200 if status["loaded"] else 503
    return jsonify({"status": "ok" if status["loaded"] else "degraded", **status}), code


@bp.post("/admin/refresh")
def refresh() -> tuple[Response, int]:
    """Reload the Production model from the registry; return the new status."""
    status = _holder().refresh()
    code = 200 if status["loaded"] else 503
    return jsonify(status), code
