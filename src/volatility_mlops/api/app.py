"""Flask application factory for the serving layer (build-plan section 8 / Day 17).

Wires the engine and the (by-stage) model holder onto the app, installs structured
JSON error handling and per-request logging with a request id, and registers the
``/health``, ``/predict``, ``/history`` (and ``/admin/refresh``) routes.

Run the development server::

    python -m volatility_mlops.api.app        # honors API_PORT (default 8000)
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid

from flask import Flask, Response, g, request
from sqlalchemy.engine import Engine

from volatility_mlops.api.errors import register_error_handlers
from volatility_mlops.api.model_holder import ModelHolder
from volatility_mlops.api.routes.health import bp as health_bp
from volatility_mlops.api.routes.history import bp as history_bp
from volatility_mlops.api.routes.portfolio import bp as portfolio_bp
from volatility_mlops.api.routes.predict import bp as predict_bp
from volatility_mlops.db.engine import get_engine


def _register_request_logging(app: Flask) -> None:
    """Tag each request with an id and log method/path/status/latency as JSON."""

    @app.before_request
    def _start() -> None:
        g.request_id = uuid.uuid4().hex[:12]
        g.start_time = time.perf_counter()

    @app.after_request
    def _log(response: Response) -> Response:
        start = getattr(g, "start_time", time.perf_counter())
        app.logger.info(
            json.dumps(
                {
                    "request_id": getattr(g, "request_id", None),
                    "method": request.method,
                    "path": request.path,
                    "status": response.status_code,
                    "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                }
            )
        )
        response.headers["X-Request-ID"] = getattr(g, "request_id", "")
        return response


def create_app(engine: Engine | None = None, model_holder: ModelHolder | None = None) -> Flask:
    """Build the Flask app.

    Args:
        engine: Application database engine. Defaults to :func:`get_engine`.
        model_holder: A (possibly pre-populated) model holder. Defaults to a fresh
            :class:`ModelHolder`. The holder is loaded once here at startup.

    Returns:
        A configured Flask application.
    """
    app = Flask(__name__)
    app.config["ENGINE"] = engine if engine is not None else get_engine()

    holder = model_holder if model_holder is not None else ModelHolder()
    holder.load()
    app.config["MODEL_HOLDER"] = holder

    register_error_handlers(app)
    _register_request_logging(app)
    for blueprint in (health_bp, predict_bp, history_bp, portfolio_bp):
        app.register_blueprint(blueprint)
    return app


def main() -> int:
    """Run the development server (production uses gunicorn; see Day 26)."""
    logging.basicConfig(level=logging.INFO)
    port = int(os.environ.get("API_PORT", "8000"))
    create_app().run(host="0.0.0.0", port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
