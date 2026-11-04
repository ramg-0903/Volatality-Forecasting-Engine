"""In-memory cache of the current Production model (build-plan section 8.2).

Loading a model from the MLflow registry is slow, so the serving layer loads the
``Production``-stage model once and keeps it (plus its registry metadata and the
load timestamp) in memory. The model is resolved *by stage*, never from a file path
(non-negotiable rule 6), so promoting a new version in the registry and then calling
:meth:`ModelHolder.refresh` swaps the served model with no code change or restart.

Access is guarded by a lock so a ``refresh`` (which may be triggered concurrently by
an ops call) never hands out a half-loaded model.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from typing import Any

from mlflow.pyfunc import PyFuncModel

from volatility_mlops.models.registry import get_production_info, load_production_model


class ModelHolder:
    """Holds the loaded Production model and its metadata, refreshable in place."""

    def __init__(self, model_name: str | None = None) -> None:
        """Create an empty holder; call :meth:`load` to populate it.

        Args:
            model_name: Registered model name. ``None`` uses the name in
                ``config/model.yml``.
        """
        self._model_name = model_name
        self._lock = threading.Lock()
        self._model: PyFuncModel | None = None
        self._info: dict[str, Any] = {}
        self._loaded_at: datetime | None = None

    def load(self) -> None:
        """(Re)load the current Production model from the registry by stage.

        If no version is in the ``Production`` stage, the holder is left empty
        (``status()["loaded"]`` is ``False``) rather than raising, so the service can
        still start and report a degraded ``/health``.
        """
        with self._lock:
            info = get_production_info(self._model_name)
            if not info:
                self._model = None
                self._info = {}
                self._loaded_at = None
                return
            self._model = load_production_model(self._model_name)
            self._info = info
            self._loaded_at = datetime.now(UTC)

    def refresh(self) -> dict[str, Any]:
        """Reload from the registry and return the new :meth:`status`."""
        self.load()
        return self.status()

    @property
    def model(self) -> PyFuncModel | None:
        """The loaded pyfunc model, or ``None`` if nothing is in Production."""
        return self._model

    @property
    def info(self) -> dict[str, Any]:
        """Registry metadata (name/version/stage/run_id/tags) of the loaded model."""
        return self._info

    def status(self) -> dict[str, Any]:
        """A flat, JSON-serializable summary for ``/health`` and ``/admin/refresh``."""
        return {
            "loaded": self._model is not None,
            "model_name": self._info.get("name"),
            "model_version": self._info.get("version"),
            "model_stage": self._info.get("stage"),
            "run_id": self._info.get("run_id"),
            "loaded_at": self._loaded_at.isoformat() if self._loaded_at else None,
        }
