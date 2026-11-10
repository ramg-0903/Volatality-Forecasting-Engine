"""Unit tests for the serving model holder's graceful degradation (Day 17 + Day 21).

The holder must let the service start and report a degraded ``/health`` when the
registry has no Production model *or* is unreachable/missing entirely (e.g. a fresh
container with no tracking server) -- never crash at startup. Pure: the registry calls
are monkeypatched, so no MLflow/network is touched.
"""

from __future__ import annotations

import pytest

from volatility_mlops.api import model_holder as mh
from volatility_mlops.api.model_holder import ModelHolder


def test_empty_when_no_production_model(monkeypatch):
    monkeypatch.setattr(mh, "get_production_info", lambda name: {})
    monkeypatch.setattr(
        mh, "load_production_model", lambda name: pytest.fail("must not load without info")
    )
    holder = ModelHolder()
    holder.load()
    status = holder.status()
    assert status["loaded"] is False
    assert status["loaded_at"] is None
    assert holder.model is None


def test_degrades_when_registry_raises(monkeypatch):
    # The failure mode a bare CI container hits: the registered model does not exist.
    def _boom(name):
        raise RuntimeError("Registered Model with name=... not found")

    monkeypatch.setattr(mh, "get_production_info", _boom)
    holder = ModelHolder()
    holder.load()  # must not raise
    assert holder.status()["loaded"] is False
    assert holder.model is None


def test_loads_when_a_model_is_present(monkeypatch):
    info = {"name": "m", "version": "3", "stage": "Production", "run_id": "r", "tags": {}}
    sentinel = object()
    monkeypatch.setattr(mh, "get_production_info", lambda name: info)
    monkeypatch.setattr(mh, "load_production_model", lambda name: sentinel)
    holder = ModelHolder()
    holder.load()
    status = holder.status()
    assert status["loaded"] is True
    assert status["model_version"] == "3"
    assert holder.model is sentinel
    assert status["loaded_at"] is not None
