"""Integration tests for the batch predictions writer (build-plan Day 19).

Marked ``integration``: needs Postgres (features) and a Production model. The headline
guarantees: predictions are version-stamped, re-running is idempotent, resolves_on is
in the future, and a corrupt forecast is rejected rather than written.
"""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import text

from volatility_mlops.db.engine import get_engine
from volatility_mlops.flows.daily import write_predictions
from volatility_mlops.models.registry import get_production_info, load_production_model

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def engine():
    return get_engine()


@pytest.fixture(scope="module")
def model_and_info():
    info = get_production_info()
    if not info:
        pytest.skip("no Production model registered; run `make register` first")
    return load_production_model(), info


def _count(engine, version) -> int:
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT count(*) FROM predictions WHERE model_version = :v"), {"v": str(version)}
        ).scalar_one()


def test_writes_version_stamped_rows(engine, model_and_info):
    model, info = model_and_info
    n = write_predictions(engine, model, info)
    assert n > 0
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT model_name, model_version, model_stage, predicted_vol, "
                "as_of_date, resolves_on, horizon_days FROM predictions "
                "WHERE model_version = :v ORDER BY as_of_date DESC LIMIT 1"
            ),
            {"v": str(info["version"])},
        ).one()
    assert row.model_name == info["name"]
    assert row.model_version == str(info["version"])
    assert float(row.predicted_vol) > 0.0  # vols are strictly positive
    assert row.resolves_on > row.as_of_date  # horizon matures in the future
    assert row.horizon_days == 21


def test_rerun_is_idempotent(engine, model_and_info):
    model, info = model_and_info
    write_predictions(engine, model, info)
    before = _count(engine, info["version"])
    write_predictions(engine, model, info)  # same day, same model version
    after = _count(engine, info["version"])
    assert after == before  # upsert overwrote, did not append


def test_missing_stamp_is_refused(engine, model_and_info):
    model, _ = model_and_info
    with pytest.raises(ValueError, match="stamp"):
        write_predictions(engine, model, {"name": "", "version": None})


def test_corrupt_forecast_is_rejected(engine, model_and_info):
    _, info = model_and_info

    class _BadModel:
        def predict(self, x):
            out = np.ones(len(x))
            out[0] = -1.0  # a non-positive vol = corruption
            return out

    before = _count(engine, info["version"])
    with pytest.raises(ValueError, match="non-finite or non-positive"):
        write_predictions(engine, _BadModel(), info)
    assert _count(engine, info["version"]) == before  # nothing was written
