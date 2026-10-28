"""Integration test for the registry: register -> promote -> load-by-stage -> predict.

Marked ``integration`` (needs the MLflow server). Uses a throwaway registered model
name and deletes it on teardown so the real registry stays clean.
"""

import contextlib

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from volatility_mlops.models.pipeline import build_pipeline
from volatility_mlops.models.registry import (
    get_production_info,
    load_production_model,
    register_pipeline,
)
from volatility_mlops.models.tracking import configure_mlflow

pytestmark = pytest.mark.integration

TEST_MODEL = "test-volatility-challenger"


def _delete_test_model() -> None:
    configure_mlflow()
    client = MlflowClient()
    with contextlib.suppress(Exception):  # absent model is fine
        client.delete_registered_model(TEST_MODEL)


@pytest.fixture(autouse=True)
def _clean_registry():
    _delete_test_model()
    yield
    _delete_test_model()


def test_register_promote_load_and_predict_by_stage():
    cols = ["f1", "f2"]
    rng = np.random.default_rng(0)
    x = pd.DataFrame({c: rng.normal(0, 1, 200) for c in cols})
    y = np.log(np.abs(rng.normal(0.2, 0.05, 200)) + 0.01)  # log-vol target
    pipeline = build_pipeline({"max_iter": 30}).fit(x, y)

    version = register_pipeline(pipeline, cols, x.head(5), TEST_MODEL, {"test": 1})
    assert version >= 1

    info = get_production_info(TEST_MODEL)
    assert info["stage"] == "Production"
    assert info["version"] == str(version)

    # Load strictly by stage (the serving rule) and predict vol levels.
    model = load_production_model(TEST_MODEL)
    preds = np.asarray(model.predict(x.head(5)))
    assert preds.shape == (5,)
    assert np.all(preds > 0)  # exponentiated -> positive vol levels
