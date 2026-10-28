"""Unit tests for the registry wrapper (no MLflow server, no DB)."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from volatility_mlops.models.registry import VolatilityModel, load_model_config


def test_volatility_model_exponentiates_log_predictions():
    x = pd.DataFrame({"a": [0.0, 1.0, 2.0]})
    pipe = Pipeline([("m", LinearRegression())]).fit(x, np.log([0.1, 0.2, 0.3]))
    model = VolatilityModel(pipe, ["a"])

    out = model.predict(None, x)
    assert np.all(out > 0)  # vol levels are strictly positive
    assert np.allclose(out, np.exp(pipe.predict(x)))  # wrapper == exp(pipeline)


def test_model_config_loads():
    config = load_model_config()
    assert config.registered_model_name
    assert "learning_rate" in config.best_params
