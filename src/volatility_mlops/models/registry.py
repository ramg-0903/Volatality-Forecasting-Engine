"""MLflow model registry wrapper (build-plan section 7.4 / Day 15).

Trains the production challenger on all labelled data, logs it as a self-contained
MLflow pyfunc model (24 features in -> annualized vol *level* out, with the log
transform encapsulated), registers it as a new version, and promotes that version
to the ``Production`` stage. The serving layer loads strictly by stage
(``models:/<name>/Production``), never from a file path (non-negotiable rule 6).

Run as a module to (re)register and promote the production model::

    python -m volatility_mlops.models.registry
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature
from mlflow.pyfunc import PyFuncModel, PythonModel
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline
from sqlalchemy.engine import Engine

from volatility_mlops.db.engine import get_engine
from volatility_mlops.features.contract import load_feature_contract
from volatility_mlops.models.baselines import TARGET
from volatility_mlops.models.evaluate import load_labeled_panel
from volatility_mlops.models.pipeline import build_pipeline
from volatility_mlops.models.tracking import configure_mlflow, git_sha

# repo_root/src/volatility_mlops/models/registry.py -> parents[3] == repo root
MODEL_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "model.yml"
_EXPERIMENT = "volatility-registry"


@dataclass(frozen=True)
class ModelConfig:
    """Production model configuration."""

    registered_model_name: str
    best_params: dict[str, Any]


def load_model_config(path: Path = MODEL_CONFIG_PATH) -> ModelConfig:
    """Load the registered model name and hyperparameters from ``model.yml``."""
    data = yaml.safe_load(path.read_text())
    return ModelConfig(
        registered_model_name=data["registered_model_name"],
        best_params=dict(data["best_params"]),
    )


class VolatilityModel(PythonModel):
    """Pyfunc wrapper that turns the log-target pipeline into a vol-level model.

    The fitted sklearn pipeline predicts log volatility; this wrapper exponentiates
    so the registered model's contract is simply: feature rows in, annualized
    volatility levels out.
    """

    def __init__(self, pipeline: Pipeline, feature_cols: list[str]) -> None:
        self.pipeline = pipeline
        self.feature_cols = feature_cols

    def predict(self, context: Any, model_input: Any, params: Any = None) -> np.ndarray:  # noqa: ANN401, ARG002
        """Return annualized volatility levels for the input feature rows."""
        features = (
            model_input[self.feature_cols] if hasattr(model_input, "columns") else model_input
        )
        return np.exp(self.pipeline.predict(features))


def train_production_model(engine: Engine) -> tuple[Pipeline, list[str], pd.DataFrame]:
    """Fit the challenger on all labelled data; return the pipeline, columns, panel."""
    contract = load_feature_contract()
    feature_cols = contract.feature_names()
    panel = load_labeled_panel(engine, feature_cols)
    config = load_model_config()

    pipeline = build_pipeline(config.best_params)
    pipeline.fit(panel[feature_cols], np.log(panel[TARGET].to_numpy()))
    return pipeline, feature_cols, panel


def _latest_version(client: MlflowClient, name: str) -> str:
    """Return the highest version number registered under ``name``."""
    versions = client.search_model_versions(f"name='{name}'")
    return str(max(int(v.version) for v in versions))


def register_pipeline(
    pipeline: Pipeline,
    feature_cols: list[str],
    example: pd.DataFrame,
    model_name: str,
    params: dict[str, Any],
) -> int:
    """Log a fitted pipeline as a pyfunc model, register it, and promote to Production.

    Args:
        pipeline: A fitted log-target pipeline.
        feature_cols: The feature columns the model expects, in order.
        example: A few rows of features for the signature/input example.
        model_name: Registered model name.
        params: Parameters to log alongside the model.

    Returns:
        The registered version number now in ``Production``.
    """
    configure_mlflow()
    sample = example[feature_cols]
    predictions = np.exp(pipeline.predict(sample))
    signature = infer_signature(sample, predictions)

    mlflow.set_experiment(_EXPERIMENT)
    with mlflow.start_run(run_name=f"register-{model_name}"):
        mlflow.set_tag("git_sha", git_sha())
        mlflow.log_params(params)
        mlflow.log_dict({"features": feature_cols}, "feature_list.json")
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=VolatilityModel(pipeline, feature_cols),
            signature=signature,
            input_example=sample,
            registered_model_name=model_name,
            # Explicit requirements skip MLflow's slow pip-based environment inference.
            pip_requirements=["scikit-learn", "numpy", "pandas", "mlflow<3"],
        )

    client = MlflowClient()
    version = _latest_version(client, model_name)
    client.transition_model_version_stage(
        model_name, version, "Production", archive_existing_versions=True
    )
    client.set_model_version_tag(model_name, version, "git_sha", git_sha())
    return int(version)


def register_production_model(engine: Engine | None = None, model_name: str | None = None) -> int:
    """Train the production challenger and register+promote it to Production."""
    engine = engine or get_engine()
    config = load_model_config()
    name = model_name or config.registered_model_name

    pipeline, feature_cols, panel = train_production_model(engine)
    params = {
        **config.best_params,
        "feature_version": load_feature_contract().version,
        "n_train_rows": len(panel),
    }
    return register_pipeline(pipeline, feature_cols, panel.head(5), name, params)


def load_production_model(model_name: str | None = None) -> PyFuncModel:
    """Load the Production-stage model by stage (never from a file path)."""
    configure_mlflow()
    name = model_name or load_model_config().registered_model_name
    return mlflow.pyfunc.load_model(f"models:/{name}/Production")


def get_production_info(model_name: str | None = None) -> dict[str, Any]:
    """Return version/stage/run/tags of the current Production model, or ``{}``."""
    configure_mlflow()
    name = model_name or load_model_config().registered_model_name
    client = MlflowClient()
    versions = client.get_latest_versions(name, stages=["Production"])
    if not versions:
        return {}
    v = versions[0]
    return {
        "name": name,
        "version": v.version,
        "stage": v.current_stage,
        "run_id": v.run_id,
        "tags": dict(v.tags),
    }


def main() -> int:
    """Register and promote the production model, then print its registry info."""
    version = register_production_model()
    info = get_production_info()
    print(f"Registered {info.get('name')} v{version} -> stage {info.get('stage')}")
    print(f"  run_id: {info.get('run_id')}  tags: {info.get('tags')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
