ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

# Pinned to the 2.x line deliberately: the model registry stage API
# (Staging / Production) that build-plan section 7.4 and the promotion gate
# depend on was removed in MLflow 3.x in favour of aliases.
# See docs/decisions.md, entry "MLflow pinned to 2.x".
ARG MLFLOW_VERSION=2.22.0

RUN pip install --no-cache-dir \
      "mlflow==${MLFLOW_VERSION}" \
      "psycopg2-binary>=2.9" \
      "boto3>=1.34"

# Artifact root. Backed by a named volume locally; repointed at s3:// in week 4.
RUN useradd --create-home --uid 1000 mlflow \
    && mkdir -p /mlflow/artifacts \
    && chown -R mlflow:mlflow /mlflow

USER mlflow
WORKDIR /mlflow
EXPOSE 5000
