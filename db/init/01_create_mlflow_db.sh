#!/bin/bash
# Runs once, on first initialisation of an empty Postgres data directory.
#
# Creates the second logical database that MLflow uses for its backend store.
# The application schema is NOT created here - that is owned by the numbered,
# forward-only migrations in db/migrations/ (day 2).
#
# Idempotent: re-running against an existing database is a no-op, so a manual
# re-run or a re-created volume behaves identically.
set -euo pipefail

MLFLOW_DB="${MLFLOW_DB:-mlflow}"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT 'CREATE DATABASE ${MLFLOW_DB}'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${MLFLOW_DB}')\gexec

    GRANT ALL PRIVILEGES ON DATABASE ${MLFLOW_DB} TO ${POSTGRES_USER};
EOSQL

echo "init: database '${MLFLOW_DB}' ready"
