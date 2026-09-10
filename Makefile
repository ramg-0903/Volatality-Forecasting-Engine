COMPOSE := docker compose
PYTHON   := python3

.DEFAULT_GOAL := help
.PHONY: help env install up down logs ps restart test lint fmt typecheck migrate seed ingest db-shell mlflow-open clean

help:  ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
.env:
	@cp .env.template .env
	@echo "created .env from .env.template - review it before running make up"

env: .env  ## Create .env from the template if it does not exist

install:  ## Install the package and dev tooling into the active environment
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install

# -----------------------------------------------------------------------------
# Stack
# -----------------------------------------------------------------------------
up: .env  ## Start Postgres and MLflow, wait for both to report healthy
	$(COMPOSE) up -d --build --wait
	@echo ""
	@echo "  postgres  localhost:$$(grep -E '^POSTGRES_HOST_PORT=' .env | cut -d= -f2)"
	@echo "  mlflow    http://localhost:$$(grep -E '^MLFLOW_HOST_PORT=' .env | cut -d= -f2)"

down:  ## Stop services, keeping data volumes
	$(COMPOSE) down

logs:  ## Follow logs from all services
	$(COMPOSE) logs -f

ps:  ## Show service status
	$(COMPOSE) ps

restart: down up  ## Restart the stack

# -----------------------------------------------------------------------------
# Quality
# -----------------------------------------------------------------------------
test:  ## Run the test suite
	pytest

lint:  ## Check formatting and lint rules
	ruff format --check .
	ruff check .

fmt:  ## Apply formatting and autofixable lint rules
	ruff format .
	ruff check --fix .

typecheck:  ## Run mypy over src/
	mypy

# -----------------------------------------------------------------------------
# Database
# -----------------------------------------------------------------------------
migrate: .env  ## Apply forward-only SQL migrations to the application database (idempotent)
	$(PYTHON) -m volatility_mlops.db.migrate

seed: .env  ## Seed dim_ticker from config/universe.yml (idempotent)
	$(PYTHON) -m volatility_mlops.ingestion.universe

ingest: .env  ## Ingest OHLCV over a date range, e.g. make ingest START=2024-01-01 END=2024-03-01
	$(PYTHON) -m volatility_mlops.ingestion.ohlcv --start $(START) --end $(END)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
db-shell:  ## Open psql against the application database
	$(COMPOSE) exec postgres sh -c 'psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"'

mlflow-open: .env  ## Open the MLflow UI in a browser
	@open "http://localhost:$$(grep -E '^MLFLOW_HOST_PORT=' .env | cut -d= -f2)"

clean:  ## Stop services AND DELETE ALL DATA VOLUMES
	@printf "This deletes the Postgres data and MLflow artifacts. Continue? [y/N] " \
		&& read ans && [ "$$ans" = "y" ]
	$(COMPOSE) down -v
