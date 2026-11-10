# Volatality-Forecasting-Engine

[![ci](https://github.com/ramg-0903/Volatality-Forecasting-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/ramg-0903/Volatality-Forecasting-Engine/actions/workflows/ci.yml)

An end-to-end MLOps system that forecasts 21-day-ahead realized volatility, retrains
weekly behind an automated promotion gate, and monitors accuracy against ground truth
that arrives 21 trading days late. See `docs/build-plan.md` for the full specification.

## Continuous Integration

Three GitHub Actions workflows (`.github/workflows/`):

| Workflow        | Trigger                                   | What it does |
|-----------------|-------------------------------------------|--------------|
| `ci`            | every push and pull request               | ruff format check, ruff lint, mypy, and the unit test suite with coverage |
| `docker`        | push to `main`                            | build the application image and smoke-test `/health` on the running container |
| `data-contract` | PRs touching the feature module/contract  | rebuild features from a synthetic dataset and assert the columns still match `config/feature_contract.yml` |

Dependencies are pinned in hash-locked `requirements.txt` (runtime) and
`requirements-dev.txt` (runtime + dev tooling), compiled from `pyproject.toml` with
`make lock`.

### Running the checks locally

```bash
make lint        # ruff format --check + ruff check
make typecheck   # mypy
make test        # full suite (needs the Docker stack up for integration tests)
```

CI runs only the **unit** suite (`pytest -m "not integration"`), which is hermetic. The
integration tests need a live Postgres + MLflow and a registered Production model; run
them locally with the stack up:

```bash
make up          # start Postgres + MLflow
make test        # unit + integration
```
