"""Integration tests for the serving API (build-plan section 8.6 done-when).

Marked ``integration``: needs Postgres (features + universe) and the MLflow registry
(a Production model). Uses the real app factory and Flask's test client. Skips
cleanly if the preconditions (a Production model, a ticker with features) are absent,
so the suite is still meaningful on a freshly seeded environment.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from volatility_mlops.api.app import create_app
from volatility_mlops.db.engine import get_engine
from volatility_mlops.models.registry import get_production_info

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def engine():
    return get_engine()


@pytest.fixture(scope="module")
def ticker_with_features(engine):
    with engine.connect() as conn:
        tk = conn.execute(text("SELECT ticker FROM features LIMIT 1")).scalar()
    if tk is None:
        pytest.skip("no feature rows available; run `make features` first")
    return tk


@pytest.fixture(scope="module")
def client(engine):
    if not get_production_info():
        pytest.skip("no Production model registered; run `make register` first")
    app = create_app(engine=engine)
    return app.test_client()


def test_health_reports_live_model_version(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert body["loaded"] is True
    assert body["model_stage"] == "Production"
    assert body["model_version"]  # non-empty version string
    assert resp.headers.get("X-Request-ID")  # request-id middleware ran


def test_predict_returns_vol_with_model_stamp(client, ticker_with_features):
    resp = client.post("/predict", json={"tickers": [ticker_with_features]})
    assert resp.status_code == 200
    body = resp.get_json()
    pred = body["predictions"][0]
    assert pred["ticker"] == ticker_with_features
    assert isinstance(pred["predicted_vol"], float) and pred["predicted_vol"] > 0
    assert pred["model_version"] == body["model"]["model_version"]


def test_predict_is_case_insensitive(client, ticker_with_features):
    resp = client.post("/predict", json={"tickers": [ticker_with_features.lower()]})
    assert resp.status_code == 200
    assert resp.get_json()["predictions"][0]["ticker"] == ticker_with_features


def test_predict_unknown_ticker_is_400(client):
    resp = client.post("/predict", json={"tickers": ["NOTATICKER"]})
    assert resp.status_code == 400
    body = resp.get_json()
    assert "NOTATICKER" in body["details"]["unknown_tickers"]


def test_predict_empty_tickers_is_400(client):
    resp = client.post("/predict", json={"tickers": []})
    assert resp.status_code == 400
    assert "tickers" in resp.get_json()["error"]


def test_predict_bad_date_is_400(client, ticker_with_features):
    resp = client.post(
        "/predict", json={"tickers": [ticker_with_features], "as_of_date": "2024/01/01"}
    )
    assert resp.status_code == 400


def test_history_returns_ordered_series(client, ticker_with_features):
    resp = client.get(f"/history?ticker={ticker_with_features}")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ticker"] == ticker_with_features
    dates = [r["as_of_date"] for r in body["history"]]
    assert dates == sorted(dates)  # oldest first


def test_history_requires_ticker(client):
    resp = client.get("/history")
    assert resp.status_code == 400


def test_unknown_route_is_json_404(client):
    resp = client.get("/does-not-exist")
    assert resp.status_code == 404
    assert resp.is_json
    assert "error" in resp.get_json()
