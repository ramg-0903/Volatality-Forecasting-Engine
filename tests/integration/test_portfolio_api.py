"""Integration tests for /portfolio/risk (build-plan section 8.6 done-when).

Marked ``integration``: needs Postgres (features + prices) and a Production model.
Skips cleanly if preconditions are absent. The headline assertion is the build-plan
done-when: portfolio volatility is below the weighted average of holding volatilities.
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
def two_tickers(engine):
    with engine.connect() as conn:
        rows = (
            conn.execute(
                text("SELECT ticker FROM features GROUP BY ticker ORDER BY ticker LIMIT 2")
            )
            .scalars()
            .all()
        )
    if len(rows) < 2:
        pytest.skip("need at least two tickers with features")
    return list(rows)


@pytest.fixture(scope="module")
def client(engine):
    if not get_production_info():
        pytest.skip("no Production model registered; run `make register` first")
    return create_app(engine=engine).test_client()


def _portfolio(a: str, b: str) -> dict:
    return {"holdings": [{"ticker": a, "weight": 0.6}, {"ticker": b, "weight": 0.4}]}


def test_portfolio_vol_below_weighted_average(client, two_tickers):
    resp = client.post("/portfolio/risk", json=_portfolio(*two_tickers))
    if resp.status_code == 422:
        pytest.skip("insufficient return history for a correlation estimate")
    assert resp.status_code == 200
    body = resp.get_json()
    # The diversification result: portfolio vol < naive weighted average.
    assert body["portfolio_volatility"] < body["weighted_avg_volatility"]
    assert body["diversification_ratio"] > 1.0
    assert body["model"]["model_stage"] == "Production"


def test_risk_contributions_sum_to_one(client, two_tickers):
    resp = client.post("/portfolio/risk", json=_portfolio(*two_tickers))
    if resp.status_code == 422:
        pytest.skip("insufficient return history for a correlation estimate")
    assert resp.status_code == 200
    pcts = [h["risk_contribution_pct"] for h in resp.get_json()["holdings"]]
    assert sum(pcts) == pytest.approx(1.0, abs=1e-4)


def test_holdings_sorted_by_risk_contribution(client, two_tickers):
    resp = client.post("/portfolio/risk", json=_portfolio(*two_tickers))
    if resp.status_code == 422:
        pytest.skip("insufficient return history")
    contribs = [h["risk_contribution"] for h in resp.get_json()["holdings"]]
    assert contribs == sorted(contribs, reverse=True)


def test_weights_must_sum_to_one(client, two_tickers):
    a, b = two_tickers
    resp = client.post(
        "/portfolio/risk",
        json={"holdings": [{"ticker": a, "weight": 0.6}, {"ticker": b, "weight": 0.6}]},
    )
    assert resp.status_code == 400
    assert "weight_sum" in resp.get_json()["details"]


def test_negative_weight_rejected(client, two_tickers):
    a, b = two_tickers
    resp = client.post(
        "/portfolio/risk",
        json={"holdings": [{"ticker": a, "weight": 1.3}, {"ticker": b, "weight": -0.3}]},
    )
    assert resp.status_code == 400
    assert b in resp.get_json()["details"]["negative_weight_tickers"]


def test_unknown_ticker_rejected(client, two_tickers):
    a = two_tickers[0]
    resp = client.post(
        "/portfolio/risk",
        json={"holdings": [{"ticker": a, "weight": 0.5}, {"ticker": "NOTATICKER", "weight": 0.5}]},
    )
    assert resp.status_code == 400


def test_empty_holdings_rejected(client):
    resp = client.post("/portfolio/risk", json={"holdings": []})
    assert resp.status_code == 400
