"""Integration test for the feature build against the real Postgres stack.

Marked ``integration``. Builds the features table from real raw data and confirms
a deterministic rebuild leaves the same rows.
"""

import pytest
from sqlalchemy import text

from volatility_mlops.db.engine import get_engine
from volatility_mlops.features.build import build_features
from volatility_mlops.features.contract import load_feature_contract
from volatility_mlops.ingestion.universe import load_universe

pytestmark = pytest.mark.integration


@pytest.fixture
def engine():
    return get_engine()


def test_build_is_deterministic_and_populates_features(engine):
    rows_first = build_features(engine)
    assert rows_first > 0

    with engine.connect() as conn:
        count_first = conn.execute(text("SELECT count(*) FROM features")).scalar_one()
        tickers = conn.execute(text("SELECT count(DISTINCT ticker) FROM features")).scalar_one()

    # Rebuild: idempotent upsert must not change the row count.
    build_features(engine)
    with engine.connect() as conn:
        count_second = conn.execute(text("SELECT count(*) FROM features")).scalar_one()
    assert count_first == count_second

    # Every modelled equity is present (the members; SPY is the proxy only).
    assert tickers == len(load_universe().members)


def test_recent_rows_have_null_label_but_valid_features(engine):
    contract = load_feature_contract()
    with engine.connect() as conn:
        # There exist rows with a null target (the most recent 21 days per ticker)...
        null_targets = conn.execute(
            text("SELECT count(*) FROM features WHERE target_fwd_rv_21 IS NULL")
        ).scalar_one()
        # ...but their core features are populated.
        null_feature = conn.execute(
            text("SELECT count(*) FROM features WHERE log_rv_21 IS NULL")
        ).scalar_one()

    assert null_targets > 0
    assert null_feature == 0
    assert len(contract.feature_names()) >= 20
