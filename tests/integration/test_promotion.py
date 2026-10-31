"""Integration test: the gate writes its decision to promotion_log.

Marked ``integration``. Writes a promote row and a reject row under a sentinel
challenger version, verifies them, and cleans up.
"""

import pytest
from sqlalchemy import text

from volatility_mlops.db.engine import get_engine
from volatility_mlops.models.promotion import PromotionDecision, log_promotion

pytestmark = pytest.mark.integration

SENTINEL = "TEST_CHAL_v999"


@pytest.fixture
def engine():
    return get_engine()


@pytest.fixture(autouse=True)
def _clean(engine):
    def _delete():
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM promotion_log WHERE challenger_version = :v"), {"v": SENTINEL}
            )

    _delete()
    yield
    _delete()


def test_log_promotion_writes_promote_and_reject_rows(engine):
    promote = PromotionDecision(True, "all 7 criteria passed", {"primary": True}, {"x": 1.0})
    reject = PromotionDecision(False, "rejected: primary failed", {"primary": False}, {"x": 0.0})

    log_promotion(engine, promote, SENTINEL, "CHAMP_v1")
    log_promotion(engine, reject, SENTINEL, "CHAMP_v1")

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT decision, reason FROM promotion_log "
                "WHERE challenger_version = :v ORDER BY id"
            ),
            {"v": SENTINEL},
        ).all()

    assert [r.decision for r in rows] == ["promoted", "rejected"]
    assert "primary" in rows[1].reason
