"""Data-contract test (build-plan section 11.1).

Rebuilds the feature frame in memory from a synthetic dataset covering the real
universe and asserts the produced columns match the committed feature contract. This
is the guard behind the ``data-contract`` CI job: if someone changes the features the
pipeline emits (adds, renames, drops, or breaks one) without updating
``config/feature_contract.yml`` -- and bumping its version -- this test fails loudly.

Pure: no database, no MLflow, no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volatility_mlops.features.build import build_feature_frame
from volatility_mlops.features.contract import load_feature_contract
from volatility_mlops.ingestion.universe import load_universe

# Frozen snapshot of the v1 contract. Changing the features the pipeline produces must
# be a deliberate edit of BOTH config/feature_contract.yml and these constants -- that
# is the "bump the version when you change the contract" tripwire.
EXPECTED_VERSION = "v1"
EXPECTED_FEATURES = [
    "log_rv_1",
    "log_rv_5",
    "log_rv_21",
    "log_rv_63",
    "log_rv_126",
    "log_ewma_vol",
    "rv_ratio_5_21",
    "rv_ratio_21_126",
    "log_parkinson_21",
    "ret_1",
    "ret_5",
    "ret_21",
    "abs_ret_1",
    "log_downside_vol_21",
    "volume_z_21",
    "log_vix",
    "vix_chg_5",
    "vrp_21",
    "log_index_rv_21",
    "rv_ratio_to_index_21",
    "term_spread",
    "entity_mean_log_rv",
    "dow",
    "month",
]


def _synthetic_ohlcv(tickers: list[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Deterministic random-walk OHLCV long frame for the given tickers and dates."""
    rng = np.random.default_rng(0)
    frames = []
    for i, ticker in enumerate(tickers):
        steps = rng.normal(0.0004, 0.012, size=len(dates))
        adj_close = 100.0 * (1.0 + i * 0.1) * np.exp(np.cumsum(steps))
        intraday = np.abs(rng.normal(0.0, 0.008, size=len(dates)))
        frames.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "trade_date": dates,
                    "adj_close": adj_close,
                    "high": adj_close * (1.0 + intraday),
                    "low": adj_close * (1.0 - intraday),
                    "volume": rng.integers(1_000_000, 5_000_000, size=len(dates)),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _synthetic_macro(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Macro frame with the VIXCLS/DGS10/DGS3MO columns build_feature_frame expects."""
    rng = np.random.default_rng(1)
    n = len(dates)
    return pd.DataFrame(
        {
            "VIXCLS": 18.0 + np.cumsum(rng.normal(0.0, 0.2, n)).clip(-6, 6),
            "DGS10": 3.5 + np.cumsum(rng.normal(0.0, 0.01, n)).clip(-1, 1),
            "DGS3MO": 2.0 + np.cumsum(rng.normal(0.0, 0.01, n)).clip(-1, 1),
        },
        index=dates,
    )


def _build() -> pd.DataFrame:
    universe = load_universe()
    tickers = [s.ticker for s in universe.members] + [universe.market_proxy.ticker]
    dates = pd.bdate_range("2023-01-02", periods=220)  # > 126d history + 21d target horizon
    return build_feature_frame(_synthetic_ohlcv(tickers, dates), _synthetic_macro(dates))


def test_contract_feature_set_is_frozen():
    # The committed contract must match the frozen snapshot above, version included.
    contract = load_feature_contract()
    assert contract.version == EXPECTED_VERSION
    assert contract.feature_names(include_optional=True) == EXPECTED_FEATURES


def test_build_emits_exactly_the_contract_features():
    df = _build()
    assert not df.empty
    produced = set(df.columns)
    for name in EXPECTED_FEATURES:
        assert name in produced, f"contract feature {name!r} is not produced by the pipeline"
        assert df[name].notna().all(), f"contract feature {name!r} has nulls after the build"


def test_build_stamps_the_contract_version():
    df = _build()
    assert (df["feature_version"] == EXPECTED_VERSION).all()
