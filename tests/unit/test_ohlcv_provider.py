"""Unit tests for the OHLCV contract and universe parsing (no DB, no network)."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from volatility_mlops.ingestion.base import OHLCV_COLUMNS, normalize_ohlcv
from volatility_mlops.ingestion.universe import load_universe

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample_ohlcv.csv"


def test_normalize_enforces_contract_and_types():
    raw = pd.read_csv(FIXTURE)
    out = normalize_ohlcv(raw)

    assert list(out.columns) == OHLCV_COLUMNS
    assert isinstance(out.loc[0, "trade_date"], date)
    assert pd.api.types.is_numeric_dtype(out["adj_close"])
    assert len(out) == 6


def test_normalize_sorts_and_dedupes_natural_key():
    raw = pd.read_csv(FIXTURE)
    # Duplicate the first row with a different close: last write must win.
    dup = raw.iloc[[0]].copy()
    dup["adj_close"] = 999.0
    shuffled = pd.concat([raw.iloc[::-1], dup], ignore_index=True)

    out = normalize_ohlcv(shuffled)

    assert len(out) == 6  # one row per (ticker, trade_date)
    assert list(out["ticker"]) == sorted(out["ticker"])
    first = out[(out["ticker"] == "FAKEA") & (out["trade_date"] == date(2024, 1, 2))]
    assert first["adj_close"].item() == 999.0


def test_normalize_drops_rows_without_adj_close():
    raw = pd.read_csv(FIXTURE)
    raw.loc[0, "adj_close"] = None
    out = normalize_ohlcv(raw)
    assert len(out) == 5


def test_normalize_raises_on_missing_column():
    raw = pd.read_csv(FIXTURE).drop(columns=["adj_close"])
    with pytest.raises(ValueError, match="adj_close"):
        normalize_ohlcv(raw)


def test_universe_parses_full_list():
    universe = load_universe()

    tickers = universe.all_tickers()
    assert len(tickers) == 30  # 29 equities + SPY
    assert "SPY" in tickers
    assert universe.market_proxy.ticker == "SPY"
    assert universe.history_start == date(2010, 1, 1)
    assert all(spec.company_name and spec.sector for spec in universe.all_specs())
