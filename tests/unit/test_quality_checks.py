"""Unit tests for the data quality checks (no DB, no network).

Each check passes on the clean fixture and fails on a deliberate corruption --
the build-plan Day 6 done-when.
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from volatility_mlops.quality import checks as C

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample_quality_ohlcv.csv"
TICKERS = ["SPY", "AAA", "BBB"]
FRESH = date(2024, 1, 9)  # one day after the fixture's latest date


def load_clean() -> pd.DataFrame:
    df = pd.read_csv(FIXTURE)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df


def _mask(df: pd.DataFrame, ticker: str, day: date) -> pd.Series:
    return (df["ticker"] == ticker) & (df["trade_date"] == day)


def test_clean_fixture_passes_every_check():
    df = load_clean()
    assert C.check_completeness(df, "SPY", TICKERS).passed
    assert C.check_non_null_critical(df).passed
    assert C.check_positivity(df).passed
    assert C.check_ohlc_consistency(df).passed
    assert C.check_duplicate_keys(df).passed
    assert C.check_return_sanity(df, 0.5).passed
    assert C.check_staleness(df, 5, FRESH).passed
    assert C.check_volatility_floor(df, 3).passed


def test_completeness_fails_on_missing_day():
    df = load_clean()
    df = df.drop(df[_mask(df, "AAA", date(2024, 1, 4))].index)
    result = C.check_completeness(df, "SPY", TICKERS)
    assert not result.passed
    assert result.observed_value == 1


def test_non_null_fails_on_null_adj_close():
    df = load_clean()
    df.loc[_mask(df, "AAA", date(2024, 1, 2)), "adj_close"] = np.nan
    assert not C.check_non_null_critical(df).passed


def test_positivity_fails_on_nonpositive_price():
    df = load_clean()
    df.loc[_mask(df, "BBB", date(2024, 1, 3)), "close"] = -1.0
    assert not C.check_positivity(df).passed


def test_ohlc_consistency_fails_when_low_not_minimum():
    df = load_clean()
    df.loc[_mask(df, "AAA", date(2024, 1, 4)), "low"] = 999.0  # low above the other fields
    assert not C.check_ohlc_consistency(df).passed


def test_duplicate_keys_fails_on_repeated_natural_key():
    df = load_clean()
    dup = df[_mask(df, "SPY", date(2024, 1, 2))].copy()
    df = pd.concat([df, dup], ignore_index=True)
    assert not C.check_duplicate_keys(df).passed


def test_return_sanity_fails_on_huge_move():
    df = load_clean()
    df.loc[_mask(df, "AAA", date(2024, 1, 4)), "adj_close"] = 500.0  # ~5x jump, unhandled split
    assert not C.check_return_sanity(df, 0.5).passed


def test_staleness_fails_when_data_is_old():
    df = load_clean()
    assert not C.check_staleness(df, 5, date(2024, 2, 1)).passed


def test_volatility_floor_fails_on_frozen_price():
    df = load_clean()
    df.loc[df["ticker"] == "AAA", "adj_close"] = 100.0  # constant price -> zero realized vol
    result = C.check_volatility_floor(df, 3)
    assert not result.passed
    assert result.observed_value > 0
