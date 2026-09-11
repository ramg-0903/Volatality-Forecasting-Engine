"""Unit tests for the macro contract and config loading (no DB, no network)."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from volatility_mlops.ingestion.base import MACRO_COLUMNS, normalize_macro
from volatility_mlops.ingestion.macro import load_macro_series

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample_macro.csv"
CONFIG = Path(__file__).resolve().parents[2] / "config" / "macro.yml"


def test_normalize_macro_cleans_and_dedupes():
    out = normalize_macro(pd.read_csv(FIXTURE))

    assert list(out.columns) == MACRO_COLUMNS
    assert isinstance(out["obs_date"].iloc[0], date)
    # The "." missing row (2024-01-04) is dropped; the duplicate key keeps last.
    assert len(out) == 5
    dup = out[(out["series_id"] == "FAKEIDX") & (out["obs_date"] == date(2024, 1, 5))]
    assert dup["value"].iloc[0] == pytest.approx(14.15)
    assert out["value"].notna().all()


def test_normalize_macro_requires_canonical_columns():
    with pytest.raises(ValueError, match="missing columns"):
        normalize_macro(pd.DataFrame({"series_id": ["X"], "obs_date": ["2024-01-02"]}))


def test_load_macro_series_matches_build_plan():
    assert load_macro_series(CONFIG) == ["VIXCLS", "DGS10", "DGS3MO"]
