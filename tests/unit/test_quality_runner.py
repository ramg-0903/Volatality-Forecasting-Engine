"""Unit tests for the quality runner's severity + gate logic (no DB)."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from volatility_mlops.quality.base import CheckResult, Severity
from volatility_mlops.quality.runner import (
    QualityGateError,
    evaluate,
    gate_failures,
    load_quality_config,
    raise_if_failed,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample_quality_ohlcv.csv"


def _clean() -> pd.DataFrame:
    df = pd.read_csv(FIXTURE)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df


def test_evaluate_runs_all_checks_and_attaches_severity():
    config = load_quality_config()
    results = evaluate(_clean(), ["SPY", "AAA", "BBB"], config, today=date(2024, 1, 9))

    assert len(results) == 8
    by_name = {r.check_name: r for r in results}
    assert by_name["completeness"].severity is Severity.ERROR
    assert by_name["return_sanity"].severity is Severity.WARN
    # On the clean, fresh fixture no error-severity check fails.
    assert gate_failures(results) == []


def test_raise_if_failed_raises_only_on_error_severity():
    warn_fail = CheckResult("return_sanity", "raw_ohlcv", passed=False, severity=Severity.WARN)
    raise_if_failed([warn_fail])  # warn failure does not raise

    error_fail = CheckResult("positivity", "raw_ohlcv", passed=False, severity=Severity.ERROR)
    with pytest.raises(QualityGateError, match="positivity"):
        raise_if_failed([error_fail])
