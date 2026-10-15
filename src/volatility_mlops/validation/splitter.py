"""Walk-forward (rolling-origin) validation with purge and embargo (section 6.2).

Random train/test splits are invalid here: the 21-day forward label overlaps
adjacent rows, so a random split leaks the future into the past. Instead we split
along the time axis into expanding-train / forward-test folds, and enforce a gap of
``purge + embargo`` trading days between the last training date and the first test
date (leakage rules, section 4.6):

- **purge** (= the 21-day horizon) drops training dates whose forward label would
  reach into the test window.
- **embargo** adds a further safety gap after the purge.

The input is the per-row ``trade_date`` array of a panel **sorted by trade_date**
and containing only labelled rows (null-label rows are excluded upstream). Folds
carry positional row indices into that panel.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import yaml

# repo_root/src/volatility_mlops/validation/splitter.py -> parents[3] == repo root
VALIDATION_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "validation.yml"


@dataclass(frozen=True)
class WalkForwardConfig:
    """Walk-forward validation parameters (trading days)."""

    n_splits: int
    test_size: int
    purge: int
    embargo: int
    min_train_size: int


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold: positional row indices plus its date boundaries."""

    index: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_end: date
    test_start: date
    test_end: date


def load_validation_config(path: Path = VALIDATION_CONFIG_PATH) -> WalkForwardConfig:
    """Load walk-forward parameters from ``validation.yml``."""
    data = yaml.safe_load(path.read_text())
    return WalkForwardConfig(
        n_splits=data["n_splits"],
        test_size=data["test_size"],
        purge=data["purge"],
        embargo=data["embargo"],
        min_train_size=data["min_train_size"],
    )


def walk_forward_folds(
    trade_dates: Sequence[date] | np.ndarray, config: WalkForwardConfig
) -> list[Fold]:
    """Build expanding-window walk-forward folds with purge and embargo.

    Test windows of ``test_size`` unique trading days are laid end-to-end at the
    most recent part of the sample (oldest fold first). Each fold trains on all
    dates up to ``purge + embargo`` trading days before its test window.

    Args:
        trade_dates: Per-row trade dates of a panel sorted by trade date (dates
            repeat across tickers).
        config: Walk-forward parameters.

    Returns:
        ``config.n_splits`` folds with positional row indices into the panel.

    Raises:
        ValueError: If the first fold would have fewer than ``min_train_size``
            training dates, or the sample is too short for the requested splits.
    """
    dates = np.asarray(trade_dates)
    unique = np.array(sorted(set(dates.tolist())))
    n = len(unique)
    gap = config.purge + config.embargo
    ts = config.test_size

    if n < config.n_splits * ts + gap + config.min_train_size:
        raise ValueError(
            f"only {n} unique trading dates; too few for {config.n_splits} splits "
            f"of {ts} with gap {gap} and min_train_size {config.min_train_size}"
        )

    folds: list[Fold] = []
    for i in range(config.n_splits):
        splits_from_end = config.n_splits - 1 - i
        test_end_pos = n - splits_from_end * ts
        test_start_pos = test_end_pos - ts
        train_end_pos = test_start_pos - gap  # exclusive; leaves the gap of `gap` dates

        if train_end_pos < config.min_train_size:
            raise ValueError(
                f"fold {i}: {train_end_pos} training dates < min_train_size {config.min_train_size}"
            )

        train_dates = unique[:train_end_pos]
        test_dates = unique[test_start_pos:test_end_pos]

        train_idx = np.flatnonzero(np.isin(dates, train_dates))
        test_idx = np.flatnonzero(np.isin(dates, test_dates))

        folds.append(
            Fold(
                index=i,
                train_idx=train_idx,
                test_idx=test_idx,
                train_end=unique[train_end_pos - 1],
                test_start=unique[test_start_pos],
                test_end=unique[test_end_pos - 1],
            )
        )
    return folds
