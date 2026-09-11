"""Market-data provider interface and the canonical OHLCV contract.

The rest of the system depends only on :class:`MarketDataProvider` and the
:data:`OHLCV_COLUMNS` shape, never on a specific vendor. That indirection is
what lets a free development source be swapped for a licensed provider later
(build-plan section 4.1) without touching ingestion, features, or serving.

Time conventions: ``trade_date`` is a calendar date in the exchange's local
convention as delivered by the provider. ``adj_close`` is split- and
dividend-adjusted and is the only price series used for return math; the raw
OHLC fields are stored for auditing only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

#: Canonical column order every provider must return, matching ``raw_ohlcv``.
OHLCV_COLUMNS: list[str] = [
    "ticker",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]

_PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "adj_close")


class MarketDataProvider(ABC):
    """A source of daily OHLCV bars, normalized to the canonical contract.

    Concrete providers set :attr:`name` (recorded on every ingested row and in
    ``ingestion_runs``) and implement :meth:`fetch_ohlcv`.
    """

    name: str = "unknown"

    @abstractmethod
    def fetch_ohlcv(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        """Fetch daily bars for ``tickers`` over the inclusive ``[start, end]`` range.

        Args:
            tickers: Symbols to fetch.
            start: First trade date to include (inclusive).
            end: Last trade date to include (inclusive).

        Returns:
            A DataFrame conforming to :data:`OHLCV_COLUMNS`. Implementations
            should return already-normalized data (see :func:`normalize_ohlcv`).
        """
        raise NotImplementedError


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a raw OHLCV frame to the canonical contract.

    Guarantees exactly :data:`OHLCV_COLUMNS`, a python ``date`` ``trade_date``,
    numeric prices, rows sorted by ``(ticker, trade_date)``, no duplicate natural
    keys (last write wins), and no rows whose ``adj_close`` is missing (a bar
    with no usable adjusted close is not ingestable).

    Args:
        df: Provider output containing at least the canonical columns.

    Returns:
        A new normalized DataFrame; the input is not mutated.

    Raises:
        ValueError: If any canonical column is absent.
    """
    missing = set(OHLCV_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV frame is missing columns: {sorted(missing)}")

    out = df.loc[:, OHLCV_COLUMNS].copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.date
    for col in _PRICE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")

    out = out.dropna(subset=["adj_close"])
    out = out.drop_duplicates(subset=["ticker", "trade_date"], keep="last")
    return out.sort_values(["ticker", "trade_date"]).reset_index(drop=True)


#: Canonical column order every macro provider must return, matching ``raw_macro``.
MACRO_COLUMNS: list[str] = ["series_id", "obs_date", "value"]


class MacroDataProvider(ABC):
    """A source of macro time series (e.g. VIX, Treasury yields).

    Concrete providers set :attr:`name` (recorded in ``ingestion_runs``) and
    implement :meth:`fetch_macro`.
    """

    name: str = "unknown"

    @abstractmethod
    def fetch_macro(self, series_ids: list[str], start: date, end: date) -> pd.DataFrame:
        """Fetch macro observations for ``series_ids`` over ``[start, end]``.

        Args:
            series_ids: Provider series codes (e.g. ``VIXCLS``, ``DGS10``).
            start: First observation date to include (inclusive).
            end: Last observation date to include (inclusive).

        Returns:
            A DataFrame conforming to :data:`MACRO_COLUMNS`. Implementations
            should return already-normalized data (see :func:`normalize_macro`).
        """
        raise NotImplementedError


def normalize_macro(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a raw macro frame to the canonical contract.

    Guarantees exactly :data:`MACRO_COLUMNS`, a python ``date`` ``obs_date``, a
    numeric ``value``, rows sorted by ``(series_id, obs_date)``, no duplicate
    natural keys (last write wins), and no rows whose ``value`` is missing. FRED
    encodes missing observations as ``"."``; those coerce to NaN and are dropped.

    Args:
        df: Provider output containing at least the canonical columns.

    Returns:
        A new normalized DataFrame; the input is not mutated.

    Raises:
        ValueError: If any canonical column is absent.
    """
    missing = set(MACRO_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"macro frame is missing columns: {sorted(missing)}")

    out = df.loc[:, MACRO_COLUMNS].copy()
    out["obs_date"] = pd.to_datetime(out["obs_date"]).dt.date
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    out = out.dropna(subset=["value"])
    out = out.drop_duplicates(subset=["series_id", "obs_date"], keep="last")
    return out.sort_values(["series_id", "obs_date"]).reset_index(drop=True)
