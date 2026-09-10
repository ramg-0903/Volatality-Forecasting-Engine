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
