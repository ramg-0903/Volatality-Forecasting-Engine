"""Concrete market-data providers.

Currently a single free development source, yfinance (approved in
docs/decisions.md). It is unofficial and its terms restrict commercial
redistribution; this project is non-commercial and educational, no raw vendor
data is committed, and the :class:`~volatility_mlops.ingestion.base.MarketDataProvider`
interface lets a licensed provider replace it without downstream changes.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from volatility_mlops.ingestion.base import (
    OHLCV_COLUMNS,
    MarketDataProvider,
    normalize_ohlcv,
)

_YF_RENAME: dict[str, str] = {
    "Date": "trade_date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


class YFinanceProvider(MarketDataProvider):
    """Daily bars from Yahoo Finance via the ``yfinance`` package."""

    name = "yfinance"

    def fetch_ohlcv(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        """Download daily bars and normalize them to the canonical contract.

        ``auto_adjust=False`` is required so a distinct ``Adj Close`` column is
        returned; adjusted close is what all downstream return math uses. The
        yfinance ``end`` argument is exclusive, so one day is added to honor the
        inclusive ``[start, end]`` contract.

        Args:
            tickers: Symbols to fetch.
            start: First trade date to include (inclusive).
            end: Last trade date to include (inclusive).

        Returns:
            A normalized DataFrame conforming to
            :data:`~volatility_mlops.ingestion.base.OHLCV_COLUMNS`.
        """
        import yfinance as yf  # noqa: PLC0415 -- lazy: keeps module import network-free

        raw = yf.download(
            tickers=tickers,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            auto_adjust=False,
            group_by="ticker",
            progress=False,
            threads=True,
        )
        if raw is None or raw.empty:
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        return normalize_ohlcv(_to_long(raw, tickers))


def _to_long(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Reshape yfinance wide output into one long, tagged OHLCV frame.

    Handles both the multi-ticker layout (a ``(ticker, field)`` column
    MultiIndex) and the single-ticker layout (flat field columns).

    Args:
        raw: The frame returned by ``yfinance.download``.
        tickers: Symbols requested, used to tag rows in the single-index case.

    Returns:
        A long DataFrame with the canonical columns (pre-normalization).
    """
    frames: list[pd.DataFrame] = []
    multi = isinstance(raw.columns, pd.MultiIndex)
    for ticker in tickers:
        if multi:
            if ticker not in raw.columns.get_level_values(0):
                continue
            sub = raw[ticker].reset_index()
        else:
            sub = raw.reset_index()
        sub = sub.rename(columns=_YF_RENAME)
        sub["ticker"] = ticker
        keep = [c for c in OHLCV_COLUMNS if c in sub.columns]
        frames.append(sub.loc[:, keep])

    if not frames:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    return pd.concat(frames, ignore_index=True)
