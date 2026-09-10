"""Load the version-controlled universe and seed ``dim_ticker``.

``config/universe.yml`` is the single source of truth for which names the system
tracks (build-plan section 4.2). Seeding ``dim_ticker`` from it must happen
before any price ingestion, because ``raw_ohlcv.ticker`` references it.

Run as a module to (idempotently) seed the reference table::

    python -m volatility_mlops.ingestion.universe
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml
from sqlalchemy import text
from sqlalchemy.engine import Engine

from volatility_mlops.db.engine import get_engine

# repo_root/src/volatility_mlops/ingestion/universe.py -> parents[3] == repo root
UNIVERSE_PATH = Path(__file__).resolve().parents[3] / "config" / "universe.yml"

_UPSERT_TICKER = text(
    """
    INSERT INTO dim_ticker (ticker, company_name, sector, active)
    VALUES (:ticker, :company_name, :sector, true)
    ON CONFLICT (ticker) DO UPDATE SET
        company_name = EXCLUDED.company_name,
        sector = EXCLUDED.sector,
        active = true
    """
)


@dataclass(frozen=True)
class TickerSpec:
    """One universe member: its symbol, display name, and sector label."""

    ticker: str
    company_name: str
    sector: str


@dataclass(frozen=True)
class Universe:
    """The fixed tradable universe plus the market-proxy ETF and backfill start."""

    members: tuple[TickerSpec, ...]
    market_proxy: TickerSpec
    history_start: date

    def all_specs(self) -> list[TickerSpec]:
        """Return every ticker spec, the market proxy included."""
        return [*self.members, self.market_proxy]

    def all_tickers(self) -> list[str]:
        """Return every symbol, the market proxy included."""
        return [spec.ticker for spec in self.all_specs()]


def load_universe(path: Path = UNIVERSE_PATH) -> Universe:
    """Parse ``universe.yml`` into a :class:`Universe`.

    Args:
        path: Path to the universe YAML file.

    Returns:
        The parsed universe, with sectors attached to each member.
    """
    data = yaml.safe_load(path.read_text())

    members: list[TickerSpec] = []
    for sector, names in data["tickers"].items():
        for ticker, company_name in names.items():
            members.append(TickerSpec(ticker=ticker, company_name=company_name, sector=sector))

    proxy = data["market_proxy"]
    market_proxy = TickerSpec(
        ticker=proxy["ticker"],
        company_name=proxy["name"],
        sector=proxy["sector"],
    )
    history_start = date.fromisoformat(str(data["history_start"]))
    return Universe(
        members=tuple(members),
        market_proxy=market_proxy,
        history_start=history_start,
    )


def upsert_tickers(engine: Engine, specs: Iterable[TickerSpec]) -> int:
    """Idempotently upsert ticker specs into ``dim_ticker``.

    Args:
        engine: Application-database engine.
        specs: Ticker specs to insert or update.

    Returns:
        The number of rows written.
    """
    records = [
        {"ticker": s.ticker, "company_name": s.company_name, "sector": s.sector} for s in specs
    ]
    if not records:
        return 0
    with engine.begin() as conn:
        conn.execute(_UPSERT_TICKER, records)
    return len(records)


def seed_dim_ticker(engine: Engine, universe: Universe | None = None) -> int:
    """Seed ``dim_ticker`` from the universe config.

    Args:
        engine: Application-database engine.
        universe: Parsed universe; loaded from disk when omitted.

    Returns:
        The number of ticker rows written (members plus the market proxy).
    """
    universe = universe or load_universe()
    return upsert_tickers(engine, universe.all_specs())


def main() -> int:
    """Seed ``dim_ticker`` from ``universe.yml`` and print a summary."""
    engine = get_engine()
    count = seed_dim_ticker(engine)
    print(f"Seeded {count} tickers into dim_ticker.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
