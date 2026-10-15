"""Feature build: assemble the features table from raw data (build-plan Day 9).

Loads ``raw_ohlcv`` and ``raw_macro``, computes the backward-looking features
(``compute.py``) and the forward-looking target (``target.py``), applies the
section 5.4 preprocessing policy (forward-fill macro across small gaps; drop rows
without enough history rather than imputing), and idempotently upserts into
``features`` on ``(ticker, trade_date)``.

Rows in the most recent 21 trading days are kept with a null label (they receive
predictions but are excluded from training, section 5.1).

Run as a module::

    python -m volatility_mlops.features.build
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from volatility_mlops.db.engine import get_engine
from volatility_mlops.features.compute import compute_market_features, compute_ticker_features
from volatility_mlops.features.contract import load_feature_contract
from volatility_mlops.features.target import build_targets
from volatility_mlops.ingestion.universe import load_universe


def _load_ohlcv(engine: Engine) -> pd.DataFrame:
    """Load all OHLCV rows, ``trade_date`` as python date."""
    df = pd.read_sql(
        text("SELECT ticker, trade_date, adj_close, high, low, volume FROM raw_ohlcv"),
        engine,
        parse_dates=["trade_date"],
    )
    df["trade_date"] = df["trade_date"].dt.date
    return df


def _load_macro(engine: Engine) -> pd.DataFrame:
    """Load macro series pivoted to columns, indexed by observation date."""
    df = pd.read_sql(
        text("SELECT series_id, obs_date, value FROM raw_macro"),
        engine,
        parse_dates=["obs_date"],
    )
    df["obs_date"] = df["obs_date"].dt.date
    # aggfunc="first" is a no-op here (the natural key is unique) but satisfies the
    # pivot_table-over-pivot lint rule.
    return df.pivot_table(
        index="obs_date", columns="series_id", values="value", aggfunc="first"
    ).sort_index()


def build_feature_frame(ohlcv: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Assemble the full feature+target frame (no DB access).

    Args:
        ohlcv: Long OHLCV frame for the whole universe.
        macro: Macro frame with ``VIXCLS``/``DGS10``/``DGS3MO`` columns.

    Returns:
        A long frame with the contract features, the target, and metadata; rows
        without enough history are dropped, recent unlabeled rows are kept.
    """
    universe = load_universe()
    contract = load_feature_contract()
    proxy = universe.market_proxy.ticker

    spy = ohlcv[ohlcv["ticker"] == proxy].set_index("trade_date")["adj_close"]
    market = compute_market_features(macro["VIXCLS"], macro["DGS10"], macro["DGS3MO"], spy)

    frames: list[pd.DataFrame] = []
    for spec in universe.members:
        rows = ohlcv[ohlcv["ticker"] == spec.ticker].set_index("trade_date")
        feats = compute_ticker_features(rows[["adj_close", "high", "low", "volume"]])
        merged = feats.join(market, how="left")
        merged["vrp_21"] = merged["vix_level"] / 100.0 - merged["rv_21_level"]
        merged["rv_ratio_to_index_21"] = merged["rv_21_level"] / merged["index_rv_21_level"]
        merged["dow"] = [d.weekday() for d in merged.index]
        merged["month"] = [d.month for d in merged.index]
        merged["ticker"] = spec.ticker
        frames.append(merged.reset_index(names="trade_date"))

    features = pd.concat(frames, ignore_index=True)

    members = [s.ticker for s in universe.members]
    targets = build_targets(
        ohlcv[ohlcv["ticker"].isin(members)][["ticker", "trade_date", "adj_close"]]
    )
    df = features.merge(targets, on=["ticker", "trade_date"], how="left")

    feature_cols = contract.feature_names(include_optional=True)
    df = df.dropna(subset=feature_cols)  # drop insufficient-history rows; keep null-target rows
    df["feature_version"] = contract.version
    return df


def _upsert_features(engine: Engine, df: pd.DataFrame, feature_cols: list[str]) -> int:
    """Idempotently upsert the feature frame into ``features``."""
    cols = [
        "ticker",
        "trade_date",
        *feature_cols,
        "target_fwd_rv_21",
        "target_available_on",
        "feature_version",
    ]
    # astype(object) first: on a float column, .where(..., None) would otherwise
    # re-coerce None back to NaN, which Postgres numeric stores as NaN, not NULL.
    clean = df[cols].astype(object).where(df[cols].notna(), None)
    records: list[dict[str, Any]] = [
        {str(k): v for k, v in row.items()} for row in clean.to_dict("records")
    ]
    for r in records:  # dow/month are integer columns
        for k in ("dow", "month"):
            r[k] = int(r[k]) if r[k] is not None else None
    if not records:
        return 0

    placeholders = ", ".join(f":{c}" for c in cols)
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in ("ticker", "trade_date"))
    stmt = text(
        f"INSERT INTO features ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT (ticker, trade_date) DO UPDATE SET {updates}, computed_at = now()"
    )
    with engine.begin() as conn:
        conn.execute(stmt, records)
    return len(records)


def build_features(engine: Engine) -> int:
    """Build features for the whole universe and upsert them.

    Args:
        engine: Application-database engine.

    Returns:
        The number of feature rows written.
    """
    contract = load_feature_contract()
    ohlcv = _load_ohlcv(engine)
    macro = _load_macro(engine)
    df = build_feature_frame(ohlcv, macro)
    return _upsert_features(engine, df, contract.feature_names(include_optional=True))


def main() -> int:
    """Build the features table from raw data and print a summary."""
    engine = get_engine()
    rows = build_features(engine)
    print(f"Built features: {rows} rows upserted into features.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
