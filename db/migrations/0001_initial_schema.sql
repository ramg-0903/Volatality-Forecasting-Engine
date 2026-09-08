-- 0001_initial_schema.sql
-- Initial application schema for the volatility engine (build-plan section 4.3).
--
-- FORWARD-ONLY: never edit this file after it has been applied anywhere. Schema
-- changes ship as new numbered files. In particular the per-feature columns are
-- deliberately NOT created here; they are defined by the feature contract
-- (section 5.2) on day 8 and added by a later migration, so this file stays
-- valid as the catalog evolves.
--
-- Applied atomically by volatility_mlops.db.migrate inside a single transaction.
-- The natural keys below (composite PKs / unique constraints) are what make the
-- day-3 ingestion upserts idempotent.

-- ---------------------------------------------------------------------------
-- Reference data
-- ---------------------------------------------------------------------------
CREATE TABLE dim_ticker (
    ticker        text PRIMARY KEY,
    company_name  text,
    sector        text,
    active        boolean NOT NULL DEFAULT true,
    added_at      timestamptz NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Raw ingested data (stored as received; adj_close drives all return math)
-- ---------------------------------------------------------------------------
CREATE TABLE raw_ohlcv (
    ticker       text NOT NULL REFERENCES dim_ticker (ticker),
    trade_date   date NOT NULL,
    open         numeric(14, 6),
    high         numeric(14, 6),
    low          numeric(14, 6),
    close        numeric(14, 6),
    adj_close    numeric(14, 6),
    volume       bigint,
    source       text,
    ingested_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, trade_date)
);
CREATE INDEX idx_raw_ohlcv_trade_date ON raw_ohlcv (trade_date);
CREATE INDEX idx_raw_ohlcv_ticker_date_desc ON raw_ohlcv (ticker, trade_date DESC);

CREATE TABLE raw_macro (
    series_id    text NOT NULL,
    obs_date     date NOT NULL,
    value        numeric(14, 6),
    ingested_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (series_id, obs_date)
);

-- ---------------------------------------------------------------------------
-- Modeling tables
-- ---------------------------------------------------------------------------
-- Feature columns are intentionally absent (see header). Only the key, the
-- label, and provenance columns exist until the feature contract lands.
CREATE TABLE features (
    ticker               text NOT NULL REFERENCES dim_ticker (ticker),
    trade_date           date NOT NULL,
    target_fwd_rv_21     numeric,      -- 21-trading-day-ahead realized vol; null for the most recent 21 rows
    target_available_on  date,         -- date this label first becomes observable (monitoring join)
    feature_version      text NOT NULL,
    computed_at          timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, trade_date)
);

CREATE TABLE predictions (
    prediction_id  bigserial PRIMARY KEY,
    ticker         text NOT NULL REFERENCES dim_ticker (ticker),
    as_of_date     date NOT NULL,
    horizon_days   int NOT NULL DEFAULT 21,
    predicted_vol  numeric,            -- annualized volatility level
    model_name     text NOT NULL,
    model_version  text NOT NULL,
    model_stage    text,
    resolves_on    date,               -- date the outcome becomes observable
    created_at     timestamptz NOT NULL DEFAULT now(),
    UNIQUE (ticker, as_of_date, model_version)
);

CREATE TABLE prediction_outcomes (
    prediction_id  bigint PRIMARY KEY REFERENCES predictions (prediction_id),
    realized_vol   numeric,            -- annualized realized vol over the horizon
    abs_error      numeric,
    sq_error       numeric,
    qlike          numeric,
    resolved_at    timestamptz NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Operational / audit tables
-- ---------------------------------------------------------------------------
CREATE TABLE ingestion_runs (
    run_id         bigserial PRIMARY KEY,
    source         text NOT NULL,
    date_start     date,
    date_end       date,
    rows_written   integer,
    status         text NOT NULL,      -- running | success | failed
    error_message  text,
    started_at     timestamptz NOT NULL DEFAULT now(),
    finished_at    timestamptz
);

CREATE TABLE data_quality_results (
    id             bigserial PRIMARY KEY,
    check_name     text NOT NULL,
    target_table   text NOT NULL,
    run_date       date NOT NULL,
    passed         boolean NOT NULL,
    observed_value numeric,
    threshold      numeric,
    created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE drift_reports (
    id               bigserial PRIMARY KEY,
    report_date      date NOT NULL,
    reference_start  date,
    reference_end    date,
    current_start    date,
    current_end      date,
    drift_stats      jsonb,            -- per-feature drift statistics
    overall_verdict  text,
    artifact_path    text,             -- path/URI to the Evidently HTML report
    created_at       timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE promotion_log (
    id                 bigserial PRIMARY KEY,
    challenger_version text,
    champion_version   text,
    metric_comparison  jsonb,
    decision           text NOT NULL,  -- promoted | rejected
    reason             text,
    created_at         timestamptz NOT NULL DEFAULT now()
);
