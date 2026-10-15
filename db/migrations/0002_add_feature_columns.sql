-- 0002_add_feature_columns.sql
-- Adds the feature columns declared in config/feature_contract.yml (v1) to the
-- features table. Day 1 (migration 0001) deliberately left these out until the
-- feature contract existed; this is the forward-only migration that lands them.
--
-- All feature values are numeric; dow/month are small integers. Every column is
-- nullable (the earliest rows per ticker lack enough history and are dropped by
-- the build, not stored as zeros).

ALTER TABLE features
    -- HAR volatility ladder (log annualized realized vol)
    ADD COLUMN log_rv_1              numeric,
    ADD COLUMN log_rv_5              numeric,
    ADD COLUMN log_rv_21             numeric,
    ADD COLUMN log_rv_63             numeric,
    ADD COLUMN log_rv_126            numeric,
    -- EWMA volatility
    ADD COLUMN log_ewma_vol          numeric,
    -- Volatility term structure
    ADD COLUMN rv_ratio_5_21         numeric,
    ADD COLUMN rv_ratio_21_126       numeric,
    -- Range-based
    ADD COLUMN log_parkinson_21      numeric,
    -- Return-based
    ADD COLUMN ret_1                 numeric,
    ADD COLUMN ret_5                 numeric,
    ADD COLUMN ret_21                numeric,
    ADD COLUMN abs_ret_1             numeric,
    -- Downside asymmetry
    ADD COLUMN log_downside_vol_21   numeric,
    -- Volume
    ADD COLUMN volume_z_21           numeric,
    -- Market-wide
    ADD COLUMN log_vix               numeric,
    ADD COLUMN vix_chg_5             numeric,
    ADD COLUMN vrp_21                numeric,
    -- Cross-sectional context
    ADD COLUMN log_index_rv_21       numeric,
    ADD COLUMN rv_ratio_to_index_21  numeric,
    -- Macro regime
    ADD COLUMN term_spread           numeric,
    -- Entity encoding (expanding-window)
    ADD COLUMN entity_mean_log_rv    numeric,
    -- Calendar (optional)
    ADD COLUMN dow                   integer,
    ADD COLUMN month                 integer;
