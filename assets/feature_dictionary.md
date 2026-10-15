# Feature data dictionary

Generated from `feature_contract.yml` (feature_version: **v1**).

| Feature | Group | Window (days) | Optional | Definition |
|---|---|---|---|---|
| `log_rv_1` | har_ladder | 1 |  | log annualized realized vol over the trailing 1 trading day |
| `log_rv_5` | har_ladder | 5 |  | log annualized realized vol over the trailing 5 trading days |
| `log_rv_21` | har_ladder | 21 |  | log annualized realized vol over the trailing 21 trading days |
| `log_rv_63` | har_ladder | 63 |  | log annualized realized vol over the trailing 63 trading days |
| `log_rv_126` | har_ladder | 126 |  | log annualized realized vol over the trailing 126 trading days |
| `log_ewma_vol` | ewma |  |  | log annualized EWMA volatility, RiskMetrics decay lambda=0.94 |
| `rv_ratio_5_21` | term_structure |  |  | ratio of trailing 5d to 21d realized vol |
| `rv_ratio_21_126` | term_structure |  |  | ratio of trailing 21d to 126d realized vol |
| `log_parkinson_21` | range | 21 |  | log annualized Parkinson high-low volatility over 21 days |
| `ret_1` | return_based | 1 |  | cumulative log return over the trailing 1 day |
| `ret_5` | return_based | 5 |  | cumulative log return over the trailing 5 days |
| `ret_21` | return_based | 21 |  | cumulative log return over the trailing 21 days |
| `abs_ret_1` | return_based | 1 |  | absolute value of the most recent daily log return |
| `log_downside_vol_21` | downside | 21 |  | log annualized semi-volatility from negative returns only, 21 days |
| `volume_z_21` | volume | 21 |  | volume z-score vs its trailing 21-day mean and std |
| `log_vix` | market_wide |  |  | log of the VIX close (implied 30-day volatility) |
| `vix_chg_5` | market_wide | 5 |  | 5-day change in the VIX level |
| `vrp_21` | market_wide | 21 |  | variance risk premium proxy: VIX/100 minus own 21d realized vol |
| `log_index_rv_21` | index_context | 21 |  | log annualized 21d realized vol of the index ETF (SPY) |
| `rv_ratio_to_index_21` | index_context | 21 |  | ticker 21d realized vol divided by index 21d realized vol |
| `term_spread` | macro |  |  | 10-year minus 3-month Treasury yield (DGS10 - DGS3MO) |
| `entity_mean_log_rv` | entity |  |  | expanding-window mean of the ticker's log 21d realized vol |
| `dow` | calendar |  | yes | day of week, 0=Monday |
| `month` | calendar |  | yes | calendar month, 1-12 |

## Target

| Column | Definition |
|---|---|
| `target_fwd_rv_21` | annualized realized volatility of the next 21 trading days (level; modelled in logs) |
| `target_available_on` | date the label first becomes observable (t+21) |
