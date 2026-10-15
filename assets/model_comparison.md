# Model comparison

Walk-forward evaluation (git `8b22cfa`), mean across folds, sorted by QLIKE.

| model | MAE | RMSE | QLIKE | MZ slope | MZ R2 |
|---|---|---|---|---|---|
| challenger | 0.0664 | 0.0977 | 0.2406 | 1.076 | 0.423 |
| garch | 0.0675 | 0.1016 | 0.2428 | 0.759 | 0.320 |
| ewma | 0.0759 | 0.1135 | 0.2751 | 0.578 | 0.323 |
| har_rv | 0.0695 | 0.1062 | 0.3234 | 1.008 | 0.280 |
| persistence | 0.0794 | 0.1207 | 0.3387 | 0.493 | 0.267 |

**Best challenger hyperparameters:** `{'learning_rate': 0.05, 'max_iter': 300, 'max_leaf_nodes': 31, 'min_samples_leaf': 50, 'l2_regularization': 0.0}`

## Honest read

- Challenger QLIKE vs HAR-RV: **+25.6%** (negative means worse than the baseline).
- Challenger QLIKE vs GARCH: **+0.9%**.
- QLIKE is the primary metric; MZ slope near 1 indicates good calibration.
- Baselines are strong by design; the operational loop (weekly retrain, promotion gate, monitoring) is the project's real contribution.
