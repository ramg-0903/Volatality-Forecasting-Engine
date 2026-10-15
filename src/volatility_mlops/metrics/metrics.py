"""Volatility forecast metrics (build-plan section 6.3).

All functions take annualized volatility *levels* (``y_true``, ``y_pred``) as
array-likes. QLIKE is defined on the corresponding variances (the squares). Pairs
with a non-finite or non-positive value are dropped before computing.

- **MAE / RMSE**: level error.
- **QLIKE**: a variance loss robust to the noisy volatility proxy; 0 is perfect,
  lower is better. ``QLIKE = mean( v_true/v_pred - ln(v_true/v_pred) - 1 )``.
- **Mincer-Zarnowitz**: regress realized on predicted (``y_true = a + b*y_pred``);
  a well-calibrated forecast has slope ``b`` near 1 and high R-squared.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

_ArrayLike = npt.ArrayLike


@dataclass(frozen=True)
class Metrics:
    """Bundle of forecast metrics for one model/fold."""

    mae: float
    rmse: float
    qlike: float
    mz_slope: float
    mz_r2: float
    n: int


def _clean_pair(y_true: _ArrayLike, y_pred: _ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return finite, strictly-positive aligned arrays of true/predicted vols."""
    t = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(t) & np.isfinite(p) & (t > 0) & (p > 0)
    return t[mask], p[mask]


def mae(y_true: _ArrayLike, y_pred: _ArrayLike) -> float:
    """Mean absolute error of volatility levels."""
    t, p = _clean_pair(y_true, y_pred)
    return float(np.mean(np.abs(t - p)))


def rmse(y_true: _ArrayLike, y_pred: _ArrayLike) -> float:
    """Root mean squared error of volatility levels."""
    t, p = _clean_pair(y_true, y_pred)
    return float(np.sqrt(np.mean((t - p) ** 2)))


def qlike(y_true: _ArrayLike, y_pred: _ArrayLike) -> float:
    """QLIKE loss on variances (0 is perfect; lower is better)."""
    t, p = _clean_pair(y_true, y_pred)
    ratio = np.square(t) / np.square(p)
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def mincer_zarnowitz(y_true: _ArrayLike, y_pred: _ArrayLike) -> tuple[float, float]:
    """Return the Mincer-Zarnowitz slope and R-squared of ``y_true ~ a + b*y_pred``."""
    t, p = _clean_pair(y_true, y_pred)
    design = np.vstack([np.ones_like(p), p]).T
    coef, *_ = np.linalg.lstsq(design, t, rcond=None)
    slope = float(coef[1])
    fitted = design @ coef
    ss_res = float(np.sum((t - fitted) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, r2


def compute_metrics(y_true: _ArrayLike, y_pred: _ArrayLike) -> Metrics:
    """Compute all metrics at once."""
    t, p = _clean_pair(y_true, y_pred)
    slope, r2 = mincer_zarnowitz(t, p)
    return Metrics(
        mae=mae(t, p),
        rmse=rmse(t, p),
        qlike=qlike(t, p),
        mz_slope=slope,
        mz_r2=r2,
        n=int(t.size),
    )
