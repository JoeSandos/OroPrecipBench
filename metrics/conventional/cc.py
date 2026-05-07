"""Spatial Pearson Correlation Coefficient (CC).

Per-time spatial Pearson CC: for each time step, flatten the 2D fields and
compute Pearson r between prediction and truth. Also computes CC on the
climatology (time-mean) fields.

Reference:
  Aich et al. (2024/2026, GMD, doi:10.5194/gmd-19-1791-2026)
  Murphy (1988, MWR) for spatial pattern verification.
"""
from __future__ import annotations

import numpy as np


def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two 1D arrays, NaN-aware."""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return np.nan
    a_m, b_m = a[mask], b[mask]
    a_m = a_m - a_m.mean()
    b_m = b_m - b_m.mean()
    denom = np.sqrt(np.sum(a_m * a_m) * np.sum(b_m * b_m))
    if denom == 0:
        return np.nan
    return float(np.sum(a_m * b_m) / denom)


def compute_cc(
    pred: np.ndarray,
    truth: np.ndarray,
) -> dict:
    """Compute per-time spatial Pearson CC.

    Parameters
    ----------
    pred : (T, H, W) ensemble-mean prediction
    truth : (T, H, W) ground truth

    Returns
    -------
    dict with 'cc_per_time' (T,), 'cc_mean' scalar, 'cc_climatology' scalar
    """
    T = pred.shape[0]
    cc_t = np.empty(T, dtype=np.float32)

    for t in range(T):
        cc_t[t] = _pearson_r(pred[t].ravel(), truth[t].ravel())

    # Climatology CC: Pearson r between time-mean fields
    pred_clim = np.nanmean(pred, axis=0)
    truth_clim = np.nanmean(truth, axis=0)
    cc_clim = _pearson_r(pred_clim.ravel(), truth_clim.ravel())

    return {
        "cc_per_time": cc_t,
        "cc_mean": float(np.nanmean(cc_t)),
        "cc_climatology": float(cc_clim),
    }
