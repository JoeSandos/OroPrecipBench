"""Root Mean Square Error (RMSE).

Per-time spatial RMSE between ensemble-mean prediction and truth.

Reference: Standard; Harris et al. (2022, JAMES).
Source: metrics/calc_metrics_config.py:224-226
"""
from __future__ import annotations

import numpy as np


def compute_rmse(
    pred: np.ndarray,
    truth: np.ndarray,
) -> dict:
    """Compute RMSE per time step, spatially averaged.

    Parameters
    ----------
    pred : (T, H, W) ensemble-mean prediction
    truth : (T, H, W) ground truth

    Returns
    -------
    dict with 'rmse_per_time' (T,) and 'rmse_mean' scalar
    """
    diff = pred - truth
    rmse_t = np.sqrt(np.nanmean(diff * diff, axis=(-2, -1)))
    return {
        "rmse_per_time": rmse_t.astype(np.float32),
        "rmse_mean": float(np.nanmean(rmse_t)),
    }
