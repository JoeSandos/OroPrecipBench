"""Mean Absolute Error (MAE).

Per-time spatial MAE between ensemble-mean prediction and truth.

Reference: Mardani et al. (2025, Commun Earth Environ).
Source: metrics/calc_metrics_config.py:229-231
"""
from __future__ import annotations

import numpy as np


def compute_mae(
    pred: np.ndarray,
    truth: np.ndarray,
) -> dict:
    """Compute MAE per time step, spatially averaged.

    Parameters
    ----------
    pred : (T, H, W) ensemble-mean prediction
    truth : (T, H, W) ground truth

    Returns
    -------
    dict with 'mae_per_time' (T,) and 'mae_mean' scalar
    """
    mae_t = np.nanmean(np.abs(pred - truth), axis=(-2, -1))
    return {
        "mae_per_time": mae_t.astype(np.float32),
        "mae_mean": float(np.nanmean(mae_t)),
    }
