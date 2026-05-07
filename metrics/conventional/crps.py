"""Continuous Ranked Probability Score (CRPS).

Fair CRPS for ensemble predictions:
  CRPS = E|X_i - y| - 0.5 * E|X_i - X_j|

where X_i are ensemble members and y is truth.

Reference:
  Gneiting & Raftery (2007, JASA).
  Harris et al. (2022, JAMES).
Source: metrics/calc_metrics_config.py:52-77
"""
from __future__ import annotations

import numpy as np


def compute_crps(
    pred: np.ndarray,
    truth: np.ndarray,
) -> dict:
    """Compute CRPS for ensemble predictions, spatially averaged per time.

    Parameters
    ----------
    pred : (T, E, H, W) full ensemble prediction
    truth : (T, H, W) ground truth

    Returns
    -------
    dict with 'crps_per_time' (T,) and 'crps_mean' scalar
    """
    if pred.ndim != 4 or truth.ndim != 3:
        raise ValueError(
            f"Expected pred (T,E,H,W) and truth (T,H,W), "
            f"got {pred.shape} and {truth.shape}"
        )

    m = pred.shape[1]  # ensemble size

    # Term 1: mean |X_i - y| over ensemble
    term1 = np.nanmean(np.abs(pred - truth[:, None, :, :]), axis=1)  # (T, H, W)

    if m <= 1:
        crps_grid = term1
    else:
        # Term 2: mean |X_i - X_j| over all ordered pairs
        pair_sum = np.zeros_like(term1)
        for i in range(m):
            for j in range(i + 1, m):
                pair_sum += np.abs(pred[:, i] - pred[:, j])
        mean_ij = (2.0 / (m * m)) * pair_sum
        crps_grid = term1 - 0.5 * mean_ij

    crps_t = np.nanmean(crps_grid, axis=(-2, -1))

    return {
        "crps_per_time": crps_t.astype(np.float32),
        "crps_mean": float(np.nanmean(crps_t)),
    }
