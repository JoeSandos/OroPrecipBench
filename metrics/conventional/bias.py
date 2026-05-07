"""Mean Bias.

Per-time spatial mean bias between ensemble-mean prediction and truth.

Reference: Standard.
"""
from __future__ import annotations

import numpy as np


def compute_bias(
    pred: np.ndarray,
    truth: np.ndarray,
) -> dict:
    """Compute mean bias per time step, spatially averaged.

    Parameters
    ----------
    pred : (T, H, W)
    truth : (T, H, W)

    Returns
    -------
    dict with 'bias_per_time' (T,) and 'bias_mean' scalar
    """
    bias_t = np.nanmean(pred - truth, axis=(-2, -1))
    return {
        "bias_per_time": bias_t.astype(np.float32),
        "bias_mean": float(np.nanmean(bias_t)),
    }
