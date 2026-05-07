"""Precipitation Intensity-Frequency Distribution (PDF).

Histogram of wet-day precipitation intensities for truth and prediction,
enabling PDF tail comparison.

Reference: Glawion et al. (2025, npj Clim Atmos Sci).
Source: metrics/plot_precip_distribution_config.py:205-234
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def compute_precip_pdf(
    pred: np.ndarray,
    truth: np.ndarray,
    *,
    wet_threshold: float = 0.01,
    bins_max: float = 200.0,
    bins_width: float = 2.0,
    ens_strategy: str = "mean",
) -> dict:
    """Compute precipitation intensity histogram.

    Parameters
    ----------
    pred : (T, E, H, W) or (T, H, W)
    truth : (T, H, W)
    wet_threshold : minimum value to include (wet-day filter)
    bins_max : histogram upper bound
    bins_width : bin width
    ens_strategy : 'mean' or 'first' for ensemble reduction

    Returns
    -------
    dict with 'bins' (N+1,), 'truth_counts' (N,), 'pred_counts' (N,),
    'bin_centers' (N,)
    """
    bins = np.arange(0.0, bins_max + bins_width, bins_width, dtype=np.float64)

    # Truth histogram
    truth_flat = truth[np.isfinite(truth)]
    truth_wet = truth_flat[truth_flat >= wet_threshold]
    truth_counts, _ = np.histogram(truth_wet, bins=bins)

    # Pred histogram
    if pred.ndim == 4:
        if ens_strategy == "mean":
            pred_2d = np.nanmean(pred, axis=1)
        else:
            pred_2d = pred[:, 0]  # first member
    else:
        pred_2d = pred

    pred_flat = pred_2d[np.isfinite(pred_2d)]
    pred_wet = pred_flat[pred_flat >= wet_threshold]
    pred_counts, _ = np.histogram(pred_wet, bins=bins)

    bin_centers = (bins[:-1] + bins[1:]) / 2

    return {
        "bins": bins.astype(np.float64),
        "bin_centers": bin_centers.astype(np.float64),
        "truth_counts": truth_counts.astype(np.int64),
        "pred_counts": pred_counts.astype(np.int64),
        "wet_threshold": float(wet_threshold),
        "truth_n_wet": int(truth_wet.size),
        "pred_n_wet": int(pred_wet.size),
    }
