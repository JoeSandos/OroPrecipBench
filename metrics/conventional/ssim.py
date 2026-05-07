"""Structural Similarity Index (SSIM).

Per-time SSIM between ensemble-mean prediction and truth.

Reference: Wang et al. (2004, IEEE TIP).
Source: metrics/calc_metrics_config.py:266-293
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from skimage.metrics import structural_similarity as ssim_skimage


def compute_ssim(
    pred: np.ndarray,
    truth: np.ndarray,
    *,
    workers: int = 4,
) -> dict:
    """Compute SSIM per time step.

    Parameters
    ----------
    pred : (T, H, W) ensemble-mean prediction
    truth : (T, H, W) ground truth
    workers : parallel threads for SSIM computation

    Returns
    -------
    dict with 'ssim_per_time' (T,) and 'ssim_mean' scalar
    """
    T = pred.shape[0]

    def _ssim_one(t: int) -> float:
        ref = np.nan_to_num(truth[t], copy=True)
        prd = np.nan_to_num(pred[t], copy=True)
        data_range = float(max(ref.max(), prd.max()) - min(ref.min(), prd.min()))
        if data_range <= 0:
            data_range = 1.0
        return float(ssim_skimage(ref, prd, data_range=data_range))

    if workers > 1 and T > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            ssim_vals = list(ex.map(_ssim_one, range(T)))
    else:
        ssim_vals = [_ssim_one(t) for t in range(T)]

    ssim_t = np.array(ssim_vals, dtype=np.float32)
    return {
        "ssim_per_time": ssim_t,
        "ssim_mean": float(np.nanmean(ssim_t)),
    }
