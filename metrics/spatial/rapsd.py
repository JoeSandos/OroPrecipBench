"""Radially Averaged Power Spectral Density (RAPSD).

Computes RAPSD using pysteps for both prediction and truth, then evaluates
the log-MSE spectral distance. Also computes normalized RAPSD (per-image sum=1).

Reference:
  Glawion et al. (2025, npj Clim Atmos Sci).
  Harris et al. (2022, JAMES).
Source: metrics/plot_psd_config.py:270-409
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_EPS = 1e-15


def _require_pysteps():
    try:
        from pysteps.utils import spectral
        return spectral
    except ImportError as exc:
        raise ImportError("pysteps required for RAPSD: pip install pysteps") from exc


def compute_rapsd(
    pred: np.ndarray,
    truth: np.ndarray,
    *,
    pixel_size_km: float = 3.0,
    fft_backend: str = "scipy",
) -> dict:
    """Compute RAPSD metrics.

    Parameters
    ----------
    pred : (T, E, H, W) full ensemble prediction
    truth : (T, H, W) ground truth
    pixel_size_km : grid spacing in km
    fft_backend : 'scipy', 'numpy', or 'pyfftw'

    Returns
    -------
    dict with:
      'score_log_mse' : float, mean log-MSE across time
      'freqs' : (N,) frequency bins
      'wavelengths' : (N,) in km
      'raps_gt_mean' : (N,) time-averaged GT RAPSD
      'raps_pred_mean' : (N,) time-averaged pred (ens-mean) RAPSD
      'raps_gt_norm_mean' : (N,) normalized GT
      'raps_pred_norm_mean' : (N,) normalized pred
    """
    spectral = _require_pysteps()
    fft_method = _get_fft_method(fft_backend)

    T = truth.shape[0]
    E = pred.shape[1]
    H, W = truth.shape[-2:]

    # Get frequency bins
    dummy = np.zeros((H, W), dtype=np.float32)
    _, freqs = spectral.rapsd(dummy, fft_method=fft_method, return_freq=True, d=pixel_size_km)
    n_freqs = len(freqs)

    raps_gt_accum = np.zeros(n_freqs, dtype=np.float64)
    raps_gt_norm_accum = np.zeros(n_freqs, dtype=np.float64)
    raps_pred_accum = np.zeros(n_freqs, dtype=np.float64)
    raps_pred_norm_accum = np.zeros(n_freqs, dtype=np.float64)

    log_mse_scores = []
    valid_gt = 0
    valid_pred = 0

    ens_raps = np.zeros(n_freqs, dtype=np.float64)

    for t in range(T):
        # Ground truth
        gt_frame = np.nan_to_num(truth[t], nan=0.0).astype(np.float64)
        gt_raps, _ = spectral.rapsd(gt_frame, fft_method=fft_method, return_freq=True, d=pixel_size_km)
        gt_sum = np.sum(gt_raps)

        if gt_sum > _EPS:
            raps_gt_accum += gt_raps
            raps_gt_norm_accum += gt_raps / gt_sum
            valid_gt += 1

        # Prediction ensemble
        ens_raps.fill(0.0)
        for e in range(E):
            pred_frame = np.nan_to_num(pred[t, e], nan=0.0).astype(np.float64)
            pred_raps, _ = spectral.rapsd(pred_frame, fft_method=fft_method, return_freq=True, d=pixel_size_km)
            pred_sum = np.sum(pred_raps)

            raps_pred_accum += pred_raps
            if pred_sum > _EPS:
                raps_pred_norm_accum += pred_raps / pred_sum
            valid_pred += 1
            ens_raps += pred_raps

        # Log-MSE between ensemble-mean RAPSD and GT
        if gt_sum > _EPS:
            ens_mean_raps = ens_raps / max(E, 1)
            mse = float(np.mean(
                (np.log10(ens_mean_raps + _EPS) - np.log10(gt_raps + _EPS)) ** 2
            ))
            log_mse_scores.append(mse)

    div_gt = max(valid_gt, 1)
    div_pred = max(valid_pred, 1)

    wavelengths = np.full_like(freqs, np.inf)
    mask = freqs > _EPS
    wavelengths[mask] = 1.0 / freqs[mask]

    return {
        "score_log_mse": float(np.nanmean(log_mse_scores)) if log_mse_scores else float("nan"),
        "freqs": freqs.astype(np.float64),
        "wavelengths": wavelengths.astype(np.float64),
        "raps_gt_mean": (raps_gt_accum / div_gt).astype(np.float64),
        "raps_pred_mean": (raps_pred_accum / div_pred).astype(np.float64),
        "raps_gt_norm_mean": (raps_gt_norm_accum / div_gt).astype(np.float64),
        "raps_pred_norm_mean": (raps_pred_norm_accum / div_pred).astype(np.float64),
    }


def _get_fft_method(backend: str):
    backend = backend.lower().strip()
    if backend == "scipy":
        import scipy.fft as fft_mod
        return fft_mod
    if backend == "numpy":
        import numpy.fft as fft_mod
        return fft_mod
    if backend == "pyfftw":
        try:
            import pyfftw
            pyfftw.interfaces.cache.enable()
            from pyfftw.interfaces import scipy_fft as fft_mod
            return fft_mod
        except ImportError:
            logger.warning("pyfftw not available, falling back to scipy")
            import scipy.fft as fft_mod
            return fft_mod
    raise ValueError(f"Unknown FFT backend: {backend}")
