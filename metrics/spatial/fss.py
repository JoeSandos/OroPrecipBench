"""Fractions Skill Score (FSS).

Computes FSS using pysteps at multiple spatial scales.

Reference:
  Roberts & Lean (2008, MWR).
Source: metrics/calc_fss_config.py:60-311
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SCALES = [1, 3, 5, 9, 17, 33, 65]


def _require_pysteps_fss():
    try:
        from pysteps.verification.spatialscores import fss
        return fss
    except ImportError as exc:
        raise ImportError("pysteps required for FSS: pip install pysteps") from exc


def compute_fss(
    pred: np.ndarray,
    truth: np.ndarray,
    *,
    scales: Sequence[int] = DEFAULT_SCALES,
    threshold: float = 0.01,
    ens_strategy: str = "mean",
) -> dict:
    """Compute FSS at multiple spatial scales.

    Parameters
    ----------
    pred : (T, E, H, W) or (T, H, W)
    truth : (T, H, W)
    scales : list of square window sizes in grid points
    threshold : precipitation threshold
    ens_strategy : 'mean' (ens-mean then FSS) or 'member_avg' (FSS per member, avg)

    Returns
    -------
    dict with 'scales', 'fss_scores' (list of floats), 'threshold'
    """
    fss_func = _require_pysteps_fss()

    if pred.ndim == 3:
        pred = pred[:, None, :, :]
    pred = pred.astype(np.float32)
    truth = truth.astype(np.float32)

    T, E, H, W = pred.shape
    scores = []

    if ens_strategy == "mean":
        pred_det = np.mean(pred, axis=1)  # (T, H, W)
        for s in scales:
            fss_vals = []
            for t in range(T):
                f = np.nan_to_num(pred_det[t], nan=0.0)
                o = np.nan_to_num(truth[t], nan=0.0)
                fss_vals.append(float(fss_func(f, o, float(threshold), int(s))))
            scores.append(float(np.nanmean(fss_vals)))

    elif ens_strategy == "member_avg":
        for s in scales:
            member_scores = []
            for m in range(E):
                fss_vals = []
                for t in range(T):
                    f = np.nan_to_num(pred[t, m], nan=0.0)
                    o = np.nan_to_num(truth[t], nan=0.0)
                    fss_vals.append(float(fss_func(f, o, float(threshold), int(s))))
                member_scores.append(float(np.nanmean(fss_vals)))
            scores.append(float(np.mean(member_scores)))
    else:
        raise ValueError(f"Unknown ens_strategy: {ens_strategy}")

    return {
        "scales": list(scales),
        "fss_scores": scores,
        "threshold": float(threshold),
        "ens_strategy": ens_strategy,
    }
