"""Probability of Detection (POD) and False Alarm Ratio (FAR).

Fixed-threshold binary verification metrics with macro aggregation
(per-time scores then average).

Reference: Wilks (2011, Statistical Methods in Atmospheric Sciences).
Source: metrics/calc_metrics_config.py:306-364
"""
from __future__ import annotations

import numpy as np


def _contingency_per_time(
    pred: np.ndarray,
    truth: np.ndarray,
    threshold: float,
) -> dict:
    """Compute per-time contingency table counts.

    Parameters
    ----------
    pred : (T, H, W)
    truth : (T, H, W)
    threshold : rain threshold

    Returns
    -------
    dict of (T,) arrays: hits, misses, false_alarms, correct_negatives
    """
    finite = np.isfinite(truth) & np.isfinite(pred)
    truth_event = (truth >= threshold) & finite
    pred_event = (pred >= threshold) & finite

    hits = (truth_event & pred_event).sum(axis=(-2, -1))
    misses = (truth_event & ~pred_event).sum(axis=(-2, -1))
    false_alarms = (~truth_event & pred_event).sum(axis=(-2, -1))
    correct_negatives = (~truth_event & ~pred_event & finite).sum(axis=(-2, -1))

    return {
        "hits": hits.astype(np.float64),
        "misses": misses.astype(np.float64),
        "false_alarms": false_alarms.astype(np.float64),
        "correct_negatives": correct_negatives.astype(np.float64),
    }


def compute_pod_far(
    pred: np.ndarray,
    truth: np.ndarray,
    threshold: float = 0.01,
) -> dict:
    """Compute POD and FAR with macro aggregation (per-time then average).

    Parameters
    ----------
    pred : (T, H, W) ensemble-mean prediction
    truth : (T, H, W) ground truth
    threshold : rain threshold (mm/h or mm/day)

    Returns
    -------
    dict with 'pod_per_time' (T,), 'far_per_time' (T,),
    'pod_mean', 'far_mean' scalars
    """
    counts = _contingency_per_time(pred, truth, threshold)
    h = counts["hits"]
    m = counts["misses"]
    fa = counts["false_alarms"]

    with np.errstate(divide="ignore", invalid="ignore"):
        pod_t = np.where(h + m > 0, h / (h + m), 0.0)
        far_t = np.where(h + fa > 0, fa / (h + fa), 0.0)

    return {
        "pod_per_time": pod_t.astype(np.float32),
        "far_per_time": far_t.astype(np.float32),
        "pod_mean": float(np.mean(pod_t)),
        "far_mean": float(np.mean(far_t)),
        "threshold": float(threshold),
    }
