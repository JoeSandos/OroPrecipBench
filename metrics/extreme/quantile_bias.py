"""Quantile Bias (QB).

Uses climatology (train data) to define quantile thresholds, then evaluates
how well predictions reproduce the exceedance of these thresholds in the
test period.

For each grid cell (i,j) and quantile q:
  1. Q_clim(i,j) = percentile_q(train[:,i,j])  -- climatological threshold
  2. Q_pred(i,j) = percentile_q(pred[:,i,j])    -- predicted quantile in test period
  3. Absolute bias = Q_pred(i,j) - Q_clim(i,j)
  4. Mehran QB ratio = count(pred > Q_clim) / count(truth > Q_clim)

Reference:
  Mehran et al. (2014, JGR-Atmos, doi:10.1002/2013JD021152).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def compute_quantile_bias(
    pred: np.ndarray,
    truth: np.ndarray,
    *,
    quantiles: Sequence[float] = (0.90, 0.95, 0.99),
    ens_strategy: str = "per_member",
    climatology: np.ndarray | None = None,
) -> dict:
    """Compute per-pixel quantile bias against climatological thresholds.

    Parameters
    ----------
    pred : (T, E, H, W) or (T, H, W)
    truth : (T, H, W)  -- test-period truth
    quantiles : quantile levels (e.g. 0.90, 0.95, 0.99)
    ens_strategy : 'mean' (ens-mean then quantile) or 'per_member' (quantile
                   per member then average)
    climatology : (T_train, H, W) training data for computing climatological
                  quantile thresholds. If None, falls back to using truth
                  (test data) itself.

    Returns
    -------
    dict with per-quantile results:
      'q{q}_abs_bias_map' : (H, W) absolute bias (pred quantile - clim quantile)
      'q{q}_abs_bias_mean' : float, spatial mean of absolute bias
      'q{q}_ratio_map' : (H, W) Mehran QB ratio
      'q{q}_ratio_mean' : float, spatial mean of QB ratio
      'q{q}_clim_map' : (H, W) climatological quantile threshold
      'q{q}_pred_map' : (H, W) pred quantile
    """
    if pred.ndim == 3:
        pred = pred[:, None, :, :]
    T, E, H, W = pred.shape

    # Determine baseline for quantile thresholds
    clim_data = climatology if climatology is not None else truth

    results = {"quantiles": list(quantiles)}

    for q in quantiles:
        q_label = f"q{q:.2f}"

        # Climatological quantile threshold per pixel (from train data)
        clim_q = np.nanpercentile(clim_data, q * 100, axis=0)  # (H, W)

        # Predicted quantile in test period
        if ens_strategy == "mean":
            pred_mean = np.nanmean(pred, axis=1)  # (T, H, W)
            pred_q = np.nanpercentile(pred_mean, q * 100, axis=0)  # (H, W)
        elif ens_strategy == "per_member":
            member_qs = []
            for e in range(E):
                mq = np.nanpercentile(pred[:, e], q * 100, axis=0)
                member_qs.append(mq)
            pred_q = np.nanmean(member_qs, axis=0)  # (H, W)
        else:
            raise ValueError(f"Unknown ens_strategy: {ens_strategy}")

        # Absolute bias: pred quantile vs climatological quantile
        abs_bias = pred_q - clim_q  # (H, W)

        # Mehran QB ratio: exceedance of clim threshold in test period
        # count(pred > Q_clim) / count(truth > Q_clim)
        if ens_strategy == "mean":
            pred_exceed = np.sum(
                np.nanmean(pred, axis=1) > clim_q[None, :, :], axis=0
            )
        else:
            exceed_sum = np.zeros((H, W), dtype=np.float64)
            for e in range(E):
                exceed_sum += np.sum(pred[:, e] > clim_q[None, :, :], axis=0)
            pred_exceed = exceed_sum / E

        truth_exceed = np.sum(truth > clim_q[None, :, :], axis=0).astype(np.float64)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(truth_exceed > 0, pred_exceed / truth_exceed, np.nan)

        results[f"{q_label}_abs_bias_map"] = abs_bias.astype(np.float32)
        results[f"{q_label}_abs_bias_mean"] = float(np.nanmean(abs_bias))
        results[f"{q_label}_ratio_map"] = ratio.astype(np.float32)
        results[f"{q_label}_ratio_mean"] = float(np.nanmean(ratio[np.isfinite(ratio)]))
        results[f"{q_label}_clim_map"] = clim_q.astype(np.float32)
        results[f"{q_label}_pred_map"] = pred_q.astype(np.float32)

    return results
