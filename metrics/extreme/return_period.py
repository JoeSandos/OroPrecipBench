"""Return Period Estimation Bias.

GEV fitting on annual maxima to estimate return levels at specified return
periods, then compute bias between prediction and truth.

Part 2 (daily) only. Skips if fewer than min_years of data.

Reference:
  Coles (2001, An Introduction to Statistical Modeling of Extreme Values,
          Springer).
  Aich et al. (2026, GMD, doi:10.5194/gmd-19-1791-2026).

Implementation uses scipy.stats.genextreme.
Note: scipy's shape parameter 'c' has opposite sign to the standard
GEV convention (Coles 2001). We store both for clarity.
"""
from __future__ import annotations

import logging
import warnings
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _extract_annual_max(daily: np.ndarray, n_days_per_year: int = 365) -> np.ndarray:
    """Extract annual maxima from daily time series.

    Parameters
    ----------
    daily : (T,) daily precipitation

    Returns
    -------
    (n_years,) annual maxima
    """
    T = daily.shape[0]
    n_years = T // n_days_per_year
    if n_years < 1:
        return np.array([], dtype=np.float64)
    trimmed = daily[:n_years * n_days_per_year].reshape(n_years, n_days_per_year)
    return np.nanmax(trimmed, axis=1)


def compute_return_period_bias(
    pred: np.ndarray,
    truth: np.ndarray,
    *,
    return_periods: Sequence[float] = (2, 5, 10, 20, 50, 100),
    min_years: int = 10,
    ens_strategy: str = "per_member",
) -> dict:
    """Compute return period estimation bias.

    Parameters
    ----------
    pred : (T, E, H, W) or (T, H, W), daily precipitation
    truth : (T, H, W), daily precipitation
    return_periods : years
    min_years : minimum annual maxima needed for GEV fit
    ens_strategy : 'mean' or 'per_member'

    Returns
    -------
    dict with per-return-period bias maps and summaries, or
    error message if insufficient data.
    """
    from scipy.stats import genextreme

    if pred.ndim == 3:
        pred = pred[:, None, :, :]
    T, E, H, W = pred.shape

    # Check if enough years
    n_years = T // 365
    if n_years < min_years:
        msg = f"Insufficient data: {n_years} years < min_years={min_years}. Skipping."
        logger.warning(msg)
        return {"error": msg, "n_years": n_years, "min_years": min_years}

    rp_arr = np.array(return_periods, dtype=np.float64)
    quantiles = 1.0 - 1.0 / rp_arr
    n_rp = len(rp_arr)

    truth_rl = np.full((n_rp, H, W), np.nan, dtype=np.float32)
    pred_rl = np.full((n_rp, H, W), np.nan, dtype=np.float32)

    n_fit_success = 0
    n_fit_fail = 0

    for i in range(H):
        for j in range(W):
            # Truth
            truth_am = _extract_annual_max(truth[:, i, j])
            truth_am = truth_am[np.isfinite(truth_am)]
            if len(truth_am) < min_years:
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    c_t, loc_t, scale_t = genextreme.fit(truth_am)
                truth_rl[:, i, j] = genextreme.ppf(quantiles, c_t, loc=loc_t, scale=scale_t)
            except Exception:
                n_fit_fail += 1
                continue

            # Prediction
            if ens_strategy == "mean":
                pred_am = _extract_annual_max(np.nanmean(pred[:, :, i, j], axis=1))
                pred_am = pred_am[np.isfinite(pred_am)]
                if len(pred_am) < min_years:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        c_p, loc_p, scale_p = genextreme.fit(pred_am)
                    pred_rl[:, i, j] = genextreme.ppf(quantiles, c_p, loc=loc_p, scale=scale_p)
                    n_fit_success += 1
                except Exception:
                    n_fit_fail += 1

            elif ens_strategy == "per_member":
                member_rls = []
                for e in range(E):
                    am = _extract_annual_max(pred[:, e, i, j])
                    am = am[np.isfinite(am)]
                    if len(am) < min_years:
                        continue
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            c_p, loc_p, scale_p = genextreme.fit(am)
                        member_rls.append(genextreme.ppf(quantiles, c_p, loc=loc_p, scale=scale_p))
                    except Exception:
                        pass
                if member_rls:
                    pred_rl[:, i, j] = np.nanmean(member_rls, axis=0)
                    n_fit_success += 1
                else:
                    n_fit_fail += 1

    logger.info("GEV fitting: %d success, %d fail out of %d pixels", n_fit_success, n_fit_fail, H * W)

    # Bias maps
    bias = pred_rl - truth_rl  # (n_rp, H, W)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_bias = np.where(
            np.abs(truth_rl) > 1e-6,
            (pred_rl - truth_rl) / truth_rl * 100,
            np.nan,
        )

    results = {
        "return_periods": rp_arr.tolist(),
        "n_years": n_years,
        "n_fit_success": n_fit_success,
        "n_fit_fail": n_fit_fail,
    }

    for k, rp in enumerate(rp_arr):
        rp_label = f"rp{int(rp)}"
        results[f"{rp_label}_bias_map"] = bias[k].astype(np.float32)
        results[f"{rp_label}_bias_mean"] = float(np.nanmean(bias[k]))
        results[f"{rp_label}_rel_bias_mean"] = float(np.nanmean(rel_bias[k][np.isfinite(rel_bias[k])]))
        results[f"{rp_label}_truth_rl_mean"] = float(np.nanmean(truth_rl[k][np.isfinite(truth_rl[k])]))
        results[f"{rp_label}_pred_rl_mean"] = float(np.nanmean(pred_rl[k][np.isfinite(pred_rl[k])]))

    return results
