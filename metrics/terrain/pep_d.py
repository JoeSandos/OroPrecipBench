"""PEP-D: Precipitation-Elevation Profile Discrepancy.

Evaluates how well a model reproduces the precipitation–elevation coupling by
binning pixels by DEM elevation and comparing the *shape* of the precipitation
profile (not absolute values).

Both GT and pred profiles are normalised to sum-to-1 before computing L²
distance, so a uniform global bias cancels out and the metric is orthogonal
to L1–L3 conventional error metrics (Bias, RMSE, …).

    D = sqrt(1/K * Σ_k (P̂_pred(k) - P̂_GT(k))²)

where P̂(k) = P̄(k) / Σ_j P̄(j).

Reference:
  Härdle & Marron (1990, Ann. Statist.) — non-parametric curve comparison.
  Napoli et al. (2019, Sci. Rep.) — elevation-stratified precipitation analysis.
"""
from __future__ import annotations

import numpy as np


def compute_pep_d(
    pred: np.ndarray,
    truth: np.ndarray,
    dem: np.ndarray,
    *,
    bin_width: float = 200.0,
    land_mask: np.ndarray | None = None,
) -> dict:
    """Compute Precipitation-Elevation Profile Discrepancy.

    Parameters
    ----------
    pred : (T, E, H, W) or (T, H, W)
    truth : (T, H, W)
    dem : (H, W) elevation in meters
    bin_width : elevation bin width in meters (default 200m)
    land_mask : (H, W) bool, optional.  If provided, only land pixels
        (True) participate in binning and statistics.

    Returns
    -------
    dict with:
      'pep_d_scalar' : float, normalised L2 distance of sum-to-1 profiles
      'bin_edges' : (K+1,) bin edges
      'bin_centers' : (K,) bin centers
      'bin_labels' : list of str, e.g. ["500-700m", "700-900m"]
      'p_gt_per_bin' : (K,) mean GT precipitation per bin (raw, before norm)
      'p_pred_per_bin' : (K,) mean pred precipitation per bin (raw)
      'n_pixels_per_bin' : (K,) pixel count per bin
    """
    if pred.ndim == 4:
        pred_mean = np.nanmean(pred, axis=1)  # ensemble mean -> (T, H, W)
    elif pred.ndim == 3:
        pred_mean = pred
    else:
        raise ValueError(f"Expected pred 3D or 4D, got {pred.ndim}D")

    # Land mask defaults to all True
    if land_mask is None:
        land_mask = np.ones(dem.shape, dtype=bool)

    # Elevation bin edges (over land pixels only)
    dem_land = dem[land_mask]
    dem_min = np.floor(np.nanmin(dem_land) / bin_width) * bin_width
    dem_max = np.nanmax(dem_land)
    edges = np.arange(dem_min, dem_max + bin_width, bin_width)
    K = len(edges) - 1

    # Assign pixels to bins
    bin_idx = np.digitize(dem, edges) - 1  # (H, W), 0-indexed
    bin_idx = np.clip(bin_idx, 0, K - 1)

    # Per-bin spatiotemporal mean
    p_gt_per_bin = np.full(K, np.nan, dtype=np.float64)
    p_pred_per_bin = np.full(K, np.nan, dtype=np.float64)
    n_pixels = np.zeros(K, dtype=np.int64)

    for k in range(K):
        mask = (bin_idx == k) & land_mask  # (H, W)
        n_pixels[k] = int(mask.sum())
        if n_pixels[k] == 0:
            continue
        p_gt_per_bin[k] = np.nanmean(truth[:, mask])
        p_pred_per_bin[k] = np.nanmean(pred_mean[:, mask])

    # Filter valid bins
    valid = np.isfinite(p_gt_per_bin) & np.isfinite(p_pred_per_bin)
    K_valid = int(valid.sum())

    if K_valid == 0:
        pep_d = float("nan")
    else:
        gt_v = p_gt_per_bin[valid]
        pred_v = p_pred_per_bin[valid]

        gt_sum = gt_v.sum()
        pred_sum = pred_v.sum()

        if gt_sum < 1e-10 or pred_sum < 1e-10:
            pep_d = float("nan")
        else:
            # Normalise to probability distribution (sum-to-1)
            gt_norm = gt_v / gt_sum
            pred_norm = pred_v / pred_sum
            pep_d = float(np.sqrt(np.mean((pred_norm - gt_norm) ** 2)))

    # Bin labels
    centers = (edges[:-1] + edges[1:]) / 2
    labels = [f"{int(edges[k])}-{int(edges[k+1])}m" for k in range(K)]

    return {
        "pep_d_scalar": pep_d,
        "bin_edges": edges.astype(np.float32),
        "bin_centers": centers.astype(np.float32),
        "bin_labels": labels,
        "p_gt_per_bin": p_gt_per_bin.astype(np.float32),
        "p_pred_per_bin": p_pred_per_bin.astype(np.float32),
        "n_pixels_per_bin": n_pixels,
    }
