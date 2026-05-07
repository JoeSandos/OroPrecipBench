"""WLAB: Windward-Leeward Asymmetry Bias.

Evaluates model accuracy in reproducing rain-shadow effects by comparing
the *shape* of the windward/leeward precipitation profile.

Windward index: W(i,t) = ∇z(i) · V̂(t)
  W > 0 → wind blows upslope (windward), W < 0 → downslope (leeward)

Since W is dynamic (per-timestep), bin means are computed per timestep first,
then averaged over time to form the K=2 profile.  Both GT and pred profiles
are normalised to sum-to-1 before computing L² distance, making the metric
orthogonal to L1–L3 conventional error metrics (Bias, RMSE, …).

    WLAB = sqrt(0.5 * Σ_k (P̂_pred(k) - P̂_GT(k))²)

where P̂(k) = P̄(k) / Σ_j P̄(j), K=2 (windward, leeward).

Reference:
  Kunz & Kottmeier (2006, JAMC) — windward/leeward precipitation ratio.
  Smith & Barstad (2004, J. Atmos. Sci.) — linear orographic precipitation model.
  Härdle & Marron (1990, Ann. Statist.) — curve comparison framework.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom


def compute_windward_index(
    dem: np.ndarray,
    u10: np.ndarray,
    v10: np.ndarray,
) -> np.ndarray:
    """Compute per-pixel, per-timestep windward index.

    Parameters
    ----------
    dem : (H, W) elevation in meters
    u10 : (T, H_lr, W_lr) 10m eastward wind (m/s), low resolution
    v10 : (T, H_lr, W_lr) 10m northward wind (m/s), low resolution

    Returns
    -------
    windward_index : (T, H, W) float32
        Positive = windward, negative = leeward.
    """
    H, W = dem.shape
    T = u10.shape[0]
    H_lr, W_lr = u10.shape[1], u10.shape[2]

    # DEM gradient (static)
    dz_dy, dz_dx = np.gradient(dem)  # each (H, W)

    # Bilinear interpolate wind to HR and compute windward index
    zoom_y = H / H_lr
    zoom_x = W / W_lr
    windward = np.empty((T, H, W), dtype=np.float32)

    for t in range(T):
        u_hr = zoom(u10[t], (zoom_y, zoom_x), order=1)[:H, :W]
        v_hr = zoom(v10[t], (zoom_y, zoom_x), order=1)[:H, :W]

        # Normalize wind to unit vector
        speed = np.sqrt(u_hr ** 2 + v_hr ** 2)
        speed = np.maximum(speed, 1e-8)
        u_hat = u_hr / speed
        v_hat = v_hr / speed

        windward[t] = dz_dx * u_hat + dz_dy * v_hat

    return windward


def compute_wlab(
    pred: np.ndarray,
    truth: np.ndarray,
    windward_index: np.ndarray,
    *,
    land_mask: np.ndarray | None = None,
) -> dict:
    """Compute Windward-Leeward Asymmetry Bias.

    Per-timestep bin means are computed first, then averaged over time to
    form the K=2 profile.  Profiles are normalised to sum-to-1 so the
    metric captures shape (ww/lw balance) rather than absolute bias.

    Parameters
    ----------
    pred : (T, E, H, W) or (T, H, W)
    truth : (T, H, W)
    windward_index : (T, H, W)
    land_mask : (H, W) bool, optional.  If provided, only land pixels
        (True) participate in bin statistics.

    Returns
    -------
    dict with:
      'wlab_scalar' : float
      'p_gt_ww', 'p_gt_lw' : float, time-mean GT precip (windward / leeward)
      'p_pred_ww', 'p_pred_lw' : float, time-mean pred precip
      'ar_gt' : float, GT asymmetry ratio (ww/lw)
      'ar_pred' : float, pred asymmetry ratio
      'n_windward' : int, total windward (t,i,j) count
      'n_leeward' : int, total leeward (t,i,j) count
    """
    if pred.ndim == 4:
        pred_mean = np.nanmean(pred, axis=1)
    elif pred.ndim == 3:
        pred_mean = pred
    else:
        raise ValueError(f"Expected pred 3D or 4D, got {pred.ndim}D")

    T = truth.shape[0]

    if land_mask is None:
        land_mask = np.ones(truth.shape[1:], dtype=bool)

    # Per-timestep bin means
    gt_ww_t = np.full(T, np.nan)
    gt_lw_t = np.full(T, np.nan)
    pred_ww_t = np.full(T, np.nan)
    pred_lw_t = np.full(T, np.nan)

    n_ww_total = 0
    n_lw_total = 0

    for t in range(T):
        ww = (windward_index[t] > 0) & land_mask
        lw = (windward_index[t] < 0) & land_mask
        n_ww = int(ww.sum())
        n_lw = int(lw.sum())
        n_ww_total += n_ww
        n_lw_total += n_lw
        if n_ww > 0 and n_lw > 0:
            gt_ww_t[t] = truth[t][ww].mean()
            gt_lw_t[t] = truth[t][lw].mean()
            pred_ww_t[t] = pred_mean[t][ww].mean()
            pred_lw_t[t] = pred_mean[t][lw].mean()

    valid = np.isfinite(gt_ww_t)
    n_valid = int(valid.sum())

    if n_valid == 0:
        return {
            "wlab_scalar": float("nan"),
            "p_gt_ww": float("nan"), "p_gt_lw": float("nan"),
            "p_pred_ww": float("nan"), "p_pred_lw": float("nan"),
            "ar_gt": float("nan"), "ar_pred": float("nan"),
            "n_windward": 0, "n_leeward": 0,
        }

    # Time-averaged profile (K=2)
    p_gt_ww = float(np.mean(gt_ww_t[valid]))
    p_gt_lw = float(np.mean(gt_lw_t[valid]))
    p_pred_ww = float(np.mean(pred_ww_t[valid]))
    p_pred_lw = float(np.mean(pred_lw_t[valid]))

    # Normalise profiles to sum-to-1
    gt_sum = p_gt_ww + p_gt_lw
    pred_sum = p_pred_ww + p_pred_lw

    if gt_sum < 1e-10 or pred_sum < 1e-10:
        wlab = float("nan")
    else:
        gt_norm = np.array([p_gt_ww / gt_sum, p_gt_lw / gt_sum])
        pred_norm = np.array([p_pred_ww / pred_sum, p_pred_lw / pred_sum])
        wlab = float(np.sqrt(np.mean((pred_norm - gt_norm) ** 2)))

    # Asymmetry ratio AR = P_ww / P_lw
    ar_gt = p_gt_ww / p_gt_lw if p_gt_lw > 1e-10 else float("nan")
    ar_pred = p_pred_ww / p_pred_lw if p_pred_lw > 1e-10 else float("nan")

    return {
        "wlab_scalar": wlab,
        "p_gt_ww": p_gt_ww,
        "p_gt_lw": p_gt_lw,
        "p_pred_ww": p_pred_ww,
        "p_pred_lw": p_pred_lw,
        "ar_gt": float(ar_gt),
        "ar_pred": float(ar_pred),
        "n_windward": n_ww_total,
        "n_leeward": n_lw_total,
    }
