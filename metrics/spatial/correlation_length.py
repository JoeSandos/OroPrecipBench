"""Spatial Correlation Length.

Estimates precipitation correlation length L using Spearman rank correlation
and a Gaussian decay model: rho(d) = exp(-d^2 / L^2).

For each target pixel (on stride grid), sample K neighbors within a radius,
compute Spearman rho, then fit L via linearized LS:
  y = -ln(rho) = (1/L^2) * d^2

Reference: Sinclair & Pegram (2005, JHM).
Source: metrics/corr_length/calc_corr_length.py:323-515
"""
from __future__ import annotations

import logging
import math
import zlib
from typing import Optional

import numpy as np
from scipy.stats import rankdata

logger = logging.getLogger(__name__)


def _compute_sigma_valid(field: np.ndarray):
    """Return (sigma, valid_mask) for (T,H,W) field."""
    finite_all = np.isfinite(field).all(axis=0)
    sigma = np.nanstd(field, axis=0)
    valid = finite_all & np.isfinite(sigma) & (sigma > 0)
    return sigma, valid


def _disk_offsets(radius_pix: int):
    rr = radius_pix
    dy, dx = np.meshgrid(
        np.arange(-rr, rr + 1, dtype=np.int32),
        np.arange(-rr, rr + 1, dtype=np.int32),
        indexing="ij",
    )
    d2 = dy * dy + dx * dx
    mask = (d2 <= rr * rr) & ~((dy == 0) & (dx == 0))
    offsets = np.stack([dy[mask], dx[mask]], axis=1).astype(np.int32)
    d2_pix = d2[mask].astype(np.int32)
    return offsets, d2_pix


def _sample_neighbors(
    H: int, W: int, target_yx, valid_flat, offsets, d2_pix,
    k_neighbors: int, km_per_pixel: float, seed: int,
):
    """Sample K valid neighbors within the disk for each target."""
    rng = np.random.default_rng(seed)
    n_targets = target_yx.shape[0]
    neigh_idx = np.full((n_targets, k_neighbors), -1, dtype=np.int64)
    neigh_d2 = np.full((n_targets, k_neighbors), np.nan, dtype=np.float32)
    max_attempts = 50 * k_neighbors

    for i in range(n_targets):
        ty, tx = int(target_yx[i, 0]), int(target_yx[i, 1])
        chosen = 0
        seen = set()
        for _ in range(max_attempts):
            if chosen >= k_neighbors:
                break
            j = int(rng.integers(0, offsets.shape[0]))
            ny, nx = ty + int(offsets[j, 0]), tx + int(offsets[j, 1])
            if ny < 0 or ny >= H or nx < 0 or nx >= W:
                continue
            flat = ny * W + nx
            if flat in seen or not valid_flat[flat]:
                continue
            seen.add(flat)
            neigh_idx[i, chosen] = flat
            neigh_d2[i, chosen] = float(d2_pix[j]) * km_per_pixel * km_per_pixel
            chosen += 1

    return neigh_idx, neigh_d2


def compute_correlation_length(
    field: np.ndarray,
    *,
    stride: int = 2,
    k_neighbors: int = 20,
    radius_km: float = 100.0,
    km_per_pixel: float = 3.0,
    seed: int = 0,
) -> dict:
    """Compute correlation length L on a stride grid.

    Parameters
    ----------
    field : (T, H, W) precipitation field (single member, NOT ens-mean)
    stride : stride for target grid
    k_neighbors : number of random neighbors per target
    radius_km : search radius in km
    km_per_pixel : grid spacing in km
    seed : random seed

    Returns
    -------
    dict with 'L_map' (H2, W2), 'L_stats' dict, 'stride'
    """
    if field.ndim != 3:
        raise ValueError(f"Expected (T,H,W), got shape {field.shape}")

    # Replace NaN with 0 (e.g. sea pixels masked by land-sea mask) then
    # filter to timesteps that were all-finite *before* fill.
    nan_mask = ~np.isfinite(field)
    if nan_mask.any():
        field = field.copy()
        field[nan_mask] = 0.0
    finite_t = np.isfinite(field).all(axis=(1, 2))
    if finite_t.sum() < 3:
        raise RuntimeError(f"Too few all-finite timesteps ({finite_t.sum()}) for Spearman")
    field = field[finite_t]

    T, H, W = field.shape
    sigma, valid_mask = _compute_sigma_valid(field)
    valid_flat = valid_mask.ravel()

    # Build targets
    ys = np.arange(0, H, stride, dtype=np.int32)
    xs = np.arange(0, W, stride, dtype=np.int32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    target_yx = np.stack([yy.ravel(), xx.ravel()], axis=1)
    H2, W2 = len(ys), len(xs)

    # Disk offsets
    radius_pix = max(1, int(math.floor(radius_km / max(km_per_pixel, 1e-12))))
    offsets, d2_pix = _disk_offsets(radius_pix)

    # Sample neighbors
    neigh_idx, neigh_d2 = _sample_neighbors(
        H, W, target_yx, valid_flat, offsets, d2_pix,
        k_neighbors, km_per_pixel, seed,
    )

    # Precompute ranks and z-scores for all valid pixels
    col_to_pix = np.nonzero(valid_flat)[0].astype(np.int64)
    pix_to_col = np.full(H * W, -1, dtype=np.int32)
    pix_to_col[col_to_pix] = np.arange(len(col_to_pix), dtype=np.int32)

    flat_field = field.reshape(T, H * W)[:, col_to_pix]
    ranks = rankdata(flat_field, axis=0, method="average").astype(np.float32)
    mean_r = ranks.mean(axis=0, keepdims=True)
    std_r = ranks.std(axis=0, ddof=1, keepdims=True)
    std_r = np.where(std_r == 0, np.nan, std_r)
    z = ((ranks - mean_r) / std_r).astype(np.float32)

    # Compute L for each target
    target_flat = (target_yx[:, 0].astype(np.int64) * W + target_yx[:, 1].astype(np.int64))
    target_col = pix_to_col[target_flat]
    neigh_col = np.where(
        neigh_idx >= 0,
        pix_to_col[np.clip(neigh_idx, 0, H * W - 1)],
        -1,
    )

    L_low = np.full((H2, W2), np.nan, dtype=np.float32)
    n_targets = target_yx.shape[0]
    K = k_neighbors
    denom = float(max(T - 1, 1))

    chunk = 512
    for start in range(0, n_targets, chunk):
        end = min(start + chunk, n_targets)
        tc = target_col[start:end]
        nc = neigh_col[start:end]
        d2 = neigh_d2[start:end]
        low_y = target_yx[start:end, 0] // stride
        low_x = target_yx[start:end, 1] // stride

        valid_t = tc >= 0
        tc_safe = np.where(valid_t, tc, 0)
        zt = z[:, tc_safe]  # (T, chunk)
        nc_safe = np.where(nc >= 0, nc, 0)
        zn = z[:, nc_safe]  # (T, chunk, K)

        rho = (zt[:, :, None] * zn).sum(axis=0) / denom  # (chunk, K)
        valid_pair = valid_t[:, None] & (nc >= 0) & np.isfinite(d2)
        valid_rho = valid_pair & np.isfinite(rho) & (rho > 0) & (rho < 1)

        y_fit = np.where(valid_rho, -np.log(np.clip(rho, 1e-12, 1 - 1e-12)), 0.0).astype(np.float32)
        x_fit = d2.astype(np.float32)

        num = np.sum(np.where(valid_rho, x_fit * y_fit, 0.0), axis=1)
        den = np.sum(np.where(valid_rho, x_fit * x_fit, 0.0), axis=1)
        cnt = valid_rho.sum(axis=1)

        a = np.where((den > 0) & (cnt >= 2), num / den, np.nan)
        L = np.where(a > 0, 1.0 / np.sqrt(a), np.nan).astype(np.float32)

        for ii in range(end - start):
            ly, lx = int(low_y[ii]), int(low_x[ii])
            if ly < H2 and lx < W2:
                L_low[ly, lx] = L[ii]

    # Stats
    v = L_low[np.isfinite(L_low)]
    stats = {
        "mean": float(np.mean(v)) if v.size > 0 else float("nan"),
        "median": float(np.median(v)) if v.size > 0 else float("nan"),
        "std": float(np.std(v)) if v.size > 0 else float("nan"),
    }

    return {
        "L_map": L_low,
        "L_stats": stats,
        "stride": stride,
        "T_used": int(T),
        "shape_full": (H, W),
    }


def select_single_member(
    pred: np.ndarray,
    label: str = "",
    seed: int = 0,
) -> np.ndarray:
    """Select one random ensemble member deterministically.

    Parameters
    ----------
    pred : (T, E, H, W)
    label : used for deterministic member selection
    seed : base seed

    Returns
    -------
    (T, H, W) single member field
    """
    if pred.ndim != 4:
        raise ValueError(f"Expected (T,E,H,W), got {pred.shape}")
    E = pred.shape[1]
    if E <= 1:
        return pred[:, 0]
    seed2 = (seed + int(zlib.crc32(label.encode("utf-8")))) % (2 ** 32)
    member = int(np.random.default_rng(seed2).integers(0, E))
    logger.info("Selected ensemble member %d/%d for '%s'", member, E, label)
    return pred[:, member]
