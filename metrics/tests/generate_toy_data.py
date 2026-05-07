#!/usr/bin/env python3
"""Generate toy NetCDF datasets for benchmark metrics testing.

Creates a complete test domain directory with:
  - train_data_out.nc  (history, for return period reference)
  - test_data_out.nc   (truth)
  - static_variables.nc (elevation + lsm)
  - bilinear_pred.nc   (deterministic baseline, ensemble=1)
  - baseline1_pred.nc  (deterministic model, ensemble=1)
  - baseline2_pred.nc  (ensemble model, ensemble=4)

All files follow dataset_policy Chapter 5 format:
  Dims: (time, [ensemble,] latitude, longitude)
  Variable: tp (mm/h)
  Coords: 2D lat/lon arrays, time as datetime64

Usage:
    python -m metrics.tests.generate_toy_data \
        --out-dir /tmp/toy_domain --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def _make_coords(H: int, W: int, lat0: float = 39.0, lon0: float = 254.0, dx_deg: float = 0.03):
    """Create 2D lat/lon coordinate arrays (matching dataset_policy)."""
    lat_1d = np.linspace(lat0, lat0 + H * dx_deg, H, dtype=np.float32)
    lon_1d = np.linspace(lon0, lon0 + W * dx_deg, W, dtype=np.float32)
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
    return lat_2d, lon_2d


def _make_time(n: int, start: str = "2024-01-01", freq: str = "h"):
    """Create datetime64 time coordinate."""
    return np.arange(
        np.datetime64(start),
        np.datetime64(start) + np.timedelta64(n, freq),
        np.timedelta64(1, freq),
    )[:n]


def _make_elevation(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    """Synthetic elevation: smooth Gaussian peaks to simulate mountains."""
    y = np.linspace(0, 1, H, dtype=np.float32)
    x = np.linspace(0, 1, W, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    elev = (
        2000 * np.exp(-((yy - 0.3) ** 2 + (xx - 0.5) ** 2) / 0.05)
        + 1500 * np.exp(-((yy - 0.7) ** 2 + (xx - 0.3) ** 2) / 0.08)
        + 500
        + rng.normal(0, 50, (H, W))
    )
    return np.clip(elev, 0, None).astype(np.float32)


def _make_truth(T: int, H: int, W: int, elev: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate synthetic precipitation truth with orographic enhancement.

    Uses exponential distribution with spatially varying rate influenced by
    elevation. Adds temporal autocorrelation via a simple AR(1) process.
    """
    elev_norm = (elev - elev.min()) / (elev.max() - elev.min() + 1e-6)
    base_rate = 2.0 + 4.0 * elev_norm  # higher precip at high elevation

    truth = np.zeros((T, H, W), dtype=np.float32)
    for t in range(T):
        # ~40% dry timesteps
        if rng.random() < 0.4:
            truth[t] = 0.0
        else:
            field = rng.exponential(base_rate)
            # Spatial smoothing (simple box filter)
            if H >= 3 and W >= 3:
                from scipy.ndimage import uniform_filter
                field = uniform_filter(field, size=3)
            truth[t] = np.clip(field, 0, None).astype(np.float32)

    return truth


def _make_bilinear(truth: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Smoothed + slightly biased version of truth (simulates bilinear interp)."""
    from scipy.ndimage import gaussian_filter
    T, H, W = truth.shape
    pred = np.zeros_like(truth)
    for t in range(T):
        pred[t] = gaussian_filter(truth[t], sigma=2.0)
    # Small bias
    pred = pred * 0.95 + 0.02
    return np.clip(pred, 0, None).astype(np.float32)


def _make_baseline1(truth: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Deterministic model: truth + small noise (better than bilinear)."""
    noise = rng.normal(0, 0.3, truth.shape).astype(np.float32)
    pred = truth + noise
    return np.clip(pred, 0, None).astype(np.float32)


def _make_baseline2(truth: np.ndarray, n_ens: int, rng: np.random.Generator) -> np.ndarray:
    """Ensemble model: truth + per-member noise."""
    T, H, W = truth.shape
    pred = np.zeros((T, n_ens, H, W), dtype=np.float32)
    for e in range(n_ens):
        noise = rng.normal(0, 0.5, (T, H, W)).astype(np.float32)
        pred[:, e] = np.clip(truth + noise, 0, None)
    return pred


def _make_wind(T: int, H_lr: int, W_lr: int, rng: np.random.Generator):
    """Generate synthetic wind field with varying direction.

    Creates a slowly rotating wind pattern so that windward/leeward
    assignments change over time (realistic for testing WLAB).
    """
    # Base wind angle rotates slowly (one full rotation over T steps)
    angles = np.linspace(0, 2 * np.pi, T, dtype=np.float32)
    base_speed = 5.0  # m/s

    u10 = np.zeros((T, H_lr, W_lr), dtype=np.float32)
    v10 = np.zeros((T, H_lr, W_lr), dtype=np.float32)

    for t in range(T):
        # Spatially uniform direction + small spatial noise
        u10[t] = base_speed * np.cos(angles[t]) + rng.normal(0, 0.5, (H_lr, W_lr))
        v10[t] = base_speed * np.sin(angles[t]) + rng.normal(0, 0.5, (H_lr, W_lr))

    return u10.astype(np.float32), v10.astype(np.float32)


def _save_input_nc(
    path: Path,
    variables: dict,
    lat_2d: np.ndarray,
    lon_2d: np.ndarray,
    time: np.ndarray,
):
    """Save multi-variable input NC (low-res) following dataset_policy."""
    data_vars = {}
    for var_name, data in variables.items():
        data_vars[var_name] = (["time", "latitude", "longitude"], data)

    ds = xr.Dataset(
        data_vars,
        coords={
            "time": time,
            "latitude": (["latitude", "longitude"], lat_2d),
            "longitude": (["latitude", "longitude"], lon_2d),
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, engine="netcdf4")


def _save_nc(
    path: Path,
    data: np.ndarray,
    lat_2d: np.ndarray,
    lon_2d: np.ndarray,
    time: np.ndarray,
    var_name: str = "tp",
):
    """Save precipitation array to NetCDF following dataset_policy."""
    H, W = lat_2d.shape

    if data.ndim == 3:
        # (time, latitude, longitude)
        ds = xr.Dataset(
            {
                var_name: (["time", "latitude", "longitude"], data),
            },
            coords={
                "time": time,
                "latitude": (["latitude", "longitude"], lat_2d),
                "longitude": (["latitude", "longitude"], lon_2d),
            },
        )
    elif data.ndim == 4:
        # (time, ensemble, latitude, longitude)
        n_ens = data.shape[1]
        ds = xr.Dataset(
            {
                var_name: (["time", "ensemble", "latitude", "longitude"], data),
            },
            coords={
                "time": time,
                "ensemble": np.arange(n_ens),
                "latitude": (["latitude", "longitude"], lat_2d),
                "longitude": (["latitude", "longitude"], lon_2d),
            },
        )
    else:
        raise ValueError(f"data must be 3D or 4D, got {data.ndim}D")

    ds[var_name].attrs["units"] = "mm/h"
    ds[var_name].attrs["long_name"] = "Total Precipitation"
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, engine="netcdf4")


def _save_static(path: Path, elev: np.ndarray, lsm: np.ndarray, lat_2d: np.ndarray, lon_2d: np.ndarray):
    """Save static variables following dataset_policy."""
    ds = xr.Dataset(
        {
            "elevation": (["latitude", "longitude"], elev),
            "lsm": (["latitude", "longitude"], lsm),
        },
        coords={
            "latitude": (["latitude", "longitude"], lat_2d),
            "longitude": (["latitude", "longitude"], lon_2d),
        },
    )
    ds["elevation"].attrs["units"] = "m"
    ds["lsm"].attrs["long_name"] = "Land-Sea Mask"
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, engine="netcdf4")


def generate(
    out_dir: Path,
    *,
    T_train: int = 200,
    T_test: int = 50,
    H: int = 32,
    W: int = 32,
    H_lr: int = 8,
    W_lr: int = 8,
    n_ens: int = 4,
    seed: int = 42,
) -> Path:
    """Generate all toy data files.

    Parameters
    ----------
    out_dir : output directory (acts as domain_dir)
    T_train : number of training time steps (for history/train_data_out.nc)
    T_test : number of test time steps
    H, W : high-resolution spatial grid size
    H_lr, W_lr : low-resolution spatial grid size (for test_data_in.nc)
    n_ens : ensemble size for baseline2
    seed : random seed

    Returns
    -------
    out_dir Path
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    print(f"Generating toy data: T_train={T_train} T_test={T_test} H={H} W={W} n_ens={n_ens}")

    # Coordinates
    lat_2d, lon_2d = _make_coords(H, W)
    time_train = _make_time(T_train, start="2020-01-01")
    time_test = _make_time(T_test, start="2024-01-01")

    # Static
    elev = _make_elevation(H, W, rng)
    lsm = (elev > 600).astype(np.float32)
    _save_static(out_dir / "static_variables.nc", elev, lsm, lat_2d, lon_2d)
    print(f"  static_variables.nc: elevation {elev.shape}, lsm {lsm.shape}")

    # History (train output, for return period baseline)
    truth_train = _make_truth(T_train, H, W, elev, rng)
    _save_nc(out_dir / "train_data_out.nc", truth_train, lat_2d, lon_2d, time_train)
    print(f"  train_data_out.nc: {truth_train.shape}")

    # Truth (test output)
    truth_test = _make_truth(T_test, H, W, elev, rng)
    _save_nc(out_dir / "test_data_out.nc", truth_test, lat_2d, lon_2d, time_test)
    print(f"  test_data_out.nc: {truth_test.shape}")

    # Low-resolution input (test_data_in.nc with u10, v10, tp)
    lat_2d_lr, lon_2d_lr = _make_coords(H_lr, W_lr)
    u10, v10 = _make_wind(T_test, H_lr, W_lr, rng)
    # Coarsen truth to LR for tp input
    from scipy.ndimage import zoom
    tp_lr = np.zeros((T_test, H_lr, W_lr), dtype=np.float32)
    for t in range(T_test):
        tp_lr[t] = zoom(truth_test[t], (H_lr / H, W_lr / W), order=1)
    _save_input_nc(
        out_dir / "test_data_in.nc",
        {"tp": tp_lr, "u10": u10, "v10": v10},
        lat_2d_lr, lon_2d_lr, time_test,
    )
    print(f"  test_data_in.nc: tp {tp_lr.shape} u10 {u10.shape} v10 {v10.shape} (LR)")

    # Bilinear prediction (ensemble=1, deterministic)
    bilinear = _make_bilinear(truth_test, rng)
    bilinear_4d = bilinear[:, None, :, :]  # add ensemble dim
    _save_nc(out_dir / "bilinear_pred.nc", bilinear_4d, lat_2d, lon_2d, time_test)
    print(f"  bilinear_pred.nc: {bilinear_4d.shape} (E=1)")

    # Baseline1: deterministic model (ensemble=1)
    baseline1 = _make_baseline1(truth_test, rng)
    baseline1_4d = baseline1[:, None, :, :]
    _save_nc(out_dir / "baseline1_pred.nc", baseline1_4d, lat_2d, lon_2d, time_test)
    print(f"  baseline1_pred.nc: {baseline1_4d.shape} (E=1)")

    # Baseline2: ensemble model (ensemble=n_ens)
    baseline2 = _make_baseline2(truth_test, n_ens, rng)
    _save_nc(out_dir / "baseline2_pred.nc", baseline2, lat_2d, lon_2d, time_test)
    print(f"  baseline2_pred.nc: {baseline2.shape} (E={n_ens})")

    print(f"\nAll files saved to: {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Generate toy benchmark test data")
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("/path/to/OroPrecipBench/metrics/tests/toy_domain"),
    )
    parser.add_argument("--T-train", type=int, default=200)
    parser.add_argument("--T-test", type=int, default=50)
    parser.add_argument("--H", type=int, default=32)
    parser.add_argument("--W", type=int, default=32)
    parser.add_argument("--H-lr", type=int, default=8)
    parser.add_argument("--W-lr", type=int, default=8)
    parser.add_argument("--n-ens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(
        args.out_dir,
        T_train=args.T_train, T_test=args.T_test,
        H=args.H, W=args.W, H_lr=args.H_lr, W_lr=args.W_lr,
        n_ens=args.n_ens, seed=args.seed,
    )


if __name__ == "__main__":
    main()
