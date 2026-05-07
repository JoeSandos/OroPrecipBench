"""Data loading utilities for benchmark evaluation.

Loads prediction and truth NetCDF files following the dataset_policy (Chapter 5):
- Truth: test_data_out.nc with dims (time, latitude, longitude), variable 'tp'
- Prediction: same format with optional ensemble dim (time, ensemble, latitude, longitude)
- Static: static_variables.nc with elevation + lsm
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

_DIM_ALIASES_TIME = {"time", "valid_time", "sample", "t"}
_DIM_ALIASES_ENS = {"ensemble", "ens", "member", "realization"}
_DIM_ALIASES_Y = {"latitude", "lat", "y", "y_grid", "south_north"}
_DIM_ALIASES_X = {"longitude", "lon", "x", "x_grid", "west_east"}


def _find_dim(da: xr.DataArray, aliases: set[str]) -> Optional[str]:
    for d in da.dims:
        if str(d).lower() in aliases:
            return str(d)
    return None


def _standardize_dims(da: xr.DataArray, *, expect_ensemble: bool) -> xr.DataArray:
    """Rename dims to standard names: time, ensemble, latitude, longitude."""
    rename = {}
    for d in da.dims:
        dl = str(d).lower()
        if dl in _DIM_ALIASES_TIME:
            rename[d] = "time"
        elif dl in _DIM_ALIASES_ENS:
            rename[d] = "ensemble"
        elif dl in _DIM_ALIASES_Y:
            rename[d] = "latitude"
        elif dl in _DIM_ALIASES_X:
            rename[d] = "longitude"
    if rename:
        da = da.rename(rename)

    # Fallback: last two unnamed dims are spatial
    if "latitude" not in da.dims or "longitude" not in da.dims:
        if da.ndim >= 2:
            d0, d1 = da.dims[-2], da.dims[-1]
            fb = {}
            if d0 not in {"time", "ensemble"} and "latitude" not in da.dims:
                fb[d0] = "latitude"
            if d1 not in {"time", "ensemble"} and "longitude" not in da.dims:
                fb[d1] = "longitude"
            if fb:
                da = da.rename(fb)

    if expect_ensemble and "ensemble" not in da.dims:
        da = da.expand_dims({"ensemble": [0]})

    return da


def load_truth(domain_dir: Path | str) -> Tuple[np.ndarray, dict]:
    """Load test_data_out.nc from a domain directory.

    Returns
    -------
    data : np.ndarray, shape (T, H, W), float32
    meta : dict with 'time' (datetime64 array), 'shape', 'path'
    """
    domain_dir = Path(domain_dir)
    path = domain_dir / "test_data_out.nc"
    if not path.exists():
        raise FileNotFoundError(f"Truth file not found: {path}")

    ds = xr.open_dataset(path)
    var_name = "tp"
    if var_name not in ds:
        candidates = list(ds.data_vars)
        if len(candidates) == 1:
            var_name = candidates[0]
            logger.warning("Variable 'tp' not found, using '%s'", var_name)
        else:
            raise ValueError(f"Variable 'tp' not found in {path}, available: {candidates}")

    da = _standardize_dims(ds[var_name], expect_ensemble=False)

    # Squeeze ensemble if present
    if "ensemble" in da.dims:
        da = da.isel(ensemble=0, drop=True)

    da = da.transpose("time", "latitude", "longitude")
    time_values = da["time"].values if "time" in da.coords else np.arange(da.sizes["time"])

    data = da.values.astype(np.float32)
    ds.close()

    return data, {
        "time": time_values,
        "shape": data.shape,
        "path": str(path),
    }


def load_prediction(
    pred_path: Path | str,
    max_ens: Optional[int] = None,
) -> Tuple[np.ndarray, dict]:
    """Load a model prediction NetCDF file.

    Parameters
    ----------
    pred_path : path to prediction .nc file
    max_ens : if set, cap ensemble members to first max_ens

    Returns
    -------
    data : np.ndarray, shape (T, E, H, W), float32
    meta : dict with 'time', 'n_ensemble', 'shape', 'path'
    """
    pred_path = Path(pred_path)
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    ds = xr.open_dataset(pred_path)
    var_name = "tp"
    if var_name not in ds:
        candidates = list(ds.data_vars)
        if len(candidates) == 1:
            var_name = candidates[0]
            logger.warning("Variable 'tp' not found, using '%s'", var_name)
        else:
            raise ValueError(f"Variable 'tp' not found in {pred_path}, available: {candidates}")

    da = _standardize_dims(ds[var_name], expect_ensemble=True)
    da = da.transpose("time", "ensemble", "latitude", "longitude")

    if max_ens is not None and da.sizes["ensemble"] > max_ens:
        logger.info("Capping ensemble from %d to %d", da.sizes["ensemble"], max_ens)
        da = da.isel(ensemble=slice(0, max_ens))

    time_values = da["time"].values if "time" in da.coords else np.arange(da.sizes["time"])
    n_ens = int(da.sizes["ensemble"])
    data = da.values.astype(np.float32)
    ds.close()

    return data, {
        "time": time_values,
        "n_ensemble": n_ens,
        "shape": data.shape,
        "path": str(pred_path),
    }


def load_train(domain_dir: Path | str) -> Tuple[np.ndarray, dict]:
    """Load train_data_out.nc from a domain directory.

    Returns
    -------
    data : np.ndarray, shape (T, H, W), float32
    meta : dict with 'time', 'shape', 'path'
    """
    domain_dir = Path(domain_dir)
    path = domain_dir / "train_data_out.nc"
    if not path.exists():
        raise FileNotFoundError(f"Train file not found: {path}")

    ds = xr.open_dataset(path)
    var_name = "tp"
    if var_name not in ds:
        candidates = list(ds.data_vars)
        if len(candidates) == 1:
            var_name = candidates[0]
            logger.warning("Variable 'tp' not found, using '%s'", var_name)
        else:
            raise ValueError(f"Variable 'tp' not found in {path}, available: {candidates}")

    da = _standardize_dims(ds[var_name], expect_ensemble=False)
    if "ensemble" in da.dims:
        da = da.isel(ensemble=0, drop=True)
    da = da.transpose("time", "latitude", "longitude")
    time_values = da["time"].values if "time" in da.coords else np.arange(da.sizes["time"])

    data = da.values.astype(np.float32)
    ds.close()

    return data, {
        "time": time_values,
        "shape": data.shape,
        "path": str(path),
    }


def load_static(domain_dir: Path | str) -> dict:
    """Load static_variables.nc.

    Returns dict with 'elevation' and 'lsm' as (H, W) float32 arrays.
    """
    domain_dir = Path(domain_dir)
    path = domain_dir / "static_variables.nc"
    if not path.exists():
        raise FileNotFoundError(f"Static file not found: {path}")

    ds = xr.open_dataset(path)
    result = {}
    for var in ["elevation", "lsm"]:
        if var in ds:
            result[var] = ds[var].values.astype(np.float32)
        else:
            logger.warning("Variable '%s' not found in %s", var, path)
    ds.close()
    return result


def load_land_mask(domain_dir: Path | str, threshold: float = 0.5) -> np.ndarray | None:
    """Load land-sea mask from static_variables.nc.

    Returns
    -------
    mask : np.ndarray of bool, shape (H, W), True = land; or None if lsm unavailable.
    """
    static = load_static(domain_dir)
    lsm = static.get("lsm")
    if lsm is None:
        return None
    return lsm >= threshold


def load_wind(domain_dir: Path | str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load u10, v10 wind components from test_data_in.nc.

    Returns
    -------
    u10 : np.ndarray, shape (T, H_lr, W_lr), float32, 10m eastward wind (m/s)
    v10 : np.ndarray, shape (T, H_lr, W_lr), float32, 10m northward wind (m/s)
    time : np.ndarray, datetime64 time coordinate
    """
    domain_dir = Path(domain_dir)
    path = domain_dir / "test_data_in.nc"
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    ds = xr.open_dataset(path)

    # Find u10 variable
    u10_aliases = {"u10", "10u", "u10m", "eastward_wind_10m"}
    v10_aliases = {"v10", "10v", "v10m", "northward_wind_10m"}

    u10_name = None
    v10_name = None
    for var in ds.data_vars:
        vl = str(var).lower()
        if vl in u10_aliases:
            u10_name = var
        elif vl in v10_aliases:
            v10_name = var

    if u10_name is None:
        raise ValueError(f"u10 variable not found in {path}, available: {list(ds.data_vars)}")
    if v10_name is None:
        raise ValueError(f"v10 variable not found in {path}, available: {list(ds.data_vars)}")

    da_u = _standardize_dims(ds[u10_name], expect_ensemble=False)
    da_v = _standardize_dims(ds[v10_name], expect_ensemble=False)

    if "ensemble" in da_u.dims:
        da_u = da_u.isel(ensemble=0, drop=True)
    if "ensemble" in da_v.dims:
        da_v = da_v.isel(ensemble=0, drop=True)

    da_u = da_u.transpose("time", "latitude", "longitude")
    da_v = da_v.transpose("time", "latitude", "longitude")

    time_values = da_u["time"].values if "time" in da_u.coords else np.arange(da_u.sizes["time"])

    u10 = da_u.values.astype(np.float32)
    v10 = da_v.values.astype(np.float32)
    ds.close()

    return u10, v10, time_values


def align_time(
    pred: np.ndarray,
    truth: np.ndarray,
    pred_time: np.ndarray,
    truth_time: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align prediction and truth arrays on common time steps.

    Returns
    -------
    pred_aligned, truth_aligned, common_time
    """
    if pred_time.dtype != truth_time.dtype:
        # Both might be datetime64 or int; try intersection
        try:
            pred_time = pred_time.astype("datetime64[ns]")
            truth_time = truth_time.astype("datetime64[ns]")
        except (TypeError, ValueError):
            pass

    common, pred_idx, truth_idx = np.intersect1d(
        pred_time, truth_time, return_indices=True
    )
    if len(common) == 0:
        raise ValueError("No overlapping time steps between prediction and truth")

    if len(common) < len(truth_time):
        logger.info(
            "Time alignment: %d common out of pred=%d truth=%d",
            len(common), len(pred_time), len(truth_time),
        )

    return pred[pred_idx], truth[truth_idx], common


def reduce_ensemble(
    pred: np.ndarray,
    strategy: str = "mean",
    member_index: int = 0,
) -> np.ndarray:
    """Reduce ensemble dimension of prediction array.

    Parameters
    ----------
    pred : (T, E, H, W)
    strategy : 'mean' or 'per_member' (returns unchanged) or 'first'

    Returns
    -------
    (T, H, W) for 'mean'/'first', or (T, E, H, W) unchanged for 'per_member'
    """
    if pred.ndim != 4:
        raise ValueError(f"Expected 4D pred (T,E,H,W), got shape {pred.shape}")

    if strategy == "mean":
        return np.nanmean(pred, axis=1)
    elif strategy == "first":
        return pred[:, member_index]
    elif strategy == "per_member":
        return pred
    else:
        raise ValueError(f"Unknown ensemble strategy: {strategy}")



# ── Part 1 / Part 2 threshold presets ────────────────────────────────────────

PART_DEFAULTS = {
    "hourly": {  # Part 1: HRRR→HRRR, mm/h
        "rain_threshold": 0.01,
        "fss_threshold": 0.01,
        "wet_threshold": 0.01,
        "pdf_bins_max": 100.0,
        "pdf_bins_width": 1.0,
    },
    "daily": {   # Part 2: ERA5→PRISM, mm/day
        "rain_threshold": 1.0,
        "fss_threshold": 1.0,
        "wet_threshold": 1.0,
        "pdf_bins_max": 200.0,
        "pdf_bins_width": 2.0,
    },
}


def detect_temporal_resolution(domain_dir: Path | str) -> str:
    """Infer 'hourly' or 'daily' from the time coordinate in test_data_out.nc.

    Uses the median time delta of the first 100 steps.
    Returns 'hourly' if median delta < 6 hours, else 'daily'.
    """
    domain_dir = Path(domain_dir)
    path = domain_dir / "test_data_out.nc"
    if not path.exists():
        raise FileNotFoundError(f"Cannot detect resolution: {path} not found")

    ds = xr.open_dataset(path)
    # Find the time coordinate
    time_var = None
    for name in ds.coords:
        if str(name).lower() in _DIM_ALIASES_TIME:
            time_var = name
            break
    if time_var is None:
        for name in ds.dims:
            if str(name).lower() in _DIM_ALIASES_TIME:
                time_var = name
                break
    if time_var is None:
        ds.close()
        logger.warning("No time coordinate found, defaulting to 'hourly'")
        return "hourly"

    time_vals = ds[time_var].values
    ds.close()

    if len(time_vals) < 2:
        logger.warning("Only %d time step(s), defaulting to 'hourly'", len(time_vals))
        return "hourly"

    # Use first 100 steps for efficiency
    subset = time_vals[:min(100, len(time_vals))]
    try:
        deltas = np.diff(subset.astype("datetime64[ns]"))
        median_delta = np.median(deltas.astype("timedelta64[s]").astype(np.float64))
        hours = median_delta / 3600.0
    except (TypeError, ValueError):
        logger.warning("Cannot parse time deltas, defaulting to 'hourly'")
        return "hourly"

    resolution = "hourly" if hours < 6.0 else "daily"
    logger.info(
        "Detected temporal resolution: %s (median delta = %.1f hours)",
        resolution, hours,
    )
    return resolution


def resolve_part_defaults(
    args,
    resolution: str,
    threshold_fields: list[str] | None = None,
) -> None:
    """Fill in None-valued threshold args from PART_DEFAULTS[resolution].

    Modifies args in place.  Only overwrites attributes that are None
    (i.e. user did not explicitly set them on the CLI).
    """
    defaults = PART_DEFAULTS[resolution]
    if threshold_fields is None:
        threshold_fields = list(defaults.keys())
    for field in threshold_fields:
        if field in defaults and getattr(args, field, None) is None:
            setattr(args, field, defaults[field])


def cache_key(path: Path | str, **params) -> str:
    """Generate a cache key from file path (mtime) and parameters."""
    path = Path(path)
    mtime = path.stat().st_mtime if path.exists() else 0
    key_str = f"{path}:{mtime}:" + json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(key_str.encode()).hexdigest()[:16]
