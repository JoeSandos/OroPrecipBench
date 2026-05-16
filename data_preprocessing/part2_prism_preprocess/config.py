"""Shared configuration for the PRISM Part 2 preprocessing pipeline."""

from datetime import date
import os
from pathlib import Path

import numpy as np
import xarray as xr


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WORK_ROOT = SCRIPT_DIR / "work"

# Paths. Override these with environment variables for shared or HPC runs.
PRISM_RAW_ROOT = _path_from_env(
    "OROPRECIPBENCH_PRISM_RAW_ROOT",
    DEFAULT_WORK_ROOT / "raw" / "prism" / "ppt" / "daily",
)
STAGING_ROOT = _path_from_env("OROPRECIPBENCH_PRISM_STAGING_ROOT", DEFAULT_WORK_ROOT / "staging")
OUTPUT_ROOT = _path_from_env("OROPRECIPBENCH_PRISM_OUTPUT_ROOT", DEFAULT_WORK_ROOT / "dataset_root")
COORD_DIR = STAGING_ROOT / "coords"

# PRISM grid parameters.
PRISM_NROWS = 621
PRISM_NCOLS = 1405
PRISM_DX = 0.041666666667
PRISM_LAT_START = 49.9375000000005   # north edge of first pixel (from GeoTIFF transform)
PRISM_LON_START = -125.0208333333335  # west edge of first pixel
# Pixel centers:
PRISM_LAT_CENTER_START = PRISM_LAT_START - PRISM_DX / 2  # 49.9167
PRISM_LON_CENTER_START = PRISM_LON_START + PRISM_DX / 2  # -125.0

NODATA = -9999.0
CORRUPTED_DATES = frozenset({date(2018, 8, 22)})

# Domains.
DOMAINS = ["Rocky", "Cascades", "Nevada", "Appalachian", "Great_Plains"]

# Time splits.
TIME_SPLITS = {
    "train": (date(2000, 1, 1), date(2021, 12, 31)),
    "val": (date(2022, 1, 1), date(2022, 12, 31)),
    "test": (date(2023, 1, 1), date(2024, 12, 31)),
}

def prism_lat_array() -> np.ndarray:
    """1D array of PRISM pixel center latitudes, descending north to south."""
    return PRISM_LAT_CENTER_START - np.arange(PRISM_NROWS) * PRISM_DX


def prism_lon_array() -> np.ndarray:
    """1D array of PRISM pixel center longitudes, ascending west to east."""
    return PRISM_LON_CENTER_START + np.arange(PRISM_NCOLS) * PRISM_DX


def load_target_coords(domain: str) -> tuple[np.ndarray, np.ndarray]:
    """Load 192x192 curvilinear target coordinates from HRRR reference.

    Returns (lat_2d, lon_2d) as float64 arrays.
    Longitude is returned in -180 convention for PRISM interpolation.
    """
    coord_file = COORD_DIR / f"part1-US-{domain}_lat_lon_192_192.nc"
    ds = xr.open_dataset(coord_file, drop_variables=["step", "heightAboveGround"])
    lat_2d = ds.latitude.values.astype(np.float64)
    lon_2d = ds.longitude.values.astype(np.float64)
    ds.close()
    # Convert 0-360 to -180 convention.
    lon_2d = np.where(lon_2d > 180, lon_2d - 360, lon_2d)
    return lat_2d, lon_2d


def load_target_coords_360(domain: str) -> tuple[np.ndarray, np.ndarray]:
    """Load 192x192 target coordinates with longitude in 0-360 convention (for NetCDF output)."""
    coord_file = COORD_DIR / f"part1-US-{domain}_lat_lon_192_192.nc"
    ds = xr.open_dataset(coord_file, drop_variables=["step", "heightAboveGround"])
    lat_2d = ds.latitude.values.astype(np.float32)
    lon_2d = ds.longitude.values.astype(np.float32)
    ds.close()
    return lat_2d, lon_2d


def domain_dir(domain: str) -> Path:
    """Output directory for a domain, e.g. dataset_root/part2-US-Rocky/"""
    return OUTPUT_ROOT / f"part2-US-{domain}"


def staging_dir(domain: str) -> Path:
    """Staging directory for a domain's intermediate daily .npy files."""
    return STAGING_ROOT / domain / "daily"


def prism_zip_path(d: date) -> Path:
    """Path to the PRISM daily ZIP archive for a given date."""
    s = d.strftime("%Y%m%d")
    return PRISM_RAW_ROOT / str(d.year) / f"prism_ppt_us_25m_{s}.zip"


def prism_tif_in_zip(d: date) -> str:
    """Filename of the GeoTIFF inside the PRISM ZIP archive."""
    s = d.strftime("%Y%m%d")
    return f"prism_ppt_us_25m_{s}.tif"
