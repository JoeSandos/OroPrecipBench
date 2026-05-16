"""Step 2: Assemble per-day .npy files into train/val/test_data_out.nc."""

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from config import (
    CORRUPTED_DATES,
    DOMAINS,
    TIME_SPLITS,
    domain_dir,
    load_target_coords_360,
    staging_dir,
)


def dates_in_range(start: date, end: date) -> list[date]:
    dates = []
    d = start
    while d <= end:
        dates.append(d)
        d += timedelta(days=1)
    return dates


def assemble_split(domain: str, split: str) -> None:
    start, end = TIME_SPLITS[split]
    all_dates = dates_in_range(start, end)

    # Filter to available .npy files
    available_dates = []
    npy_paths = []
    for d in all_dates:
        if d in CORRUPTED_DATES:
            continue
        p = staging_dir(domain) / str(d.year) / f"{d.strftime('%Y%m%d')}.npy"
        if p.exists():
            available_dates.append(d)
            npy_paths.append(p)

    n = len(available_dates)
    if n == 0:
        print(f"    [WARN] {domain}/{split}: no .npy files found!")
        return

    # Pre-allocate and load
    data = np.empty((n, 192, 192), dtype=np.float32)
    for i, p in enumerate(npy_paths):
        data[i] = np.load(p)

    # Coordinates
    lat_2d, lon_2d = load_target_coords_360(domain)
    time_arr = pd.DatetimeIndex(available_dates)

    # Build dataset
    ds = xr.Dataset(
        {"tp": (["time", "latitude", "longitude"], data)},
        coords={
            "time": time_arr,
            "latitude": (["latitude", "longitude"], lat_2d),
            "longitude": (["latitude", "longitude"], lon_2d),
        },
    )

    out_path = domain_dir(domain) / f"{split}_data_out.nc"
    ds.to_netcdf(out_path, format="NETCDF4")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"    {split}: {n} days, shape={data.shape}, {size_mb:.1f} MB -> {out_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Step 2: Assemble NetCDF files")
    parser.add_argument("--domains", default=",".join(DOMAINS))
    args = parser.parse_args()

    domains = args.domains.split(",")

    print("=" * 60)
    print("Step 2: Assemble NetCDF")
    print("=" * 60)

    for domain in domains:
        print(f"\n  {domain}:")
        t0 = time.time()
        for split in ["train", "val", "test"]:
            assemble_split(domain, split)
        print(f"    Done in {time.time() - t0:.1f}s")

    print("\nStep 2 complete.")


if __name__ == "__main__":
    main()
