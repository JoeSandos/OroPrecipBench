"""Convert prediction files from legacy format to dataset_policy Section 5 format.

Legacy format:  group='prediction', variables 'pred' and 'truth', dims (time, ensemble, lat, lon)
Target format:  flat (no group), variable 'tp', dims (time, ensemble, latitude, longitude)

Output is saved as pred.nc alongside the original test.nc (which is NOT modified).
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

RUNS_ROOT = Path(os.environ.get("OROPRECIP_RUNS_ROOT", "baselines/runs"))


def convert_one(src: Path, dst: Path, *, dry_run: bool = False) -> bool:
    """Convert a single prediction file. Returns True on success."""
    if dst.exists():
        logger.info("SKIP (already exists): %s", dst)
        return True

    try:
        ds = xr.open_dataset(src, group="prediction")
    except Exception:
        # Maybe already flat format — check if tp exists at root
        try:
            ds = xr.open_dataset(src)
            if "tp" in ds.data_vars:
                logger.info("SKIP (already compliant): %s", src)
                ds.close()
                return True
        except Exception:
            pass
        logger.error("FAILED to open: %s", src)
        return False

    if "pred" not in ds.data_vars:
        logger.error("No 'pred' variable in %s", src)
        ds.close()
        return False

    # Rename pred -> tp, keep only prediction (drop truth)
    da = ds["pred"].rename("tp")

    # Ensure standard dim names
    rename_map = {}
    for d in da.dims:
        dl = str(d).lower()
        if dl in {"lat"}:
            rename_map[d] = "latitude"
        elif dl in {"lon"}:
            rename_map[d] = "longitude"
        elif dl in {"ens", "member", "realization"}:
            rename_map[d] = "ensemble"
    if rename_map:
        da = da.rename(rename_map)

    # Ensure dim order: (time, ensemble, latitude, longitude)
    expected_order = [d for d in ["time", "ensemble", "latitude", "longitude"] if d in da.dims]
    da = da.transpose(*expected_order)

    # Copy coordinate variables (lat/lon 2D arrays, time, ensemble)
    coords = {}
    for c in ["time", "ensemble", "latitude", "longitude"]:
        if c in ds.coords:
            coords[c] = ds.coords[c]

    out_ds = da.to_dataset()

    # Preserve original attributes
    for attr in ["method", "part", "region", "split", "config", "ckpt",
                 "input_file", "stats_file", "stats_source", "ensemble_size"]:
        if attr in ds.attrs:
            out_ds.attrs[attr] = ds.attrs[attr]

    if dry_run:
        logger.info("DRY RUN: would write %s  (tp shape=%s)", dst, da.shape)
        ds.close()
        return True

    # Write with compression
    encoding = {
        "tp": {"zlib": True, "complevel": 4, "dtype": "float32"},
    }
    out_ds.to_netcdf(dst, encoding=encoding)
    logger.info("OK: %s  (tp shape=%s)", dst, da.shape)

    ds.close()
    return True


def find_prediction_files(root: Path) -> list[Path]:
    """Find all predictions/test.nc files under root."""
    return sorted(root.rglob("predictions/test.nc"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root", type=Path, default=RUNS_ROOT,
        help="Root directory containing experiment runs",
    )
    parser.add_argument(
        "--output-name", default="pred.nc",
        help="Output filename (saved alongside test.nc, default: pred.nc)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without writing files",
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Only process paths containing this substring (e.g. 'part1' or 'resnet')",
    )
    args = parser.parse_args()

    files = find_prediction_files(args.runs_root)
    logger.info("Found %d prediction files under %s", len(files), args.runs_root)

    if args.filter:
        files = [f for f in files if args.filter in str(f)]
        logger.info("After filter '%s': %d files", args.filter, len(files))

    ok, fail = 0, 0
    total = len(files)
    for i, src in enumerate(files, 1):
        dst = src.parent / args.output_name
        logger.info("[%d/%d] Processing: %s", i, total, src.relative_to(args.runs_root))
        success = convert_one(src, dst, dry_run=args.dry_run)
        if success:
            ok += 1
        else:
            fail += 1

    logger.info("Done: %d OK, %d FAILED out of %d total", ok, fail, total)
    if fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
