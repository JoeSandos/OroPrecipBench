#!/usr/bin/env python3
"""Step 3: check regional hourly ERA5 files for missing variables."""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    from netCDF4 import Dataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency netCDF4. Install it first: conda install -c conda-forge netcdf4"
    ) from exc


ERA5_DATA_ROOT = Path("./era5_data")
REGIONS = [
    "part1-US-Rocky",
    "part1-US-Cascades",
    "part1-US-Nevada",
    "part1-US-Appalachian",
    "part1-US-Great_Plains",
]
EXPECTED_VARS = ["msl", "ssrd", "t2m", "tisr", "tp", "u10", "u100", "v10", "v100"]
FILENAME_RE = re.compile(r"^era5_(?P<region>.+)_(?P<date>\d{8})_(?P<hour>\d{4})\.nc$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check regional ERA5 NetCDF files for missing variables")
    parser.add_argument("--root", type=Path, default=ERA5_DATA_ROOT)
    parser.add_argument("--regions", nargs="+", default=REGIONS)
    parser.add_argument("--expected-vars", nargs="+", default=EXPECTED_VARS)
    parser.add_argument("--start", type=str, default=None, help="Start time: YYYYMMDDHH")
    parser.add_argument("--end", type=str, default=None, help="End time: YYYYMMDDHH, inclusive")
    parser.add_argument(
        "--write-list",
        type=Path,
        default=None,
        help="Write the missing-variable file list to a text file.",
    )
    return parser.parse_args()


def parse_bound(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.strptime(value, "%Y%m%d%H")


def parse_timestamp(path: Path) -> datetime | None:
    match = FILENAME_RE.match(path.name)
    if not match:
        return None
    return datetime.strptime(match.group("date") + match.group("hour"), "%Y%m%d%H%M")


def data_variables(path: Path) -> set[str]:
    with Dataset(path) as ds:
        return {
            name
            for name, var in ds.variables.items()
            if name not in {"time", "latitude", "longitude"}
            and set(var.dimensions) == {"time", "latitude", "longitude"}
        }


def main() -> int:
    args = parse_args()
    start = parse_bound(args.start)
    end = parse_bound(args.end)
    expected = set(args.expected_vars)

    missing_files: list[tuple[Path, list[str]]] = []
    by_region: dict[str, int] = defaultdict(int)
    by_var: dict[str, int] = defaultdict(int)
    checked = 0

    for region in args.regions:
        region_dir = args.root / region
        if not region_dir.exists():
            print(f"[WARN] Region directory does not exist: {region_dir}")
            continue
        for path in sorted(region_dir.glob("era5_*.nc")):
            ts = parse_timestamp(path)
            if ts is None:
                continue
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            checked += 1
            present = data_variables(path)
            missing = sorted(expected - present)
            if missing:
                missing_files.append((path, missing))
                by_region[region] += 1
                for var in missing:
                    by_var[var] += 1

    print(f"[INFO] Checked files: {checked}")
    print(f"[INFO] Files with missing variables: {len(missing_files)}")
    for region in args.regions:
        print(f"[INFO] {region}: {by_region.get(region, 0)} files with missing variables")
    for var in sorted(by_var):
        print(f"[INFO] Missing {var}: {by_var[var]} files")

    for path, missing in missing_files[:50]:
        print(f"[MISS] {path} missing: {', '.join(missing)}")
    if len(missing_files) > 50:
        print(f"[INFO] Showing only the first 50; remaining {len(missing_files) - 50} not shown")

    if args.write_list:
        args.write_list.parent.mkdir(parents=True, exist_ok=True)
        with args.write_list.open("w", encoding="utf-8") as f:
            for path, missing in missing_files:
                f.write(f"{path}\t{','.join(missing)}\n")
        print(f"[INFO] Wrote missing-variable list: {args.write_list}")

    return 0 if not missing_files else 1


if __name__ == "__main__":
    raise SystemExit(main())
