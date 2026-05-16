#!/usr/bin/env python3
"""Step 2: crop global hourly ERA5 files to US regions and merge variables.

Primary input is checked first; supplemental input is used only when a file is
missing. Outputs are one NetCDF per region per hour.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from netCDF4 import Dataset, date2num
except ImportError as exc:  # pragma: no cover - user environment check
    raise SystemExit(
        "Missing dependency netCDF4. Install it before running:\n"
        "  conda install -c conda-forge netcdf4\n"
        "or:\n"
        "  pip install netCDF4"
    ) from exc


SOURCE_ROOT = Path(
    "/path/to/era5/hourly/global"
)
SUPPLEMENTAL_SOURCE_ROOT = Path("./era5_source")
OUTPUT_ROOT = Path("./era5_data")
DEFAULT_WORKERS = min(32, os.cpu_count() or 1)

FILENAME_RE = re.compile(r"^(?P<var>.+)_(?P<date>\d{8})_(?P<hour>\d{4})\.nc$")
OUTPUT_FILENAME_RE = re.compile(
    r"^era5_(?P<region>.+)_(?P<date>\d{8})_(?P<hour>\d{4})\.nc$"
)


@dataclass(frozen=True)
class Region:
    dirname: str
    lat_min: float
    lat_max: float
    west_min: float
    west_max: float

    @property
    def lon_min_0360(self) -> float:
        return 360.0 - self.west_max

    @property
    def lon_max_0360(self) -> float:
        return 360.0 - self.west_min


REGIONS = [
    Region("part1-US-Rocky", 36.0, 42.0, 103.0, 110.5),
    Region("part1-US-Cascades", 43.25, 49.75, 117.0, 126.5),
    Region("part1-US-Nevada", 34.75, 41.25, 115.75, 123.75),
    Region("part1-US-Appalachian", 34.25, 40.5, 76.0, 83.75),
    Region("part1-US-Great_Plains", 37.25, 42.75, 95.0, 102.25),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop hourly ERA5 global files to US regions and merge variables."
    )
    parser.add_argument("--input-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument(
        "--supplemental-input-root",
        type=Path,
        default=SUPPLEMENTAL_SOURCE_ROOT,
        help=(
            "Supplemental global hourly files; layout must match input-root."
            "Default: ./era5_source."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--variables",
        nargs="+",
        default=None,
        help="Variable names; default: all variable directories under input-root.",
    )
    parser.add_argument("--start", type=str, default=None, help="Start time: YYYYMMDDHH")
    parser.add_argument("--end", type=str, default=None, help="End time: YYYYMMDDHH, inclusive")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument(
        "--repair-missing-output-vars",
        action="store_true",
        help=(
            "Scan generated regional NetCDF files under output-root, "
            "reprocess only timestamps with missing variables, and overwrite those outputs."
        ),
    )
    parser.add_argument(
        "--no-auto-diff",
        action="store_true",
        help="Disable automatic accumulated-variable detection and differencing.",
    )
    parser.add_argument(
        "--accum-vars",
        nargs="*",
        default=None,
        help="Manually specify accumulated variables to difference; overrides auto-detection.",
    )
    parser.add_argument(
        "--max-hours",
        type=int,
        default=None,
        help="Process only the first N hours, useful for testing.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of worker processes; default: {DEFAULT_WORKERS}。",
    )
    parser.add_argument(
        "--max-pending",
        type=int,
        default=None,
        help="Maximum pending tasks; default workers*4 to avoid submitting too many tasks at once.",
    )
    return parser.parse_args()


def parse_timestamp(path: Path) -> datetime | None:
    match = FILENAME_RE.match(path.name)
    if not match:
        return None
    return datetime.strptime(match.group("date") + match.group("hour"), "%Y%m%d%H%M")


def parse_output_timestamp(path: Path) -> datetime | None:
    match = OUTPUT_FILENAME_RE.match(path.name)
    if not match:
        return None
    return datetime.strptime(match.group("date") + match.group("hour"), "%Y%m%d%H%M")


def parse_bound(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.strptime(value, "%Y%m%d%H")


def input_roots(primary_root: Path, supplemental_root: Path | None) -> list[Path]:
    roots = [primary_root]
    if supplemental_root and supplemental_root != primary_root:
        roots.append(supplemental_root)
    return roots


def discover_variables(input_roots_: list[Path], variables: list[str] | None) -> list[str]:
    if variables:
        return variables
    discovered: set[str] = set()
    for root in input_roots_:
        if not root.exists():
            continue
        discovered.update(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    return sorted(discovered)


def is_readable_netcdf(path: Path) -> bool:
    if path.stat().st_size >= 1024:
        return True
    print(f"[WARN] Skipping suspected corrupt NetCDF: {path} (file is too small)", file=sys.stderr)
    try:
        with Dataset(path):
            return True
    except Exception:
        return False


def build_inventory(
    input_roots_: list[Path],
    variables: Iterable[str],
    start: datetime | None,
    end: datetime | None,
) -> dict[datetime, dict[str, Path]]:
    inventory: dict[datetime, dict[str, Path]] = defaultdict(dict)
    for root in reversed(input_roots_):
        for var in variables:
            var_root = root / var
            if not var_root.exists():
                continue
            for path in sorted(var_root.glob("*/*/*.nc")):
                ts = parse_timestamp(path)
                if ts is None:
                    continue
                if start and ts < start:
                    continue
                if end and ts > end:
                    continue
                if not is_readable_netcdf(path):
                    continue
                inventory[ts][var] = path

    for var in variables:
        if not any((root / var).exists() for root in input_roots_):
            roots = ", ".join(str(root / var) for root in input_roots_)
            print(f"[WARN] Variable directory missing, skipping: {roots}", file=sys.stderr)
    return dict(sorted(inventory.items()))


def output_present_variables(path: Path, expected_variables: set[str]) -> set[str]:
    with Dataset(path) as nc:
        present: set[str] = set()
        for name in expected_variables:
            if name not in nc.variables:
                continue
            dims = tuple(nc.variables[name].dimensions)
            if dims == ("time", "latitude", "longitude"):
                present.add(name)
        return present


def timestamps_with_missing_output_vars(
    output_root: Path,
    variables: list[str],
    start: datetime | None,
    end: datetime | None,
) -> set[datetime]:
    expected = set(variables)
    repair_timestamps: set[datetime] = set()
    missing_file_count = 0
    missing_by_var: dict[str, int] = defaultdict(int)
    checked = 0

    for region in REGIONS:
        region_dir = output_root / region.dirname
        if not region_dir.exists():
            continue
        for path in sorted(region_dir.glob("era5_*.nc")):
            ts = parse_output_timestamp(path)
            if ts is None:
                continue
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            checked += 1
            try:
                present = output_present_variables(path, expected)
            except Exception as exc:
                print(f"[WARN] Cannot read output file; will reprocess: {path} ({exc})", file=sys.stderr)
                present = set()
            missing = sorted(expected - present)
            if missing:
                repair_timestamps.add(ts)
                missing_file_count += 1
                for var in missing:
                    missing_by_var[var] += 1

    print(f"[INFO] Checked regional output files: {checked}", file=sys.stderr)
    print(f"[INFO] Output files with missing variables: {missing_file_count}", file=sys.stderr)
    for var in sorted(missing_by_var):
        print(f"[INFO] Output files missing {var}: {missing_by_var[var]} ", file=sys.stderr)
    return repair_timestamps


def get_coord_name(nc: Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in nc.variables:
            return name
    raise KeyError(f"Coordinate variable not found: {candidates}")


def get_data_var_name(nc: Dataset, expected: str) -> str:
    if expected in nc.variables:
        return expected
    for name, var in nc.variables.items():
        dims = set(var.dimensions)
        if {"latitude", "longitude"}.issubset(dims) and name not in {
            "latitude",
            "longitude",
            "valid_time",
            "time",
        }:
            return name
    raise KeyError(f"Data variable not found: {expected}")


def read_global_coords(sample_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Dataset(sample_path) as nc:
        lat_name = get_coord_name(nc, ("latitude", "lat"))
        lon_name = get_coord_name(nc, ("longitude", "lon"))
        lat = np.asarray(nc.variables[lat_name][:])
        lon = np.asarray(nc.variables[lon_name][:])
    lon = np.mod(lon, 360.0)
    return lat, lon


def region_indices(
    lat: np.ndarray, lon: np.ndarray, region: Region
) -> tuple[np.ndarray, np.ndarray]:
    lat_idx = np.where((lat >= region.lat_min) & (lat <= region.lat_max))[0]
    lon_idx = np.where(
        (lon >= region.lon_min_0360) & (lon <= region.lon_max_0360)
    )[0]
    if lat_idx.size == 0 or lon_idx.size == 0:
        raise ValueError(f"{region.dirname} has no matching latitude/longitude grid points")
    return lat_idx, lon_idx


def is_accumulated_by_metadata(path: Path) -> bool:
    with Dataset(path) as nc:
        history = str(getattr(nc, "history", ""))
    return '"stepType": ["accum"]' in history or "stepType" in history and "accum" in history


def day_sample_files(input_roots_: list[Path], var: str) -> list[Path]:
    files_by_time: dict[datetime, Path] = {}
    for root in reversed(input_roots_):
        for path in sorted((root / var).glob("*/*/*.nc")):
            ts = parse_timestamp(path)
            if ts is not None:
                files_by_time[ts] = path
    by_day: dict[str, list[Path]] = defaultdict(list)
    for ts, path in sorted(files_by_time.items()):
        by_day[ts.strftime("%Y%m%d")].append(path)
    for day_files in by_day.values():
        if len(day_files) >= 24:
            return sorted(day_files)[:24]
    return []


def detect_accumulated_variables(
    input_roots_: list[Path],
    variables: Iterable[str],
    sample_region: Region,
    sample_lat: np.ndarray,
    sample_lon: np.ndarray,
) -> set[str]:
    accum_vars: set[str] = set()
    lat_idx, lon_idx = region_indices(sample_lat, sample_lon, sample_region)
    for var in variables:
        sample_files = day_sample_files(input_roots_, var)
        if not sample_files:
            continue
        if not is_accumulated_by_metadata(sample_files[0]):
            continue

        means: list[float] = []
        for path in sample_files:
            with Dataset(path) as nc:
                data_name = get_data_var_name(nc, var)
                arr = read_cropped_array(nc, data_name, lat_idx, lon_idx)
                means.append(float(np.nanmean(arr)))

        diffs = np.diff(means)
        non_decreasing_fraction = float(np.mean(diffs >= -1.0e-10)) if diffs.size else 0.0
        grows_over_day = means[-1] >= means[0]
        if non_decreasing_fraction >= 0.95 and grows_over_day:
            accum_vars.add(var)
            print(
                f"[INFO] {var}: detected as within-day accumulation; writing hourly differences",
                file=sys.stderr,
            )
        else:
            print(
                f"[INFO] {var}: metadata says accum, but sample day is not stably increasing; keeping raw values",
                file=sys.stderr,
            )
    return accum_vars


def compact_indexer(indices: np.ndarray) -> slice | np.ndarray:
    if indices.size and np.all(np.diff(indices) == 1):
        return slice(int(indices[0]), int(indices[-1]) + 1)
    return indices


def read_cropped_array(
    nc: Dataset, data_name: str, lat_idx: np.ndarray, lon_idx: np.ndarray
) -> np.ndarray:
    var = nc.variables[data_name]
    lat_selector = compact_indexer(lat_idx)
    lon_selector = compact_indexer(lon_idx)
    slices: list[object] = []
    for dim in var.dimensions:
        if dim in {"valid_time", "time"}:
            slices.append(0)
        elif dim in {"latitude", "lat"}:
            slices.append(lat_selector)
        elif dim in {"longitude", "lon"}:
            slices.append(lon_selector)
        else:
            slices.append(0)
    data = np.asarray(var[tuple(slices)], dtype=np.float32)
    if data.ndim != 2:
        data = np.squeeze(data)
    if data.ndim != 2:
        raise ValueError(f"{data_name} cropped data is not 2-D; actual shape: {data.shape}")
    return data


def read_variable(
    path: Path, expected_var: str, lat_idx: np.ndarray, lon_idx: np.ndarray
) -> tuple[np.ndarray, dict[str, object]]:
    with Dataset(path) as nc:
        data_name = get_data_var_name(nc, expected_var)
        data = read_cropped_array(nc, data_name, lat_idx, lon_idx)
        attrs = {
            key: nc.variables[data_name].getncattr(key)
            for key in nc.variables[data_name].ncattrs()
        }
    return data, attrs


def hourly_from_accumulation(
    var: str,
    timestamp: datetime,
    current: np.ndarray,
    previous_state: dict[tuple[str, str], tuple[datetime, np.ndarray]],
    region_name: str,
) -> np.ndarray:
    key = (region_name, var)
    prev = previous_state.get(key)
    previous_state[key] = (timestamp, current.copy())
    if prev is None or prev[0].date() != timestamp.date():
        return current
    diff = current - prev[1]
    return np.where(diff >= 0.0, diff, current).astype(np.float32)


def hourly_from_region_state(
    var: str,
    timestamp: datetime,
    current: np.ndarray,
    previous_state: dict[str, tuple[datetime, np.ndarray]],
) -> np.ndarray:
    prev = previous_state.get(var)
    previous_state[var] = (timestamp, current.copy())
    if prev is None or prev[0].date() != timestamp.date():
        return current
    diff = current - prev[1]
    return np.where(diff >= 0.0, diff, current).astype(np.float32)


def normalize_precipitation_units(
    name: str, data: np.ndarray, attrs: dict[str, object]
) -> tuple[np.ndarray, dict[str, object]]:
    """Convert ERA5 total precipitation from metres to mm/h for hourly files."""
    units = str(attrs.get("units", "")).strip().lower()
    long_name = str(attrs.get("long_name", "")).lower()
    is_total_precip = name == "tp" or "total precipitation" in long_name
    if not is_total_precip or units not in {"m", "metre", "meter", "meters", "metres"}:
        return data, attrs

    updated_attrs = dict(attrs)
    updated_attrs["original_units"] = attrs.get("units", "m")
    updated_attrs["units"] = "mm h-1"
    updated_attrs["long_name"] = "Total precipitation rate"
    updated_attrs["unit_conversion"] = "ERA5 precipitation converted from m to mm/h"
    return (data * np.float32(1000.0)).astype(np.float32), updated_attrs


def copy_variable_attrs(out_var, attrs: dict[str, object]) -> None:
    for key, value in attrs.items():
        if key.startswith("_") or key in {"coordinates"}:
            continue
        try:
            out_var.setncattr(key, value)
        except TypeError:
            pass


def write_output(
    output_path: Path,
    timestamp: datetime,
    lat_values: np.ndarray,
    lon_values: np.ndarray,
    variables: dict[str, tuple[np.ndarray, dict[str, object]]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    nlat = lat_values.size
    nlon = lon_values.size
    lon2d, lat2d = np.meshgrid(lon_values.astype(np.float32), lat_values.astype(np.float32))

    try:
        with Dataset(tmp_path, "w", format="NETCDF4") as out:
            out.createDimension("time", 1)
            out.createDimension("latitude", nlat)
            out.createDimension("longitude", nlon)

            time_var = out.createVariable("time", "f8", ("time",))
            time_var.units = "seconds since 1970-01-01 00:00:00"
            time_var.calendar = "standard"
            time_var.standard_name = "time"
            time_var.long_name = "time"
            time_var[:] = date2num([timestamp], time_var.units, time_var.calendar)

            lat_var = out.createVariable(
                "latitude", "f4", ("latitude", "longitude"), zlib=True, complevel=4
            )
            lat_var.units = "degrees_north"
            lat_var.standard_name = "latitude"
            lat_var.long_name = "latitude"
            lat_var[:, :] = lat2d

            lon_var = out.createVariable(
                "longitude", "f4", ("latitude", "longitude"), zlib=True, complevel=4
            )
            lon_var.units = "degrees_east"
            lon_var.standard_name = "longitude"
            lon_var.long_name = "longitude"
            lon_var[:, :] = lon2d

            for name, (data, attrs) in variables.items():
                out_var = out.createVariable(
                    name,
                    "f4",
                    ("time", "latitude", "longitude"),
                    zlib=True,
                    complevel=4,
                    fill_value=np.float32(np.nan),
                )
                copy_variable_attrs(out_var, attrs)
                out_var.coordinates = "latitude longitude"
                out_var[0, :, :] = data.astype(np.float32)

            out.Conventions = "CF-1.7"
            out.history = (
                f"Created by process_era5_us_regions.py at "
                f"{datetime.utcnow():%Y-%m-%dT%H:%M:%SZ}"
            )

        os.replace(tmp_path, output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def process_region_timestamp(
    region: Region,
    timestamp: datetime,
    files_by_var: dict[str, Path],
    accum_vars: set[str],
    lat_idx: np.ndarray,
    lon_idx: np.ndarray,
    lat_values: np.ndarray,
    lon_values: np.ndarray,
    output_root: Path,
    overwrite: bool,
    previous_state: dict[str, tuple[datetime, np.ndarray]],
) -> tuple[str, dict[str, tuple[datetime, np.ndarray]]]:
    stamp = timestamp.strftime("%Y%m%d_%H%M")
    merged: dict[str, tuple[np.ndarray, dict[str, object]]] = {}

    for var, path in sorted(files_by_var.items()):
        data, attrs = read_variable(path, var, lat_idx, lon_idx)
        if var in accum_vars:
            data = hourly_from_region_state(var, timestamp, data, previous_state)
            attrs = dict(attrs)
            attrs["hourly_from_accumulation"] = "true"
        data, attrs = normalize_precipitation_units(var, data, attrs)
        merged[var] = (data, attrs)

    output_path = output_root / region.dirname / f"era5_{region.dirname}_{stamp}.nc"
    if not output_path.exists() or overwrite:
        write_output(output_path, timestamp, lat_values, lon_values, merged)

    return region.dirname, previous_state


def submit_region_task(
    executor: ProcessPoolExecutor,
    region: Region,
    timestamp: datetime,
    files_by_var: dict[str, Path],
    accum_vars: set[str],
    region_lookup: dict[str, tuple[np.ndarray, np.ndarray]],
    global_lat: np.ndarray,
    global_lon: np.ndarray,
    output_root: Path,
    overwrite: bool,
    previous_state: dict[str, tuple[datetime, np.ndarray]] | None = None,
):
    lat_idx, lon_idx = region_lookup[region.dirname]
    return executor.submit(
        process_region_timestamp,
        region,
        timestamp,
        files_by_var,
        accum_vars,
        lat_idx,
        lon_idx,
        global_lat[lat_idx],
        global_lon[lon_idx],
        output_root,
        overwrite,
        previous_state or {},
    )


def process_without_accumulation_parallel(
    timestamps: list[datetime],
    inventory: dict[datetime, dict[str, Path]],
    variables: list[str],
    region_lookup: dict[str, tuple[np.ndarray, np.ndarray]],
    global_lat: np.ndarray,
    global_lon: np.ndarray,
    args: argparse.Namespace,
) -> None:
    if args.workers == 1:
        total_tasks = len(timestamps) * len(REGIONS)
        completed = 0
        for timestamp in timestamps:
            files_by_var = inventory[timestamp]
            stamp = timestamp.strftime("%Y%m%d_%H%M")
            missing = sorted(set(variables) - set(files_by_var))
            if missing:
                print(f"[WARN] {stamp} missing variables: {', '.join(missing)}", file=sys.stderr)

            for region in REGIONS:
                lat_idx, lon_idx = region_lookup[region.dirname]
                process_region_timestamp(
                    region,
                    timestamp,
                    files_by_var,
                    set(),
                    lat_idx,
                    lon_idx,
                    global_lat[lat_idx],
                    global_lon[lon_idx],
                    args.output_root,
                    args.overwrite,
                    {},
                )
                completed += 1
                if completed == 1 or completed % 1000 == 0 or completed == total_tasks:
                    print(f"[INFO] Processed tasks {completed}/{total_tasks}", file=sys.stderr)
        return

    max_pending = args.max_pending or args.workers * 4
    total_tasks = len(timestamps) * len(REGIONS)
    completed = 0
    pending = set()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for timestamp in timestamps:
            files_by_var = inventory[timestamp]
            stamp = timestamp.strftime("%Y%m%d_%H%M")
            missing = sorted(set(variables) - set(files_by_var))
            if missing:
                print(f"[WARN] {stamp} missing variables: {', '.join(missing)}", file=sys.stderr)

            for region in REGIONS:
                pending.add(
                    submit_region_task(
                        executor,
                        region,
                        timestamp,
                        files_by_var,
                        set(),
                        region_lookup,
                        global_lat,
                        global_lon,
                        args.output_root,
                        args.overwrite,
                    )
                )

                if len(pending) >= max_pending:
                    done = next(as_completed(pending))
                    done.result()
                    pending.remove(done)
                    completed += 1
                    if completed == 1 or completed % 1000 == 0:
                        print(
                            f"[INFO] Processed tasks {completed}/{total_tasks}",
                            file=sys.stderr,
                        )

        for future in as_completed(pending):
            future.result()
            completed += 1
            if completed == 1 or completed % 1000 == 0 or completed == total_tasks:
                print(f"[INFO] Processed tasks {completed}/{total_tasks}", file=sys.stderr)


def process_with_accumulation_ordered(
    timestamps: list[datetime],
    inventory: dict[datetime, dict[str, Path]],
    variables: list[str],
    accum_vars: set[str],
    region_lookup: dict[str, tuple[np.ndarray, np.ndarray]],
    global_lat: np.ndarray,
    global_lon: np.ndarray,
    args: argparse.Namespace,
) -> None:
    previous_accum: dict[str, dict[str, tuple[datetime, np.ndarray]]] = {
        region.dirname: {} for region in REGIONS
    }
    total = len(timestamps)
    executor = (
        ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    )
    try:
        for index, timestamp in enumerate(timestamps, start=1):
            files_by_var = inventory[timestamp]
            stamp = timestamp.strftime("%Y%m%d_%H%M")
            missing = sorted(set(variables) - set(files_by_var))
            if missing:
                print(f"[WARN] {stamp} missing variables: {', '.join(missing)}", file=sys.stderr)

            if executor is None:
                for region in REGIONS:
                    lat_idx, lon_idx = region_lookup[region.dirname]
                    region_name, next_state = process_region_timestamp(
                        region,
                        timestamp,
                        files_by_var,
                        accum_vars,
                        lat_idx,
                        lon_idx,
                        global_lat[lat_idx],
                        global_lon[lon_idx],
                        args.output_root,
                        args.overwrite,
                        previous_accum[region.dirname],
                    )
                    previous_accum[region_name] = next_state
            else:
                futures = []
                for region in REGIONS:
                    futures.append(
                        submit_region_task(
                            executor,
                            region,
                            timestamp,
                            files_by_var,
                            accum_vars,
                            region_lookup,
                            global_lat,
                            global_lon,
                            args.output_root,
                            args.overwrite,
                            previous_accum[region.dirname],
                        )
                    )
                for future in as_completed(futures):
                    region_name, next_state = future.result()
                    previous_accum[region_name] = next_state

            if index == 1 or index % 100 == 0 or index == total:
                print(f"[INFO] Processed {index}/{total}: {stamp}", file=sys.stderr)
    finally:
        if executor is not None:
            executor.shutdown()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        print("[ERROR] --workers must be >= 1", file=sys.stderr)
        return 2
    if args.max_pending is not None and args.max_pending < args.workers:
        print("[ERROR] --max-pending must be >= --workers", file=sys.stderr)
        return 2

    roots = input_roots(args.input_root, args.supplemental_input_root)
    variables = discover_variables(roots, args.variables)
    if not variables:
        print(
            f"[ERROR] No variable directories found: {', '.join(str(root) for root in roots)}",
            file=sys.stderr,
        )
        return 2

    start = parse_bound(args.start)
    end = parse_bound(args.end)
    inventory = build_inventory(roots, variables, start, end)
    if not inventory:
        print("[ERROR] No matching input files found", file=sys.stderr)
        return 2

    first_path = next(iter(next(iter(inventory.values())).values()))
    global_lat, global_lon = read_global_coords(first_path)
    region_lookup = {
        region.dirname: region_indices(global_lat, global_lon, region)
        for region in REGIONS
    }

    if args.accum_vars is not None:
        accum_vars = set(args.accum_vars)
    elif args.no_auto_diff:
        accum_vars = set()
    else:
        accum_vars = detect_accumulated_variables(
            roots, variables, REGIONS[0], global_lat, global_lon
        )

    for region in REGIONS:
        (args.output_root / region.dirname).mkdir(parents=True, exist_ok=True)

    timestamps = list(inventory)
    if args.repair_missing_output_vars:
        repair_timestamps = timestamps_with_missing_output_vars(
            args.output_root, variables, start, end
        )
        if not repair_timestamps:
            print("[INFO] No regional output files with missing variables; no repair needed.", file=sys.stderr)
            return 0
        timestamps = [ts for ts in timestamps if ts in repair_timestamps]
        missing_sources = sorted(repair_timestamps - set(timestamps))
        if missing_sources:
            for ts in missing_sources[:20]:
                print(
                    f"[WARN] {ts:%Y%m%d_%H%M} has missing output variables, but the input source still lacks this timestamp",
                    file=sys.stderr,
                )
            if len(missing_sources) > 20:
                print(
                    f"[WARN] Additionally, {len(missing_sources) - 20} missing-variable timestamps still have no input source",
                    file=sys.stderr,
                )
        args.overwrite = True
        print(
            f"[INFO] repair mode: reprocessing {len(timestamps)} timestamps with missing variables and overwriting corresponding outputs.",
            file=sys.stderr,
        )
        if not timestamps:
            print("[WARN] No input timestamps available for repair; check whether supplemental downloads are complete.", file=sys.stderr)
            return 1

    if args.max_hours is not None:
        timestamps = timestamps[: args.max_hours]

    if accum_vars:
        print(
            f"[INFO] Detected accumulated variables: {', '.join(sorted(accum_vars))}; using time-ordered parallel mode",
            file=sys.stderr,
        )
        process_with_accumulation_ordered(
            timestamps,
            inventory,
            variables,
            accum_vars,
            region_lookup,
            global_lat,
            global_lon,
            args,
        )
    else:
        print(
            f"[INFO] No accumulated variables detected; using global parallel mode, workers={args.workers}",
            file=sys.stderr,
        )
        process_without_accumulation_parallel(
            timestamps,
            inventory,
            variables,
            region_lookup,
            global_lat,
            global_lon,
            args,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

