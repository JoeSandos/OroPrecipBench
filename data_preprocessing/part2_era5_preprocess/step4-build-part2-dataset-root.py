#!/usr/bin/env python3
"""Step 4: build train/val/test NetCDF files from regional ERA5 files.

Use --scales part2 for daily ERA5 part2 datasets. Daily sums are used for
tp/ssrd/tisr; other variables are averaged.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np

try:
    from netCDF4 import Dataset, date2num
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency netCDF4. Install it first: conda install -c conda-forge netcdf4"
    ) from exc


ERA5_DATA_ROOT = Path("./era5_data")
DATASET_ROOT = Path("./dataset_root")
FILENAME_RE = re.compile(r"^era5_(?P<region>.+)_(?P<date>\d{8})_(?P<hour>\d{4})\.nc$")
COORD_VARS = {"time", "valid_time", "latitude", "longitude"}

PART1_SPLITS = {
    "train": (datetime(2015, 1, 1, 0), datetime(2022, 12, 31, 23)),
    "val": (datetime(2023, 1, 1, 0), datetime(2023, 12, 31, 23)),
    "test": (datetime(2024, 1, 1, 0), datetime(2024, 12, 31, 23)),
}
PART2_SPLITS = {
    "train": (datetime(2010, 1, 1, 0), datetime(2021, 12, 31, 23)),
    "val": (datetime(2022, 1, 1, 0), datetime(2022, 12, 31, 23)),
    "test": (datetime(2023, 1, 1, 0), datetime(2024, 12, 31, 23)),
}
DEFAULT_PART2_EXCLUDE_DATES = ["2018-08-22"]


@dataclass
class RunningStats:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    min_value: float = math.inf
    max_value: float = -math.inf

    def update(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return
        self.count += int(finite.size)
        self.total += float(finite.sum())
        self.total_sq += float(np.square(finite).sum())
        self.min_value = min(self.min_value, float(finite.min()))
        self.max_value = max(self.max_value, float(finite.max()))

    def to_dict(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
        mean = self.total / self.count
        variance = max(self.total_sq / self.count - mean * mean, 0.0)
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.min_value,
            "max": self.max_value,
        }


@dataclass
class RegionMeta:
    lat_values: np.ndarray
    lon_values: np.ndarray
    data_vars: list[str]
    var_attrs: dict[str, dict[str, object]]


@dataclass
class DatasetWriter:
    path: Path
    meta: RegionMeta
    time_units: str
    calendar: str = "standard"
    dataset: Dataset | None = None
    time_index: int = 0

    def __enter__(self) -> "DatasetWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(f".{self.path.name}.tmp.{os.getpid()}")
        self.dataset = Dataset(tmp_path, "w", format="NETCDF4")
        self.path = self.path
        self._tmp_path = tmp_path
        self._create_schema()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self.dataset is not None
        self.dataset.close()
        if exc_type is None:
            os.replace(self._tmp_path, self.path)
        elif self._tmp_path.exists():
            self._tmp_path.unlink()

    def _create_schema(self) -> None:
        assert self.dataset is not None
        ds = self.dataset
        nlat, nlon = self.meta.lat_values.shape
        ds.createDimension("time", None)
        ds.createDimension("latitude", nlat)
        ds.createDimension("longitude", nlon)

        time_var = ds.createVariable("time", "f8", ("time",))
        time_var.units = self.time_units
        time_var.calendar = self.calendar
        time_var.standard_name = "time"
        time_var.long_name = "time"

        lat_var = ds.createVariable(
            "latitude", "f4", ("latitude", "longitude"), zlib=True, complevel=4
        )
        lat_var.units = "degrees_north"
        lat_var.standard_name = "latitude"
        lat_var.long_name = "latitude"
        lat_var[:, :] = self.meta.lat_values.astype(np.float32)

        lon_var = ds.createVariable(
            "longitude", "f4", ("latitude", "longitude"), zlib=True, complevel=4
        )
        lon_var.units = "degrees_east"
        lon_var.standard_name = "longitude"
        lon_var.long_name = "longitude"
        lon_var[:, :] = self.meta.lon_values.astype(np.float32)

        for name in self.meta.data_vars:
            out_var = ds.createVariable(
                name,
                "f4",
                ("time", "latitude", "longitude"),
                zlib=True,
                complevel=4,
                fill_value=np.float32(np.nan),
            )
            for attr, value in self.meta.var_attrs.get(name, {}).items():
                if attr.startswith("_") or attr == "coordinates":
                    continue
                try:
                    out_var.setncattr(attr, value)
                except Exception:
                    pass
            out_var.coordinates = "latitude longitude"

        ds.Conventions = "CF-1.7"
        ds.history = f"Created by step4-build-part2-dataset-root.py at {datetime.utcnow():%Y-%m-%dT%H:%M:%SZ}"

    def write_step(self, timestamp: datetime, arrays: dict[str, np.ndarray]) -> None:
        assert self.dataset is not None
        ds = self.dataset
        ds.variables["time"][self.time_index] = date2num(
            timestamp, ds.variables["time"].units, ds.variables["time"].calendar
        )
        for name in self.meta.data_vars:
            ds.variables[name][self.time_index, :, :] = arrays[name].astype(np.float32)
        self.time_index += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dataset_root structure from regional ERA5 files")
    parser.add_argument("--input-root", type=Path, default=ERA5_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DATASET_ROOT)
    parser.add_argument(
        "--regions",
        nargs="+",
        default=None,
        help="Region directory names; default: part1-US-* under input-root.",
    )
    parser.add_argument("--variables", nargs="+", default=None, help="Variable list; default: read from a sample file.")
    parser.add_argument(
        "--scales",
        nargs="+",
        choices=["part1", "part2"],
        default=["part1", "part2"],
        help="Output hourly part1 and daily part2; default: both.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument(
        "--coarsen-factor",
        type=int,
        default=1,
        help="Spatial coarsening factor; 1 means no coarsening, 2 means 2x2 average.",
    )
    parser.add_argument(
        "--daily-sum-vars",
        nargs="+",
        default=["tp", "ssrd", "tisr"],
        help="Daily sum variables; default: tp ssrd tisr.",
    )
    parser.add_argument("--daily-max-vars", nargs="*", default=[], help="Daily max variables")
    parser.add_argument("--daily-min-vars", nargs="*", default=[], help="Daily min variables")
    parser.add_argument(
        "--allow-incomplete-days",
        action="store_true",
        help="Allow daily aggregation with fewer than 24 hours; default: skip incomplete days.",
    )
    parser.add_argument(
        "--part2-exclude-dates",
        nargs="*",
        default=DEFAULT_PART2_EXCLUDE_DATES,
        help="Dates to exclude from part2 daily data, format YYYY-MM-DD; default excludes 2018-08-22.",
    )
    return parser.parse_args()


def parse_timestamp(path: Path) -> datetime | None:
    match = FILENAME_RE.match(path.name)
    if not match:
        return None
    return datetime.strptime(match.group("date") + match.group("hour"), "%Y%m%d%H%M")


def parse_exclude_dates(values: list[str]) -> set[date]:
    return {datetime.strptime(value, "%Y-%m-%d").date() for value in values}


def discover_regions(input_root: Path, regions: list[str] | None) -> list[str]:
    if regions:
        return regions
    return sorted(p.name for p in input_root.iterdir() if p.is_dir() and p.name.startswith("part1-US-"))


def files_for_region(region_dir: Path) -> list[tuple[datetime, Path]]:
    pairs: list[tuple[datetime, Path]] = []
    for path in sorted(region_dir.glob("era5_*.nc")):
        ts = parse_timestamp(path)
        if ts is not None:
            pairs.append((ts, path))
    return pairs


def in_range(ts: datetime, bounds: tuple[datetime, datetime]) -> bool:
    return bounds[0] <= ts <= bounds[1]


def read_region_meta(sample_path: Path, variables: list[str] | None, coarsen_factor: int) -> RegionMeta:
    with Dataset(sample_path) as ds:
        lat = np.asarray(ds.variables["latitude"][:], dtype=np.float32)
        lon = np.asarray(ds.variables["longitude"][:], dtype=np.float32)
        data_vars = [
            name
            for name, var in ds.variables.items()
            if name not in COORD_VARS and tuple(var.dimensions) == ("time", "latitude", "longitude")
        ]
        if variables:
            data_vars = [name for name in variables if name in data_vars]
        var_attrs = {
            name: {attr: ds.variables[name].getncattr(attr) for attr in ds.variables[name].ncattrs()}
            for name in data_vars
        }

    lat = coarsen_2d(lat, coarsen_factor)
    lon = coarsen_2d(lon, coarsen_factor)
    return RegionMeta(lat_values=lat, lon_values=lon, data_vars=data_vars, var_attrs=var_attrs)


def coarsen_2d(values: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return values
    nlat = values.shape[0] // factor * factor
    nlon = values.shape[1] // factor * factor
    trimmed = values[:nlat, :nlon]
    return np.nanmean(trimmed.reshape(nlat // factor, factor, nlon // factor, factor), axis=(1, 3))


def coarsen_data(values: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return values
    nlat = values.shape[0] // factor * factor
    nlon = values.shape[1] // factor * factor
    trimmed = values[:nlat, :nlon]
    return np.nanmean(trimmed.reshape(nlat // factor, factor, nlon // factor, factor), axis=(1, 3))


def read_arrays(path: Path, variables: list[str], coarsen_factor: int) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    with Dataset(path) as ds:
        for name in variables:
            data = np.asarray(ds.variables[name][0, :, :], dtype=np.float32)
            arrays[name] = coarsen_data(data, coarsen_factor)
    return arrays


def write_stats(
    path: Path,
    scale: str,
    split_bounds: dict[str, tuple[datetime, datetime]],
    stats: dict[str, RunningStats],
    excluded_dates: set[date] | None = None,
) -> None:
    payload = {
        "scale": scale,
        "normalization_source": "train",
        "splits": {
            name: {
                "start": bounds[0].strftime("%Y-%m-%d %H:%M:%S"),
                "end": bounds[1].strftime("%Y-%m-%d %H:%M:%S"),
            }
            for name, bounds in split_bounds.items()
        },
        "variables": {name: stat.to_dict() for name, stat in sorted(stats.items())},
    }
    if excluded_dates:
        payload["excluded_dates"] = [day.isoformat() for day in sorted(excluded_dates)]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def time_units_for_split(bounds: tuple[datetime, datetime]) -> str:
    return f"seconds since {bounds[0].year:04d}-01-01 00:00:00"


def build_part1(region: str, region_files: list[tuple[datetime, Path]], meta: RegionMeta, args: argparse.Namespace) -> None:
    out_dir = args.output_root / region
    out_dir.mkdir(parents=True, exist_ok=True)
    train_stats = {name: RunningStats() for name in meta.data_vars}

    for split, bounds in PART1_SPLITS.items():
        selected = [(ts, path) for ts, path in region_files if in_range(ts, bounds)]
        out_path = out_dir / f"{split}_data_in.nc"
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] Already exists: {out_path}")
            continue
        print(f"[INFO] {region} part1 {split}: {len(selected)} hourly steps")
        with DatasetWriter(out_path, meta, time_units_for_split(bounds)) as writer:
            for ts, path in selected:
                arrays = read_arrays(path, meta.data_vars, args.coarsen_factor)
                writer.write_step(ts, arrays)
                if split == "train":
                    for name, arr in arrays.items():
                        train_stats[name].update(arr)

    write_stats(out_dir / "stats.json", "part1-hourly", PART1_SPLITS, train_stats)


def update_daily_attrs(meta: RegionMeta, sum_vars: set[str]) -> RegionMeta:
    attrs = {name: dict(value) for name, value in meta.var_attrs.items()}
    for name in sum_vars:
        if name not in attrs:
            continue
        if name == "tp":
            attrs[name]["units"] = "mm day-1"
        attrs[name]["daily_aggregation"] = "sum"
    for name in attrs:
        attrs[name].setdefault("daily_aggregation", "mean")
    return RegionMeta(meta.lat_values, meta.lon_values, meta.data_vars, attrs)


def daily_groups(region_files: list[tuple[datetime, Path]]) -> dict[datetime, list[tuple[datetime, Path]]]:
    groups: dict[datetime, list[tuple[datetime, Path]]] = defaultdict(list)
    for ts, path in region_files:
        groups[datetime(ts.year, ts.month, ts.day)].append((ts, path))
    return {day: sorted(items) for day, items in sorted(groups.items())}


def aggregate_day(
    hourly_items: list[tuple[datetime, Path]],
    variables: list[str],
    coarsen_factor: int,
    sum_vars: set[str],
    max_vars: set[str],
    min_vars: set[str],
) -> dict[str, np.ndarray]:
    stacks: dict[str, list[np.ndarray]] = {name: [] for name in variables}
    for _, path in hourly_items:
        arrays = read_arrays(path, variables, coarsen_factor)
        for name, arr in arrays.items():
            stacks[name].append(arr)

    daily: dict[str, np.ndarray] = {}
    for name, values in stacks.items():
        stack = np.stack(values, axis=0)
        if name in sum_vars:
            daily[name] = np.nansum(stack, axis=0).astype(np.float32)
        elif name in max_vars:
            daily[name] = np.nanmax(stack, axis=0).astype(np.float32)
        elif name in min_vars:
            daily[name] = np.nanmin(stack, axis=0).astype(np.float32)
        else:
            daily[name] = np.nanmean(stack, axis=0).astype(np.float32)
    return daily


def build_part2(region: str, region_files: list[tuple[datetime, Path]], meta: RegionMeta, args: argparse.Namespace) -> None:
    out_region = region.replace("part1-", "part2-", 1)
    out_dir = args.output_root / out_region
    out_dir.mkdir(parents=True, exist_ok=True)

    sum_vars = set(args.daily_sum_vars)
    max_vars = set(args.daily_max_vars)
    min_vars = set(args.daily_min_vars)
    excluded_dates = parse_exclude_dates(args.part2_exclude_dates)
    daily_meta = update_daily_attrs(meta, sum_vars)
    groups = daily_groups(region_files)
    train_stats = {name: RunningStats() for name in daily_meta.data_vars}

    for split, bounds in PART2_SPLITS.items():
        selected_days = [
            (day, items)
            for day, items in groups.items()
            if bounds[0].date() <= day.date() <= bounds[1].date()
            and day.date() not in excluded_dates
            and (args.allow_incomplete_days or len(items) == 24)
        ]
        skipped = sum(
            1
            for day, items in groups.items()
            if bounds[0].date() <= day.date() <= bounds[1].date()
            and len(items) != 24
        )
        excluded_in_split = sum(
            1
            for day in groups
            if bounds[0].date() <= day.date() <= bounds[1].date()
            and day.date() in excluded_dates
        )
        if skipped:
            print(f"[WARN] {out_region} part2 {split}: Skipped {skipped} incomplete days")
        if excluded_in_split:
            print(f"[INFO] {out_region} part2 {split}: Excluded {excluded_in_split} specified dates")
        out_path = out_dir / f"{split}_data_in.nc"
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] Already exists: {out_path}")
            continue
        print(f"[INFO] {out_region} part2 {split}: {len(selected_days)} daily steps")
        with DatasetWriter(out_path, daily_meta, time_units_for_split(bounds)) as writer:
            for day, items in selected_days:
                arrays = aggregate_day(
                    items,
                    daily_meta.data_vars,
                    args.coarsen_factor,
                    sum_vars,
                    max_vars,
                    min_vars,
                )
                writer.write_step(day, arrays)
                if split == "train":
                    for name, arr in arrays.items():
                        train_stats[name].update(arr)

    write_stats(out_dir / "stats.json", "part2-daily", PART2_SPLITS, train_stats, excluded_dates)


def main() -> int:
    args = parse_args()
    if args.coarsen_factor < 1:
        raise SystemExit("--coarsen-factor must be >= 1")

    regions = discover_regions(args.input_root, args.regions)
    if not regions:
        raise SystemExit(f"No region directories found: {args.input_root}/part1-US-*")

    args.output_root.mkdir(parents=True, exist_ok=True)
    for region in regions:
        region_dir = args.input_root / region
        region_files = files_for_region(region_dir)
        if not region_files:
            print(f"[WARN] {region} has no NetCDF files, skipping")
            continue
        meta = read_region_meta(region_files[0][1], args.variables, args.coarsen_factor)
        print(f"[INFO] Processing {region}: {len(region_files)} hourly files, vars={','.join(meta.data_vars)}")
        if "part1" in args.scales:
            build_part1(region, region_files, meta, args)
        if "part2" in args.scales:
            build_part2(region, region_files, meta, args)

    print(f"[DONE] dataset_root generated: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
