#!/usr/bin/env python3
"""Step 5: prepend 2000-2009 part2 train data and rewrite stats.json."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from netCDF4 import Dataset, date2num, num2date
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency netCDF4. Install it first: conda install -c conda-forge netcdf4"
    ) from exc


SRC_ROOT = Path("./dataset_root2000")
DST_ROOT = Path("./dataset_root")
BACKUP_SUFFIX = ".before_dataset_root2000_merge"
LOG_EPS = 1.0e-6


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

    def to_dict(self) -> dict[str, float | None]:
        if self.count == 0:
            return {"mean": None, "std": None, "min": None, "max": None}
        mean = self.total / self.count
        variance = max(self.total_sq / self.count - mean * mean, 0.0)
        return {
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.min_value,
            "max": self.max_value,
        }


@dataclass
class PairStats:
    raw: RunningStats
    log: RunningStats | None = None

    def to_dict(self, include_log: bool = False, log_eps: float = LOG_EPS) -> dict[str, float | None]:
        payload = self.raw.to_dict()
        if include_log:
            log_stats = self.log.to_dict() if self.log else RunningStats().to_dict()
            payload.update(
                {
                    "log_mean": log_stats["mean"],
                    "log_std": log_stats["std"],
                    "log_min": log_stats["min"],
                    "log_max": log_stats["max"],
                    "log_eps": log_eps,
                }
            )
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepend dataset_root2000 part2 train*.nc files to dataset_root and rewrite stats.json."
    )
    parser.add_argument("--src-root", type=Path, default=SRC_ROOT)
    parser.add_argument("--dst-root", type=Path, default=DST_ROOT)
    parser.add_argument("--regions", nargs="+", default=None)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--log-eps", type=float, default=LOG_EPS)
    parser.add_argument("--stats-only", action="store_true", help="Rewrite stats.json in the new format only; do not merge NetCDF files")
    parser.add_argument("--force", action="store_true", help="Force merge even if target train already starts from 2000")
    parser.add_argument("--no-backup", action="store_true", help="Do not back up overwritten files")
    return parser.parse_args()


def data_vars(nc: Dataset) -> list[str]:
    return [
        name
        for name, var in nc.variables.items()
        if name not in {"time", "latitude", "longitude"}
        and tuple(var.dimensions) == ("time", "latitude", "longitude")
    ]


def decoded_times(nc: Dataset):
    time_var = nc.variables["time"]
    return num2date(
        time_var[:],
        time_var.units,
        getattr(time_var, "calendar", "standard"),
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )


def first_year(nc: Dataset) -> int:
    return decoded_times(nc)[0].year


def copy_attrs(src, dst) -> None:
    for attr in src.ncattrs():
        if attr == "_FillValue":
            continue
        try:
            dst.setncattr(attr, src.getncattr(attr))
        except Exception:
            pass


def create_like(out: Dataset, name: str, src_var, dimensions: tuple[str, ...]):
    kwargs = {"zlib": True, "complevel": 4}
    if "_FillValue" in src_var.ncattrs():
        kwargs["fill_value"] = src_var.getncattr("_FillValue")
    try:
        out_var = out.createVariable(name, src_var.datatype, dimensions, **kwargs)
    except TypeError:
        out_var = out.createVariable(name, src_var.datatype, dimensions)
    copy_attrs(src_var, out_var)
    return out_var


def backup_once(path: Path) -> None:
    backup_path = path.with_name(path.name + BACKUP_SUFFIX)
    if backup_path.exists():
        return
    shutil.copy2(path, backup_path)
    print(f"[BACKUP] {backup_path}")


def validate_compatible(src_nc: Dataset, dst_nc: Dataset, src_path: Path, dst_path: Path) -> list[str]:
    src_vars = data_vars(src_nc)
    dst_vars = data_vars(dst_nc)
    if src_vars != dst_vars:
        raise ValueError(f"Variable mismatch: {src_path} {src_vars} vs {dst_path} {dst_vars}")
    for dim in ("latitude", "longitude"):
        if len(src_nc.dimensions[dim]) != len(dst_nc.dimensions[dim]):
            raise ValueError(
                f"{dim} dimension mismatch; cannot concatenate: {src_path}={len(src_nc.dimensions[dim])}, "
                f"{dst_path}={len(dst_nc.dimensions[dim])}. Rebuild 2010-2021 part2 with the same spatial bounds first."
            )
    if not np.allclose(src_nc.variables["latitude"][:], dst_nc.variables["latitude"][:]):
        raise ValueError(f"latitude coordinate mismatch; cannot concatenate: {src_path} vs {dst_path}")
    if not np.allclose(src_nc.variables["longitude"][:], dst_nc.variables["longitude"][:]):
        raise ValueError(f"longitude coordinate mismatch; cannot concatenate: {src_path} vs {dst_path}")
    return src_vars


def update_stats_from_values(stats: dict[str, PairStats], name: str, values: np.ndarray, log_eps: float) -> None:
    if name not in stats:
        stats[name] = PairStats(raw=RunningStats(), log=RunningStats() if name == "tp" else None)
    stats[name].raw.update(values)
    if name == "tp" and stats[name].log is not None:
        stats[name].log.update(np.log1p(np.maximum(values, 0.0) / log_eps))


def copy_data_block(
    out_var,
    src_var,
    output_offset: int,
    chunk_size: int,
    stats: dict[str, PairStats],
    name: str,
    log_eps: float,
) -> None:
    total = src_var.shape[0]
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        values = np.asarray(src_var[start:end, :, :], dtype=np.float32)
        out_var[output_offset + start : output_offset + end, :, :] = values
        update_stats_from_values(stats, name, values, log_eps)


def write_time_values(out: Dataset, src_nc: Dataset, dst_nc: Dataset) -> None:
    out_time = out.variables["time"]
    all_times = list(decoded_times(src_nc)) + list(decoded_times(dst_nc))
    out_time[:] = date2num(
        all_times,
        out_time.units,
        getattr(out_time, "calendar", "standard"),
    )


def merge_train_file(src_path: Path, dst_path: Path, args: argparse.Namespace) -> bool:
    with Dataset(src_path) as src_nc, Dataset(dst_path) as dst_nc:
        if first_year(dst_nc) <= 2000 and not args.force:
            print(f"[SKIP] Target train already starts from 2000: {dst_path}")
            return False

        variables = validate_compatible(src_nc, dst_nc, src_path, dst_path)
        src_len = len(src_nc.dimensions["time"])
        dst_len = len(dst_nc.dimensions["time"])
        tmp_path = dst_path.with_name(f".{dst_path.name}.tmp.{os.getpid()}")
        stats: dict[str, PairStats] = {}

        with Dataset(tmp_path, "w", format="NETCDF4") as out:
            out.createDimension("time", None)
            out.createDimension("latitude", len(dst_nc.dimensions["latitude"]))
            out.createDimension("longitude", len(dst_nc.dimensions["longitude"]))

            time_var = out.createVariable("time", "f8", ("time",))
            copy_attrs(dst_nc.variables["time"], time_var)
            time_var.units = "seconds since 2000-01-01 00:00:00"
            time_var.calendar = getattr(dst_nc.variables["time"], "calendar", "standard")

            for coord in ("latitude", "longitude"):
                coord_var = create_like(out, coord, dst_nc.variables[coord], dst_nc.variables[coord].dimensions)
                coord_var[:, :] = dst_nc.variables[coord][:]

            for name in variables:
                out_var = create_like(out, name, dst_nc.variables[name], dst_nc.variables[name].dimensions)
                copy_data_block(out_var, src_nc.variables[name], 0, args.chunk_size, stats, name, args.log_eps)
                copy_data_block(
                    out_var,
                    dst_nc.variables[name],
                    src_len,
                    args.chunk_size,
                    stats,
                    name,
                    args.log_eps,
                )

            write_time_values(out, src_nc, dst_nc)
            for attr in dst_nc.ncattrs():
                try:
                    out.setncattr(attr, dst_nc.getncattr(attr))
                except Exception:
                    pass
            out.history = (
                f"Merged dataset_root2000 train data before existing train data at "
                f"{datetime.utcnow():%Y-%m-%dT%H:%M:%SZ}"
            )

        if not args.no_backup:
            backup_once(dst_path)
        os.replace(tmp_path, dst_path)

    print(f"[MERGED] {dst_path} time={src_len + dst_len} ({src_len}+{dst_len})")
    return True


def compute_file_stats(path: Path, log_eps: float, chunk_size: int) -> dict[str, PairStats]:
    stats: dict[str, PairStats] = {}
    with Dataset(path) as nc:
        for name in data_vars(nc):
            var = nc.variables[name]
            total = var.shape[0]
            for start in range(0, total, chunk_size):
                end = min(start + chunk_size, total)
                update_stats_from_values(stats, name, np.asarray(var[start:end, :, :], dtype=np.float32), log_eps)
    return stats


def stats_payload(
    input_stats: dict[str, PairStats],
    output_stats: dict[str, PairStats],
    log_eps: float,
) -> dict[str, dict[str, dict[str, float | None]]]:
    return {
        "input": {
            name: stat.to_dict(include_log=(name == "tp"), log_eps=log_eps)
            for name, stat in sorted(input_stats.items())
        },
        "output": {
            name: stat.to_dict(include_log=(name == "tp"), log_eps=log_eps)
            for name, stat in sorted(output_stats.items())
        },
    }


def rewrite_stats_json(region_dir: Path, args: argparse.Namespace) -> None:
    train_in = region_dir / "train_data_in.nc"
    train_out = region_dir / "train_data_out.nc"
    input_stats = compute_file_stats(train_in, args.log_eps, args.chunk_size) if train_in.exists() else {}
    output_stats = compute_file_stats(train_out, args.log_eps, args.chunk_size) if train_out.exists() else {}
    stats_path = region_dir / "stats.json"
    if stats_path.exists() and not args.no_backup:
        backup_once(stats_path)
    stats_path.write_text(
        json.dumps(stats_payload(input_stats, output_stats, args.log_eps), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[STATS] {stats_path}")


def discover_regions(src_root: Path, dst_root: Path, regions: list[str] | None) -> list[str]:
    if regions:
        return regions
    src_regions = {p.name for p in src_root.iterdir() if p.is_dir() and p.name.startswith("part2-")}
    dst_regions = {p.name for p in dst_root.iterdir() if p.is_dir()}
    return sorted(src_regions & dst_regions)


def main() -> int:
    args = parse_args()
    regions = discover_regions(args.src_root, args.dst_root, args.regions)
    for region in regions:
        region_dir = args.dst_root / region
        if not args.stats_only:
            src_path = args.src_root / region / "train_data_in.nc"
            dst_path = region_dir / "train_data_in.nc"
            if not src_path.exists():
                print(f"[WARN] Source file missing, skip merge: {src_path}")
            elif not dst_path.exists():
                print(f"[WARN] Target file missing, skip merge: {dst_path}")
            else:
                try:
                    merge_train_file(src_path, dst_path, args)
                except Exception as exc:
                    print(f"[ERROR] {region} merge failed: {exc}")
        rewrite_stats_json(region_dir, args)

    if args.regions is None:
        for region_dir in sorted(args.dst_root.iterdir()):
            if not region_dir.is_dir() or region_dir.name in regions:
                continue
            if (region_dir / "train_data_in.nc").exists() or (region_dir / "train_data_out.nc").exists():
                rewrite_stats_json(region_dir, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
