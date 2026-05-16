#!/usr/bin/env python3
"""Step 6: interpolate part2 ERA5 inputs to the 24x24 HRRR low-resolution grid."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

try:
    from netCDF4 import Dataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency netCDF4. Install it first: conda install -c conda-forge netcdf4"
    ) from exc


SRC_ROOT = Path("./dataset_root")
TARGET_ROOT = Path("./hrrr_dataset_root")
OUT_ROOT = Path("./dataset_root_new")
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
class VarStats:
    raw: RunningStats
    log: RunningStats | None = None

    def update(self, name: str, values: np.ndarray, log_eps: float) -> None:
        self.raw.update(values)
        if name == "tp" and self.log is not None:
            self.log.update(np.log1p(np.maximum(values, 0.0) / log_eps))

    def to_dict(self, include_log: bool, log_eps: float) -> dict[str, float | None]:
        payload = self.raw.to_dict()
        if include_log:
            log_payload = self.log.to_dict() if self.log else RunningStats().to_dict()
            payload.update(
                {
                    "log_mean": log_payload["mean"],
                    "log_std": log_payload["std"],
                    "log_min": log_payload["min"],
                    "log_max": log_payload["max"],
                    "log_eps": log_eps,
                }
            )
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interpolate dataset_root/part2-* ERA5 inputs to the Part1 HRRR 24x24 low-resolution grid."
    )
    parser.add_argument("--src-root", type=Path, default=SRC_ROOT)
    parser.add_argument("--target-root", type=Path, default=TARGET_ROOT)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--regions", nargs="+", default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--log-eps", type=float, default=LOG_EPS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def part1_region_name(part2_region: str) -> str:
    return part2_region.replace("part2-", "part1-", 1)


def data_vars(nc: Dataset) -> list[str]:
    return [
        name
        for name, var in nc.variables.items()
        if name not in {"time", "latitude", "longitude"}
        and tuple(var.dimensions) == ("time", "latitude", "longitude")
    ]


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


def read_target_grid(target_file: Path) -> tuple[np.ndarray, np.ndarray]:
    with Dataset(target_file) as nc:
        lat = np.asarray(nc.variables["latitude"][:], dtype=np.float64)
        lon = np.mod(np.asarray(nc.variables["longitude"][:], dtype=np.float64), 360.0)
    return lat, lon


def regular_axes(src_nc: Dataset) -> tuple[np.ndarray, np.ndarray, bool]:
    lat2d = np.asarray(src_nc.variables["latitude"][:], dtype=np.float64)
    lon2d = np.mod(np.asarray(src_nc.variables["longitude"][:], dtype=np.float64), 360.0)
    lat_axis = lat2d[:, 0]
    lon_axis = lon2d[0, :]
    reverse_lat = bool(lat_axis[0] > lat_axis[-1])
    if reverse_lat:
        lat_axis = lat_axis[::-1]
    if lon_axis[0] > lon_axis[-1]:
        raise ValueError("This script assumes the source longitude axis is increasing")
    return lat_axis, lon_axis, reverse_lat


def init_stats(var_names: list[str]) -> dict[str, VarStats]:
    return {
        name: VarStats(raw=RunningStats(), log=RunningStats() if name == "tp" else None)
        for name in var_names
    }


def interpolate_chunk(
    values: np.ndarray,
    lat_axis: np.ndarray,
    lon_axis: np.ndarray,
    target_points: np.ndarray,
    target_shape: tuple[int, int],
    reverse_lat: bool,
) -> np.ndarray:
    if reverse_lat:
        values = values[:, ::-1, :]
    values_for_interp = np.moveaxis(values, 0, -1)  # lat, lon, time
    interpolator = RegularGridInterpolator(
        (lat_axis, lon_axis),
        values_for_interp,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    interpolated = interpolator(target_points)  # points, time
    return interpolated.T.reshape(values.shape[0], *target_shape).astype(np.float32)


def interpolate_file(
    src_file: Path,
    target_file: Path,
    out_file: Path,
    chunk_size: int,
    log_eps: float,
    overwrite: bool,
) -> dict[str, VarStats]:
    if out_file.exists() and not overwrite:
        print(f"[SKIP] Already exists: {out_file}")
        with Dataset(out_file) as nc:
            return compute_stats(nc, log_eps, chunk_size)

    target_lat, target_lon = read_target_grid(target_file)
    target_points = np.column_stack([target_lat.ravel(), target_lon.ravel()])
    target_shape = target_lat.shape
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = out_file.with_name(f".{out_file.name}.tmp.{os.getpid()}")

    with Dataset(src_file) as src_nc:
        var_names = data_vars(src_nc)
        stats = init_stats(var_names)
        lat_axis, lon_axis, reverse_lat = regular_axes(src_nc)
        ntime = len(src_nc.dimensions["time"])

        with Dataset(tmp_file, "w", format="NETCDF4") as out_nc:
            out_nc.createDimension("time", None)
            out_nc.createDimension("latitude", target_shape[0])
            out_nc.createDimension("longitude", target_shape[1])

            time_var = out_nc.createVariable("time", "f8", ("time",))
            copy_attrs(src_nc.variables["time"], time_var)
            time_var[:] = src_nc.variables["time"][:]

            lat_var = out_nc.createVariable(
                "latitude", "f4", ("latitude", "longitude"), zlib=True, complevel=4
            )
            copy_attrs(src_nc.variables["latitude"], lat_var)
            lat_var[:, :] = target_lat.astype(np.float32)

            lon_var = out_nc.createVariable(
                "longitude", "f4", ("latitude", "longitude"), zlib=True, complevel=4
            )
            copy_attrs(src_nc.variables["longitude"], lon_var)
            lon_var[:, :] = target_lon.astype(np.float32)

            out_vars = {
                name: create_like(out_nc, name, src_nc.variables[name], ("time", "latitude", "longitude"))
                for name in var_names
            }

            for attr in src_nc.ncattrs():
                try:
                    out_nc.setncattr(attr, src_nc.getncattr(attr))
                except Exception:
                    pass
            out_nc.history = "Interpolated ERA5 part2 input to Part1 24x24 HRRR grid"

            for start in range(0, ntime, chunk_size):
                end = min(start + chunk_size, ntime)
                for name in var_names:
                    chunk = np.asarray(src_nc.variables[name][start:end, :, :], dtype=np.float32)
                    interp = interpolate_chunk(
                        chunk, lat_axis, lon_axis, target_points, target_shape, reverse_lat
                    )
                    out_vars[name][start:end, :, :] = interp
                    stats[name].update(name, interp, log_eps)
                print(f"[PROGRESS] {out_file} {end}/{ntime}", flush=True)

    os.replace(tmp_file, out_file)
    return stats


def compute_stats(nc: Dataset, log_eps: float, chunk_size: int) -> dict[str, VarStats]:
    var_names = data_vars(nc)
    stats = init_stats(var_names)
    total = len(nc.dimensions["time"])
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        for name in var_names:
            stats[name].update(name, np.asarray(nc.variables[name][start:end, :, :]), log_eps)
    return stats


def stats_payload(input_stats: dict[str, VarStats], output_stats: dict[str, VarStats], log_eps: float):
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


def rewrite_stats(region_out: Path, train_input_stats: dict[str, VarStats], args: argparse.Namespace) -> None:
    output_stats: dict[str, VarStats] = {}
    train_out = region_out / "train_data_out.nc"
    if train_out.exists():
        with Dataset(train_out) as nc:
            output_stats = compute_stats(nc, args.log_eps, args.chunk_size)
    payload = stats_payload(train_input_stats, output_stats, args.log_eps)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    (region_out / "stats.json").write_text(text, encoding="utf-8")
    (region_out / "status.json").write_text(text, encoding="utf-8")


def discover_regions(src_root: Path, regions: list[str] | None) -> list[str]:
    if regions:
        return regions
    return sorted(p.name for p in src_root.iterdir() if p.is_dir() and p.name.startswith("part2-"))


def main() -> int:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    for region in discover_regions(args.src_root, args.regions):
        src_region = args.src_root / region
        target_file = args.target_root / part1_region_name(region) / "train_data_in.nc"
        out_region = args.out_root / region
        if not target_file.exists():
            print(f"[WARN] Target HRRR grid file missing, skipping: {target_file}")
            continue

        print(f"[REGION] {region}", flush=True)
        train_stats: dict[str, VarStats] = {}
        for split in args.splits:
            src_file = src_region / f"{split}_data_in.nc"
            out_file = out_region / f"{split}_data_in.nc"
            if not src_file.exists():
                print(f"[WARN] Source file missing, skipping: {src_file}")
                continue
            stats = interpolate_file(
                src_file,
                target_file,
                out_file,
                args.chunk_size,
                args.log_eps,
                args.overwrite,
            )
            if split == "train":
                train_stats = stats

        if train_stats:
            rewrite_stats(out_region, train_stats, args)
            print(f"[STATS] {out_region / 'stats.json'} and status.json", flush=True)

    print(f"[DONE] Output directory: {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
