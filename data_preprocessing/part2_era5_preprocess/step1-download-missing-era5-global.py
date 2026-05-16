#!/usr/bin/env python3
"""Step 1: download missing global hourly ERA5 files from CDS.

The script scans <root>/<var>/<YYYY>/<MM>/<var>_YYYYMMDD_HH00.nc,
then writes missing hours to --output-root using the same layout.
No credentials are stored here; pass .cdsapirc paths with --cdsapirc-files.
"""
from __future__ import annotations

import argparse
import os
import re
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock


try:
    import cdsapi
except ImportError:  # pragma: no cover - runtime dependency message
    cdsapi = None

try:
    from netCDF4 import Dataset
except ImportError as exc:  # pragma: no cover - runtime dependency message
    raise SystemExit(
        "Missing dependency netCDF4. Install it first: conda install -c conda-forge netcdf4"
    ) from exc


SOURCE_ROOT = Path(
    "/path/to/era5/hourly/global"
)
OUTPUT_ROOT = Path("./era5_source")

VARIABLES = ["msl", "ssrd", "t2m", "tisr", "tp", "u10", "u100", "v10", "v100"]

CDS_VARIABLES = {
    "msl": "mean_sea_level_pressure",
    "ssrd": "surface_solar_radiation_downwards",
    "t2m": "2m_temperature",
    "tisr": "toa_incident_solar_radiation",
    "tp": "total_precipitation",
    "u10": "10m_u_component_of_wind",
    "u100": "100m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "v100": "100m_v_component_of_wind",
}

FILENAME_RE = re.compile(r"^(?P<var>.+)_(?P<date>\d{8})_(?P<hour>\d{4})\.nc$")
NETCDF_SPLIT_LOCK = Lock()


@dataclass(frozen=True)
class MissingDay:
    var: str
    day: datetime
    hours: tuple[int, ...]


@dataclass(frozen=True)
class CdsConfig:
    path: Path
    url: str
    key: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan and download missing hourly global ERA5 files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=SOURCE_ROOT,
        help="Root directory used to scan existing ERA5 source files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory for downloaded supplemental files; default: ./era5_source.",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=VARIABLES,
        choices=VARIABLES,
        help="Variables to check/download; default: all variables.",
    )
    parser.add_argument("--start", type=str, default=None, help="Start time: YYYYMMDDHH")
    parser.add_argument("--end", type=str, default=None, help="End time: YYYYMMDDHH, inclusive")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute downloads; without this flag, only print the missing list.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing hourly files; default is fill missing files only.",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=None,
        help="Maximum number of var-day tasks to download, useful for small tests.",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=None,
        help="Temporary download directory; default: <output-root>/.tmp_cds_downloads.",
    )
    parser.add_argument(
        "--cds-format-style",
        choices=["new", "old"],
        default="new",
        help="CDS request field style; use new for the current CDS API and old for legacy API.",
    )
    parser.add_argument(
        "--cdsapirc-files",
        nargs="+",
        type=Path,
        default=None,
        help=(
            "Multiple CDS config file paths; default search: "
            "~/.cdsapirc0 to ~/.cdsapirc5."
        ),
    )
    parser.add_argument(
        "--parallel-downloads",
        type=int,
        default=6,
        help="Number of parallel download groups; default: 6, matching six .cdsapirc files.",
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


def is_readable_netcdf(path: Path) -> bool:
    if path.stat().st_size >= 1024:
        return True
    try:
        with Dataset(path):
            return True
    except Exception:
        return False


def existing_timestamps(root: Path, var: str) -> set[datetime]:
    timestamps: set[datetime] = set()
    var_root = root / var
    if not var_root.exists():
        return timestamps
    for path in var_root.glob("*/*/*.nc"):
        ts = parse_timestamp(path)
        if ts is not None and is_readable_netcdf(path):
            timestamps.add(ts)
    return timestamps


def infer_time_range(root: Path, variables: list[str]) -> tuple[datetime, datetime]:
    all_times: list[datetime] = []
    for var in variables:
        all_times.extend(existing_timestamps(root, var))
    if not all_times:
        raise SystemExit("Could not infer time range from existing files; specify --start and --end.")
    return min(all_times), max(all_times)


def iter_hours(start: datetime, end: datetime):
    current = start
    while current <= end:
        yield current
        current += timedelta(hours=1)


def expected_path(root: Path, var: str, timestamp: datetime) -> Path:
    return (
        root
        / var
        / f"{timestamp:%Y}"
        / f"{timestamp:%m}"
        / f"{var}_{timestamp:%Y%m%d_%H00}.nc"
    )


def group_missing_by_day(
    scan_root: Path,
    output_root: Path,
    variables: list[str],
    start: datetime,
    end: datetime,
    overwrite: bool,
) -> list[MissingDay]:
    missing: list[MissingDay] = []
    all_hours = list(iter_hours(start, end))
    for var in variables:
        source_exists = existing_timestamps(scan_root, var)
        output_exists = existing_timestamps(output_root, var)
        by_day: dict[datetime, list[int]] = defaultdict(list)
        for ts in all_hours:
            if ts in source_exists:
                continue
            out_path = expected_path(output_root, var, ts)
            if not overwrite and (ts in output_exists or (out_path.exists() and is_readable_netcdf(out_path))):
                continue
            day = datetime(ts.year, ts.month, ts.day)
            by_day[day].append(ts.hour)
        for day, hours in sorted(by_day.items()):
            missing.append(MissingDay(var, day, tuple(sorted(hours))))
    return missing


def build_cds_request(task: MissingDay, style: str) -> dict[str, object]:
    request: dict[str, object] = {
        "product_type": ["reanalysis"],
        "variable": [CDS_VARIABLES[task.var]],
        "year": [f"{task.day:%Y}"],
        "month": [f"{task.day:%m}"],
        "day": [f"{task.day:%d}"],
        "time": [f"{hour:02d}:00" for hour in task.hours],
    }
    if style == "new":
        request["data_format"] = "netcdf"
        request["download_format"] = "unarchived"
    else:
        request["format"] = "netcdf"
    return request


def parse_cdsapirc(path: Path) -> CdsConfig:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip("'\"")
    if "url" not in values or "key" not in values:
        raise ValueError(f"{path} is missing url or key fields")
    return CdsConfig(path=path, url=values["url"], key=values["key"])


def discover_cdsapirc_files(files: list[Path] | None) -> list[Path]:
    if files:
        return [path.expanduser() for path in files]

    discovered: list[Path] = []
    base = Path.home()
    for index in range(6):
        path = base / f".cdsapirc{index}"
        if path.exists() and path not in discovered:
            discovered.append(path)
    path = base / ".cdsapirc"
    if path.exists() and path not in discovered:
        discovered.append(path)
    return discovered


def load_cds_configs(files: list[Path] | None, limit: int) -> list[CdsConfig]:
    paths = discover_cdsapirc_files(files)
    configs = [parse_cdsapirc(path) for path in paths]
    if limit > 0:
        configs = configs[:limit]
    if not configs:
        raise SystemExit(
            "No CDS config file found. Create ~/.cdsapirc or pass --cdsapirc-files."
        )
    return configs


def find_time_dim(nc: Dataset) -> str:
    for name in ("valid_time", "time"):
        if name in nc.dimensions:
            return name
    for name in nc.dimensions:
        if "time" in name.lower():
            return name
    raise ValueError(f"No time dimension found in downloaded file，dims={list(nc.dimensions)}")


def data_var_name(nc: Dataset, short_name: str) -> str:
    if short_name in nc.variables:
        return short_name
    for name, var in nc.variables.items():
        dims = set(var.dimensions)
        if {"latitude", "longitude"}.issubset(dims) and (
            "time" in dims or "valid_time" in dims
        ):
            return name
    raise ValueError(f"No data variable found in downloaded file: {short_name}")


def copy_attrs(src, dst) -> None:
    for attr in src.ncattrs():
        if attr == "_FillValue":
            continue
        try:
            dst.setncattr(attr, src.getncattr(attr))
        except Exception:
            pass


def create_var_like(out: Dataset, name: str, src_var, dimensions: tuple[str, ...]):
    fill_value = src_var.getncattr("_FillValue") if "_FillValue" in src_var.ncattrs() else None
    kwargs = {"zlib": True, "complevel": 4}
    if fill_value is not None:
        kwargs["fill_value"] = fill_value
    try:
        dst_var = out.createVariable(name, src_var.datatype, dimensions, **kwargs)
    except TypeError:
        dst_var = out.createVariable(name, src_var.datatype, dimensions)
    copy_attrs(src_var, dst_var)
    return dst_var


def is_string_or_vlen_variable(var) -> bool:
    dtype = getattr(var, "dtype", None)
    dtype_kind = getattr(dtype, "kind", None)
    datatype = getattr(var, "datatype", None)
    return datatype is str or dtype is str or dtype_kind in {"O", "S", "U"}


def copy_whole_variable(src_var, dst_var) -> None:
    if src_var.shape == ():
        dst_var.assignValue(src_var.getValue())
    else:
        dst_var[:] = src_var[:]


def copy_variable_slice(
    src_nc: Dataset,
    out_nc: Dataset,
    src_name: str,
    dst_name: str,
    src_time_dim: str,
    time_index: int,
) -> None:
    src_var = src_nc.variables[src_name]
    dst_dims: list[str] = []
    selectors: list[object] = []
    for dim in src_var.dimensions:
        if dim == src_time_dim:
            dst_dims.append("valid_time")
            selectors.append(slice(time_index, time_index + 1))
        else:
            dst_dims.append(dim)
            selectors.append(slice(None))

    dst_var = create_var_like(out_nc, dst_name, src_var, tuple(dst_dims))
    dst_var[:] = src_var[tuple(selectors)]


def write_hourly_file(
    src_nc: Dataset,
    out_path: Path,
    task: MissingDay,
    time_dim: str,
    time_index: int,
    hour: int,
    overwrite: bool,
) -> None:
    if out_path.exists() and not overwrite and is_readable_netcdf(out_path):
        return
    if out_path.exists() and not is_readable_netcdf(out_path):
        print(f"[WARN] Overwriting unreadable NetCDF: {out_path}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_path.with_name(f".{out_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with Dataset(tmp_out, "w", format="NETCDF4") as out_nc:
            for attr in src_nc.ncattrs():
                try:
                    out_nc.setncattr(attr, src_nc.getncattr(attr))
                except Exception:
                    pass

            for dim_name, dim in src_nc.dimensions.items():
                if dim_name == time_dim:
                    out_nc.createDimension("valid_time", 1)
                else:
                    out_nc.createDimension(dim_name, len(dim))

            for name, var in src_nc.variables.items():
                if name == time_dim:
                    copy_variable_slice(src_nc, out_nc, name, "valid_time", time_dim, time_index)
                elif name == task.var:
                    copy_variable_slice(src_nc, out_nc, name, task.var, time_dim, time_index)
                elif name in {"latitude", "longitude"}:
                    dst_var = create_var_like(out_nc, name, var, var.dimensions)
                    copy_whole_variable(var, dst_var)
                elif time_dim not in var.dimensions:
                    if is_string_or_vlen_variable(var):
                        print(f"[INFO] Skipping string auxiliary variable: {name}", flush=True)
                        continue
                    dst_var = create_var_like(out_nc, name, var, var.dimensions)
                    copy_whole_variable(var, dst_var)

            out_nc.downloaded_hour_utc = f"{hour:02d}:00"

        os.replace(tmp_out, out_path)
        print(f"[WRITE] {out_path}", flush=True)
    except Exception:
        if tmp_out.exists():
            tmp_out.unlink()
        raise


def split_daily_file(tmp_file: Path, root: Path, task: MissingDay, overwrite: bool) -> None:
    with NETCDF_SPLIT_LOCK:
        with Dataset(tmp_file) as src_nc:
            time_dim = find_time_dim(src_nc)
            if len(src_nc.dimensions[time_dim]) != len(task.hours):
                raise ValueError(
                    f"{tmp_file} time-step count {len(src_nc.dimensions[time_dim])} does not match requested hour count "
                    f"{len(task.hours)} mismatch"
                )
            data_var_name(src_nc, task.var)

            for index, hour in enumerate(task.hours):
                ts = task.day.replace(hour=hour)
                out_path = expected_path(root, task.var, ts)
                write_hourly_file(
                    src_nc, out_path, task, time_dim, index, hour, overwrite=overwrite
                )


def group_tasks_by_variable(tasks: list[MissingDay]) -> dict[str, list[MissingDay]]:
    grouped: dict[str, list[MissingDay]] = defaultdict(list)
    for task in tasks:
        grouped[task.var].append(task)
    return dict(sorted(grouped.items()))


def download_variable_tasks(
    var: str,
    tasks: list[MissingDay],
    config: CdsConfig,
    tmp_dir: Path,
    root: Path,
    overwrite: bool,
    cds_format_style: str,
) -> tuple[str, int]:
    client = cdsapi.Client(url=config.url, key=config.key)
    completed = 0
    for index, task in enumerate(tasks, start=1):
        tmp_file = tmp_dir / f"{task.var}_{task.day:%Y%m%d}_{uuid.uuid4().hex}.nc"
        request = build_cds_request(task, cds_format_style)
        print(
            f"[DOWNLOAD] {config.path.name} {var} {index}/{len(tasks)} "
            f"{task.day:%Y-%m-%d} hours={','.join(f'{h:02d}' for h in task.hours)}",
            flush=True,
        )
        client.retrieve("reanalysis-era5-single-levels", request, str(tmp_file))
        split_daily_file(tmp_file, root, task, overwrite=overwrite)
        print(f"[TMP] {config.path.name} Temporary download file kept at: {tmp_file}", flush=True)
        completed += 1
    return var, completed


def download_config_bucket(
    config: CdsConfig,
    assignments: list[tuple[str, list[MissingDay]]],
    tmp_dir: Path,
    root: Path,
    overwrite: bool,
    cds_format_style: str,
) -> list[tuple[str, int]]:
    results: list[tuple[str, int]] = []
    for var, var_tasks in assignments:
        results.append(
            download_variable_tasks(
                var,
                var_tasks,
                config,
                tmp_dir,
                root,
                overwrite,
                cds_format_style,
            )
        )
    return results


def print_summary(tasks: list[MissingDay]) -> None:
    total_hours = sum(len(task.hours) for task in tasks)
    print(f"[INFO] Missing var-day tasks: {len(tasks)}")
    print(f"[INFO] Missing hourly files: {total_hours}")
    by_var: dict[str, int] = defaultdict(int)
    for task in tasks:
        by_var[task.var] += len(task.hours)
    for var in sorted(by_var):
        print(f"[INFO] {var}: {by_var[var]} hourly files missing")


def main() -> int:
    args = parse_args()
    start = parse_bound(args.start)
    end = parse_bound(args.end)
    if start is None or end is None:
        inferred_start, inferred_end = infer_time_range(args.root, args.variables)
        start = start or inferred_start
        end = end or inferred_end

    if start > end:
        raise SystemExit("--start must not be later than --end")

    tasks = group_missing_by_day(
        args.root,
        args.output_root,
        list(args.variables),
        start,
        end,
        overwrite=args.overwrite,
    )
    if args.max_days is not None:
        tasks = tasks[: args.max_days]

    print(f"[INFO] Scanning source root: {args.root}")
    print(f"[INFO] Download output root: {args.output_root}")
    print(f"[INFO] Check range: {start:%Y-%m-%d %H}:00 to {end:%Y-%m-%d %H}:00")
    print_summary(tasks)
    for task in tasks[:20]:
        hours = ",".join(f"{hour:02d}" for hour in task.hours)
        print(f"[MISS] {task.var} {task.day:%Y-%m-%d}: {hours}")
    if len(tasks) > 20:
        print(f"[INFO] Showing only the first 20 missing var-day tasks; remaining {len(tasks) - 20} not shown")

    if not args.execute:
        print("[DRY-RUN] --execute not set; no download or write will be performed.")
        return 0

    if cdsapi is None:
        raise SystemExit(
            "Missing dependency cdsapi. Install it first:\n"
            "  pip install cdsapi\n"
            "and configure ~/.cdsapirc or a CDS API key."
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = args.tmp_dir or (args.output_root / ".tmp_cds_downloads")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    configs = load_cds_configs(args.cdsapirc_files, args.parallel_downloads)
    grouped = list(group_tasks_by_variable(tasks).items())
    buckets: list[list[tuple[str, list[MissingDay]]]] = [[] for _ in configs]
    for index, item in enumerate(grouped):
        buckets[index % len(configs)].append(item)

    print(
        "[INFO] Using CDS configs for parallel downloads: "
        + ", ".join(config.path.name for config in configs),
        flush=True,
    )
    for config, bucket in zip(configs, buckets):
        var_names = ", ".join(var for var, _ in bucket) or "no tasks"
        print(f"[INFO] {config.path.name}: {var_names}", flush=True)

    with ThreadPoolExecutor(max_workers=len(configs)) as executor:
        futures = [
            executor.submit(
                download_config_bucket,
                config,
                bucket,
                tmp_dir,
                args.output_root,
                args.overwrite,
                args.cds_format_style,
            )
            for config, bucket in zip(configs, buckets)
            if bucket
        ]
        for future in as_completed(futures):
            for var, completed in future.result():
                print(f"[DONE] {var}: completed {completed} var-day download tasks", flush=True)

    print("[DONE] Finished downloading missing global hourly ERA5 files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
