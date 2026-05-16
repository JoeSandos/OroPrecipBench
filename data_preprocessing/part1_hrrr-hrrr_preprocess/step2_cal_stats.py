"""
计算 HRRR-HRRR 训练集各变量的归一化统计量, 保存为 stats.json
"""

import os
import json
import argparse
from typing import List, Optional

import numpy as np
import xarray as xr


def log_transform(x, eps=1.0):
    if eps == 1.0:
        return np.log1p(x)
    return np.log(x + eps)


def compute_var_stats(arr: np.ndarray, is_log_var: bool = False, log_eps: float = 1.0) -> dict:
    stats = {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
    }
    if is_log_var:
        arr_log = log_transform(arr, eps=log_eps)
        stats["log_mean"] = float(np.nanmean(arr_log))
        stats["log_std"] = float(np.nanstd(arr_log))
        stats["log_min"] = float(np.nanmin(arr_log))
        stats["log_max"] = float(np.nanmax(arr_log))
        stats["log_eps"] = log_eps
    return stats


def cal_stats(
    region_dir: str,
    output_path: Optional[str] = None,
    input_vars: Optional[List[str]] = None,
    output_vars: Optional[List[str]] = None,
    invariant_vars: Optional[List[str]] = None,
    log_vars: Optional[List[str]] = None,
    log_eps: float = 1.0,
) -> dict:
    region_dir = os.path.abspath(region_dir)

    in_path = os.path.join(region_dir, "train_data_in.nc")
    out_path = os.path.join(region_dir, "train_data_out.nc")
    static_path = os.path.join(region_dir, "static_variables.nc")
    output_path = output_path or os.path.join(region_dir, "stats.json")

    ds_in = xr.open_dataset(in_path)
    ds_out = xr.open_dataset(out_path)
    ds_static = xr.open_dataset(static_path)

    if input_vars is None:
        input_vars = list(ds_in.data_vars)
    if output_vars is None:
        output_vars = list(ds_out.data_vars)
    if invariant_vars is None:
        invariant_vars = list(ds_static.data_vars)
    if log_vars is None:
        log_vars = ["tp"]

    print(f"Input vars:      {input_vars}")
    print(f"Output vars:     {output_vars}")
    print(f"Invariant vars:  {invariant_vars}")
    print(f"Log vars:        {log_vars}")

    result = {"input": {}, "output": {}, "invariant": {}}

    for var in input_vars:
        print(f"  Processing input variable: {var}")
        arr = ds_in[var].values
        result["input"][var] = compute_var_stats(arr, var in log_vars, log_eps)

    for var in output_vars:
        print(f"  Processing output variable: {var}")
        arr = ds_out[var].values
        result["output"][var] = compute_var_stats(arr, var in log_vars, log_eps)

    for var in invariant_vars:
        print(f"  Processing invariant variable: {var}")
        arr = ds_static[var].values
        result["invariant"][var] = compute_var_stats(arr, var in log_vars, log_eps)

    ds_in.close()
    ds_out.close()
    ds_static.close()

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[OK] Saved stats to {output_path}")

    return result


def parse_args():
    p = argparse.ArgumentParser(description="Calculate normalization statistics for HRRR-HRRR dataset")
    p.add_argument("--region-dir", required=True, help="Path to region directory containing train/*.nc files")
    p.add_argument("--output-path", default=None, help="Output JSON path (default: {region_dir}/stats.json)")
    p.add_argument("--input-vars", nargs="+", default=None)
    p.add_argument("--output-vars", nargs="+", default=None)
    p.add_argument("--invariant-vars", nargs="+", default=None)
    p.add_argument("--log-vars", nargs="+", default=["tp"])
    p.add_argument("--log-eps", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    cal_stats(
        region_dir=args.region_dir,
        output_path=args.output_path,
        input_vars=args.input_vars,
        output_vars=args.output_vars,
        invariant_vars=args.invariant_vars,
        log_vars=args.log_vars,
        log_eps=args.log_eps,
    )


def run():
    base = "/path/to/project_climate/Benchmark/preprocess/hrrr-hrrr/dataset_root"
    regions = [
        "part1-US-Rocky",
        "part1-US-Cascades",
        "part1-US-Nevada",
        # "part1-US-Appalachian",
        # "part1-US-Great_Plains",
    ]
    for region in regions:
        region_dir = os.path.join(base, region)
        if not os.path.isdir(region_dir):
            print(f"[SKIP] {region_dir} does not exist")
            continue
        print(f"\n{'=' * 50}")
        print(f"Computing stats for: {region}")
        print(f"{'=' * 50}")
        cal_stats(region_dir)


if __name__ == "__main__":
    main()
