"""Step 4: Validate all output files for format correctness and data sanity."""

import argparse
import json
import netCDF4
import numpy as np
import xarray as xr

from config import (
    COORD_DIR,
    CORRUPTED_DATES,
    DOMAINS,
    TIME_SPLITS,
    domain_dir,
)

EXPECTED_COUNTS = {
    "train": None,  # computed dynamically (depends on corrupted dates)
    "val": 365,
    "test": 731,
}


def count_expected_days(split: str) -> int:
    start, end = TIME_SPLITS[split]
    d = start
    count = 0
    from datetime import timedelta

    while d <= end:
        if d not in CORRUPTED_DATES:
            count += 1
        d += timedelta(days=1)
    return count


def validate_domain(domain: str) -> list[dict]:
    """Validate all files for one domain. Returns list of check results."""
    checks = []
    out_dir = domain_dir(domain)

    # File existence.
    expected_files = [
        "train_data_out.nc", "val_data_out.nc", "test_data_out.nc", "stats.json",
    ]
    for fname in expected_files:
        path = out_dir / fname
        checks.append({
            "check": f"file_exists:{fname}",
            "ok": path.exists(),
            "detail": str(path),
        })

    if not all(c["ok"] for c in checks):
        return checks

    # Load HRRR reference coords.
    ref_path = COORD_DIR / f"part1-US-{domain}_lat_lon_192_192.nc"
    ds_ref = xr.open_dataset(ref_path, drop_variables=["step", "heightAboveGround"])
    lat_ref = ds_ref.latitude.values
    lon_ref = ds_ref.longitude.values
    # Convert to 0-360 for comparison
    lon_ref_360 = np.where(lon_ref > 0, lon_ref, lon_ref + 360)
    ds_ref.close()

    # Validate data_out.nc files.
    for split in ["train", "val", "test"]:
        fname = f"{split}_data_out.nc"
        path = out_dir / fname
        expected_n = count_expected_days(split)

        ds = xr.open_dataset(path)
        tp = ds.tp.values

        # Dimensions
        dims = list(ds.tp.dims)
        checks.append({
            "check": f"{fname}:dims",
            "ok": dims == ["time", "latitude", "longitude"],
            "detail": str(dims),
        })

        # Shape
        shape_ok = tp.shape[0] == expected_n and tp.shape[1:] == (192, 192)
        checks.append({
            "check": f"{fname}:shape",
            "ok": shape_ok,
            "detail": f"expected ({expected_n}, 192, 192), got {tp.shape}",
        })

        # Dtype
        checks.append({
            "check": f"{fname}:dtype",
            "ok": tp.dtype == np.float32,
            "detail": str(tp.dtype),
        })

        # Values
        checks.append({
            "check": f"{fname}:no_nan",
            "ok": not np.any(np.isnan(tp)),
            "detail": f"NaN count: {np.isnan(tp).sum()}",
        })
        checks.append({
            "check": f"{fname}:non_negative",
            "ok": bool(tp.min() >= 0),
            "detail": f"min={tp.min():.6f}",
        })

        # Coordinates match reference
        lat_ds = ds.latitude.values
        lon_ds = ds.longitude.values
        lat_match = np.allclose(lat_ds, lat_ref, atol=1e-3)
        lon_match = np.allclose(lon_ds, lon_ref_360, atol=1e-3)
        checks.append({
            "check": f"{fname}:lat_match_ref",
            "ok": lat_match,
            "detail": f"max diff: {np.abs(lat_ds - lat_ref).max():.6f}",
        })
        checks.append({
            "check": f"{fname}:lon_match_ref",
            "ok": lon_match,
            "detail": f"max diff: {np.abs(lon_ds - lon_ref_360).max():.6f}",
        })

        # Time sorted, no duplicates
        times = ds.time.values
        time_sorted = np.all(times[1:] > times[:-1])
        checks.append({
            "check": f"{fname}:time_sorted",
            "ok": bool(time_sorted),
            "detail": "",
        })

        # No compression
        nc = netCDF4.Dataset(str(path), "r")
        filt = nc.variables["tp"].filters()
        no_compress = not filt.get("zlib", False)
        checks.append({
            "check": f"{fname}:no_compression",
            "ok": no_compress,
            "detail": str(filt),
        })
        nc.close()

        ds.close()

    # Validate stats.json.
    path = out_dir / "stats.json"
    with open(path) as f:
        stats = json.load(f)

    # Required keys
    checks.append({
        "check": "stats:output_exists",
        "ok": "output" in stats,
        "detail": "",
    })

    if "output" in stats and "tp" in stats["output"]:
        tp_stats = stats["output"]["tp"]
        required = ["mean", "std", "min", "max", "log_mean", "log_std", "log_min", "log_max", "log_eps"]
        for key in required:
            checks.append({
                "check": f"stats:output.tp.{key}",
                "ok": key in tp_stats and np.isfinite(tp_stats[key]),
                "detail": str(tp_stats.get(key)),
            })
        checks.append({
            "check": "stats:output.tp.std>0",
            "ok": tp_stats.get("std", 0) > 0,
            "detail": str(tp_stats.get("std")),
        })

    # Sanity: annual mean precip.
    ds_train = xr.open_dataset(out_dir / "train_data_out.nc")
    annual_mean = float(ds_train.tp.mean())
    ds_train.close()
    reasonable = 0.1 < annual_mean < 10
    checks.append({
        "check": "sanity:annual_mean_precip",
        "ok": reasonable,
        "detail": f"{annual_mean:.3f} mm/day",
    })

    return checks


def main():
    parser = argparse.ArgumentParser(description="Step 4: Validate outputs")
    parser.add_argument("--domains", default=",".join(DOMAINS))
    args = parser.parse_args()

    domains = args.domains.split(",")

    print("=" * 60)
    print("Step 4: Validate outputs")
    print("=" * 60)

    all_results = {}
    all_ok = True

    for domain in domains:
        print(f"\n  {domain}:")
        checks = validate_domain(domain)
        all_results[domain] = checks

        n_pass = sum(1 for c in checks if c["ok"])
        n_fail = sum(1 for c in checks if not c["ok"])
        all_ok = all_ok and (n_fail == 0)

        for c in checks:
            status = "PASS" if c["ok"] else "FAIL"
            detail = f" ({c['detail']})" if c["detail"] else ""
            if not c["ok"]:
                print(f"    [{status}] {c['check']}{detail}")

        print(f"    {n_pass} passed, {n_fail} failed")

    # Write reports
    from config import STAGING_ROOT

    report_json = STAGING_ROOT / "validation_report.json"
    with open(report_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    report_md = STAGING_ROOT / "validation_report.md"
    with open(report_md, "w") as f:
        f.write("# PRISM Part 2 Validation Report\n\n")
        for domain, checks in all_results.items():
            n_pass = sum(1 for c in checks if c["ok"])
            n_fail = sum(1 for c in checks if not c["ok"])
            f.write(f"## {domain}: {n_pass} pass, {n_fail} fail\n\n")
            for c in checks:
                status = "PASS" if c["ok"] else "**FAIL**"
                detail = f" - {c['detail']}" if c["detail"] else ""
                f.write(f"- [{status}] {c['check']}{detail}\n")
            f.write("\n")

    status_str = "ALL PASSED" if all_ok else "FAILURES DETECTED"
    print(f"\n  Overall: {status_str}")
    print(f"  Reports: {report_json}")
    print(f"           {report_md}")
    print("\nStep 4 complete.")


if __name__ == "__main__":
    main()
