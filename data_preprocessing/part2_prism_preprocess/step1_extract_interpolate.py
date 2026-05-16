"""Step 1: Extract daily PRISM GeoTIFFs and interpolate onto HRRR curvilinear grids."""

import argparse
import time
from datetime import date, timedelta
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import rasterio
from scipy.interpolate import RegularGridInterpolator

from config import (
    CORRUPTED_DATES,
    DOMAINS,
    NODATA,
    load_target_coords,
    prism_lat_array,
    prism_lon_array,
    prism_tif_in_zip,
    prism_zip_path,
    staging_dir,
)

# Built once at module level after init
_LAT_ASC = None  # 1D ascending lat
_LON_ASC = None  # 1D ascending lon
_DOMAIN_TARGETS = None  # dict: domain -> (N, 2) query points


def _init_shared():
    global _LAT_ASC, _LON_ASC, _DOMAIN_TARGETS
    lat_desc = prism_lat_array()  # descending north to south
    _LAT_ASC = lat_desc[::-1].copy()  # ascending for interpolator
    _LON_ASC = prism_lon_array()  # already ascending west to east


def _init_domain_targets(domains):
    global _DOMAIN_TARGETS
    _DOMAIN_TARGETS = {}
    for domain in domains:
        lat_2d, lon_2d = load_target_coords(domain)
        pts = np.column_stack([lat_2d.ravel(), lon_2d.ravel()])
        _DOMAIN_TARGETS[domain] = pts


def process_date(d: date) -> dict:
    """Process one date: read PRISM ZIP, interpolate to all domains, save .npy files.

    Returns dict with 'date', 'status', 'error'.
    """
    if d in CORRUPTED_DATES:
        return {"date": d, "status": "skipped", "error": "corrupted"}

    zip_path = prism_zip_path(d)
    if not zip_path.exists():
        return {"date": d, "status": "missing", "error": f"ZIP not found: {zip_path}"}

    tif_name = prism_tif_in_zip(d)
    vsi_path = f"/vsizip/{zip_path}/{tif_name}"

    try:
        with rasterio.open(vsi_path) as src:
            data = src.read(1)  # (621, 1405) float32
    except Exception as e:
        return {"date": d, "status": "failed", "error": str(e)}

    # Replace nodata with NaN, flip to ascending lat
    data = data.astype(np.float64)
    data[data == NODATA] = np.nan
    data_flipped = data[::-1, :]  # flip rows to match ascending lat

    # Create interpolator
    interp = RegularGridInterpolator(
        (_LAT_ASC, _LON_ASC),
        data_flipped,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    date_str = d.strftime("%Y%m%d")
    year_str = str(d.year)

    for domain, pts in _DOMAIN_TARGETS.items():
        result = interp(pts).reshape(192, 192)
        # NaN to 0 for ocean/outside PRISM, then clip negatives.
        result = np.nan_to_num(result, nan=0.0)
        result = np.clip(result, 0.0, None)

        out_path = staging_dir(domain) / year_str / f"{date_str}.npy"
        np.save(out_path, result.astype(np.float32))

    return {"date": d, "status": "ok", "error": ""}


def _worker_init(domains):
    """Initialize shared data in each worker process."""
    _init_shared()
    _init_domain_targets(domains)


def main():
    parser = argparse.ArgumentParser(description="Step 1: Extract & interpolate PRISM daily data")
    parser.add_argument("--start-date", default="2000-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--domains", default=",".join(DOMAINS))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    domains = args.domains.split(",")

    # Build date list
    dates = []
    d = start
    while d <= end:
        dates.append(d)
        d += timedelta(days=1)

    print("=" * 60)
    print("Step 1: Extract & interpolate PRISM")
    print("=" * 60)
    print(f"  Date range: {start} to {end} ({len(dates)} days)")
    print(f"  Domains: {domains}")
    print(f"  Workers: {args.workers}")

    t0 = time.time()
    results = []

    if args.workers <= 1:
        _init_shared()
        _init_domain_targets(domains)
        for i, d in enumerate(dates):
            r = process_date(d)
            results.append(r)
            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(dates)}] {elapsed:.0f}s elapsed")
    else:
        with Pool(
            processes=args.workers,
            initializer=_worker_init,
            initargs=(domains,),
        ) as pool:
            for i, r in enumerate(pool.imap(process_date, dates, chunksize=50)):
                results.append(r)
                if (i + 1) % 200 == 0:
                    elapsed = time.time() - t0
                    print(f"  [{i+1}/{len(dates)}] {elapsed:.0f}s elapsed")

    elapsed = time.time() - t0

    # Summarize
    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    missing = sum(1 for r in results if r["status"] == "missing")
    failed = sum(1 for r in results if r["status"] == "failed")

    print(f"\n  Done in {elapsed:.1f}s: {ok} ok, {skipped} skipped, {missing} missing, {failed} failed")

    # Write missing dates per domain
    failed_dates = [r for r in results if r["status"] in ("missing", "failed", "skipped")]
    for domain in domains:
        missing_file = staging_dir(domain).parent / "missing_dates.txt"
        with open(missing_file, "w") as f:
            for r in failed_dates:
                f.write(f"{r['date'].isoformat()}\t{r['status']}\t{r['error']}\n")

    if failed > 0:
        print("\n  Failed dates:")
        for r in results:
            if r["status"] == "failed":
                print(f"    {r['date']}: {r['error']}")


if __name__ == "__main__":
    main()
