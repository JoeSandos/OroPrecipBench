#!/usr/bin/env python3
"""Master entry point: run all metric categories.

IMPORTANT: This module must be run from the *parent* of the ``benchmark/``
directory (i.e. ``~/chy/climate/``) so that ``python -m metrics.…``
can resolve the package.  Alternatively, set PYTHONPATH:

    export PYTHONPATH=/path/to/OroPrecipBench

Usage:
    cd /path/to/OroPrecipBench          # or set PYTHONPATH as above
    python -m metrics.run_metrics \
        --domain-dir /path/to/part1-US-Rocky \
        --pred-path /path/to/pred.nc \
        --out-dir outputs/metrics/part1-US-Rocky/model \
        --categories conventional spatial extreme terrain \
        --pixel-size-km 3.0
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger("metrics")

ALL_CATEGORIES = ["conventional", "spatial", "extreme", "terrain"]


def setup_logging(out_dir: Path, level: str = "INFO") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    logger.setLevel(getattr(logging, level))
    logger.handlers.clear()

    fh = logging.FileHandler(out_dir / "run_metrics.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all benchmark evaluation metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Categories:\n"
            "  conventional  RMSE, MAE, Bias, CC, CRPS, SSIM, POD/FAR\n"
            "  spatial        RAPSD, FSS, Correlation Length\n"
            "  extreme        Quantile Bias, Precip PDF, Return Period\n"
        ),
    )
    parser.add_argument("--domain-dir", type=Path, required=True, help="Domain directory with test_data_out.nc")
    parser.add_argument("--pred-path", type=Path, required=True, help="Prediction NetCDF file")
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("outputs/metrics"),
        help="Base output directory",
    )
    parser.add_argument(
        "--categories", nargs="+", choices=ALL_CATEGORIES, default=ALL_CATEGORIES,
        help="Metric categories to run",
    )

    # Part detection (auto-sets thresholds for hourly vs daily)
    parser.add_argument(
        "--part", default="auto", choices=["part1", "part2", "auto"],
        help="Experiment part: part1=hourly, part2=daily, auto=detect from data",
    )

    # Shared (None = inherit from --part preset)
    parser.add_argument("--max-ens", type=int, default=None, help="Cap ensemble size")
    parser.add_argument("--ens-strategy", default="mean", choices=["mean", "per_member", "first"])
    parser.add_argument("--pixel-size-km", type=float, default=3.0)
    parser.add_argument("--rain-threshold", type=float, default=None, help="POD/FAR threshold (auto from --part)")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--log-level", default="INFO")

    # Conventional-specific
    parser.add_argument("--ssim-workers", type=int, default=4)

    # Spatial-specific
    parser.add_argument("--fft-backend", default="scipy", choices=["scipy", "numpy", "pyfftw"])
    parser.add_argument("--fss-scales", type=int, nargs="+", default=[1, 3, 5, 9, 17, 33, 65])
    parser.add_argument("--fss-threshold", type=float, default=None, help="FSS threshold (auto from --part)")
    parser.add_argument("--corr-stride", type=int, default=2)
    parser.add_argument("--corr-k", type=int, default=20)
    parser.add_argument("--corr-radius-km", type=float, default=100.0)
    parser.add_argument("--corr-seed", type=int, default=0)

    # Extreme-specific
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.90, 0.95, 0.99])
    parser.add_argument("--return-periods", type=float, nargs="+", default=[2, 5, 10, 20, 50, 100])
    parser.add_argument("--min-years", type=int, default=10)
    parser.add_argument("--wet-threshold", type=float, default=None, help="Wet-day threshold (auto from --part)")
    parser.add_argument("--pdf-bins-max", type=float, default=None)
    parser.add_argument("--pdf-bins-width", type=float, default=None)

    # Terrain-specific
    parser.add_argument("--bin-width", type=float, default=200.0, help="PEP-D elevation bin width (m)")

    args = parser.parse_args()
    base_out = Path(args.out_dir)
    setup_logging(base_out, args.log_level)

    # ── Resolve Part 1/Part 2 defaults ──
    from metrics.io_utils import (
        PART_DEFAULTS,
        detect_temporal_resolution,
        resolve_part_defaults,
    )
    if args.part == "auto":
        resolution = detect_temporal_resolution(args.domain_dir)
    else:
        resolution = "hourly" if args.part == "part1" else "daily"
    resolve_part_defaults(args, resolution)
    logger.info(
        "Part: %s (%s) → rain=%.3f fss=%.3f wet=%.3f bins_max=%.0f bins_width=%.1f",
        args.part, resolution,
        args.rain_threshold, args.fss_threshold, args.wet_threshold,
        args.pdf_bins_max, args.pdf_bins_width,
    )
    logger.info("Running categories: %s", args.categories)

    all_summaries = {}
    t_start = time.time()

    if "conventional" in args.categories:
        from metrics.conventional.run_conventional import run as run_conv
        conv_args = argparse.Namespace(
            domain_dir=args.domain_dir,
            pred_path=args.pred_path,
            out_dir=base_out / "conventional",
            metrics=None,  # all
            ens_strategy=args.ens_strategy,
            max_ens=args.max_ens,
            rain_threshold=args.rain_threshold,
            ssim_workers=args.ssim_workers,
            log_level=args.log_level,
        )
        logger.info("--- Running conventional metrics ---")
        t0 = time.time()
        res = run_conv(conv_args)
        logger.info("Conventional done in %.1fs", time.time() - t0)
        all_summaries["conventional"] = {
            k: v for name, r in res.items()
            for k, v in r.items() if isinstance(v, (int, float))
        }

    if "spatial" in args.categories:
        from metrics.spatial.run_spatial import run as run_spat
        spat_args = argparse.Namespace(
            domain_dir=args.domain_dir,
            pred_path=args.pred_path,
            out_dir=base_out / "spatial",
            metrics=None,
            max_ens=args.max_ens,
            ens_strategy=args.ens_strategy,
            pixel_size_km=args.pixel_size_km,
            fft_backend=args.fft_backend,
            no_cache=args.no_cache,
            fss_scales=args.fss_scales,
            fss_threshold=args.fss_threshold,
            corr_stride=args.corr_stride,
            corr_k=args.corr_k,
            corr_radius_km=args.corr_radius_km,
            corr_seed=args.corr_seed,
            log_level=args.log_level,
        )
        logger.info("--- Running spatial metrics ---")
        t0 = time.time()
        run_spat(spat_args)
        logger.info("Spatial done in %.1fs", time.time() - t0)

    if "extreme" in args.categories:
        from metrics.extreme.run_extreme import run as run_ext
        ext_args = argparse.Namespace(
            domain_dir=args.domain_dir,
            pred_path=args.pred_path,
            out_dir=base_out / "extreme",
            metrics=None,
            max_ens=args.max_ens,
            ens_strategy="per_member" if args.ens_strategy == "per_member" else args.ens_strategy,
            no_cache=args.no_cache,
            quantiles=args.quantiles,
            wet_threshold=args.wet_threshold,
            pdf_bins_max=args.pdf_bins_max,
            pdf_bins_width=args.pdf_bins_width,
            return_periods=args.return_periods,
            min_years=args.min_years,
            log_level=args.log_level,
        )
        logger.info("--- Running extreme metrics ---")
        t0 = time.time()
        run_ext(ext_args)
        logger.info("Extreme done in %.1fs", time.time() - t0)

    if "terrain" in args.categories:
        from metrics.terrain.run_terrain import run as run_terr
        terr_args = argparse.Namespace(
            domain_dir=args.domain_dir,
            pred_path=args.pred_path,
            out_dir=base_out / "terrain",
            metrics=None,
            max_ens=args.max_ens,
            bin_width=args.bin_width,
            no_cache=args.no_cache,
            log_level=args.log_level,
        )
        logger.info("--- Running terrain metrics ---")
        t0 = time.time()
        run_terr(terr_args)
        logger.info("Terrain done in %.1fs", time.time() - t0)

    logger.info("All done in %.1fs. Output: %s", time.time() - t_start, base_out)


if __name__ == "__main__":
    main()
