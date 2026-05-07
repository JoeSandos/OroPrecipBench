#!/usr/bin/env python3
"""Run extreme precipitation metrics (design.md Section 5.3).

Usage:
    python -m metrics.extreme.run_extreme \
        --domain-dir /path/to/part2-US-Rocky \
        --pred-path /path/to/pred.nc \
        --out-dir outputs/metrics/extreme/part2-US-Rocky/model \
        --metrics quantile_bias precip_pdf return_period \
        --quantiles 0.90 0.95 0.99 --ens-strategy per_member
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from metrics.io_utils import align_time, cache_key, load_prediction, load_train, load_truth

ALL_METRICS = ["quantile_bias", "precip_pdf", "return_period"]

logger = logging.getLogger("metrics.extreme")


def setup_logging(out_dir: Path, level: str = "INFO") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    logger.setLevel(getattr(logging, level))
    logger.handlers.clear()

    fh = logging.FileHandler(out_dir / "run_extreme.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)


def run(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir, args.log_level)

    metrics_to_run = args.metrics or ALL_METRICS
    logger.info("Metrics to run: %s", metrics_to_run)

    t0 = time.time()
    truth, truth_meta = load_truth(args.domain_dir)
    pred, pred_meta = load_prediction(args.pred_path, max_ens=args.max_ens)
    logger.info("Loaded truth %s, pred %s in %.1fs", truth.shape, pred.shape, time.time() - t0)

    pred, truth, common_time = align_time(pred, truth, pred_meta["time"], truth_meta["time"])
    logger.info("Aligned: T=%d", truth.shape[0])

    # Mask sea pixels with NaN
    from metrics.io_utils import load_land_mask
    land = load_land_mask(args.domain_dir)
    if land is not None:
        sea = ~land
        n_sea = int(sea.sum())
        if n_sea > 0:
            truth[:, sea] = np.nan
            pred[:, :, sea] = np.nan
            logger.info("Masked %d sea pixels (%.1f%%)", n_sea, 100 * n_sea / sea.size)

    # Load train data for climatological quantile thresholds
    climatology = None
    if "quantile_bias" in metrics_to_run:
        try:
            train_data, _ = load_train(args.domain_dir)
            climatology = train_data
            if land is not None and n_sea > 0:
                climatology[:, sea] = np.nan
            logger.info("Loaded train data %s for climatological quantiles", train_data.shape)
        except FileNotFoundError:
            logger.warning("train_data_out.nc not found, falling back to test data for quantile thresholds")

    results = {}

    if "quantile_bias" in metrics_to_run:
        from metrics.extreme.quantile_bias import compute_quantile_bias
        t0 = time.time()
        res = compute_quantile_bias(
            pred, truth,
            quantiles=args.quantiles,
            ens_strategy=args.ens_strategy,
            climatology=climatology,
        )
        results["quantile_bias"] = res
        for q in args.quantiles:
            ql = f"q{q:.2f}"
            logger.info(
                "  QB %s: abs_bias=%.4f ratio=%.4f",
                ql, res[f"{ql}_abs_bias_mean"], res[f"{ql}_ratio_mean"],
            )
        logger.info("Quantile bias computed in %.1fs", time.time() - t0)

    if "precip_pdf" in metrics_to_run:
        from metrics.extreme.precip_pdf import compute_precip_pdf
        t0 = time.time()
        res = compute_precip_pdf(
            pred, truth,
            wet_threshold=args.wet_threshold,
            bins_max=args.pdf_bins_max,
            bins_width=args.pdf_bins_width,
            ens_strategy=args.ens_strategy if args.ens_strategy != "per_member" else "mean",
        )
        results["precip_pdf"] = res
        logger.info(
            "Precip PDF: truth_wet=%d pred_wet=%d (%.1fs)",
            res["truth_n_wet"], res["pred_n_wet"], time.time() - t0,
        )

    if "return_period" in metrics_to_run:
        logger.info("Skipping return_period (disabled by default — requires multi-decade daily data for valid GEV fitting)")
        results["return_period"] = {"skipped": True, "reason": "disabled by default"}

    _save_results(results, out_dir)
    logger.info("Results saved to %s", out_dir)
    return results


def _save_results(results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for name, res in results.items():
        for k, v in res.items():
            if isinstance(v, (int, float)):
                summary[f"{name}/{k}"] = v
            elif isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
                summary[f"{name}/{k}"] = v

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    rows = []
    for k, v in summary.items():
        if isinstance(v, (int, float)):
            rows.append({"metric": k, "value": v})
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)

    for name, res in results.items():
        with open(out_dir / f"{name}_detail.pkl", "wb") as f:
            pickle.dump(res, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run extreme precipitation metrics (Section 5.3)")
    parser.add_argument("--domain-dir", type=Path, required=True)
    parser.add_argument("--pred-path", type=Path, required=True)
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("outputs/metrics/extreme"),
    )
    parser.add_argument("--part", default="auto", choices=["part1", "part2", "auto"],
                        help="Experiment part: part1=hourly, part2=daily, auto=detect")
    parser.add_argument("--metrics", nargs="+", choices=ALL_METRICS, default=None)
    parser.add_argument("--max-ens", type=int, default=None)
    parser.add_argument("--ens-strategy", default="per_member", choices=["mean", "per_member"])
    parser.add_argument("--no-cache", action="store_true")

    # Quantile bias
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.90, 0.95, 0.99])

    # Precip PDF
    parser.add_argument("--wet-threshold", type=float, default=None,
                        help="Wet-day threshold (auto from --part if omitted)")
    parser.add_argument("--pdf-bins-max", type=float, default=None)
    parser.add_argument("--pdf-bins-width", type=float, default=None)

    # Return period
    parser.add_argument("--return-periods", type=float, nargs="+", default=[2, 5, 10, 20, 50, 100])
    parser.add_argument("--min-years", type=int, default=10)

    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    # Resolve Part defaults
    from metrics.io_utils import detect_temporal_resolution, resolve_part_defaults
    if args.part == "auto":
        resolution = detect_temporal_resolution(args.domain_dir)
    else:
        resolution = "hourly" if args.part == "part1" else "daily"
    resolve_part_defaults(args, resolution, ["wet_threshold", "pdf_bins_max", "pdf_bins_width"])

    run(args)


if __name__ == "__main__":
    main()
