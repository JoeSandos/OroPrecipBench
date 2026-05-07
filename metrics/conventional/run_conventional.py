#!/usr/bin/env python3
"""Run conventional metrics (design.md Section 5.1).

Supports selecting a subset of metrics and configurable ensemble strategy.

Usage:
    python -m metrics.conventional.run_conventional \
        --domain-dir /path/to/part1-US-Rocky \
        --pred-path /path/to/pred.nc \
        --out-dir outputs/metrics/conventional/part1-US-Rocky/model \
        --metrics rmse mae crps cc ssim pod_far \
        --ens-strategy mean --rain-threshold 0.01
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
from metrics.io_utils import (
    align_time,
    load_prediction,
    load_truth,
    reduce_ensemble,
)

ALL_METRICS = ["rmse", "mae", "bias", "cc", "crps", "ssim", "pod_far"]

logger = logging.getLogger("metrics.conventional")


def setup_logging(out_dir: Path, level: str = "INFO") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    logger.setLevel(getattr(logging, level))
    logger.handlers.clear()

    fh = logging.FileHandler(out_dir / "run_conventional.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)


def run(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out_dir)
    setup_logging(out_dir, args.log_level)

    metrics_to_run = args.metrics or ALL_METRICS
    logger.info("Metrics to run: %s", metrics_to_run)

    # Load data
    t0 = time.time()
    truth, truth_meta = load_truth(args.domain_dir)
    pred, pred_meta = load_prediction(args.pred_path, max_ens=args.max_ens)
    logger.info("Loaded truth %s, pred %s in %.1fs", truth.shape, pred.shape, time.time() - t0)

    # Align time
    pred, truth, common_time = align_time(pred, truth, pred_meta["time"], truth_meta["time"])
    T, E, H, W = pred.shape
    logger.info("Aligned: T=%d E=%d H=%d W=%d", T, E, H, W)

    # Prepare ensemble-mean for scalar metrics
    pred_mean = reduce_ensemble(pred, strategy=args.ens_strategy)

    # Mask sea pixels with NaN
    from metrics.io_utils import load_land_mask
    land = load_land_mask(args.domain_dir)
    if land is not None:
        sea = ~land
        n_sea = int(sea.sum())
        if n_sea > 0:
            truth[:, sea] = np.nan
            pred[:, :, sea] = np.nan
            pred_mean[:, sea] = np.nan
            logger.info("Masked %d sea pixels (%.1f%%)", n_sea, 100 * n_sea / sea.size)

    results = {}

    if "rmse" in metrics_to_run:
        from metrics.conventional.rmse import compute_rmse
        t0 = time.time()
        res = compute_rmse(pred_mean, truth)
        results["rmse"] = res
        logger.info("RMSE: %.4f (%.1fs)", res["rmse_mean"], time.time() - t0)

    if "mae" in metrics_to_run:
        from metrics.conventional.mae import compute_mae
        t0 = time.time()
        res = compute_mae(pred_mean, truth)
        results["mae"] = res
        logger.info("MAE: %.4f (%.1fs)", res["mae_mean"], time.time() - t0)

    if "bias" in metrics_to_run:
        from metrics.conventional.bias import compute_bias
        t0 = time.time()
        res = compute_bias(pred_mean, truth)
        results["bias"] = res
        logger.info("Bias: %.4f (%.1fs)", res["bias_mean"], time.time() - t0)

    if "cc" in metrics_to_run:
        from metrics.conventional.cc import compute_cc
        t0 = time.time()
        res = compute_cc(pred_mean, truth)
        results["cc"] = res
        logger.info(
            "CC: mean=%.4f clim=%.4f (%.1fs)",
            res["cc_mean"], res["cc_climatology"], time.time() - t0,
        )

    if "crps" in metrics_to_run:
        from metrics.conventional.crps import compute_crps
        t0 = time.time()
        res = compute_crps(pred, truth)  # uses full ensemble
        results["crps"] = res
        logger.info("CRPS: %.4f (%.1fs)", res["crps_mean"], time.time() - t0)

    if "ssim" in metrics_to_run:
        from metrics.conventional.ssim import compute_ssim
        t0 = time.time()
        res = compute_ssim(pred_mean, truth, workers=args.ssim_workers)
        results["ssim"] = res
        logger.info("SSIM: %.4f (%.1fs)", res["ssim_mean"], time.time() - t0)

    if "pod_far" in metrics_to_run:
        from metrics.conventional.pod_far import compute_pod_far
        t0 = time.time()
        res = compute_pod_far(pred_mean, truth, threshold=args.rain_threshold)
        results["pod_far"] = res
        logger.info(
            "POD: %.4f FAR: %.4f (%.1fs)",
            res["pod_mean"], res["far_mean"], time.time() - t0,
        )

    # Save results
    _save_results(results, out_dir)
    logger.info("Results saved to %s", out_dir)
    return results


def _save_results(results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Summary JSON (scalar values only)
    summary = {}
    for name, res in results.items():
        for k, v in res.items():
            if isinstance(v, (int, float)):
                summary[f"{name}/{k}"] = v
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Summary CSV
    rows = []
    for name, res in results.items():
        row = {"metric": name}
        for k, v in res.items():
            if isinstance(v, (int, float)):
                row[k] = v
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)

    # Detail PKL (includes per-time arrays)
    for name, res in results.items():
        detail = {}
        for k, v in res.items():
            if isinstance(v, np.ndarray):
                detail[k] = v
            else:
                detail[k] = v
        with open(out_dir / f"{name}_detail.pkl", "wb") as f:
            pickle.dump(detail, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run conventional metrics (Section 5.1)")
    parser.add_argument("--domain-dir", type=Path, required=True, help="Domain directory with test_data_out.nc")
    parser.add_argument("--pred-path", type=Path, required=True, help="Prediction NetCDF file")
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("outputs/metrics/conventional"),
        help="Output directory",
    )
    parser.add_argument(
        "--metrics", nargs="+", choices=ALL_METRICS, default=None,
        help=f"Metrics to compute (default: all). Choices: {ALL_METRICS}",
    )
    parser.add_argument("--part", default="auto", choices=["part1", "part2", "auto"],
                        help="Experiment part: part1=hourly, part2=daily, auto=detect")
    parser.add_argument("--ens-strategy", default="mean", choices=["mean", "per_member", "first"])
    parser.add_argument("--max-ens", type=int, default=None, help="Cap ensemble size")
    parser.add_argument("--rain-threshold", type=float, default=None,
                        help="Threshold for POD/FAR (auto from --part if omitted)")
    parser.add_argument("--ssim-workers", type=int, default=4, help="Threads for SSIM")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    # Resolve Part defaults
    from metrics.io_utils import detect_temporal_resolution, resolve_part_defaults
    if args.part == "auto":
        resolution = detect_temporal_resolution(args.domain_dir)
    else:
        resolution = "hourly" if args.part == "part1" else "daily"
    resolve_part_defaults(args, resolution, ["rain_threshold"])

    run(args)


if __name__ == "__main__":
    main()
