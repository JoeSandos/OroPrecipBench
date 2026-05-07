#!/usr/bin/env python3
"""Run spatial structure metrics (design.md Section 5.2).

Usage:
    python -m metrics.spatial.run_spatial \
        --domain-dir /path/to/part1-US-Rocky \
        --pred-path /path/to/pred.nc \
        --out-dir outputs/metrics/spatial/part1-US-Rocky/model \
        --metrics rapsd fss correlation_length \
        --pixel-size-km 3.0
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
from metrics.io_utils import align_time, cache_key, load_prediction, load_truth

ALL_METRICS = ["rapsd", "fss", "correlation_length"]

logger = logging.getLogger("metrics.spatial")


def setup_logging(out_dir: Path, level: str = "INFO") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    logger.setLevel(getattr(logging, level))
    logger.handlers.clear()

    fh = logging.FileHandler(out_dir / "run_spatial.log")
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
    T, E, H, W = pred.shape
    logger.info("Aligned: T=%d E=%d H=%d W=%d", T, E, H, W)

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

    results = {}

    if "rapsd" in metrics_to_run:
        from metrics.spatial.rapsd import compute_rapsd
        t0 = time.time()

        # Check cache
        ck = cache_key(args.pred_path, metric="rapsd", pixel_size_km=args.pixel_size_km)
        cached = cache_dir / f"rapsd_{ck}.pkl"
        if cached.exists() and not args.no_cache:
            logger.info("Loading cached RAPSD from %s", cached)
            with open(cached, "rb") as f:
                res = pickle.load(f)
        else:
            res = compute_rapsd(
                pred, truth,
                pixel_size_km=args.pixel_size_km,
                fft_backend=args.fft_backend,
            )
            with open(cached, "wb") as f:
                pickle.dump(res, f)

        results["rapsd"] = res
        logger.info("RAPSD log-MSE: %.6f (%.1fs)", res["score_log_mse"], time.time() - t0)

    if "fss" in metrics_to_run:
        from metrics.spatial.fss import compute_fss
        t0 = time.time()

        ck = cache_key(
            args.pred_path, metric="fss",
            scales=args.fss_scales, threshold=args.fss_threshold,
            ens_strategy=args.ens_strategy,
        )
        cached = cache_dir / f"fss_{ck}.pkl"
        if cached.exists() and not args.no_cache:
            logger.info("Loading cached FSS from %s", cached)
            with open(cached, "rb") as f:
                res = pickle.load(f)
        else:
            res = compute_fss(
                pred, truth,
                scales=args.fss_scales,
                threshold=args.fss_threshold,
                ens_strategy=args.ens_strategy,
            )
            with open(cached, "wb") as f:
                pickle.dump(res, f)

        results["fss"] = res
        for s, v in zip(res["scales"], res["fss_scores"]):
            logger.info("  FSS scale=%d: %.4f", s, v)
        logger.info("FSS computed in %.1fs", time.time() - t0)

    if "correlation_length" in metrics_to_run:
        from metrics.spatial.correlation_length import (
            compute_correlation_length,
            select_single_member,
        )
        t0 = time.time()

        # Select single member (do NOT use ensemble mean for correlation length)
        pred_single = select_single_member(pred, label="pred", seed=args.corr_seed)

        ck = cache_key(
            args.pred_path, metric="corr_length",
            stride=args.corr_stride, k=args.corr_k,
            radius=args.corr_radius_km, km_per_pixel=args.pixel_size_km,
            seed=args.corr_seed,
        )
        cached_pred = cache_dir / f"corr_length_pred_{ck}.pkl"
        cached_truth = cache_dir / f"corr_length_truth_{ck}.pkl"

        if cached_truth.exists() and not args.no_cache:
            logger.info("Loading cached truth corr length from %s", cached_truth)
            with open(cached_truth, "rb") as f:
                truth_cl = pickle.load(f)
        else:
            truth_cl = compute_correlation_length(
                truth,
                stride=args.corr_stride,
                k_neighbors=args.corr_k,
                radius_km=args.corr_radius_km,
                km_per_pixel=args.pixel_size_km,
                seed=args.corr_seed,
            )
            with open(cached_truth, "wb") as f:
                pickle.dump(truth_cl, f)

        if cached_pred.exists() and not args.no_cache:
            logger.info("Loading cached pred corr length from %s", cached_pred)
            with open(cached_pred, "rb") as f:
                pred_cl = pickle.load(f)
        else:
            pred_cl = compute_correlation_length(
                pred_single,
                stride=args.corr_stride,
                k_neighbors=args.corr_k,
                radius_km=args.corr_radius_km,
                km_per_pixel=args.pixel_size_km,
                seed=args.corr_seed,
            )
            with open(cached_pred, "wb") as f:
                pickle.dump(pred_cl, f)

        # Compute bias on intersection
        L_t = truth_cl["L_map"]
        L_p = pred_cl["L_map"]
        inter = np.isfinite(L_t) & np.isfinite(L_p)
        bias = float(np.nanmean(L_p[inter] - L_t[inter])) if inter.any() else float("nan")

        results["correlation_length"] = {
            "truth_stats": truth_cl["L_stats"],
            "pred_stats": pred_cl["L_stats"],
            "bias_mean_L": bias,
            "truth_L_map": L_t,
            "pred_L_map": L_p,
        }
        logger.info(
            "Corr Length: truth_mean=%.1f pred_mean=%.1f bias=%.1f (%.1fs)",
            truth_cl["L_stats"]["mean"], pred_cl["L_stats"]["mean"],
            bias, time.time() - t0,
        )

    _save_results(results, out_dir)
    logger.info("Results saved to %s", out_dir)
    return results


def _save_results(results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for name, res in results.items():
        if name == "rapsd":
            summary["rapsd/score_log_mse"] = res["score_log_mse"]
        elif name == "fss":
            for s, v in zip(res["scales"], res["fss_scores"]):
                summary[f"fss/scale_{s}"] = v
        elif name == "correlation_length":
            for k, v in res.get("truth_stats", {}).items():
                summary[f"corr_length/truth_{k}"] = v
            for k, v in res.get("pred_stats", {}).items():
                summary[f"corr_length/pred_{k}"] = v
            summary["corr_length/bias_mean_L"] = res.get("bias_mean_L", float("nan"))

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    rows = [{"metric": k, "value": v} for k, v in summary.items()]
    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)

    for name, res in results.items():
        with open(out_dir / f"{name}_detail.pkl", "wb") as f:
            pickle.dump(res, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run spatial structure metrics (Section 5.2)")
    parser.add_argument("--domain-dir", type=Path, required=True)
    parser.add_argument("--pred-path", type=Path, required=True)
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("outputs/metrics/spatial"),
    )
    parser.add_argument("--metrics", nargs="+", choices=ALL_METRICS, default=None)
    parser.add_argument("--part", default="auto", choices=["part1", "part2", "auto"],
                        help="Experiment part: part1=hourly, part2=daily, auto=detect")
    parser.add_argument("--max-ens", type=int, default=None)
    parser.add_argument("--ens-strategy", default="mean", choices=["mean", "member_avg"])
    parser.add_argument("--pixel-size-km", type=float, default=3.0)
    parser.add_argument("--fft-backend", default="scipy", choices=["scipy", "numpy", "pyfftw"])
    parser.add_argument("--no-cache", action="store_true", help="Disable cache loading")

    # FSS
    parser.add_argument("--fss-scales", type=int, nargs="+", default=[1, 3, 5, 9, 17, 33, 65])
    parser.add_argument("--fss-threshold", type=float, default=None,
                        help="FSS threshold (auto from --part if omitted)")

    # Correlation length
    parser.add_argument("--corr-stride", type=int, default=2)
    parser.add_argument("--corr-k", type=int, default=20)
    parser.add_argument("--corr-radius-km", type=float, default=100.0)
    parser.add_argument("--corr-seed", type=int, default=0)

    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    # Resolve Part defaults
    from metrics.io_utils import detect_temporal_resolution, resolve_part_defaults
    if args.part == "auto":
        resolution = detect_temporal_resolution(args.domain_dir)
    else:
        resolution = "hourly" if args.part == "part1" else "daily"
    resolve_part_defaults(args, resolution, ["fss_threshold"])

    run(args)


if __name__ == "__main__":
    main()
