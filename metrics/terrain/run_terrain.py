#!/usr/bin/env python3
"""Run terrain-specific metrics (design.md Section 5.4).

Usage:
    python -m metrics.terrain.run_terrain \
        --domain-dir /path/to/part1-US-Rocky \
        --pred-path /path/to/pred.nc \
        --out-dir outputs/metrics/terrain/part1-US-Rocky/model \
        --metrics pep_d
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
    cache_key,
    load_prediction,
    load_static,
    load_truth,
    load_wind,
)

# WLAB disabled: K=2 profile normalization lacks discriminative power at 3km DEM gradient scale.
# Code preserved in wlab.py for future multi-scale gradient approach.
ALL_METRICS = ["pep_d"]  # ["pep_d", "wlab"]

logger = logging.getLogger("metrics.terrain")


def setup_logging(out_dir: Path, level: str = "INFO") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    logger.setLevel(getattr(logging, level))
    logger.handlers.clear()

    fh = logging.FileHandler(out_dir / "run_terrain.log")
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
    static = load_static(args.domain_dir)
    dem = static["elevation"]
    logger.info(
        "Loaded truth %s, pred %s, dem %s in %.1fs",
        truth.shape, pred.shape, dem.shape, time.time() - t0,
    )

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
            dem[sea] = np.nan
            logger.info("Masked %d sea pixels (%.1f%%)", n_sea, 100 * n_sea / sea.size)

    results = {}

    # ── PEP-D ──
    if "pep_d" in metrics_to_run:
        from metrics.terrain.pep_d import compute_pep_d
        from metrics.terrain.plot_terrain import plot_pep_d_bars

        t0 = time.time()
        ck = cache_key(args.pred_path, metric="pep_d", bin_width=args.bin_width)
        cached = cache_dir / f"pep_d_{ck}.pkl"

        if cached.exists() and not args.no_cache:
            logger.info("Loading cached PEP-D from %s", cached)
            with open(cached, "rb") as f:
                res = pickle.load(f)
        else:
            res = compute_pep_d(pred, truth, dem, bin_width=args.bin_width, land_mask=land)
            with open(cached, "wb") as f:
                pickle.dump(res, f)

        results["pep_d"] = res
        logger.info("PEP-D = %.6f (%.1fs)", res["pep_d_scalar"], time.time() - t0)

        plot_pep_d_bars(res, save_path=out_dir / "pep_d_bars.png")
        logger.info("Saved pep_d_bars.png")

    # ── WLAB (disabled) ──
    # K=2 profile normalization lacks discriminative power; local 3km DEM gradient
    # does not align with macro-scale rain-shadow structure (e.g. Cascades).
    # Code preserved in wlab.py for future multi-scale gradient approach.
    #
    # if "wlab" in metrics_to_run:
    #     from metrics.terrain.wlab import compute_wlab, compute_windward_index
    #     from metrics.terrain.plot_terrain import plot_wlab_bars
    #     t0 = time.time()
    #     wi_ck = cache_key(args.domain_dir / "test_data_in.nc", metric="windward_index")
    #     wi_cached = cache_dir / f"windward_index_{wi_ck}.npy"
    #     if wi_cached.exists() and not args.no_cache:
    #         windward_idx = np.load(wi_cached)
    #     else:
    #         u10, v10, wind_time = load_wind(args.domain_dir)
    #         if len(wind_time) != truth.shape[0]:
    #             common_w, w_idx, t_idx = np.intersect1d(wind_time, common_time, return_indices=True)
    #             u10, v10 = u10[w_idx], v10[w_idx]
    #         windward_idx = compute_windward_index(dem, u10, v10)
    #         np.save(wi_cached, windward_idx)
    #     wlab_ck = cache_key(args.pred_path, metric="wlab")
    #     wlab_cached = cache_dir / f"wlab_{wlab_ck}.pkl"
    #     if wlab_cached.exists() and not args.no_cache:
    #         with open(wlab_cached, "rb") as f: res = pickle.load(f)
    #     else:
    #         res = compute_wlab(pred, truth, windward_idx, land_mask=land)
    #         with open(wlab_cached, "wb") as f: pickle.dump(res, f)
    #     results["wlab"] = res
    #     logger.info("WLAB = %.6f  AR(GT)=%.3f AR(Pred)=%.3f (%.1fs)",
    #                 res["wlab_scalar"], res["ar_gt"], res["ar_pred"], time.time() - t0)
    #     plot_wlab_bars(res, save_path=out_dir / "wlab_bars.png")

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

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    rows = [{"metric": k, "value": v} for k, v in summary.items()]
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)

    for name, res in results.items():
        with open(out_dir / f"{name}_detail.pkl", "wb") as f:
            pickle.dump(res, f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run terrain-specific metrics (Section 5.4)",
    )
    parser.add_argument("--domain-dir", type=Path, required=True)
    parser.add_argument("--pred-path", type=Path, required=True)
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("outputs/metrics/terrain"),
    )
    parser.add_argument("--metrics", nargs="+", choices=ALL_METRICS, default=None)
    parser.add_argument("--max-ens", type=int, default=None)
    parser.add_argument("--bin-width", type=float, default=200.0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
