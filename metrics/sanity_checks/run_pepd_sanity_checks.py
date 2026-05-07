#!/usr/bin/env python3
"""Run PEP-D sanity-check experiments for paper Section 3.4.

Experiments
-----------
1. Cross-domain comparison:
   Compute bilinear-baseline PEP-D for 5 Part 1 US domains and save
   a ranked bar chart.
2. Temporal subsampling stability:
   Repeatedly split a test set into two random 50% temporal subsets,
   compute PEP-D on each subset, and summarize the variability.

Outputs are written under ``paper/figures`` by default.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from metrics.io_utils import align_time, load_land_mask, load_prediction, load_static, load_truth
from metrics.terrain.pep_d import compute_pep_d


DATA_ROOT = Path(os.environ.get("OROPRECIP_DATA_ROOT", "data/dataset_root"))
BILINEAR_ROOT = Path(
    os.environ.get(
        "OROPRECIP_BILINEAR_ROOT",
        "baselines/runs/grid_part1_hrrr_50m_dem0/statistical/bilinear",
    )
)

DOMAINS = [
    {
        "dataset": "part1-US-Rocky",
        "baseline": "Rocky",
        "label": "Rocky",
    },
    {
        "dataset": "part1-US-Cascades",
        "baseline": "Cascades",
        "label": "Cascades",
    },
    {
        "dataset": "part1-US-Nevada",
        "baseline": "Nevada",
        "label": "Sierra Nevada",
    },
    {
        "dataset": "part1-US-Appalachian",
        "baseline": "Appalachian",
        "label": "Appalachian",
    },
    {
        "dataset": "part1-US-Great_Plains",
        "baseline": "Great_Plains",
        "label": "Great Plains",
    },
]


def _load_domain_inputs(domain: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    domain_dir = DATA_ROOT / domain["dataset"]
    pred_path = BILINEAR_ROOT / domain["baseline"] / "default" / "predictions" / "pred.nc"

    truth, truth_meta = load_truth(domain_dir)
    pred, pred_meta = load_prediction(pred_path)
    pred, truth, _ = align_time(pred, truth, pred_meta["time"], truth_meta["time"])

    static = load_static(domain_dir)
    dem = static["elevation"]
    land_mask = load_land_mask(domain_dir)

    return pred, truth, dem, land_mask


def run_cross_domain(bin_width: float) -> pd.DataFrame:
    rows = []
    for domain in DOMAINS:
        print(f"[cross-domain] loading {domain['label']} ...", flush=True)
        pred, truth, dem, land_mask = _load_domain_inputs(domain)
        result = compute_pep_d(pred, truth, dem, bin_width=bin_width, land_mask=land_mask)
        rows.append(
            {
                "domain": domain["label"],
                "dataset_dir": domain["dataset"],
                "baseline_dir": domain["baseline"],
                "pep_d": float(result["pep_d_scalar"]),
                "n_time": int(truth.shape[0]),
                "n_land_pixels": int(land_mask.sum()) if land_mask is not None else int(np.prod(dem.shape)),
                "elevation_min_m": float(np.nanmin(dem[land_mask])) if land_mask is not None else float(np.nanmin(dem)),
                "elevation_max_m": float(np.nanmax(dem[land_mask])) if land_mask is not None else float(np.nanmax(dem)),
            }
        )
    df = pd.DataFrame(rows).sort_values("pep_d", ascending=False).reset_index(drop=True)
    return df


def plot_cross_domain(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    colors = []
    mountain_domains = {"Rocky", "Cascades", "Sierra Nevada"}
    for domain in df["domain"]:
        if domain in mountain_domains:
            colors.append("#C44E52")
        elif domain == "Great Plains":
            colors.append("#55A868")
        else:
            colors.append("#8172B2")

    bars = ax.bar(df["domain"], df["pep_d"], color=colors, alpha=0.92)
    ax.set_ylabel("PEP-D")
    ax.set_xlabel("Domain")
    ax.set_title("Cross-domain PEP-D of Bilinear Baseline")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=20)

    for bar, value in zip(bars, df["pep_d"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_temporal_stability(bin_width: float, repeats: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for domain in DOMAINS:
        print(f"[temporal-stability] loading {domain['label']} ...", flush=True)
        pred, truth, dem, land_mask = _load_domain_inputs(domain)
        n_time = truth.shape[0]
        if n_time < 2:
            raise ValueError(f"Need at least 2 time steps for temporal split, got {n_time} for {domain['label']}")

        split_a = []
        split_b = []

        for repeat_idx in range(repeats):
            perm = rng.permutation(n_time)
            mid = n_time // 2
            idx_a = np.sort(perm[:mid])
            idx_b = np.sort(perm[mid:])

            res_a = compute_pep_d(
                pred[idx_a],
                truth[idx_a],
                dem,
                bin_width=bin_width,
                land_mask=land_mask,
            )
            res_b = compute_pep_d(
                pred[idx_b],
                truth[idx_b],
                dem,
                bin_width=bin_width,
                land_mask=land_mask,
            )

            split_a.append(float(res_a["pep_d_scalar"]))
            split_b.append(float(res_b["pep_d_scalar"]))
            print(
                f"  repeat {repeat_idx + 1:02d}/{repeats}: "
                f"A={split_a[-1]:.6f}, B={split_b[-1]:.6f}",
                flush=True,
            )

        values = np.asarray(split_a + split_b, dtype=np.float64)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        rel_std = float(100.0 * std / mean) if mean != 0.0 else float("nan")
        pair_abs_diff_mean = float(np.mean(np.abs(np.asarray(split_a) - np.asarray(split_b))))

        rows.append(
            {
                "domain": domain["label"],
                "pepd_mean": mean,
                "pepd_std": std,
                "relative_std_percent": rel_std,
                "mean_abs_diff_between_halves": pair_abs_diff_mean,
                "n_repeats": repeats,
                "n_time": n_time,
            }
        )

    df = pd.DataFrame(rows).sort_values("pepd_mean", ascending=False).reset_index(drop=True)
    return df


def plot_temporal_stability(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(df))
    bars = ax.bar(
        x,
        df["pepd_mean"],
        yerr=df["pepd_std"],
        capsize=5,
        color="#4C72B0",
        alpha=0.9,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["domain"], rotation=20)
    ax.set_ylabel("PEP-D")
    ax.set_xlabel("Domain")
    ax.set_title("Temporal 50% Subsampling Stability of PEP-D")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)

    for bar, mean, std, rel_std in zip(
        bars,
        df["pepd_mean"],
        df["pepd_std"],
        df["relative_std_percent"],
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{mean:.4f}\n±{std:.4f}\n({rel_std:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PEP-D sanity-check experiments.")
    parser.add_argument("--bin-width", type=float, default=200.0)
    parser.add_argument("--temporal-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("paper/figures"),
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cross_df = run_cross_domain(bin_width=args.bin_width)
    cross_csv = out_dir / "sanity_check_cross_domain_pepd.csv"
    cross_json = out_dir / "sanity_check_cross_domain_pepd.json"
    cross_fig = out_dir / "sanity_check_cross_domain_pepd.png"
    cross_df.to_csv(cross_csv, index=False)
    cross_df.to_json(cross_json, orient="records", indent=2)
    plot_cross_domain(cross_df, cross_fig)

    stability_df = run_temporal_stability(
        bin_width=args.bin_width,
        repeats=args.temporal_repeats,
        seed=args.seed,
    )
    stability_csv = out_dir / "sanity_check_temporal_stability.csv"
    stability_json = out_dir / "sanity_check_temporal_stability.json"
    stability_fig = out_dir / "sanity_check_temporal_stability.png"
    stability_df.to_csv(stability_csv, index=False)
    stability_df.to_json(stability_json, orient="records", indent=2)
    plot_temporal_stability(stability_df, stability_fig)

    summary = {
        "cross_domain_csv": str(cross_csv),
        "cross_domain_json": str(cross_json),
        "cross_domain_figure": str(cross_fig),
        "temporal_stability_csv": str(stability_csv),
        "temporal_stability_json": str(stability_json),
        "temporal_stability_figure": str(stability_fig),
    }
    with open(out_dir / "sanity_check_pepd_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nCross-domain PEP-D")
    print(cross_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nTemporal stability")
    formatted = stability_df.copy()
    formatted["summary"] = formatted.apply(
        lambda r: f"{r['pepd_mean']:.6f} ± {r['pepd_std']:.6f} ({r['relative_std_percent']:.2f}%)",
        axis=1,
    )
    print(formatted[["domain", "summary", "mean_abs_diff_between_halves"]].to_string(index=False))


if __name__ == "__main__":
    main()
