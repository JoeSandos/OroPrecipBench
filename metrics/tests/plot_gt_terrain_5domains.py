#!/usr/bin/env python3
"""Plot GT-only PEP-D and WLAB for all 5 US Part 1 domains.

Generates two figures:
  1. PEP-D: 5 subplots (one per domain), each showing GT elevation-precipitation profile
  2. WLAB: 5 subplots, each showing GT windward vs leeward precipitation

Usage:
    conda run -n climate python metrics/tests/plot_gt_terrain_5domains.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from metrics.io_utils import load_truth, load_static, load_wind
from metrics.terrain.pep_d import compute_pep_d
from metrics.terrain.wlab import compute_windward_index, compute_wlab

DATA_ROOT = Path(os.environ.get("OROPRECIP_DATA_ROOT", "data/dataset_root"))
DOMAINS = [
    "part1-US-Rocky",
    "part1-US-Cascades",
    "part1-US-Nevada",
    "part1-US-Appalachian",
    "part1-US-Great_Plains",
]
LABELS = ["Rocky Mtns", "Cascades", "Sierra Nevada", "Appalachians", "Great Plains"]
OUT_DIR = Path(__file__).resolve().parent / "gt_terrain_plots"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pepd_results = {}
    wlab_results = {}

    for domain, label in zip(DOMAINS, LABELS):
        domain_dir = DATA_ROOT / domain
        print(f"\n{'='*50}")
        print(f"  {label} ({domain})")
        print(f"{'='*50}")

        # Load data
        truth, truth_meta = load_truth(domain_dir)
        static = load_static(domain_dir)
        dem = static["elevation"]
        print(f"  truth: {truth.shape}, dem: {dem.shape}")
        print(f"  elevation range: {dem.min():.0f} - {dem.max():.0f} m")

        # PEP-D (GT only: use truth as both pred and truth)
        truth_4d = truth[:, None, :, :]  # fake ensemble dim
        pepd = compute_pep_d(truth_4d, truth, dem, bin_width=200.0)
        pepd_results[label] = pepd
        print(f"  PEP-D (self): {pepd['pep_d_scalar']:.6f} (should be ~0)")
        print(f"  Bins: {len(pepd['bin_centers'])}, p_gt_bar: {pepd['p_gt_bar']:.4f}")

        # WLAB
        u10, v10, wind_time = load_wind(domain_dir)
        print(f"  wind: u10 {u10.shape}")

        # Align wind with truth time (same length for part1)
        T = min(truth.shape[0], u10.shape[0])
        windward_idx = compute_windward_index(dem, u10[:T], v10[:T])
        wlab = compute_wlab(truth_4d[:T], truth[:T], windward_idx)
        wlab_results[label] = wlab
        print(f"  WLAB (self): {wlab['wlab_scalar']:.6f} (should be ~0)")
        print(f"  AR(GT): {wlab['ar_gt']:.4f}")
        print(f"  P_ww={wlab['p_gt_ww']:.4f}, P_lw={wlab['p_gt_lw']:.4f}")

    # ── Plot PEP-D: 5 subplots ──
    print("\n\nPlotting PEP-D...")
    fig, axes = plt.subplots(1, 5, figsize=(24, 5), sharey=False)
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

    for i, (label, pepd) in enumerate(pepd_results.items()):
        ax = axes[i]
        valid = np.isfinite(pepd["p_gt_per_bin"])
        idx = np.where(valid)[0]
        labels_v = [pepd["bin_labels"][j] for j in idx]
        vals = pepd["p_gt_per_bin"][idx]

        ax.bar(range(len(vals)), vals, color=colors[i], alpha=0.85)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels_v, rotation=60, ha="right", fontsize=7)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean Precip (mm/h)" if i == 0 else "")
        ax.set_xlabel("Elevation Bin")

    fig.suptitle("GT Precipitation-Elevation Profile (Part 1, all 5 US domains)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gt_pepd_5domains.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {OUT_DIR / 'gt_pepd_5domains.png'}")

    # ── Plot WLAB: 5 subplots ──
    print("Plotting WLAB...")
    fig2, axes2 = plt.subplots(1, 5, figsize=(18, 5), sharey=True)

    for i, (label, wlab) in enumerate(wlab_results.items()):
        ax = axes2[i]
        x = [0, 1]
        vals = [wlab["p_gt_ww"], wlab["p_gt_lw"]]
        ax.bar(x[0], vals[0], color=colors[i], alpha=0.9, width=0.5)
        ax.bar(x[1], vals[1], color=colors[i], alpha=0.5, width=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(["Windward", "Leeward"])
        ax.set_title(f"{label}\nAR={wlab['ar_gt']:.3f}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Mean Precip (mm/h)" if i == 0 else "")

        # Annotate values
        for j, v in enumerate(vals):
            ax.text(j, v + 0.001, f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    fig2.suptitle("GT Windward vs Leeward Precipitation (Part 1, all 5 US domains)", fontsize=13)
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "gt_wlab_5domains.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {OUT_DIR / 'gt_wlab_5domains.png'}")

    print("\nDone!")


if __name__ == "__main__":
    main()
