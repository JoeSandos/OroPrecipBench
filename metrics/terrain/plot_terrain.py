"""Visualization for terrain-specific metrics (PEP-D and WLAB).

Generates grouped bar charts comparing GT vs prediction precipitation
profiles across elevation bins (PEP-D) or windward/leeward categories (WLAB).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_pep_d_bars(
    results: dict,
    *,
    ax: plt.Axes | None = None,
    title: str = "PEP-D",
    pred_label: str = "Pred",
    gt_color: str = "#555555",
    pred_color: str = "#4C9BE8",
    save_path: Path | str | None = None,
) -> Figure:
    """Plot grouped bar chart of precipitation-elevation profile.

    Parameters
    ----------
    results : dict from compute_pep_d()
    ax : optional axes for multi-panel embedding
    title : figure title
    pred_label : label for prediction bars
    save_path : if provided, save figure to this path
    """
    labels = results["bin_labels"]
    p_gt = results["p_gt_per_bin"]
    p_pred = results["p_pred_per_bin"]
    pep_d = results["pep_d_scalar"]
    K = len(labels)

    # Filter valid bins
    valid = np.isfinite(p_gt) & np.isfinite(p_pred)
    idx = np.where(valid)[0]
    if len(idx) == 0:
        fig, ax_ = (ax.figure, ax) if ax is not None else plt.subplots(1, 1, figsize=(8, 5))
        ax_.text(0.5, 0.5, "No valid bins", ha="center", va="center", transform=ax_.transAxes)
        return fig

    labels_v = [labels[i] for i in idx]
    p_gt_v = p_gt[idx]
    p_pred_v = p_pred[idx]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(max(6, len(idx) * 0.8 + 2), 5))
    else:
        fig = ax.figure

    x = np.arange(len(labels_v))
    width = 0.35

    ax.bar(x - width / 2, p_gt_v, width, label="GT", color=gt_color)
    ax.bar(x + width / 2, p_pred_v, width, label=pred_label, color=pred_color)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_v, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Precipitation")
    ax.set_title(f"{title}  (D = {pep_d:.4f})")
    ax.legend()

    if own_fig:
        fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig


def plot_wlab_bars(
    results: dict,
    *,
    ax: plt.Axes | None = None,
    title: str = "WLAB",
    pred_label: str = "Pred",
    gt_color: str = "#555555",
    pred_color: str = "#E8784C",
    save_path: Path | str | None = None,
) -> Figure:
    """Plot grouped bar chart of windward/leeward precipitation.

    Parameters
    ----------
    results : dict from compute_wlab()
    ax : optional axes for multi-panel embedding
    title : figure title
    pred_label : label for prediction bars
    save_path : if provided, save figure to this path
    """
    wlab = results["wlab_scalar"]
    ar_gt = results["ar_gt"]
    ar_pred = results["ar_pred"]

    categories = ["Windward", "Leeward"]
    gt_vals = [results["p_gt_ww"], results["p_gt_lw"]]
    pred_vals = [results["p_pred_ww"], results["p_pred_lw"]]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    else:
        fig = ax.figure

    x = np.arange(len(categories))
    width = 0.3

    ax.bar(x - width / 2, gt_vals, width, label="GT", color=gt_color)
    ax.bar(x + width / 2, pred_vals, width, label=pred_label, color=pred_color)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Mean Precipitation")
    ax.set_title(f"{title}  (WLAB = {wlab:.4f})")
    ax.legend()

    # Annotate AR values
    y_max = max(max(gt_vals), max(pred_vals))
    ax.text(
        0.98, 0.95,
        f"AR(GT)={ar_gt:.2f}\nAR(Pred)={ar_pred:.2f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
    )

    if own_fig:
        fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig
