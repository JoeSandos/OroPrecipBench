#!/usr/bin/env python3
"""Redraw publication-style PEP-D sanity-check figures from existing CSV files."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIG_DIR = Path("paper/figures")
CROSS_CSV = FIG_DIR / "sanity_check_cross_domain_pepd.csv"
STAB_CSV = FIG_DIR / "sanity_check_temporal_stability.csv"

DOMAIN_ORDER = [
    "Cascades",
    "Appalachian",
    "Rocky",
    "Sierra Nevada",
    "Great Plains",
]

DOMAIN_COLORS = {
    "Cascades": "#1f77b4",
    "Appalachian": "#ff7f0e",
    "Rocky": "#2ca02c",
    "Sierra Nevada": "#d62728",
    "Great Plains": "#9467bd",
}

DISPLAY_LABELS = {
    "Cascades": "Cascades",
    "Appalachian": "Appala-\nchian",
    "Rocky": "Rocky",
    "Sierra Nevada": "Sierra\nNevada",
    "Great Plains": "Great\nPlains",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 30,
            "axes.titlesize": 34,
            "axes.labelsize": 32,
            "xtick.labelsize": 29,
            "ytick.labelsize": 29,
            "legend.fontsize": 28,
            "axes.linewidth": 1.4,
            "xtick.major.width": 1.3,
            "ytick.major.width": 1.3,
            "xtick.major.size": 7.0,
            "ytick.major.size": 7.0,
        }
    )


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    cross_df = pd.read_csv(CROSS_CSV)
    stab_df = pd.read_csv(STAB_CSV)

    cross_df["domain"] = pd.Categorical(cross_df["domain"], DOMAIN_ORDER, ordered=True)
    stab_df["domain"] = pd.Categorical(stab_df["domain"], DOMAIN_ORDER, ordered=True)

    cross_df = cross_df.sort_values("domain").reset_index(drop=True)
    stab_df = stab_df.sort_values("domain").reset_index(drop=True)
    return cross_df, stab_df


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.9, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.5,
        -0.20,
        label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=32,
        fontweight="bold",
    )


def draw_cross_domain(ax: plt.Axes, cross_df: pd.DataFrame) -> None:
    colors = [DOMAIN_COLORS[d] for d in cross_df["domain"].astype(str)]
    x = np.arange(len(cross_df))
    bars = ax.bar(
        x,
        cross_df["pep_d"],
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        width=0.72,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_LABELS[d] for d in cross_df["domain"].astype(str)])
    for label in ax.get_xticklabels():
        label.set_fontsize(26)
        label.set_linespacing(0.95)
    ax.set_ylabel("PEP-D")
    ax.set_xlabel("")
    ax.set_title("Cross-domain comparison", pad=10)
    ax.set_ylim(0.0, max(cross_df["pep_d"]) * 1.22)
    _style_axis(ax)

    for bar, value in zip(bars, cross_df["pep_d"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.00045,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=25,
        )

    _panel_label(ax, "(a)")


def draw_stability(
    ax: plt.Axes,
    cross_df: pd.DataFrame,
    stab_df: pd.DataFrame,
    *,
    square_like: bool,
) -> None:
    merged = cross_df[["domain", "pep_d"]].merge(
        stab_df[["domain", "pepd_mean", "pepd_std", "mean_abs_diff_between_halves"]],
        on="domain",
        how="inner",
    )

    max_val = float(
        np.max(
            np.concatenate(
                [
                    merged["pep_d"].to_numpy(),
                    (merged["pepd_mean"] + merged["pepd_std"]).to_numpy(),
                ]
            )
        )
    )
    lim = max_val * 1.18
    band = 0.0010

    ax.fill_between(
        [0.0, lim],
        [0.0 - band, lim - band],
        [0.0 + band, lim + band],
        color="#d0d0d0",
        alpha=0.28,
        zorder=0,
    )
    ax.plot([0.0, lim], [0.0, lim], linestyle="--", color="black", linewidth=1.4, zorder=1)

    offsets = {
        "Cascades": (10, 8),
        "Appalachian": (10, -16),
        "Rocky": (-48, 10),
        "Sierra Nevada": (10, 10),
        "Great Plains": (10, 10),
    }

    for _, row in merged.iterrows():
        domain = str(row["domain"])
        color = DOMAIN_COLORS[domain]
        ax.errorbar(
            row["pep_d"],
            row["pepd_mean"],
            yerr=row["pepd_std"],
            fmt="o",
            markersize=7,
            color=color,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.8,
            elinewidth=2.4,
            capsize=6,
            zorder=3,
        )
        dx, dy = offsets[domain]
        ax.annotate(
            DISPLAY_LABELS[domain].replace("\n", " "),
            (row["pep_d"], row["pepd_mean"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=25,
            color=color,
            fontweight="bold",
        )

    ax.set_xlim(0.0, lim)
    ax.set_ylim(0.0, lim)
    if square_like:
        ax.set_box_aspect(0.95)
    ax.set_xlabel("Full-test PEP-D")
    ax.set_ylabel("50% subsample PEP-D")
    ax.set_title("Temporal subsample agreement", pad=10)
    _style_axis(ax)

    max_diff = merged["mean_abs_diff_between_halves"].max()
    ax.text(
        0.04,
        0.96,
        f"Dashed line: y = x\nGray band: ±0.001\nMax mean |A-B| = {max_diff:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=24,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#b0b0b0", "alpha": 0.92},
    )

    _panel_label(ax, "(b)")


def main() -> None:
    configure_style()
    cross_df, stab_df = load_tables()

    single_cross_pdf = FIG_DIR / "sanity_check_cross_domain_pepd_publication.pdf"
    single_stab_pdf = FIG_DIR / "sanity_check_temporal_stability_publication.pdf"
    combined_pdf = FIG_DIR / "sanity_check_pepd_publication_combined.pdf"

    fig1, ax1 = plt.subplots(figsize=(9.6, 6.2))
    draw_cross_domain(ax1, cross_df)
    fig1.tight_layout()
    fig1.savefig(single_cross_pdf, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(9.6, 9.2))
    draw_stability(ax2, cross_df, stab_df, square_like=True)
    fig2.tight_layout()
    fig2.savefig(single_stab_pdf, bbox_inches="tight")
    plt.close(fig2)

    fig, axes = plt.subplots(2, 1, figsize=(11.2, 17.2))
    draw_cross_domain(axes[0], cross_df)
    draw_stability(axes[1], cross_df, stab_df, square_like=False)
    fig.tight_layout(h_pad=1.8)
    fig.savefig(combined_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {single_cross_pdf}")
    print(f"Saved: {single_stab_pdf}")
    print(f"Saved: {combined_pdf}")


if __name__ == "__main__":
    main()
