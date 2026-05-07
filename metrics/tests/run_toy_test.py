#!/usr/bin/env python3
"""End-to-end test: generate toy data, run all metrics, verify outputs.

Usage:
    python -m metrics.tests.run_toy_test
"""
from __future__ import annotations

import json
import sys
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TOY_DIR = Path(__file__).resolve().parent / "toy_domain"
OUT_BASE = Path(__file__).resolve().parent / "toy_output"


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    # Clean up previous runs
    if TOY_DIR.exists():
        shutil.rmtree(TOY_DIR)
    if OUT_BASE.exists():
        shutil.rmtree(OUT_BASE)

    # ── Step 1: Generate toy data ─────────────────────────────────
    section("Step 1: Generate toy data")
    from metrics.tests.generate_toy_data import generate

    generate(TOY_DIR, T_train=200, T_test=50, H=32, W=32, H_lr=8, W_lr=8, n_ens=4, seed=42)

    # Verify files exist
    for name in [
        "test_data_out.nc", "train_data_out.nc", "test_data_in.nc",
        "static_variables.nc",
        "bilinear_pred.nc", "baseline1_pred.nc", "baseline2_pred.nc",
    ]:
        p = TOY_DIR / name
        assert p.exists(), f"Missing: {p}"
        print(f"  OK: {name} ({p.stat().st_size / 1024:.0f} KB)")

    # ── Step 2: Load and verify ───────────────────────────────────
    section("Step 2: Load and verify data shapes")
    from metrics.io_utils import load_truth, load_prediction, load_static

    truth, truth_meta = load_truth(TOY_DIR)
    print(f"  truth: {truth.shape} dtype={truth.dtype}")

    bilinear, bilinear_meta = load_prediction(TOY_DIR / "bilinear_pred.nc")
    print(f"  bilinear: {bilinear.shape} (E={bilinear_meta['n_ensemble']})")

    baseline1, b1_meta = load_prediction(TOY_DIR / "baseline1_pred.nc")
    print(f"  baseline1: {baseline1.shape} (E={b1_meta['n_ensemble']})")

    baseline2, b2_meta = load_prediction(TOY_DIR / "baseline2_pred.nc")
    print(f"  baseline2: {baseline2.shape} (E={b2_meta['n_ensemble']})")

    static = load_static(TOY_DIR)
    print(f"  static: elevation={static['elevation'].shape} lsm={static['lsm'].shape}")

    # Test max_ens
    b2_capped, b2c_meta = load_prediction(TOY_DIR / "baseline2_pred.nc", max_ens=2)
    assert b2_capped.shape[1] == 2, f"Expected E=2, got {b2_capped.shape[1]}"
    print(f"  baseline2 (max_ens=2): {b2_capped.shape}")

    # ── Step 3: Conventional metrics ──────────────────────────────
    section("Step 3: Conventional metrics")
    from metrics.io_utils import reduce_ensemble, align_time
    from metrics.conventional.rmse import compute_rmse
    from metrics.conventional.mae import compute_mae
    from metrics.conventional.bias import compute_bias
    from metrics.conventional.cc import compute_cc
    from metrics.conventional.crps import compute_crps
    from metrics.conventional.ssim import compute_ssim
    from metrics.conventional.pod_far import compute_pod_far

    preds = {
        "bilinear": bilinear,
        "baseline1": baseline1,
        "baseline2": baseline2,
    }

    for name, pred in preds.items():
        pred_a, truth_a, _ = align_time(pred, truth, bilinear_meta["time"], truth_meta["time"])
        pred_mean = reduce_ensemble(pred_a, strategy="mean")

        r = compute_rmse(pred_mean, truth_a)
        m = compute_mae(pred_mean, truth_a)
        b = compute_bias(pred_mean, truth_a)
        c = compute_cc(pred_mean, truth_a)
        s = compute_ssim(pred_mean, truth_a, workers=1)
        pf = compute_pod_far(pred_mean, truth_a, threshold=0.5)
        cr = compute_crps(pred_a, truth_a)

        print(f"  {name:>10s}: RMSE={r['rmse_mean']:.3f} MAE={m['mae_mean']:.3f} "
              f"Bias={b['bias_mean']:.3f} CC={c['cc_mean']:.3f} "
              f"SSIM={s['ssim_mean']:.3f} POD={pf['pod_mean']:.3f} "
              f"FAR={pf['far_mean']:.3f} CRPS={cr['crps_mean']:.3f}")

        assert r["rmse_per_time"].shape == (truth_a.shape[0],)
        assert np.isfinite(r["rmse_mean"])

    # ── Step 4: Spatial metrics ───────────────────────────────────
    section("Step 4: Spatial metrics")
    from metrics.spatial.rapsd import compute_rapsd
    from metrics.spatial.fss import compute_fss
    from metrics.spatial.correlation_length import (
        compute_correlation_length,
        select_single_member,
    )

    for name, pred in preds.items():
        pred_a, truth_a, _ = align_time(pred, truth, bilinear_meta["time"], truth_meta["time"])

        # RAPSD
        rapsd = compute_rapsd(pred_a, truth_a, pixel_size_km=3.0)
        print(f"  {name:>10s} RAPSD: log-MSE={rapsd['score_log_mse']:.6f} "
              f"freqs={rapsd['freqs'].shape}")

        # FSS (use smaller scales for 32x32 grid)
        fss = compute_fss(pred_a, truth_a, scales=[1, 3, 5, 9], threshold=0.5)
        fss_str = " ".join(f"s{s}={v:.3f}" for s, v in zip(fss["scales"], fss["fss_scores"]))
        print(f"  {name:>10s} FSS: {fss_str}")

        # Correlation length (single member)
        single = select_single_member(pred_a, label=name, seed=0)
        cl = compute_correlation_length(
            single, stride=4, k_neighbors=8,
            radius_km=30.0, km_per_pixel=3.0, seed=0,
        )
        print(f"  {name:>10s} CorrLen: mean={cl['L_stats']['mean']:.1f} "
              f"L_map={cl['L_map'].shape}")

    # ── Step 5: Extreme metrics ───────────────────────────────────
    section("Step 5: Extreme metrics")
    from metrics.extreme.quantile_bias import compute_quantile_bias
    from metrics.extreme.precip_pdf import compute_precip_pdf
    from metrics.extreme.return_period import compute_return_period_bias
    from metrics.io_utils import load_train

    # Load train data for climatological quantile thresholds
    train_data, _ = load_train(TOY_DIR)
    print(f"  Loaded train data for climatology: {train_data.shape}")

    for name, pred in preds.items():
        pred_a, truth_a, _ = align_time(pred, truth, bilinear_meta["time"], truth_meta["time"])

        # Quantile bias (using train data as climatology)
        qb = compute_quantile_bias(
            pred_a, truth_a, quantiles=[0.90, 0.95, 0.99],
            climatology=train_data,
        )
        for q in [0.90, 0.95, 0.99]:
            ql = f"q{q:.2f}"
            print(f"  {name:>10s} QB {ql}: abs={qb[f'{ql}_abs_bias_mean']:.3f} "
                  f"ratio={qb[f'{ql}_ratio_mean']:.3f}")

        # Precip PDF
        pdf = compute_precip_pdf(pred_a, truth_a, wet_threshold=0.1)
        print(f"  {name:>10s} PDF: truth_wet={pdf['truth_n_wet']} "
              f"pred_wet={pdf['pred_n_wet']} bins={pdf['bins'].shape}")

    # Return period (need enough years — use T=50 as "daily" here, skip if <10 yrs)
    print("\n  Return Period (requires daily data with >=10 years):")
    pred_a, truth_a, _ = align_time(baseline2, truth, b2_meta["time"], truth_meta["time"])
    rp = compute_return_period_bias(pred_a, truth_a, return_periods=[2, 5], min_years=10)
    if "error" in rp:
        print(f"  (Expected) Skipped: {rp['error']}")
    else:
        for rp_val in [2, 5]:
            rpl = f"rp{rp_val}"
            print(f"    RP {rp_val}-yr: bias={rp[f'{rpl}_bias_mean']:.2f}")

    # ── Step 6: Test CLI runners ──────────────────────────────────
    section("Step 6: Test CLI runners (conventional)")
    import argparse
    from metrics.conventional.run_conventional import run as run_conv

    conv_out = OUT_BASE / "conventional" / "baseline2"
    conv_args = argparse.Namespace(
        domain_dir=TOY_DIR,
        pred_path=TOY_DIR / "baseline2_pred.nc",
        out_dir=conv_out,
        metrics=["rmse", "mae", "crps", "cc", "pod_far"],
        ens_strategy="mean",
        max_ens=None,
        rain_threshold=0.5,
        ssim_workers=1,
        log_level="WARNING",
    )
    results = run_conv(conv_args)
    assert (conv_out / "summary.json").exists()
    assert (conv_out / "summary.csv").exists()
    with open(conv_out / "summary.json") as f:
        summary = json.load(f)
    print(f"  summary.json keys: {list(summary.keys())}")
    print(f"  RMSE={summary.get('rmse/rmse_mean', '?'):.4f}")
    print(f"  CRPS={summary.get('crps/crps_mean', '?'):.4f}")

    # ── Step 7: Test spatial CLI ──────────────────────────────────
    section("Step 7: Test CLI runners (spatial)")
    from metrics.spatial.run_spatial import run as run_spat

    spat_out = OUT_BASE / "spatial" / "baseline1"
    spat_args = argparse.Namespace(
        domain_dir=TOY_DIR,
        pred_path=TOY_DIR / "baseline1_pred.nc",
        out_dir=spat_out,
        metrics=["rapsd", "fss"],
        max_ens=None,
        ens_strategy="mean",
        pixel_size_km=3.0,
        fft_backend="scipy",
        no_cache=True,
        fss_scales=[1, 3, 5, 9],
        fss_threshold=0.5,
        corr_stride=4,
        corr_k=8,
        corr_radius_km=30.0,
        corr_seed=0,
        log_level="WARNING",
    )
    run_spat(spat_args)
    assert (spat_out / "summary.json").exists()
    with open(spat_out / "summary.json") as f:
        summary = json.load(f)
    print(f"  summary.json keys: {list(summary.keys())}")

    # ── Step 8: Test extreme CLI ──────────────────────────────────
    section("Step 8: Test CLI runners (extreme)")
    from metrics.extreme.run_extreme import run as run_ext

    ext_out = OUT_BASE / "extreme" / "baseline2"
    ext_args = argparse.Namespace(
        domain_dir=TOY_DIR,
        pred_path=TOY_DIR / "baseline2_pred.nc",
        out_dir=ext_out,
        metrics=["quantile_bias", "precip_pdf"],
        max_ens=None,
        ens_strategy="per_member",
        no_cache=True,
        quantiles=[0.90, 0.95, 0.99],
        wet_threshold=0.1,
        pdf_bins_max=50.0,
        pdf_bins_width=1.0,
        return_periods=[2, 5],
        min_years=10,
        log_level="WARNING",
    )
    run_ext(ext_args)
    assert (ext_out / "summary.json").exists()
    with open(ext_out / "summary.json") as f:
        summary = json.load(f)
    print(f"  summary.json keys: {list(summary.keys())}")

    # ── Step 9: Terrain metrics ──────────────────────────────────
    section("Step 9: Terrain metrics (PEP-D only; WLAB disabled)")
    from metrics.terrain.pep_d import compute_pep_d

    dem = static["elevation"]

    for name, pred in preds.items():
        pred_a, truth_a, _ = align_time(pred, truth, bilinear_meta["time"], truth_meta["time"])

        # PEP-D
        pepd = compute_pep_d(pred_a, truth_a, dem, bin_width=200.0)
        print(f"  {name:>10s} PEP-D={pepd['pep_d_scalar']:.6f} "
              f"K={len(pepd['bin_centers'])} bins")
        assert np.isfinite(pepd["pep_d_scalar"])

    # ── Step 10: Test terrain CLI ─────────────────────────────────
    section("Step 10: Test CLI runners (terrain)")
    from metrics.terrain.run_terrain import run as run_terr

    terr_out = OUT_BASE / "terrain" / "baseline1"
    terr_args = argparse.Namespace(
        domain_dir=TOY_DIR,
        pred_path=TOY_DIR / "baseline1_pred.nc",
        out_dir=terr_out,
        metrics=["pep_d"],
        max_ens=None,
        bin_width=200.0,
        no_cache=True,
        log_level="WARNING",
    )
    run_terr(terr_args)
    assert (terr_out / "summary.json").exists()
    assert (terr_out / "pep_d_bars.png").exists()
    with open(terr_out / "summary.json") as f:
        summary = json.load(f)
    print(f"  summary.json keys: {list(summary.keys())}")
    print(f"  PEP-D={summary.get('pep_d/pep_d_scalar', '?'):.6f}")

    # ── Done ──────────────────────────────────────────────────────
    section("ALL TESTS PASSED")
    print(f"Toy data: {TOY_DIR}")
    print(f"Outputs:  {OUT_BASE}")


if __name__ == "__main__":
    main()
