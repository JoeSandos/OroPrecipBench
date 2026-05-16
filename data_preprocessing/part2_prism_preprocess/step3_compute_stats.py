"""Step 3: Compute precipitation output normalization statistics."""

import argparse
import json

import numpy as np
import xarray as xr

from config import DOMAINS, domain_dir


def compute_stats(domain: str) -> dict:
    """Compute stats.json content for one domain."""
    out_dir = domain_dir(domain)

    # Output: tp from training data.
    ds = xr.open_dataset(out_dir / "train_data_out.nc")
    tp = ds.tp.values  # (N, 192, 192) float32
    ds.close()

    tp_flat = tp.ravel()
    log_tp = np.log1p(tp_flat.astype(np.float64))

    output_stats = {
        "tp": {
            "mean": float(tp_flat.mean()),
            "std": float(tp_flat.std()),
            "min": float(tp_flat.min()),
            "max": float(tp_flat.max()),
            "log_mean": float(log_tp.mean()),
            "log_std": float(log_tp.std()),
            "log_min": float(log_tp.min()),
            "log_max": float(log_tp.max()),
            "log_eps": 1.0,
        }
    }

    return {"output": output_stats}


def main():
    parser = argparse.ArgumentParser(description="Step 3: Compute stats.json")
    parser.add_argument("--domains", default=",".join(DOMAINS))
    args = parser.parse_args()

    domains = args.domains.split(",")

    print("=" * 60)
    print("Step 3: Compute stats.json")
    print("=" * 60)

    for domain in domains:
        stats = compute_stats(domain)
        out_path = domain_dir(domain) / "stats.json"
        with open(out_path, "w") as f:
            json.dump(stats, f, indent=4)

        tp = stats["output"]["tp"]
        print(f"\n  {domain}:")
        print(f"    tp: mean={tp['mean']:.4f}, std={tp['std']:.3f}, max={tp['max']:.1f} mm/day")
        print(f"    tp log: mean={tp['log_mean']:.4f}, std={tp['log_std']:.4f}")

    print("\nStep 3 complete.")


if __name__ == "__main__":
    main()
