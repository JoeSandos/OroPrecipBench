"""Step 0: Verify HRRR coordinate files and create directory structure."""

import argparse
import xarray as xr

from config import (
    COORD_DIR,
    DOMAINS,
    OUTPUT_ROOT,
    STAGING_ROOT,
    domain_dir,
    staging_dir,
)


def verify_coord_files():
    for domain in DOMAINS:
        fname = f"part1-US-{domain}_lat_lon_192_192.nc"
        path = COORD_DIR / fname
        assert path.exists(), f"Missing: {path}"
        ds = xr.open_dataset(path, drop_variables=["step", "heightAboveGround"])
        assert ds.latitude.shape == (192, 192), f"{domain} lat shape: {ds.latitude.shape}"
        assert ds.longitude.shape == (192, 192), f"{domain} lon shape: {ds.longitude.shape}"
        lat = ds.latitude.values
        lon = ds.longitude.values
        print(
            f"  [ok]   {domain}: lat [{lat.min():.3f}, {lat.max():.3f}], "
            f"lon [{lon.min():.3f}, {lon.max():.3f}]"
        )
        ds.close()


def create_directories():
    for domain in DOMAINS:
        domain_dir(domain).mkdir(parents=True, exist_ok=True)
        for year in range(2000, 2025):
            (staging_dir(domain) / str(year)).mkdir(parents=True, exist_ok=True)
    (STAGING_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    print(f"  [ok]   Created output dirs under {OUTPUT_ROOT}")
    print(f"  [ok]   Created staging dirs under {STAGING_ROOT}")


def main():
    parser = argparse.ArgumentParser(description="Step 0: Prepare directories and verify coordinate files")
    parser.parse_args()

    print("=" * 60)
    print("Step 0: Prepare")
    print("=" * 60)

    print(f"  Coordinate directory: {COORD_DIR}")

    print("\n1. Verifying coordinate files...")
    verify_coord_files()

    print("\n2. Creating directory structure...")
    create_directories()

    print("\nStep 0 complete.")


if __name__ == "__main__":
    main()
