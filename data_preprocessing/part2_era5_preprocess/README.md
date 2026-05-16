# Part2 ERA5 Preproces

This directory contains the ERA5 part2 preprocessing workflow. Scripts are ordered from `step1-*` to `step6-*`.

## Dependencies

```bash
conda install -c conda-forge netcdf4 numpy scipy cdsapi
```

`cdsapi` is only required for Step 1. Do not store CDS credentials in code. Keep credentials in local `.cdsapirc` files and pass them with `--cdsapirc-files` when needed.

## Step 1: Download missing global hourly ERA5 files

Script: `step1-download-missing-era5-global.py`

Expected input layout:

```text
<era5-global-root>/<var>/<YYYY>/<MM>/<var>_YYYYMMDD_HH00.nc
```

Example:

```bash
python step1-download-missing-era5-global.py   --root /path/to/era5/hourly/global   --output-root ./era5_source   --start 2010010100   --end 2024123123   --cdsapirc-files ~/.cdsapirc0 ~/.cdsapirc1   --parallel-downloads 2   --execute
```

Downloaded files are written to `--output-root` with the same layout as the original global hourly ERA5 files. The original source directory is not modified.

## Step 2: Crop US regions and merge variables

Script: `step2-crop-merge-era5-us-regions.py`

This step crops global hourly ERA5 files to five US regions, merges nine variables, and differences accumulated variables when needed. `tp` is converted from meters to `mm/h`.

Example:

```bash
python step2-crop-merge-era5-us-regions.py   --input-root /path/to/era5/hourly/global   --supplemental-input-root ./era5_source   --output-root ./era5_data   --start 2010010100   --end 2024123123   --workers 16
```

To repair already generated files with missing variables:

```bash
python step2-crop-merge-era5-us-regions.py   --input-root /path/to/era5/hourly/global   --supplemental-input-root ./era5_source   --output-root ./era5_data   --repair-missing-output-vars   --workers 16
```

## Step 3: Check regional files for missing variables

Script: `step3-check-region-missing-vars.py`

Example:

```bash
python step3-check-region-missing-vars.py   --root ./era5_data   --write-list ./missing_region_vars.txt
```

## Step 4: Build the part2 daily dataset_root

Script: `step4-build-part2-dataset-root.py`

This step converts hourly regional files into `train/val/test_data_in.nc`. For part2 daily aggregation, `tp/ssrd/tisr` are summed, other variables are averaged, and `2018-08-22` is excluded by default.

Example:

```bash
python step4-build-part2-dataset-root.py   --input-root ./era5_data   --output-root ./dataset_root   --scales part2
```

Part2 splits: train=2010-2021, val=2022, test=2023-2024.

## Step 5: Merge 2000-2009 train data and rewrite stats

Script: `step5-merge-2000-2009-train-and-rewrite-stats.py`

This step prepends older 2000-2009 part2 train files to the current train files and recomputes `stats.json`.

Example:

```bash
python step5-merge-2000-2009-train-and-rewrite-stats.py   --src-root ./dataset_root2000   --dst-root ./dataset_root
```

To rewrite statistics only:

```bash
python step5-merge-2000-2009-train-and-rewrite-stats.py   --dst-root ./dataset_root   --stats-only
```

## Step 6: Interpolate to the HRRR 24x24 low-resolution grid

Script: `step6-interpolate-part2-to-hrrr-24x24.py`

This step interpolates part2 ERA5 0.25-degree regular-grid inputs to the Part 1 24x24 HRRR curvilinear grid and recomputes `stats.json` and `status.json`.

Example:

```bash
python step6-interpolate-part2-to-hrrr-24x24.py   --src-root ./dataset_root   --target-root /path/to/hrrr/dataset_root   --out-root ./dataset_root_new   --overwrite
```

Interpolation uses `RegularGridInterpolator(method="linear")`. Output coordinates are written directly from the Part 1 input 2D `latitude(24,24)` and `longitude(24,24)` arrays, with longitudes kept in 0-360 degrees.

## Final output

Recommended final layout:

```text
dataset_root_new/
  part2-US-Appalachian/
    train_data_in.nc
    val_data_in.nc
    test_data_in.nc
    stats.json
    status.json
  part2-US-Cascades/
  part2-US-Great_Plains/
  part2-US-Nevada/
  part2-US-Rocky/
```

## Privacy

This cleaned copy does not include CDS keys, user names, or machine-specific absolute working paths. Local paths should be passed through command-line arguments.
