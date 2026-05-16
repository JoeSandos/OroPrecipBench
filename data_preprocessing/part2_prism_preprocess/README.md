# PRISM Part 2 Preprocessing

This directory contains the reproducible preprocessing scripts for the
OroPrecipBench Part 2 PRISM target dataset. The pipeline maps daily PRISM
precipitation onto the five 192 x 192 HRRR benchmark domains, assembles
train/validation/test NetCDF files, computes precipitation normalization
statistics, and validates the outputs.

## Contents

- `config.py`: shared domains, PRISM grid metadata, time splits, and path configuration.
- `step0_prepare.py`: verifies HRRR coordinate files and creates staging/output directories.
- `step1_extract_interpolate.py`: reads daily PRISM ZIP archives and interpolates precipitation to each domain.
- `step2_assemble_netcdf.py`: assembles daily `.npy` files into split-level NetCDF files.
- `step3_compute_stats.py`: computes `stats.json` from training precipitation outputs.
- `step4_validate.py`: checks file existence, shapes, coordinates, dtypes, ranges, and summary statistics.

Shell wrappers, caches, and intermediate data are intentionally not included in
the repository.

## Inputs

The pipeline expects:

- Daily PRISM precipitation ZIP archives named like
  `prism_ppt_us_25m_YYYYMMDD.zip`, organized by year.
- HRRR reference coordinate files named
  `part1-US-<Domain>_lat_lon_192_192.nc`.

The five domains are:

- `Rocky`
- `Cascades`
- `Nevada`
- `Appalachian`
- `Great_Plains`

The default time split is:

- train: 2000-01-01 through 2021-12-31
- val: 2022-01-01 through 2022-12-31
- test: 2023-01-01 through 2024-12-31

The known corrupted PRISM date `2018-08-22` is skipped.

## Configuration

Paths are controlled with environment variables. If they are not set, outputs go
under `data_preprocessing/part2_prism_preprocess/work/`, which is ignored by Git.

```bash
export OROPRECIPBENCH_PRISM_RAW_ROOT=/path/to/prism/raw/ppt/daily
export OROPRECIPBENCH_PRISM_STAGING_ROOT=/path/to/staging
export OROPRECIPBENCH_PRISM_OUTPUT_ROOT=/path/to/dataset_root
```

## Usage

Run from this directory or from the repository root with explicit script paths.
Place the five coordinate files in `$OROPRECIPBENCH_PRISM_STAGING_ROOT/coords/`
before running `step0_prepare.py`.

```bash
python data_preprocessing/part2_prism_preprocess/step0_prepare.py
python data_preprocessing/part2_prism_preprocess/step1_extract_interpolate.py --workers 8
python data_preprocessing/part2_prism_preprocess/step2_assemble_netcdf.py
python data_preprocessing/part2_prism_preprocess/step3_compute_stats.py
python data_preprocessing/part2_prism_preprocess/step4_validate.py
```

For a subset of domains:

```bash
python data_preprocessing/part2_prism_preprocess/step1_extract_interpolate.py --domains Rocky,Cascades
python data_preprocessing/part2_prism_preprocess/step2_assemble_netcdf.py --domains Rocky,Cascades
```

## Outputs

Each domain writes to:

```text
$OROPRECIPBENCH_PRISM_OUTPUT_ROOT/part2-US-<Domain>/
```

Expected files:

- `train_data_out.nc`
- `val_data_out.nc`
- `test_data_out.nc`
- `stats.json`

Validation reports are written to:

```text
$OROPRECIPBENCH_PRISM_STAGING_ROOT/validation_report.json
$OROPRECIPBENCH_PRISM_STAGING_ROOT/validation_report.md
```

## Python Dependencies

The scripts use `numpy`, `pandas`, `xarray`, `netCDF4`, `rasterio`, and `scipy`.
