# OroPrecipBench

Official repository for **"OroPrecipBench: a km-scale benchmark for spatial precipitation downscaling over complex terrain"**.

OroPrecipBench provides:
- A reproducible **data preprocessing** pipeline to construct benchmark-ready inputs/targets.
- A set of standardized **evaluation metrics** tailored to precipitation downscaling over complex terrain.

**Baselines**: in the main text, we also evaluate the SOTA baselines and will release the implementation soon.

---

## Quick Start

### 1) Install

```bash
pip install -r requirements.txt
```

### 2) Run metrics on an existing domain

From the repository root:

```bash
python -m metrics.run_metrics \
    --domain-dir /path/to/part1-US-Rocky \
    --pred-path /path/to/model_pred.nc \
    --out-dir outputs/metrics/part1-US-Rocky/model_name \
    --categories conventional spatial extreme terrain \
    --part part1 \
    --pixel-size-km 3.0 \
    --ens-strategy mean \
    --max-ens 8
```

### 3) Preprocess data

- **Part 1 (HRRR→HRRR)**: `data_preprocessing/part1_hrrr-hrrr_preprocess/guide.md`
- **Part 2 inputs (ERA5)**: `data_preprocessing/part2_era5_preprocess/README.md`
- **Part 2 targets (PRISM)**: `data_preprocessing/part2_prism_preprocess/README.md`

---

## Repository Layout

- `metrics/`: metric implementations + CLI runners.
- `data_preprocessing/`: preprocessing pipelines.

---

## Metrics

The `metrics/` package implements four metric groups (see `metrics/README.md` for full details):

- **Conventional**: RMSE, MAE, Bias, Pearson correlation (CC), CRPS, SSIM, POD/FAR.
- **Spatial**: RAPSD, Fractions Skill Score (FSS), correlation length.
- **Extreme**: quantile bias, precipitation intensity-frequency PDF, return-period utilities.
- **Terrain**: terrain-conditioned precipitation–elevation profile distance (PEP-D).

### Expected data format

Metrics assume the benchmark NetCDF policy described in `metrics/README.md`. Typical files are:

- `{domain-dir}/test_data_out.nc` with `tp` ground-truth precipitation.
- `{domain-dir}/train_data_out.nc` with `tp` used to compute climatological thresholds/quantiles (for some extreme metrics).
- `{domain-dir}/test_data_in.nc` low-resolution inputs.
- `{domain-dir}/static_variables.nc` with `elevation`, `lsm`, etc.
- `--pred-path` prediction NetCDF with `tp` and an `ensemble` dimension (deterministic predictions can use `ensemble=1`).

Common dimension aliases (`valid_time`, `sample`, `ens`, `member`, `lat`, `lon`, `x`, `y`) are standardized automatically by the loaders.

### Running metrics

Run all categories at once:

```bash
python -m metrics.run_metrics \
    --domain-dir /path/to/part1-US-Rocky \
    --pred-path /path/to/model_pred.nc \
    --out-dir outputs/metrics/part1-US-Rocky/model_name \
    --categories conventional spatial extreme terrain \
    --part auto
```

Or run a specific category runner:

```bash
# Conventional
python -m metrics.conventional.run_conventional \
    --domain-dir /path/to/part1-US-Rocky \
    --pred-path /path/to/model_pred.nc \
    --out-dir outputs/metrics/conventional/part1-US-Rocky/model_name \
    --metrics rmse mae crps cc ssim pod_far \
    --part auto

# Spatial
python -m metrics.spatial.run_spatial \
    --domain-dir /path/to/part1-US-Rocky \
    --pred-path /path/to/model_pred.nc \
    --out-dir outputs/metrics/spatial/part1-US-Rocky/model_name \
    --metrics rapsd fss correlation_length \
    --part auto \
    --pixel-size-km 3.0 \
    --fss-scales 1 3 5 9 17 33 65

# Extreme
python -m metrics.extreme.run_extreme \
    --domain-dir /path/to/part2-US-Rocky \
    --pred-path /path/to/model_pred.nc \
    --out-dir outputs/metrics/extreme/part2-US-Rocky/model_name \
    --metrics quantile_bias precip_pdf return_period \
    --part auto \
    --quantiles 0.90 0.95 0.99

# Terrain
python -m metrics.terrain.run_terrain \
    --domain-dir /path/to/part1-US-Rocky \
    --pred-path /path/to/model_pred.nc \
    --out-dir outputs/metrics/terrain/part1-US-Rocky/model_name \
    --metrics pep_d \
    --bin-width 200.0
```

### Part presets (hourly vs daily)

All runners support `--part auto`, `--part part1` (hourly), and `--part part2` (daily). Presets control thresholds and binning defaults (e.g., POD/FAR thresholds, PDF parameters). Explicit CLI arguments always override presets.

### Outputs

Each runner writes to `--out-dir`:

```text
{out_dir}/
├── summary.json
├── summary.csv
├── {metric}_detail.pkl
├── run_*.log
└── cache/
```

`summary.json` / `summary.csv` contain scalar values for quick comparison; detail pickle files contain arrays/maps for deeper analysis and plotting.

---

## Data Preprocessing

OroPrecipBench includes **three** reproducible preprocessing workflows covering Part 1 (hourly) and Part 2 (daily). Each workflow produces a domain-centric `dataset_root/` that is directly consumed by metric runners and (future) baselines.

### Part 1 (HRRR → HRRR) hourly benchmark construction

- Location: `data_preprocessing/part1_hrrr-hrrr_preprocess/`
- Main guide: `data_preprocessing/part1_hrrr-hrrr_preprocess/guide.md`

**Goal.** Build the **Part 1 hourly** downscaling benchmark where:
- **Low-resolution inputs** come from HRRR 3 km fields spatially **coarsened to 25 km** (typical shape `24×24`; variables such as `tp`, `t2m`, `u10`, `v10`).
- **High-resolution targets** are HRRR 3 km precipitation `tp` on the native high-res grid (typical shape `192×192`).
- **Static variables** (e.g., `elevation`, `lsm`) are provided at high resolution (`192×192`) and reused across splits.

**Expected output layout (per domain).**

`dataset_root/part1-US-<Domain>/` typically contains:
- `train_data_in.nc`, `val_data_in.nc`, `test_data_in.nc` (low-res predictors)
- `train_data_out.nc`, `val_data_out.nc`, `test_data_out.nc` (high-res precipitation target, `tp`, units `mm/h`)
- `static_variables.nc` (e.g., `elevation`, `lsm`)
- `stats.json` (normalization statistics; precipitation commonly includes `log(1+x)` stats)

**How to run (CLI entrypoints).**

The workflow is implemented as step scripts (see the guide for details):

```bash
python data_preprocessing/part1_hrrr-hrrr_preprocess/step1_build_hrrr-hrrr.py --help
python data_preprocessing/part1_hrrr-hrrr_preprocess/step2_cal_stats.py --help
```

Notes:
- HRRR variables may have missing timestamps; when merging variables, the workflow uses the **intersection of valid times** to keep time alignment consistent.
- Some scripts ship with local default paths; for reproducibility, prefer passing your own data roots via CLI args.

### Part 2 inputs (ERA5) daily dataset construction

- Location: `data_preprocessing/part2_era5_preprocess/`
- Documentation: `data_preprocessing/part2_era5_preprocess/README.md`

**Goal.** Construct **Part 2 daily** *inputs* from ERA5, organized by the same benchmark domains, and optionally interpolate inputs onto the Part 1 low-resolution HRRR curvilinear grid.

**Pipeline summary (step1 → step6).**

1. Download missing global hourly ERA5 files (optional; uses `cdsapi`).
2. Crop global ERA5 to the five US benchmark domains and merge variables; convert `tp` to `mm/h`.
3. Check for missing variables in regional outputs.
4. Build the Part 2 **daily** `dataset_root` (`train/val/test_data_in.nc`) via daily aggregation:
   - sum: `tp`, `ssrd`, `tisr`
   - mean: other variables
   (the corrupted date `2018-08-22` is excluded by default)
5. Optionally prepend older train years (2000–2009) and recompute `stats.json`.
6. Interpolate Part 2 inputs to the **Part 1 24×24** low-resolution HRRR grid and recompute metadata.

**How to run.** Follow the exact commands in the workflow README:

- `data_preprocessing/part2_era5_preprocess/README.md`

### Part 2 targets (PRISM) daily precipitation mapping

- Location: `data_preprocessing/part2_prism_preprocess/`
- Documentation: `data_preprocessing/part2_prism_preprocess/README.md`

**Goal.** Build the **Part 2 PRISM target** dataset by mapping daily PRISM precipitation to each of the five 192×192 benchmark domains, then assembling split-level NetCDF files and computing normalization statistics.

**Inputs.**

- Daily PRISM ZIP archives, organized by year, named like:
  `prism_ppt_us_25m_YYYYMMDD.zip`
- HRRR reference coordinate files:
  `part1-US-<Domain>_lat_lon_192_192.nc` (used as the target grid)

**Default time split.**

- train: 2000-01-01 → 2021-12-31
- val: 2022-01-01 → 2022-12-31
- test: 2023-01-01 → 2024-12-31

The corrupted PRISM date `2018-08-22` is skipped.

**Configuration (environment variables).**

```bash
export OROPRECIPBENCH_PRISM_RAW_ROOT=/path/to/prism/raw/ppt/daily
export OROPRECIPBENCH_PRISM_STAGING_ROOT=/path/to/staging
export OROPRECIPBENCH_PRISM_OUTPUT_ROOT=/path/to/dataset_root
```

**How to run (step0 → step4).**

```bash
python data_preprocessing/part2_prism_preprocess/step0_prepare.py
python data_preprocessing/part2_prism_preprocess/step1_extract_interpolate.py --workers 8
python data_preprocessing/part2_prism_preprocess/step2_assemble_netcdf.py
python data_preprocessing/part2_prism_preprocess/step3_compute_stats.py
python data_preprocessing/part2_prism_preprocess/step4_validate.py
```

**Outputs (per domain).**

`$OROPRECIPBENCH_PRISM_OUTPUT_ROOT/part2-US-<Domain>/`:
- `train_data_out.nc`, `val_data_out.nc`, `test_data_out.nc`
- `stats.json`

Validation reports are written under the staging root.
