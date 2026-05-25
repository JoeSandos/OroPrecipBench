# OroPrecipBench

Official repository for **"OroPrecipBench: a km-scale benchmark for spatial precipitation downscaling over complex terrain"**.

OroPrecipBench provides:
- A reproducible **data preprocessing** pipeline to construct benchmark-ready inputs/targets.
- A set of standardized **evaluation metrics** tailored to precipitation downscaling over complex terrain.

> **Baselines**: will be released after the paper is accepted.

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

### 3) Preprocess data (Part 2 / ERA5)

See `data_preprocessing/part2_era5_preprocess/README.md` for the full step-by-step workflow.

---

## Repository Layout

- `metrics/`: metric implementations + CLI runners.
- `data_preprocessing/`: preprocessing pipelines (e.g., ERA5 Part 2 workflow).

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

### Part 2 (ERA5) workflow

A complete step-by-step preprocessing pipeline is provided under:

- `data_preprocessing/part2_era5_preprocess/`

The workflow is organized as `step1-*` through `step6-*` scripts:

1. **Download** missing global hourly ERA5 files (optional; uses `cdsapi`).
2. **Crop & merge** to multiple US regions; merge variables; difference accumulated variables when needed; convert `tp` from meters to `mm/h`.
3. **Check** for missing variables.
4. **Build** Part 2 daily `dataset_root` (daily aggregation: sum for `tp/ssrd/tisr`, mean for other variables; train/val/test splits).
5. **Merge** additional training years (2000–2009) and rewrite `stats.json` (optional).
6. **Interpolate** ERA5 inputs to the Part 1 HRRR 24×24 low-resolution curvilinear grid.

For exact commands and expected directory layouts, follow:

- `data_preprocessing/part2_era5_preprocess/README.md`

### Recommended output layout

After preprocessing, the repository expects a domain-centric `dataset_root` layout, e.g.:

```text
dataset_root/
  part2-US-Rocky/
    train_data_in.nc
    val_data_in.nc
    test_data_in.nc
    stats.json
    status.json
  part2-US-Cascades/
  ...
```

---

## Citation

If you use OroPrecipBench in your research, please cite the corresponding paper:

```bibtex
@article{oroprecipbench,
  title   = {OroPrecipBench: a km-scale benchmark for spatial precipitation downscaling over complex terrain},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO}
}
```

---

## License

See `LICENSE` (if provided).

---

## Contact

Please open a GitHub Issue for questions or bug reports.
