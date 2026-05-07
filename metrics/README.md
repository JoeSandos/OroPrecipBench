# Metrics

Evaluation metrics for OroPrecipBench precipitation downscaling experiments over complex terrain.

This package provides four metric groups:

- `conventional`: RMSE, MAE, Bias, Pearson correlation, CRPS, SSIM, POD/FAR.
- `spatial`: RAPSD, Fractions Skill Score, and correlation length.
- `extreme`: Quantile bias, precipitation intensity-frequency PDF, and return-period utilities.
- `terrain`: Terrain-conditioned precipitation-elevation profile distance (PEP-D). WLAB code is kept for future multi-scale terrain-gradient work but is not enabled by default.

## Directory Layout

```text
metrics/
├── io_utils.py
├── run_metrics.py
├── convert_predictions.py
├── conventional/
├── spatial/
├── extreme/
├── terrain/
├── sanity_checks/
└── tests/
```

The package is intentionally initialized with source, documentation, and lightweight test scripts only. Generated outputs, toy NetCDF files, caches, logs, figures, and bytecode are excluded from version control.

## Data Format

Inputs follow the benchmark data policy:

| File | Dimensions | Variable | Purpose |
| --- | --- | --- | --- |
| `test_data_out.nc` | `(time, latitude, longitude)` | `tp` | Ground truth precipitation, in mm/h or mm/day |
| `train_data_out.nc` | `(time, latitude, longitude)` | `tp` | Historical precipitation for climatological quantiles |
| `test_data_in.nc` | `(time, latitude, longitude)` | `tp`, `u10`, `v10`, ... | Low-resolution inputs; wind fields are used by WLAB utilities |
| Prediction `.nc` | `(time, ensemble, latitude, longitude)` | `tp` | Model predictions; `ensemble=1` is deterministic |
| `static_variables.nc` | `(latitude, longitude)` | `elevation`, `lsm` | Static topographic and land-sea variables |

Dimension aliases such as `valid_time`, `sample`, `ens`, `member`, `lat`, `lon`, `y`, and `x` are standardized automatically.

## Running Metrics

Run from the repository root:

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

You can also set `PYTHONPATH` and run from another directory:

```bash
export PYTHONPATH=/path/to/OroPrecipBench
python -m metrics.run_metrics ...
```

### Category Runners

Conventional metrics:

```bash
python -m metrics.conventional.run_conventional \
    --domain-dir /path/to/part1-US-Rocky \
    --pred-path /path/to/model_pred.nc \
    --out-dir outputs/metrics/conventional/part1-US-Rocky/model_name \
    --metrics rmse mae crps cc ssim pod_far \
    --part part1 \
    --ens-strategy mean
```

Spatial metrics:

```bash
python -m metrics.spatial.run_spatial \
    --domain-dir /path/to/part1-US-Rocky \
    --pred-path /path/to/model_pred.nc \
    --out-dir outputs/metrics/spatial/part1-US-Rocky/model_name \
    --metrics rapsd fss correlation_length \
    --part part1 \
    --pixel-size-km 3.0 \
    --fss-scales 1 3 5 9 17 33 65
```

Extreme precipitation metrics:

```bash
python -m metrics.extreme.run_extreme \
    --domain-dir /path/to/part2-US-Rocky \
    --pred-path /path/to/model_pred.nc \
    --out-dir outputs/metrics/extreme/part2-US-Rocky/model_name \
    --metrics quantile_bias precip_pdf return_period \
    --part part2 \
    --quantiles 0.90 0.95 0.99 \
    --ens-strategy per_member
```

Terrain metrics:

```bash
python -m metrics.terrain.run_terrain \
    --domain-dir /path/to/part1-US-Rocky \
    --pred-path /path/to/model_pred.nc \
    --out-dir outputs/metrics/terrain/part1-US-Rocky/model_name \
    --metrics pep_d \
    --bin-width 200.0
```

## Part Presets

All runners support `--part auto`, `--part part1`, and `--part part2`.

`auto` infers the temporal resolution from `test_data_out.nc`:

- Median interval below 6 hours: hourly Part 1 defaults.
- Median interval at least 6 hours: daily Part 2 defaults.

| Parameter | Part 1 hourly | Part 2 daily | Use |
| --- | ---: | ---: | --- |
| `rain_threshold` | `0.01` mm/h | `1.0` mm/day | POD/FAR threshold |
| `fss_threshold` | `0.01` mm/h | `1.0` mm/day | FSS threshold |
| `wet_threshold` | `0.01` mm/h | `1.0` mm/day | PDF wet-event filter |
| `pdf_bins_max` | `100.0` | `200.0` | PDF upper bound |
| `pdf_bins_width` | `1.0` | `2.0` | PDF bin width |

Explicit CLI values override presets.

## Python API

```python
from metrics.io_utils import load_truth, load_train, load_prediction, load_static, align_time
from metrics.conventional.crps import compute_crps
from metrics.spatial.fss import compute_fss
from metrics.extreme.quantile_bias import compute_quantile_bias
from metrics.terrain.pep_d import compute_pep_d

truth, truth_meta = load_truth("/path/to/part1-US-Rocky")
pred, pred_meta = load_prediction("/path/to/pred.nc", max_ens=8)
train, _ = load_train("/path/to/part1-US-Rocky")
pred, truth, _ = align_time(pred, truth, pred_meta["time"], truth_meta["time"])

crps = compute_crps(pred, truth)
fss = compute_fss(pred, truth, scales=[1, 3, 5, 9], threshold=0.5)
qb = compute_quantile_bias(pred, truth, quantiles=[0.90, 0.95, 0.99], climatology=train)

static = load_static("/path/to/part1-US-Rocky")
pepd = compute_pep_d(pred, truth, static["elevation"], bin_width=200.0)
```

## Outputs

Each runner writes to `--out-dir`:

```text
{out_dir}/
├── summary.json
├── summary.csv
├── {metric}_detail.pkl
├── run_*.log
└── cache/
```

`summary.json` and `summary.csv` contain scalar values for quick comparison. Detail pickle files contain arrays and maps needed for deeper analysis and plotting. The cache directory stores intermediate RAPSD, FSS, correlation-length, return-period, and PEP-D results.

These files are generated artifacts and should not be committed.

## Configuration Paths

Scripts that need external data use anonymized defaults and can be configured with environment variables:

- `OROPRECIP_DATA_ROOT`: dataset root, default `data/dataset_root`.
- `OROPRECIP_RUNS_ROOT`: baseline runs root, default `baselines/runs`.
- `OROPRECIP_BILINEAR_ROOT`: bilinear baseline root for sanity checks, default `baselines/runs/grid_part1_hrrr_50m_dem0/statistical/bilinear`.

## Toy Test

Generate synthetic test data:

```bash
python -m metrics.tests.generate_toy_data \
    --out-dir metrics/tests/toy_domain \
    --T-train 200 --T-test 50 --H 32 --W 32 --n-ens 4 --seed 42
```

Run the end-to-end toy workflow:

```bash
python -m metrics.tests.run_toy_test
```

The toy workflow generates temporary NetCDF inputs, runs all metric categories against three toy baselines, checks CLI runner outputs, and reports `ALL TESTS PASSED` on success. Its generated `toy_domain/` and `toy_output/` directories are ignored by Git.

## Metric Summary

| Group | Metrics | Notes |
| --- | --- | --- |
| Conventional | RMSE, MAE, Bias, CC, CRPS, SSIM, POD/FAR | CRPS consumes the full ensemble; most other metrics use the selected ensemble reduction |
| Spatial | RAPSD, FSS, correlation length | RAPSD compares scale spectra; FSS evaluates thresholded spatial fractions; correlation length uses a deterministic single member |
| Extreme | Quantile bias, precipitation PDF, return-period utilities | Quantile bias uses climatological thresholds from training data when available |
| Terrain | PEP-D | Compares normalized precipitation-elevation profiles over land pixels |

## References

- Aich et al. (2026), GMD.
- Coles (2001), *An Introduction to Statistical Modeling of Extreme Values*.
- Glawion et al. (2025), npj Climate and Atmospheric Science.
- Gneiting and Raftery (2007), JASA.
- Harris et al. (2022), JAMES.
- Härdle and Marron (1990), Annals of Statistics.
- Mehran et al. (2014), JGR Atmospheres.
- Napoli et al. (2019), Scientific Reports.
- Roberts and Lean (2008), MWR.
- Sinclair and Pegram (2005), JHM.
- Wang et al. (2004), IEEE TIP.
- Wilks (2011), *Statistical Methods in the Atmospheric Sciences*.

