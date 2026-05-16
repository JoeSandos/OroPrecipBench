
# Downscaling Models

This repository provides three downscaling models (Regression / Diffusion / Residual Diffusion) along with training and inference scripts. Configuration is managed with Hydra (see `conf/`).

## Models (3)

### 1) Regression (UNet Regression)

- **Goal**: learn a deterministic mapping from low-resolution inputs (upsampled + static invariants) to high-resolution outputs.
- **Training entry**: `train_regression.py`
- **Inference entry**: `generate_regression.py`
- **Configs**: `conf/regression/`
- **Output**: a single prediction (saved with `ensemble=1`).

### 2) Diffusion (EDM Conditional Diffusion Super-Resolution)

- **Goal**: conditional diffusion generation of high-resolution outputs; can generate multiple ensemble members per sample.
- **Training entry**: `train_diffusion.py`
- **Inference entry**: `generate_diffusion.py`
- **Configs**: `conf/diffusion/`
- **Output**: multi-ensemble predictions (`ensemble=num_ensembles`).

### 3) Residual Diffusion / CorrDiff (Regression + Residual Diffusion)

- **Goal**: first predict a mean field with a regression model, then generate a residual with diffusion and add them together; improves stochasticity and extreme-event representation.
- **Training entry**: `train_residual_diffusion.py` (requires a regression checkpoint)
- **Inference entry**: `generate_residual_diffusion.py` (requires both regression and residual-diffusion checkpoints)
- **Configs**: `conf/residual_diffusion/`
- **Output**: multi-ensemble predictions (`ensemble=num_ensembles`).

## Installation

It is recommended to use a Python virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

OroPrecipBench:

https://www.modelscope.cn/datasets/OroPrecipBench/OroPrecipBench

## Training

All training runs are launched via Hydra configs under `conf/*/config_training_*.yaml`, where `*` denotes the model type (e.g., `diffusion`, `regression`, `residual_diffusion`). You can edit YAML files directly or override specific fields from the command line (e.g., checkpoint paths).

Below are example commands for `part1_US_Rocky`.

### Train Regression

```bash
torchrun --standalone --nproc_per_node=1 train_regression.py --config-name=config_training_part1_US_Rocky.yaml
```

### Train Diffusion

```bash
torchrun --standalone --nproc_per_node=1 train_diffusion.py --config-name=config_training_part1_US_Rocky.yaml
```

### Train Residual Diffusion (requires a regression checkpoint)

1) Train Regression first and locate `network-snapshot-*.pkl` in the output directory.
2) Train Residual Diffusion and pass `regression_checkpoint_path`:

```bash
torchrun --standalone --nproc_per_node=1 train_residual_diffusion.py \
  --config-name=config_training_part1_US_Rocky.yaml \
  regression_checkpoint_path="<path/to/regression/network-snapshot-XXXXXX.pkl>"
```

## Inference / Generation

For inference, it is recommended to use `conf/*/config_generate_*.yaml` (these include fields like `generation.checkpoint_dir` and output paths). Predictions are saved as NetCDF (`.nc`). By default, only the prediction group is saved (inputs/ground truth can be obtained from the original dataset).

### Regression inference

```bash
python generate_regression.py \
  --config-name=config_generate_part1_US_Rocky.yaml \
  generation.checkpoint_dir="<path/to/regression/network-snapshot-XXXXXX.pkl>"
```

The output path is controlled by `generation.output_path`. The script automatically organizes outputs by dataset name, input filename, and checkpoint directory name.

### Diffusion inference (ensemble generation)

```bash
torchrun --standalone --nproc_per_node=1 generate_diffusion.py \
  --config-name=config_generate_part1_US_Rocky.yaml \
  generation.checkpoint_dir="<path/to/diffusion/network-snapshot-XXXXXX.pkl>"
```

Common inference parameters:

- `generation.num_ensembles`: number of ensemble members per sample
- `generation.seed_batch_size`: number of seeds generated per batch (recommended: `nproc_per_node * seed_batch_size = num_ensembles`)
- `generation.sampler_type`: `deterministic` or `stochastic`

### Residual Diffusion inference (Regression + Residual Diffusion)

```bash
torchrun --standalone --nproc_per_node=1 generate_residual_diffusion.py \
  --config-name=config_generate_part1_US_Rocky.yaml \
  generation.regression_checkpoint_path="<path/to/regression/network-snapshot-XXXXXX.pkl>" \
  generation.residual_checkpoint_path="<path/to/residual_diffusion/network-snapshot-YYYYYY.pkl>"
```

## FAQ

- **Cannot find `dataset_root/...` files**: ensure your working directory is the repository root, or override `dataset.*_path` with absolute paths.
- **SwanLab**: training scripts initialize `swanlab` logging (saved under `outputs/swanlog/`). If you do not use SwanLab, please install/login first or configure it for offline mode in your environment.

