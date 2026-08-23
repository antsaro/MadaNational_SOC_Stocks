# Reproducibility notes

This file documents the exact settings, execution order, and known caveats needed to
reproduce the results in this repository, beyond what's in the README.

## Execution order

The scripts are not meant to be run standalone in arbitrary order. The intended sequence is:

1. **`trees_models_fitting.ipynb`** calls `train_soc_dualgpu.py`, which for each depth
   interval runs the nested cross-validation (10-fold outer x 10 repeats, 4-fold inner)
   with Optuna hyperparameter search (TPE sampler, `MedianPruner`) for both RandomForest
   (cuML, GPU) and XGBoost (`device='cuda'`), fits the final model on the full training
   set with the best parameters found, and writes it to
   `ML_SOC_Results_GPU/results_<depth>/models/<ModelName>.pkl`. RandomForest and XGBoost
   are trained in parallel on separate GPUs via `subprocess` + `CUDA_VISIBLE_DEVICES`
   (see `MODEL_GPU_MAP` near the top of `train_soc_dualgpu.py`).
2. **`rf_bootstrap_fil.py`** loads the RandomForest model produced in step 1
   (`MODEL_PKL` constant near the top of the file), fits `B = 100` bootstrap replicate
   models (each with `random_state = RANDOM_SEED + b`), and runs tiled raster inference
   over the national covariate stack using cuML's ForestInference (FIL) to produce
   per-pixel mean, standard deviation, and coefficient of variation.
3. **`xgb_bootstrap_fil.py`** does the same for XGBoost, but splits GPU inference across
   two devices using `gpu_worker_module.py` as the worker function (one process per GPU,
   tiles distributed through a multiprocessing queue).

`gpu_worker_module.py` is not meant to be run directly — it is imported by
`xgb_bootstrap_fil.py` and spawned once per GPU worker process.

## Fixed seeds

Every stochastic step in the pipeline is seeded, but note the caveat on GPU determinism
below before assuming this makes runs bit-for-bit reproducible.

| Component | Seed / setting |
|---|---|
| Optuna sampler | `TPESampler(seed=42)` |
| Optuna pruner | `MedianPruner(n_warmup_steps=1)` |
| Inner CV folds (hyperparameter search) | `KFold(n_splits=4, shuffle=True, random_state=0)` |
| Outer CV folds (nested CV evaluation) | `KFold(n_splits=10, shuffle=True, random_state=rep * 100 + 42)` for repeat `rep` |
| RandomForest final fit | `random_state=42` |
| XGBoost final fit | `random_state=42` |
| Bootstrap replicate `b` (RF and XGB) | `random_state = 42 + b`, for `b` in `0..99` |

## Known non-determinism

- **cuML RandomForest** does not guarantee bit-identical results across runs even with a
  fixed `random_state`, because tree-building on GPU uses non-deterministic reductions.
  Expect metric-level reproducibility (RMSE/R²/MAE within a small tolerance across runs),
  not identical trees or identical `.pkl` files.
- **XGBoost with `tree_method='hist', device='cuda'`** is deterministic given the same
  data order, seed, and XGBoost/CUDA version, but results can drift across different GPU
  architectures or CUDA versions.
- Bootstrap standard deviation and coefficient-of-variation maps are therefore expected
  to be close to, but not pixel-identical to, the manuscript figures if re-run on
  different hardware or library versions.

## Compute environment used for the original runs

- **Hyperparameter search and training**: Kaggle notebook, free tier, dual NVIDIA T4
  GPUs (16 GB VRAM each), CUDA driver supporting CUDA 12.x.
- **Bootstrap raster inference**: same environment; `rf_bootstrap_fil.py` uses a single
  GPU with CPU-side joblib parallelism for the bootstrap model fitting step, then GPU FIL
  for tiled inference; `xgb_bootstrap_fil.py` splits inference across both GPUs.
- **Covariate preprocessing**: Google Earth Engine (not included in this repository).

## Version pinning

`environment-gpu.yml` pins `cuml=24.10` and `cudatoolkit=12.2` as a reasonable default,
but RAPIDS release cadence and CUDA compatibility change frequently. If you hit an
install or runtime error, check the currently supported combinations at
https://rapids.ai/start.html and adjust the pin before filing an issue against this
repository, since it may be a RAPIDS-side compatibility issue rather than a bug here.

## Data paths

All scripts currently hardcode Kaggle-style input/output paths (`/kaggle/input/...`,
`/kaggle/working/...`) near the top of each file, rather than reading them from a config
file or command-line argument. To run outside Kaggle, edit these path constants directly
(they are grouped under a `CONFIGURATION` comment block near the top of each script)
before running.
