import os
import sys
import gc
import json
import time
import csv
import math
import warnings
import pickle
import numpy as np
import pandas as pd
import joblib
import rasterio
from rasterio.windows import Window
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed
import multiprocessing as mp
import cupy as cp
import cuml
from cuml.fil import ForestInference
from tqdm import tqdm

warnings.filterwarnings('ignore')
print(f"cuML version: {cuml.__version__}")

# ===============================================================================
#  CONFIGURATION
# ===============================================================================

TRAIN_CSV  = '/kaggle/working/DeepSOC_predictors_table_final.csv'
OUTPUT_DIR = '/kaggle/working/DeepSOC_RF'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -- Reference RandomForest model produced by the dual-GPU nested-CV pipeline --
MODEL_PKL = '/kaggle/working/ML_SOC_Results_GPU/results_DeepSOC30_100/models/RandomForest.pkl'

# -- Bootstrap -----------------------------------------------------------------
B            = 100
RANDOM_SEED  = 42
N_JOBS_TRAIN = -1

# -- GPU -----------------------------------------------------------------------
GPU_ID         = 0
FIL_BATCH_SIZE = 100_000

# -- FIL compilation batch: only BATCH_B models live on the GPU at once --------
BATCH_B = 10

# -- Tiling --------------------------------------------------------------------
TILE_SIZE  = 3840
NODATA_OUT = -9999.0

RESPONSE_VARIABLE      = 'DeepSOC30_100'
SOC_UPPER_LIMIT        = 1000
NUMERIC_PREDICTORS     = ['NDVI', 'NDWI', 'NIRI', 'elevation', 'slope',
                           'mat', 'map', 'treecover']
CATEGORICAL_PREDICTORS = ['lulc', 'soil_type']
MAT_DIVIDE_BY_10       = True

RASTERS = {
    'NDVI'      : '/kaggle/input/datasets/antsasarobidyran/merged-covariates-geotiff-v2/NDVI.tif',
    'NDWI'      : '/kaggle/input/datasets/antsasarobidyran/merged-covariates-geotiff-v2/NDWI.tif',
    'NIRI'      : '/kaggle/input/datasets/antsasarobidyran/merged-covariates-geotiff-v2/NIRI.tif',
    'elevation' : '/kaggle/input/datasets/antsasarobidyran/merged-covariates-geotiff-v6/ELEV_ALOS.tif',
    'lulc'      : '/kaggle/input/datasets/antsasarobidyran/merged-covariates-geotiff-v3/lulc.tif',
    'map'       : '/kaggle/input/datasets/antsasarobidyran/merged-covariates-geotiff-v3/map_corrected.tif',
    'mat'       : '/kaggle/input/datasets/antsasarobidyran/merged-covariates-geotiff-v3/mat_corrected.tif',
    'slope'     : '/kaggle/input/datasets/antsasarobidyran/merged-covariates-geotiff-v6/SLOPE_ALOS.tif',
    'soil_type' : '/kaggle/input/datasets/antsasarobidyran/merged-covariates-geotiff-v3/soil_type.tif',
    'treecover' : '/kaggle/input/datasets/antsasarobidyran/merged-covariates-geotiff-v3/treecover.tif',
}

BOOTSTRAP_PKL = os.path.join(OUTPUT_DIR, 'bootstrap_models.pkl')

cp.cuda.Device(GPU_ID).use()

_SKLEARN_RF_KEYS = {
    'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf',
    'max_features', 'max_leaf_nodes', 'min_impurity_decrease',
    'bootstrap', 'max_samples', 'ccp_alpha', 'random_state',
}


def load_reference_model_bundle(model_pkl_path):
    """Unpickle the real fitted RandomForest bundle from the dual-GPU
    nested-CV pipeline and pull out sklearn-compatible RF hyperparameters
    plus the feature list.
    """
    print(f'\nLoading reference model bundle -> {model_pkl_path}')
    bundle   = joblib.load(model_pkl_path)
    model    = bundle['model']
    features = bundle['features']

    raw_params = model.get_params()
    rf_params  = {k: v for k, v in raw_params.items() if k in _SKLEARN_RF_KEYS}
    dropped    = sorted(set(raw_params) - set(rf_params))
    if dropped:
        print(f'   Ignoring non-sklearn/cuML-only keys: {dropped}')

    # Bootstrap replicates each set their own random_state (seed + b) in
    # _train_one, so drop it here to avoid confusion — it gets overridden
    # regardless.
    rf_params.pop('random_state', None)

    print(f'   Features     ({len(features)}): {features}')
    print(f'   n_estimators : {getattr(model, "n_estimators", rf_params.get("n_estimators"))}')
    print(f'   RF hyperparams (sklearn-compatible) : {rf_params}')
    return rf_params, features


# ===============================================================================
#  PHASE 1 -- BOOTSTRAP TRAINING  (CPU, joblib Parallel)
# ===============================================================================

def _train_one(b, X_full, y_full, rf_params, n_samples, seed):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    rng      = np.random.RandomState(seed + b)
    boot_idx = rng.randint(0, n_samples, size=n_samples)

    oob_mask                      = np.ones(n_samples, dtype=bool)
    oob_mask[np.unique(boot_idx)] = False
    oob_idx                       = np.where(oob_mask)[0]

    X_boot = X_full[boot_idx]
    y_boot = y_full[boot_idx]

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_boot)

    params = {**rf_params, 'random_state': seed + b, 'n_jobs': 1}
    mdl = RandomForestRegressor(**params)
    mdl.fit(X_sc, y_boot)

    oob_r2 = oob_rmse = np.nan
    if len(oob_idx) > 1:
        X_oob_sc = scaler.transform(X_full[oob_idx])
        p_oob    = mdl.predict(X_oob_sc)
        y_oob    = y_full[oob_idx]
        ss_res   = float(np.sum((y_oob - p_oob) ** 2))
        ss_tot   = float(np.sum((y_oob - y_oob.mean()) ** 2))
        oob_r2   = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        oob_rmse = float(np.sqrt(np.mean((y_oob - p_oob) ** 2)))

    return dict(
        model_bytes = pickle.dumps(mdl),
        sc_mean     = scaler.mean_.astype(np.float32),
        sc_scale    = scaler.scale_.astype(np.float32),
        oob_r2      = float(oob_r2),
        oob_rmse    = float(oob_rmse),
    )


def phase1_train(features, rf_params):
    print('\n' + '='*70)
    print('  PHASE 1 -- Bootstrap Training (CPU, joblib)')
    print('='*70)

    print('Loading training CSV...')
    df = pd.read_csv(TRAIN_CSV)
    df.replace(-9999, np.nan, inplace=True)
    df = df.dropna(subset=[RESPONSE_VARIABLE])
    if SOC_UPPER_LIMIT:
        df = df[df[RESPONSE_VARIABLE] < SOC_UPPER_LIMIT].copy()

    for col in NUMERIC_PREDICTORS:
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    cat_classes = {}
    for col in CATEGORICAL_PREDICTORS:
        df[col] = df[col].astype(str).fillna('Unknown')
        le = LabelEncoder()
        le.fit(df[col])
        cat_classes[col] = le.classes_
        df[f'{col}_enc'] = le.transform(df[col])

    X_full    = df[features].values.astype(np.float32)
    y_full    = df[RESPONSE_VARIABLE].values.astype(np.float32)
    n_samples = len(y_full)
    print(f'   Samples : {n_samples:,}')
    print(f'   DeepSOC : {y_full.min():.1f} - {y_full.max():.1f} Mg C/ha')
    print(f'\nTraining {B} replicates (n_jobs={N_JOBS_TRAIN})...')
    t0 = time.time()

    results = Parallel(n_jobs=N_JOBS_TRAIN, verbose=5, backend='loky')(
        delayed(_train_one)(b, X_full, y_full, rf_params, n_samples, RANDOM_SEED)
        for b in range(B)
    )

    oob_r2s  = np.array([r['oob_r2']   for r in results])
    oob_rmse = np.array([r['oob_rmse'] for r in results])
    print(f'\nTraining complete in {(time.time()-t0)/60:.1f} min')
    print(f'   OOB R2   : {np.nanmean(oob_r2s):.4f} +/- {np.nanstd(oob_r2s):.4f}')
    print(f'   OOB RMSE : {np.nanmean(oob_rmse):.3f} +/- {np.nanstd(oob_rmse):.3f} Mg C/ha')

    oob_csv = os.path.join(OUTPUT_DIR, 'bootstrap_oob_diagnostics.csv')
    with open(oob_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['replicate', 'oob_r2', 'oob_rmse_MgCha'])
        for b in range(B):
            w.writerow([b, f'{oob_r2s[b]:.6f}', f'{oob_rmse[b]:.6f}'])
    print(f'   OOB CSV -> {oob_csv}')

    print(f'\nSaving {B} bootstrap models -> {BOOTSTRAP_PKL}')
    joblib.dump(
        {'models': results, 'features': features, 'cat_classes': cat_classes},
        BOOTSTRAP_PKL, compress=3,
    )
    print(f'   Saved: {os.path.getsize(BOOTSTRAP_PKL)/1e6:.1f} MB')
    return results, cat_classes


# ===============================================================================
#  PHASE 2 -- FIL GPU TILE PREDICTION
# ===============================================================================

def _build_cat_lut(classes_array):
    int_vals, enc_vals = [], []
    for idx, cls in enumerate(classes_array):
        try:
            int_vals.append(int(cls))
            enc_vals.append(idx)
        except (ValueError, TypeError):
            pass
    if not int_vals:
        return np.zeros(1, dtype=np.int32)
    lut = np.zeros(max(int_vals) + 1, dtype=np.int32)
    for iv, ev in zip(int_vals, enc_vals):
        if iv >= 0:
            lut[iv] = ev
    return lut


def _fil_predict(fil_model, X_gpu):
    raw = fil_model.predict(X_gpu)
    if hasattr(raw, 'values'):
        raw = raw.values
    if isinstance(raw, np.ndarray):
        raw = cp.asarray(raw)
    return raw.ravel().astype(cp.float32)


def phase2_predict(features, cat_classes):
    print('\n' + '='*70)
    print('  PHASE 2 -- FIL GPU Tile Prediction  [batch-outer / tile-inner]')
    print('='*70)

    print(f'Loading bootstrap bundle from {BOOTSTRAP_PKL} ...')
    saved  = joblib.load(BOOTSTRAP_PKL)
    models = saved['models']
    B_eff  = len(models)
    print(f'   {B_eff} replicates in bundle.')

    predict_indices = list(range(B_eff))
    N_eff           = B_eff
    print(f'   Using ALL {B_eff} bootstrap models for prediction.')

    idx_csv = os.path.join(OUTPUT_DIR, 'predict_model_indices.csv')
    with open(idx_csv, 'w', newline='') as f:
        csv.writer(f).writerow(['selected_bootstrap_index'] + predict_indices)
    print(f'   Selected indices -> {idx_csv}')

    n_batches = math.ceil(N_eff / BATCH_B)
    print(f'   GPU batch size : {BATCH_B}  ->  {n_batches} compilation batches')

    cat_luts_np = {
        c: _build_cat_lut(cat_classes[c])
        for c in CATEGORICAL_PREDICTORS
    }

    src_nodata = {}
    with rasterio.open(list(RASTERS.values())[0]) as ref:
        profile = ref.profile.copy()
        H, W    = ref.height, ref.width
        crs     = ref.crs
    for name, path in RASTERS.items():
        with rasterio.open(path) as src:
            src_nodata[name] = src.nodata

    row_starts = list(range(0, H, TILE_SIZE))
    col_starts = list(range(0, W, TILE_SIZE))
    n_tiles    = len(row_starts) * len(col_starts)
    n_cols     = len(col_starts)
    tiles = [
        (ti * n_cols + tj + 1, row_starts[ti], col_starts[tj])
        for ti in range(len(row_starts))
        for tj in range(len(col_starts))
    ]
    print(f'\nGrid {H}x{W}  |  {n_tiles} tiles ({TILE_SIZE}px)  |  CRS: {crs}')

    raster_handles = {name: rasterio.open(path) for name, path in RASTERS.items()}

    def _build_X_cpu(rs, cs):
        re  = min(rs + TILE_SIZE, H);  ce  = min(cs + TILE_SIZE, W)
        th  = re - rs;                 tw  = ce - cs
        win = Window(cs, rs, tw, th)

        bands = {}
        valid = np.ones((th, tw), dtype=bool)
        for name, src in raster_handles.items():
            arr = src.read(1, window=win).astype(np.float32)
            if arr.shape != (th, tw):
                tmp = np.full((th, tw), np.nan, np.float32)
                tmp[:arr.shape[0], :arr.shape[1]] = arr
                arr = tmp
            nd  = src_nodata[name]
            msk = ~np.isfinite(arr)
            if nd is not None:
                msk |= np.isclose(arr, nd)
            if name in ('mat', 'map'):
                msk |= (arr == 0)
            valid      &= ~msk
            bands[name] = arr

        n_v = int(valid.sum())
        if n_v == 0:
            return None, None, th, tw, win

        flat = np.where(valid.ravel())[0].astype(np.int32)

        num = []
        for f in NUMERIC_PREDICTORS:
            col = bands[f].ravel()[flat].copy()
            if f == 'mat' and MAT_DIVIDE_BY_10:
                col /= 10.0
            num.append(col)

        cat = []
        for f in CATEGORICAL_PREDICTORS:
            raw_int = bands[f].ravel()[flat].astype(np.int64)
            lut     = cat_luts_np[f]
            clipped = np.clip(raw_int, 0, len(lut) - 1)
            cat.append(lut[clipped].astype(np.float32))

        bands.clear()
        X = np.column_stack(num + cat).astype(np.float32)
        return X, flat, th, tw, win

    def _gpu_predict(fil, X_cpu, sc_mean, sc_scale):
        X_sc = (X_cpu - sc_mean) / sc_scale
        n_v  = X_cpu.shape[0]
        out  = np.empty(n_v, dtype=np.float32)
        for p0 in range(0, n_v, FIL_BATCH_SIZE):
            p1         = min(p0 + FIL_BATCH_SIZE, n_v)
            X_gpu      = cp.asarray(X_sc[p0:p1])
            out[p0:p1] = cp.asnumpy(_fil_predict(fil, X_gpu))
            del X_gpu
            cp.get_default_memory_pool().free_all_blocks()
        del X_sc
        return out

    tile_accums: dict = {}
    tile_meta:   dict = {}

    tracking_csv = os.path.join(OUTPUT_DIR, 'tile_tracking.csv')
    tracking_fields = [
        'batch_idx', 'batch_global_models',
        'tile_idx', 'row_start', 'col_start',
        'n_valid_pixels', 'models_accumulated',
        'batch_elapsed_s', 'cumulative_elapsed_min',
        'status',
    ]
    tracking_file   = open(tracking_csv, 'w', newline='')
    tracking_writer = csv.DictWriter(tracking_file, fieldnames=tracking_fields)
    tracking_writer.writeheader()
    tracking_file.flush()
    print(f'Tile tracking log -> {tracking_csv}')

    out_prof = profile.copy()
    out_prof.update(
        dtype='float32', count=1, nodata=NODATA_OUT,
        compress='lzw', predictor=2,
        tiled=True, blockxsize=512, blockysize=512, bigtiff='YES',
    )
    PATH_MEAN = os.path.join(OUTPUT_DIR, f'DeepSOC_mean_n{N_eff}.tif')
    PATH_STD  = os.path.join(OUTPUT_DIR, f'DeepSOC_std_n{N_eff}.tif')

    t0 = time.time()

    for batch_idx in range(n_batches):
        local_start   = batch_idx * BATCH_B
        local_end     = min(local_start + BATCH_B, N_eff)
        batch_locals  = list(range(local_start, local_end))
        batch_globals = [predict_indices[li] for li in batch_locals]

        print(f'\nCompiling FIL batch {batch_idx+1}/{n_batches} '
              f'(global model ids {batch_globals}) ...', flush=True)
        t_c = time.time()

        fil_batch   = []
        mean_batch  = []
        scale_batch = []
        for g_idx in batch_globals:
            sk = pickle.loads(models[g_idx]['model_bytes'])
            fil_batch.append(ForestInference.load_from_sklearn(sk))
            mean_batch.append(models[g_idx]['sc_mean'].astype(np.float32))
            scale_batch.append(models[g_idx]['sc_scale'].astype(np.float32))
            del sk
        print(f'   Compiled {len(fil_batch)} models in {time.time()-t_c:.1f}s',
              flush=True)

        pbar = tqdm(
            tiles,
            desc=f'Batch {batch_idx+1}/{n_batches}',
            unit='tile', ncols=95,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
        )

        for tidx, rs, cs in pbar:
            t_tile = time.time()

            try:
                X_cpu, flat, th, tw, win = _build_X_cpu(rs, cs)
                read_ok = True
            except Exception as e:
                tqdm.write(f'  ERROR tile {tidx} read: {e}')
                X_cpu   = None
                read_ok = False

            if X_cpu is None:
                tracking_writer.writerow(dict(
                    batch_idx            = batch_idx,
                    batch_global_models  = str(batch_globals),
                    tile_idx             = tidx,
                    row_start            = rs,
                    col_start            = cs,
                    n_valid_pixels       = 0,
                    models_accumulated   = tile_accums[tidx]['k'] if tidx in tile_accums else 0,
                    batch_elapsed_s      = f'{time.time()-t_tile:.2f}',
                    cumulative_elapsed_min = f'{(time.time()-t0)/60:.2f}',
                    status               = 'nodata' if read_ok else 'error',
                ))
                tracking_file.flush()
                continue

            n_v = X_cpu.shape[0]

            if tidx not in tile_accums:
                tile_accums[tidx] = {
                    'mean': np.zeros(n_v, np.float32),
                    'm2':   np.zeros(n_v, np.float32),
                    'k':    0,
                }
                tile_meta[tidx] = {'flat': flat, 'th': th, 'tw': tw, 'win': win}

            mean_v = tile_accums[tidx]['mean']
            m2_v   = tile_accums[tidx]['m2']
            k      = tile_accums[tidx]['k']

            for j in range(len(batch_globals)):
                try:
                    preds = _gpu_predict(fil_batch[j], X_cpu,
                                         mean_batch[j], scale_batch[j])
                except Exception as e:
                    tqdm.write(
                        f'  ERROR GPU tile {tidx} model {batch_globals[j]}: {e}')
                    continue

                k      += 1
                delta   = preds - mean_v
                mean_v  = mean_v + delta / np.float32(k)
                m2_v    = m2_v   + delta * (preds - mean_v)
                del preds

            tile_accums[tidx]['mean'] = mean_v
            tile_accums[tidx]['m2']   = m2_v
            tile_accums[tidx]['k']    = k

            tracking_writer.writerow(dict(
                batch_idx              = batch_idx,
                batch_global_models    = str(batch_globals),
                tile_idx               = tidx,
                row_start              = rs,
                col_start              = cs,
                n_valid_pixels         = n_v,
                models_accumulated     = k,
                batch_elapsed_s        = f'{time.time()-t_tile:.2f}',
                cumulative_elapsed_min = f'{(time.time()-t0)/60:.2f}',
                status                 = 'ok',
            ))
            tracking_file.flush()

            del X_cpu
            gc.collect()

        pbar.close()

        for fil in fil_batch:
            del fil
        del fil_batch, mean_batch, scale_batch
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

        elapsed = time.time() - t0
        eta     = elapsed / (batch_idx + 1) * (n_batches - batch_idx - 1)
        print(f'   Batch {batch_idx+1}/{n_batches} done | '
              f'elapsed {elapsed/60:.1f} min | ETA {eta/60:.1f} min')

    tracking_file.close()

    for h in raster_handles.values():
        h.close()
    cp.get_default_memory_pool().free_all_blocks()

    print('\nWriting output rasters ...')
    with rasterio.open(PATH_MEAN, 'w', **out_prof) as h_mean, \
         rasterio.open(PATH_STD,  'w', **out_prof) as h_std:

        for tidx, rs, cs in tqdm(tiles, desc='Writing tiles', unit='tile',
                                  ncols=80):
            re  = min(rs + TILE_SIZE, H);  ce = min(cs + TILE_SIZE, W)
            th  = re - rs;                 tw = ce - cs
            win = Window(cs, rs, tw, th)

            mt = np.full((th, tw), NODATA_OUT, np.float32)
            st = np.full((th, tw), NODATA_OUT, np.float32)

            if tidx in tile_accums:
                flat     = tile_meta[tidx]['flat']
                r2, c2   = np.unravel_index(flat, (th, tw))
                k_used   = tile_accums[tidx]['k']
                variance = tile_accums[tidx]['m2'] / max(k_used - 1, 1)
                mt[r2, c2] = tile_accums[tidx]['mean']
                st[r2, c2] = np.sqrt(np.maximum(variance, 0.0))

            h_mean.write(mt, 1, window=win)
            h_std .write(st, 1, window=win)

    tile_accums.clear()
    tile_meta.clear()
    cp.get_default_memory_pool().free_all_blocks()

    elapsed = time.time() - t0
    print(f'\nPrediction complete in {elapsed/60:.1f} min')
    print(f'   Models used : {N_eff} / {B_eff}')
    print(f'   -> {PATH_MEAN}')
    print(f'   -> {PATH_STD}')
    print(f'   -> {tracking_csv}')


# ===============================================================================
#  MAIN
# ===============================================================================

def main():
    print('='*70)
    print('  RF Bootstrap Uncertainty Map -- cuML FIL / Single GPU')
    print(f'  B={B} (all used for prediction)  BATCH_B={BATCH_B}')
    print('='*70)

    rf_params, features = load_reference_model_bundle(MODEL_PKL)

    if os.path.exists(BOOTSTRAP_PKL):
        print(f'\nBootstrap models already exist -> {BOOTSTRAP_PKL}')
        print('   Delete this file to force retraining.')
        saved       = joblib.load(BOOTSTRAP_PKL)
        cat_classes = saved['cat_classes']
    else:
        _, cat_classes = phase1_train(features, rf_params)

    phase2_predict(features, cat_classes)

    N_eff = B
    print('\n' + '='*70)
    print('  All outputs written:')
    for name in (f'DeepSOC_mean.tif', f'DeepSOC_std.tif',
                 'bootstrap_oob_diagnostics.csv', 'bootstrap_models.pkl',
                 'predict_model_indices.csv', 'tile_tracking.csv'):
        p = os.path.join(OUTPUT_DIR, name)
        if os.path.exists(p):
            print(f'   {name:<45s}  {os.path.getsize(p)/1e6:7.1f} MB')
    print('='*70)


if __name__ == '__main__':
    main()
