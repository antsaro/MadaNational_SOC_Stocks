import os
import sys
import gc
import time
import csv
import warnings
import threading
import numpy as np
import pandas as pd
import joblib
import rasterio
from rasterio.windows import Window
from sklearn.preprocessing import StandardScaler, LabelEncoder
from joblib import Parallel, delayed
import xgboost as xgb
import multiprocessing as mp
from multiprocessing import Process, Queue
from tqdm import tqdm

warnings.filterwarnings('ignore')

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

sys.path.insert(0, '/kaggle/working')
from gpu_worker_module import gpu_worker 


TRAIN_CSV = '/kaggle/working/DeepSOC_predictors_table_final.csv'

DUALGPU_OUTPUT_BASE_DIR = '/kaggle/working/ML_SOC_Results_GPU'
RESPONSE_VARIABLE_TAG   = 'DeepSOC30_100'
MODEL_PKL = os.path.join(
    DUALGPU_OUTPUT_BASE_DIR, f'results_{RESPONSE_VARIABLE_TAG}', 'models', 'XGBoost.pkl'
)

OUTPUT_DIR = '/kaggle/working/DeepSOC_XGB_DualGPU_Bootstrap'

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

# ── Bootstrap ──────────────────────────────────────────────────────────────────
B            = 100
RANDOM_SEED  = 42
N_JOBS_TRAIN = -1

# ── GPU ────────────────────────────────────────────────────────────────────────
GPU_IDS       = [0, 1]
QUEUE_MAXSIZE = 3

# ── Tiling ─────────────────────────────────────────────────────────────────────
TILE_SIZE  = 2048
NODATA_OUT = -9999.0

RESPONSE_VARIABLE      = 'DeepSOC30_100'
SOC_UPPER_LIMIT        = 1000
NUMERIC_PREDICTORS     = ['NDVI', 'NDWI', 'NIRI', 'elevation', 'slope',
                           'mat', 'map', 'treecover']
CATEGORICAL_PREDICTORS = ['lulc', 'soil_type']
MAT_DIVIDE_BY_10       = False

BOOTSTRAP_PKL = os.path.join(OUTPUT_DIR, 'bootstrap_models.pkl')

def _train_one(b, X_full, y_full, xgb_params, n_samples, seed):
    rng      = np.random.RandomState(seed + b)
    boot_idx = rng.randint(0, n_samples, size=n_samples)

    oob_mask                       = np.ones(n_samples, dtype=bool)
    oob_mask[np.unique(boot_idx)]  = False
    oob_idx                        = np.where(oob_mask)[0]

    X_boot = X_full[boot_idx]
    y_boot = y_full[boot_idx]

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_boot)

    params = {
        **xgb_params,
        'random_state': seed + b,
        'n_jobs'      : 1,
        'device'      : 'cpu',
    }
    mdl = xgb.XGBRegressor(**params)
    mdl.fit(X_sc, y_boot, verbose=False)

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
        booster_bytes = bytes(mdl.get_booster().save_raw('ubj')),
        sc_mean       = scaler.mean_.astype(np.float32),
        sc_scale      = scaler.scale_.astype(np.float32),
        oob_r2        = float(oob_r2),
        oob_rmse      = float(oob_rmse),
    )


def phase1_train(features, xgb_params):
    print('\n' + '='*70)
    print('  PHASE 1 — Bootstrap Training (CPU, joblib)')
    print('  Hyperparameters sourced from dual-GPU model:')
    print(f'    {MODEL_PKL}')
    print('='*70)

    print('📂 Loading training CSV...')
    df = pd.read_csv(TRAIN_CSV)
    df.replace(-9999, np.nan, inplace=True)
    df = df.dropna(subset=[RESPONSE_VARIABLE])
    if SOC_UPPER_LIMIT:
        df = df[df[RESPONSE_VARIABLE] < SOC_UPPER_LIMIT].copy()

    for col in NUMERIC_PREDICTORS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

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
    print(f'   SOC     : {y_full.min():.1f} – {y_full.max():.1f} Mg C/ha')
    print(f'\n🔁 Training {B} replicates (n_jobs={N_JOBS_TRAIN})...')
    t0 = time.time()

    results = Parallel(n_jobs=N_JOBS_TRAIN, verbose=5, backend='loky')(
        delayed(_train_one)(b, X_full, y_full, xgb_params, n_samples, RANDOM_SEED)
        for b in range(B)
    )

    oob_r2s  = np.array([r['oob_r2']   for r in results])
    oob_rmse = np.array([r['oob_rmse'] for r in results])
    print(f'\n✅ Training complete in {(time.time()-t0)/60:.1f} min')
    print(f'   OOB R²   : {np.nanmean(oob_r2s):.4f} ± {np.nanstd(oob_r2s):.4f}')
    print(f'   OOB RMSE : {np.nanmean(oob_rmse):.3f} ± {np.nanstd(oob_rmse):.3f} Mg C/ha')

    oob_csv = os.path.join(OUTPUT_DIR, 'bootstrap_oob_diagnostics.csv')
    with open(oob_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['replicate', 'oob_r2', 'oob_rmse_MgCha'])
        for b in range(B):
            w.writerow([b, f'{oob_r2s[b]:.6f}', f'{oob_rmse[b]:.6f}'])
    print(f'   OOB CSV  → {oob_csv}')

    print(f'\n💾 Saving {B} bootstrap models → {BOOTSTRAP_PKL}')
    joblib.dump(
        {'models': results, 'features': features, 'cat_classes': cat_classes},
        BOOTSTRAP_PKL, compress=3,
    )
    print(f'   Saved: {os.path.getsize(BOOTSTRAP_PKL)/1e6:.1f} MB')

    return results, cat_classes

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


def phase2_predict(features, cat_classes):
    print('\n' + '='*70)
    print('  PHASE 2 — GPU Tile Prediction')
    print('='*70)

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
    print(f'📐 Grid {H}×{W}  |  {n_tiles} tiles ({TILE_SIZE}px)  |  '
          f'{len(GPU_IDS)} GPUs  |  CRS: {crs}')
    print(f'   B={B} replicates  |  QUEUE_MAXSIZE={QUEUE_MAXSIZE}\n')

    cat_luts = {c: _build_cat_lut(cat_classes[c]) for c in CATEGORICAL_PREDICTORS}

    raster_handles = {name: rasterio.open(path) for name, path in RASTERS.items()}

    out_prof = profile.copy()
    out_prof.update(
        dtype='float32', count=1, nodata=NODATA_OUT,
        compress='lzw', predictor=2,
        tiled=True, blockxsize=512, blockysize=512, bigtiff='YES',
    )
    PATH_MEAN = os.path.join(OUTPUT_DIR, 'SOC_mean.tif')
    PATH_STD  = os.path.join(OUTPUT_DIR, 'SOC_std.tif')
    h_mean    = rasterio.open(PATH_MEAN, 'w', **out_prof)
    h_std     = rasterio.open(PATH_STD,  'w', **out_prof)
    wlock     = threading.Lock()
    print('✅ Output rasters created:')
    print(f'   {PATH_MEAN}')
    print(f'   {PATH_STD}\n')

    def _build_X(rs, cs):
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
            valid   &= ~msk
            bands[name] = arr

        n_v = int(valid.sum())
        if n_v == 0:
            return None, None, th, tw, win

        flat = np.where(valid.ravel())[0]

        num = []
        for f in NUMERIC_PREDICTORS:
            col = bands[f].ravel()[flat].copy()
            if f == 'mat' and MAT_DIVIDE_BY_10:
                col /= 10.0
            num.append(col)

        cat = []
        for f in CATEGORICAL_PREDICTORS:
            raw = bands[f].ravel()[flat].astype(np.int64)
            lut = cat_luts[f]
            cat.append(lut[np.clip(raw, 0, len(lut) - 1)].astype(np.float32))

        X = np.column_stack(num + cat).astype(np.float32)
        return X, flat, th, tw, win

    def _write(win, th, tw, flat, mean_p, std_p):
        mt  = np.full((th, tw), NODATA_OUT, np.float32)
        st  = np.full((th, tw), NODATA_OUT, np.float32)
        r, c = np.unravel_index(flat, (th, tw))
        mt[r, c] = mean_p
        st[r, c] = std_p
        with wlock:
            h_mean.write(mt, 1, window=win)
            h_std .write(st, 1, window=win)
        vm = mt[mt != NODATA_OUT]
        vs = st[st != NODATA_OUT]
        return (
            f'{vm.mean():.2f}' if vm.size else 'n/a',
            f'{vs.mean():.3f}' if vs.size else 'n/a',
        )

    n_w     = len(GPU_IDS)
    work_qs = [Queue(maxsize=QUEUE_MAXSIZE) for _ in GPU_IDS]
    res_q   = Queue()
    procs   = []
    for i, gid in enumerate(GPU_IDS):
        p = Process(
            target=gpu_worker,
            args=(gid, BOOTSTRAP_PKL, features, work_qs[i], res_q),
            daemon=True,
            name=f'gpu{gid}',
        )
        p.start()
        procs.append(p)
    print(f'✅ {n_w} GPU workers started (GPUs {GPU_IDS}).\n')

    tiles = [
        (ti * n_cols + tj + 1, row_starts[ti], col_starts[tj])
        for ti in range(len(row_starts))
        for tj in range(len(col_starts))
    ]

    pending   = {}
    done      = 0
    rr        = 0
    t_iter    = iter(tiles)
    exhausted = False
    t0        = time.time()

    pbar = tqdm(
        total=n_tiles, desc='Tiles', unit='tile', ncols=95,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
    )

    while done < n_tiles:

        if not exhausted:
            for _ in range(n_w * QUEUE_MAXSIZE * 2):
                wq_idx = rr % n_w
                if work_qs[wq_idx].full():
                    wq_idx = (rr + 1) % n_w
                    if work_qs[wq_idx].full():
                        break

                try:
                    tidx, rs, cs = next(t_iter)
                except StopIteration:
                    exhausted = True
                    break

                try:
                    X, flat, th, tw, win = _build_X(rs, cs)
                except Exception as e:
                    tqdm.write(f'  ❌ Tile {tidx} read error: {e}')
                    done += 1; pbar.update(1); rr += 1
                    continue

                if X is None:
                    nd = np.full((th, tw), NODATA_OUT, np.float32)
                    with wlock:
                        h_mean.write(nd, 1, window=win)
                        h_std .write(nd, 1, window=win)
                    done += 1; pbar.update(1); rr += 1
                    continue

                pending[tidx] = (flat, th, tw, win)
                work_qs[wq_idx].put((tidx, X))
                del X
                rr += 1

        got = True
        while got:
            try:
                res = res_q.get(timeout=0.25)
            except Exception:
                got = False
                break

            if res is None:
                continue

            tidx, mean_p, std_p = res

            if isinstance(mean_p, (Exception, RuntimeError)):
                tqdm.write(f'  ❌ GPU error on tile {tidx}: {mean_p}')
                if isinstance(std_p, str):
                    tqdm.write(std_p[:400])
                done += 1; pbar.update(1)
                pending.pop(tidx, None)
                continue

            if tidx not in pending:
                tqdm.write(f'  ⚠️  Unknown tile {tidx} — ignored')
                continue

            flat, th, tw, win = pending.pop(tidx)
            ms, ss = _write(win, th, tw, flat, mean_p, std_p)

            elapsed = time.time() - t0
            rate    = done / elapsed if elapsed > 0 else 1e-9
            eta     = (n_tiles - done) / rate
            tqdm.write(
                f'  [{tidx:4d}/{n_tiles}]  '
                f'mean={ms} Mg C/ha  std={ss}  '
                f'ETA {eta/60:.1f} min'
            )
            done += 1; pbar.update(1)
            gc.collect()

    pbar.close()

    for wq in work_qs:
        wq.put(None)
    for p in procs:
        p.join(timeout=120)
        if p.is_alive():
            tqdm.write(f'  ⚠️  {p.name} did not exit — terminating')
            p.terminate()

    for h in raster_handles.values():
        h.close()
    h_mean.close()
    h_std .close()

    elapsed = time.time() - t0
    print(f'\n Prediction complete in {elapsed/60:.1f} min '
          f'({elapsed/n_tiles:.1f} s/tile  ×  {n_tiles} tiles)')
    print(f'   → {PATH_MEAN}')
    print(f'   → {PATH_STD}')

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('='*70)
    print('  SOC Bootstrap Uncertainty Map — XGBoost')
    print('='*70)

    if not os.path.exists(MODEL_PKL):
        raise FileNotFoundError(
            f'Model bundle not found:\n  {MODEL_PKL}\n'
            f'Make sure train_soc_dualgpu.py has finished and produced this file '
            f'(OUTPUT_BASE_DIR/results_{{RESPONSE_VARIABLE}}/models/XGBoost.pkl).'
        )

    print(f'\n Loading dual-GPU model bundle...\n   {MODEL_PKL}')
    bundle   = joblib.load(MODEL_PKL)
    model    = bundle['model']
    features = bundle['features']

    xgb_params = model.get_params()
    for k in ('n_jobs', 'random_state', 'seed', 'device',
              'tree_method', 'gpu_id', 'predictor'):
        xgb_params.pop(k, None)

    print(f'   Features ({len(features)}): {features}')
    print(f'   n_estimators: {model.n_estimators}')
    print(f'   xgb_params  : {xgb_params}')

    if os.path.exists(BOOTSTRAP_PKL):
        print(f'\n Bootstrap models already exist → {BOOTSTRAP_PKL}')
        print('   Delete this file to force retraining.')
        print('   Loading cat_classes from saved bundle...')
        saved       = joblib.load(BOOTSTRAP_PKL)
        cat_classes = saved['cat_classes']
    else:
        _, cat_classes = phase1_train(features, xgb_params)

    phase2_predict(features, cat_classes)

    print('\n' + '='*70)
    print('  All outputs written:')
    for name in ('SOC_mean.tif', 'SOC_std.tif',
                 'bootstrap_oob_diagnostics.csv', 'bootstrap_models.pkl'):
        p = os.path.join(OUTPUT_DIR, name)
        if os.path.exists(p):
            print(f'   {name:<40s}  {os.path.getsize(p)/1e6:7.1f} MB')
    print('='*70)


if __name__ == '__main__':
    main()
