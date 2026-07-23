#!/usr/bin/env python
"""
USAGE
-----
In a Kaggle notebook cell, just run:

    !python train_soc_dualgpu.py

That's the "driver" call: no --model / --gpu flags -> it spawns
    python train_soc_dualgpu.py --model RandomForest --gpu 0
    python train_soc_dualgpu.py --model XGBoost      --gpu 1
in parallel, waits for both, and aggregates the results exactly like the
original single-process script did (summary.csv, per_LUv2_metrics_all_models.csv,
all_results_summary.xlsx, best_model_info.json, etc).

You can also debug a single model/GPU manually:

    !python train_soc_dualgpu.py --model XGBoost --gpu 1
"""

import os
import sys
import json
import time
import argparse
import subprocess
import threading
import queue
import joblib
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ============================================================
# CONFIG
# ============================================================
INPUT_FILE = '/kaggle/working/DeepSOC_predictors_table_final.csv'
OUTPUT_BASE_DIR = '/kaggle/working/ML_SOC_Results_GPU'
RESPONSE_VARIABLE = 'DeepSOC30_100'
LUV2_VARIABLE = 'Ecosystem'
N_OUTER_FOLDS = 10
N_OUTER_REPEATS = 10
N_INNER_FOLDS = 4
OPTUNA_TRIALS = 100

NUMERIC_PREDICTORS = [
    'NDVI', 'NDWI', 'NIRI', 'elevation', 'slope', 'mat', 'map', 'treecover',
]
CATEGORICAL_PREDICTORS = [
    'lulc', 'soil_type',
]

# One physical GPU per model on a Kaggle T4 x2 instance.
MODEL_GPU_MAP = {
    'RandomForest': 0,
    'XGBoost': 1,
}

FIXED_PARAMS = {
    'RandomForest': dict(random_state=42),
    'XGBoost': dict(random_state=42, verbosity=0, tree_method='hist', device='cuda'),
}

def load_data(file_path, response_variable, numeric_preds, categorical_preds,
              luv2_variable=None):

    print("=" * 70)
    print(f"Response variable : {response_variable}")
    print("=" * 70)

    df = pd.read_csv(file_path, sep=',')
    print(f"Raw shape         : {df.shape}")
    print(f"Columns           : {list(df.columns)}")

    df.replace(-9999, np.nan, inplace=True)

    before = len(df)
    df = df.dropna(subset=[response_variable])
    print(f"Dropped {before - len(df)} rows where response is NaN")
    print(f"Working dataset   : {len(df)} rows")

    if luv2_variable and luv2_variable in df.columns:
        print(f"LUv2 classes found: {sorted(df[luv2_variable].dropna().unique())}")
    else:
        print(f"[WARN] LUv2 column '{luv2_variable}' not found in CSV")

    available_num = [c for c in numeric_preds if c in df.columns]
    available_cat = [c for c in categorical_preds if c in df.columns]
    missing_num = [c for c in numeric_preds if c not in df.columns]
    missing_cat = [c for c in categorical_preds if c not in df.columns]

    if missing_num:
        print(f"[WARN] Numeric predictors not found    : {missing_num}")
    if missing_cat:
        print(f"[WARN] Categorical predictors not found: {missing_cat}")

    extra_cols = [luv2_variable] if luv2_variable and luv2_variable in df.columns else []
    keep_cols = list(dict.fromkeys([response_variable] + available_num + available_cat + extra_cols))
    df = df[keep_cols].copy()

    for col in available_num:
        if df[col].isnull().any():
            n_miss = df[col].isnull().sum()
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  Imputed {n_miss:>4} NaN in '{col}' with median={median_val:.4f}")

    encoders = {}
    encoded_cols = []
    for col in available_cat:
        df[col] = df[col].astype(str).fillna('Unknown')
        le = LabelEncoder()
        enc = f'{col}_enc'
        df[enc] = le.fit_transform(df[col])
        encoders[col] = le
        encoded_cols.append(enc)
        print(f"Encoded '{col}': {len(le.classes_)} classes -> {list(le.classes_)}")

    feature_cols = available_num + encoded_cols

    print(f"\nFinal dataset shape : {df.shape}")
    print(f"Features used       : {len(feature_cols)}")
    print(f"\n{response_variable} statistics:")
    print(df[response_variable].describe().round(4))

    return df, feature_cols, encoders


def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 2:
        return dict(n=len(y_true), rmse=np.nan, r2=np.nan, mae=np.nan,
                    bias=np.nan, rpd=np.nan, rpiq=np.nan, ccc=np.nan)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    sd = np.std(y_true)
    iqr = np.percentile(y_true, 75) - np.percentile(y_true, 25)
    bias = float(np.mean(y_pred - y_true))
    rpd = sd / rmse if rmse > 0 else 0.0
    rpiq = iqr / rmse if rmse > 0 else 0.0
    ccc_num = 2 * np.cov(y_true, y_pred, ddof=0)[0, 1]
    ccc_den = (np.var(y_true, ddof=0) + np.var(y_pred, ddof=0) +
               (np.mean(y_true) - np.mean(y_pred)) ** 2)
    ccc = ccc_num / ccc_den if ccc_den > 0 else 0.0
    return dict(n=int(len(y_true)), rmse=rmse, r2=r2, mae=mae,
                bias=bias, rpd=rpd, rpiq=rpiq, ccc=ccc)


def compute_luv2_metrics(predictions_df):
    rows = []
    for lu in sorted(predictions_df['LUv2'].dropna().unique()):
        sub = predictions_df[predictions_df['LUv2'] == lu]
        if len(sub) < 2:
            continue
        m = calc_metrics(sub['observed'].values, sub['predicted'].values)
        rows.append({'LUv2': lu, 'n': m['n'],
                     'R2': round(m['r2'], 4), 'RMSE': round(m['rmse'], 4),
                     'MAE': round(m['mae'], 4), 'Bias': round(m['bias'], 4),
                     'RPD': round(m['rpd'], 4), 'RPIQ': round(m['rpiq'], 4),
                     'CCC': round(m['ccc'], 4)})
    m_all = calc_metrics(predictions_df['observed'].values, predictions_df['predicted'].values)
    rows.append({'LUv2': 'ALL', 'n': m_all['n'],
                 'R2': round(m_all['r2'], 4), 'RMSE': round(m_all['rmse'], 4),
                 'MAE': round(m_all['mae'], 4), 'Bias': round(m_all['bias'], 4),
                 'RPD': round(m_all['rpd'], 4), 'RPIQ': round(m_all['rpiq'], 4),
                 'CCC': round(m_all['ccc'], 4)})
    return pd.DataFrame(rows)


def print_luv2_table(lu_df, model_name):
    print(f"\n  -- Per-LUv2 metrics ({model_name}) --")
    hdr = (f"  {'LUv2':<30} {'n':>5} {'R2':>7} {'RMSE':>8} "
           f"{'MAE':>8} {'Bias':>8} {'RPD':>7} {'RPIQ':>7} {'CCC':>7}")
    sep = '  ' + '-' * (len(hdr) - 2)
    print(sep); print(hdr); print(sep)
    for _, row in lu_df.iterrows():
        print(f"  {str(row['LUv2']):<30} {int(row['n']):>5} "
              f"{row['R2']:>7.4f} {row['RMSE']:>8.4f} {row['MAE']:>8.4f} "
              f"{row['Bias']:>8.4f} {row['RPD']:>7.4f} {row['RPIQ']:>7.4f} {row['CCC']:>7.4f}")
    print(sep)

def _to_host(a):
    import cupy as cp
    return cp.asnumpy(a) if isinstance(a, cp.ndarray) else np.asarray(a)


def get_model_class(model_name):
    if model_name == 'RandomForest':
        from cuml.ensemble import RandomForestRegressor as cuRandomForestRegressor
        return cuRandomForestRegressor
    elif model_name == 'XGBoost':
        import xgboost as xgb
        return xgb.XGBRegressor
    raise ValueError(f'Unknown model: {model_name}')


def make_objective(model_name, X_cp, y_cp, n_inner_folds):
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    n = X_cp.shape[0]
    model_cls = get_model_class(model_name)

    def objective(trial):
        if model_name == 'RandomForest':
            params = dict(
                n_estimators=trial.suggest_int('n_estimators', 100, 1000),
                max_depth=trial.suggest_int('max_depth', 5, 24),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 15),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 8),
                max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', 1.0]),
                random_state=42,
            )
        elif model_name == 'XGBoost':
            params = dict(
                n_estimators=trial.suggest_int('n_estimators', 100, 1000),
                max_depth=trial.suggest_int('max_depth', 3, 10),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.30, log=True),
                subsample=trial.suggest_float('subsample', 0.6, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                reg_alpha=trial.suggest_float('reg_alpha', 0.0, 2.0),
                reg_lambda=trial.suggest_float('reg_lambda', 0.5, 5.0),
                random_state=42, verbosity=0, tree_method='hist', device='cuda',
            )
        else:
            raise ValueError(model_name)

        inner_kf = KFold(n_splits=n_inner_folds, shuffle=True, random_state=0)
        scores = []
        for i, (tr, va) in enumerate(inner_kf.split(np.arange(n))):
            scaler = cuStandardScaler()
            Xtr = scaler.fit_transform(X_cp[tr])
            Xva = scaler.transform(X_cp[va])
            model = model_cls(**params)
            model.fit(Xtr, y_cp[tr])
            preds = _to_host(model.predict(Xva))
            fold_rmse = np.sqrt(mean_squared_error(_to_host(y_cp[va]), preds))
            scores.append(fold_rmse)

            trial.report(float(np.mean(scores)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(scores))
    return objective


def inner_hyperparam_search(model_name, X_cp, y_cp, n_inner_folds, n_trials):
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
    study.optimize(
        make_objective(model_name, X_cp, y_cp, n_inner_folds),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    return {**FIXED_PARAMS[model_name], **study.best_params}, study.best_value


def nested_repeated_kfold_cv(model_name, X_cp, y_cp, luv2_series=None,
                              n_outer_folds=10, n_outer_repeats=10, n_inner_folds=4,
                              n_trials=100):
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    model_class = get_model_class(model_name)

    n = X_cp.shape[0]
    fold_metrics = []
    all_obs, all_pred, all_repeat, all_fold, all_luv2 = [], [], [], [], []
    best_params_per_fold = []
    total_folds = n_outer_folds * n_outer_repeats
    run = 0

    for rep in range(n_outer_repeats):
        outer_kf = KFold(n_splits=n_outer_folds, shuffle=True, random_state=rep * 100 + 42)
        for fold, (tr_idx, va_idx) in enumerate(outer_kf.split(np.arange(n)), 1):
            run += 1
            print(f"    [{model_name}] Outer fold {run}/{total_folds}  (rep {rep+1}, fold {fold})")

            X_tr_outer = X_cp[tr_idx]
            y_tr_outer = y_cp[tr_idx]
            X_va_outer = X_cp[va_idx]
            y_va_outer = y_cp[va_idx]

            best_params, best_inner_rmse = inner_hyperparam_search(
                model_name, X_tr_outer, y_tr_outer, n_inner_folds, n_trials
            )
            best_params_per_fold.append({
                'repeat': rep + 1, 'fold': fold,
                'best_inner_rmse': best_inner_rmse, **best_params,
            })

            scaler = cuStandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr_outer)
            X_va_scaled = scaler.transform(X_va_outer)

            model = model_class(**best_params)
            model.fit(X_tr_scaled, y_tr_outer)
            preds = _to_host(model.predict(X_va_scaled))
            y_va_host = _to_host(y_va_outer)

            m = calc_metrics(y_va_host, preds)
            m['repeat'] = rep + 1
            m['fold'] = fold
            fold_metrics.append(m)

            all_obs.extend(y_va_host.tolist())
            all_pred.extend(preds.tolist())
            all_repeat.extend([rep + 1] * len(preds))
            all_fold.extend([fold] * len(preds))
            all_luv2.extend(
                luv2_series.iloc[va_idx].tolist() if luv2_series is not None else ['Unknown'] * len(preds)
            )

    metric_keys = ['rmse', 'r2', 'mae', 'bias', 'rpd', 'rpiq', 'ccc']
    mean_metrics = {k: float(np.mean([f[k] for f in fold_metrics])) for k in metric_keys}
    std_metrics = {f'{k}_std': float(np.std([f[k] for f in fold_metrics])) for k in metric_keys}

    repeat_summaries = []
    for rep in range(1, n_outer_repeats + 1):
        rep_folds = [f for f in fold_metrics if f['repeat'] == rep]
        rs = {'repeat': rep}
        for k in metric_keys:
            rs[f'mean_{k}'] = float(np.mean([f[k] for f in rep_folds]))
            rs[f'std_{k}'] = float(np.std([f[k] for f in rep_folds]))
        repeat_summaries.append(rs)

    overall = calc_metrics(all_obs, all_pred)

    predictions_df = pd.DataFrame({
        'observed': all_obs, 'predicted': all_pred,
        'repeat': all_repeat, 'fold': all_fold, 'LUv2': all_luv2,
    })

    return {
        'fold_metrics': fold_metrics,
        'mean': mean_metrics,
        'std': std_metrics,
        'repeat_summaries': repeat_summaries,
        'overall': overall,
        'predictions_df': predictions_df,
        'best_params_per_fold': pd.DataFrame(best_params_per_fold),
    }


# ============================================================
# SINGLE-MODEL PIPELINE — the whole body that runs on ONE GPU
# ============================================================
def run_single_model_pipeline(model_name, gpu_id, out_dir):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    import cupy as cp
    from cuml.preprocessing import StandardScaler as cuStandardScaler

    t0 = time.time()
    print(f"[GPU {gpu_id}] PID {os.getpid()} starting {model_name} pipeline")
    print(f"[GPU {gpu_id}] CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")

    df, feature_cols, encoders = load_data(
        INPUT_FILE, RESPONSE_VARIABLE, NUMERIC_PREDICTORS, CATEGORICAL_PREDICTORS,
        luv2_variable=LUV2_VARIABLE,
    )
    luv2_series = df[LUV2_VARIABLE].copy() if LUV2_VARIABLE in df.columns else None

    X_cp = cp.asarray(df[feature_cols].values, dtype=cp.float32)
    y_cp = cp.asarray(df[RESPONSE_VARIABLE].values, dtype=cp.float32)

    print(f"\n{'-'*65}\n  {model_name} (GPU {gpu_id})\n{'-'*65}")
    print(f"  Nested CV: outer {N_OUTER_FOLDS}-fold x {N_OUTER_REPEATS}-repeat, "
          f"inner {N_INNER_FOLDS}-fold, {OPTUNA_TRIALS} Optuna trials with pruning")

    cv = nested_repeated_kfold_cv(
        model_name, X_cp, y_cp, luv2_series=luv2_series,
        n_outer_folds=N_OUTER_FOLDS, n_outer_repeats=N_OUTER_REPEATS,
        n_inner_folds=N_INNER_FOLDS, n_trials=OPTUNA_TRIALS,
    )

    print(f"\n  -- Aggregated CV metrics (mean +/- std across "
          f"{N_OUTER_FOLDS*N_OUTER_REPEATS} outer folds) --")
    for k in ['r2', 'rmse', 'mae', 'bias', 'rpiq', 'ccc']:
        print(f"  Mean {k.upper():<6}: {cv['mean'][k]:.4f}  +/- {cv['std'][f'{k}_std']:.4f}")

    print(f"\n  -- Pooled-overall metrics --")
    for k in ['r2', 'rmse', 'mae', 'bias', 'rpd', 'rpiq', 'ccc']:
        print(f"  Overall {k.upper():<6}: {cv['overall'][k]:.4f}")

    lu_df = compute_luv2_metrics(cv['predictions_df'])
    lu_df.insert(0, 'Model', model_name)
    print_luv2_table(lu_df, model_name)

    print(f"\n  Refitting final model on full dataset with Optuna "
          f"({OPTUNA_TRIALS} trials, {N_INNER_FOLDS}-fold inner CV, pruning)...")
    final_best_params, final_best_rmse = inner_hyperparam_search(
        model_name, X_cp, y_cp, N_INNER_FOLDS, OPTUNA_TRIALS
    )
    print(f"  Final best inner RMSE : {final_best_rmse:.4f}")
    print(f"  Final best params     : {final_best_params}")

    model_class = get_model_class(model_name)
    scaler_final = cuStandardScaler()
    X_scaled = scaler_final.fit_transform(X_cp)
    final_model = model_class(**final_best_params)
    final_model.fit(X_scaled, y_cp)

    feat_imp = None
    if hasattr(final_model, 'feature_importances_'):
        importances = _to_host(final_model.feature_importances_)
        feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
        print(f"\n  Top 10 features:")
        for feat, imp in feat_imp[:10]:
            print(f"    {feat:<35}: {imp:.4f}")
    else:
        print(f"  [WARN] {model_name} model exposes no feature_importances_ — skipping.")

    # ---- persist everything to disk (cross-process aggregation reads these back) ----
    models_dir = os.path.join(out_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump({'model': final_model, 'scaler': scaler_final,
                 'encoders': encoders, 'features': feature_cols},
                os.path.join(models_dir, f'{model_name}.pkl'))

    cv['predictions_df'].to_csv(os.path.join(out_dir, f'cv_predictions_{model_name}.csv'), index=False)
    lu_df.to_csv(os.path.join(out_dir, f'per_LUv2_metrics_{model_name}.csv'), index=False)
    cv['best_params_per_fold'].to_csv(
        os.path.join(out_dir, f'best_params_per_outer_fold_{model_name}.csv'), index=False)
    pd.DataFrame(cv['fold_metrics']).to_csv(
        os.path.join(out_dir, f'fold_details_{model_name}.csv'), index=False)
    pd.DataFrame(cv['repeat_summaries']).to_csv(
        os.path.join(out_dir, f'repeat_summaries_{model_name}.csv'), index=False)
    if feat_imp:
        pd.DataFrame(feat_imp, columns=['Feature', 'Importance']).to_csv(
            os.path.join(out_dir, f'feat_importance_{model_name}.csv'), index=False)

    summary_row = {
        'Model': model_name,
        'GPU': gpu_id,
        'Mean_CV_R2': round(cv['mean']['r2'], 4),
        'Std_CV_R2': round(cv['std']['r2_std'], 4),
        'Mean_CV_RMSE': round(cv['mean']['rmse'], 4),
        'Std_CV_RMSE': round(cv['std']['rmse_std'], 4),
        'Mean_CV_MAE': round(cv['mean']['mae'], 4),
        'Std_CV_MAE': round(cv['std']['mae_std'], 4),
        'Mean_CV_Bias': round(cv['mean']['bias'], 4),
        'Std_CV_Bias': round(cv['std']['bias_std'], 4),
        'Mean_CV_RPIQ': round(cv['mean']['rpiq'], 4),
        'Std_CV_RPIQ': round(cv['std']['rpiq_std'], 4),
        'Mean_CV_CCC': round(cv['mean']['ccc'], 4),
        'Std_CV_CCC': round(cv['std']['ccc_std'], 4),
        'Overall_R2': round(cv['overall']['r2'], 4),
        'Overall_RMSE': round(cv['overall']['rmse'], 4),
        'Overall_MAE': round(cv['overall']['mae'], 4),
        'Overall_Bias': round(cv['overall']['bias'], 4),
        'Overall_RPD': round(cv['overall']['rpd'], 4),
        'Overall_RPIQ': round(cv['overall']['rpiq'], 4),
        'Overall_CCC': round(cv['overall']['ccc'], 4),
        'final_refit_best_params': {
            k: (v if not isinstance(v, np.generic) else v.item())
            for k, v in final_best_params.items()
        },
        'final_best_inner_rmse': float(final_best_rmse),
        'n_features': len(feature_cols),
        'luv2_classes': sorted(df[LUV2_VARIABLE].dropna().unique().tolist())
                        if LUV2_VARIABLE in df.columns else [],
        'runtime_seconds': round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, f'summary_{model_name}.json'), 'w') as f:
        json.dump(summary_row, f, indent=4)

    print(f"\n[GPU {gpu_id}] Finished {model_name} in {summary_row['runtime_seconds']}s")


def _driver():
    out_dir = os.path.join(OUTPUT_BASE_DIR, f'results_{RESPONSE_VARIABLE}')
    os.makedirs(out_dir, exist_ok=True)
    script_path = os.path.abspath(__file__)

    # A short, fixed-width tag per model so interleaved lines from both
    # GPUs stay easy to tell apart in one shared cell output.
    tags = {}
    for model_name in MODEL_GPU_MAP:
        tags[model_name] = f"[{model_name}]".ljust(max(len(m) for m in MODEL_GPU_MAP) + 2)

    line_queue = queue.Queue()

    def _reader(model_name, proc, log_f):
        for line in iter(proc.stdout.readline, ''):
            if line == '':
                break
            log_f.write(line)
            log_f.flush()
            line_queue.put((model_name, line.rstrip('\n')))
        proc.stdout.close()
        line_queue.put((model_name, None))  # sentinel: this stream is done

    procs = {}
    threads = []
    for model_name, gpu_id in MODEL_GPU_MAP.items():
        cmd = [sys.executable, script_path, '--model', model_name, '--gpu', str(gpu_id)]
        log_path = os.path.join(out_dir, f'log_{model_name}.txt')
        log_f = open(log_path, 'w')
        print(f"Launching {model_name} on GPU {gpu_id}  ->  log: {log_path}")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        procs[model_name] = (proc, log_f)
        th = threading.Thread(target=_reader, args=(model_name, proc, log_f), daemon=True)
        th.start()
        threads.append(th)

    print("\nBoth models are now training in parallel, one per GPU.")
    print("Live output below (lines are tagged by model as they arrive):\n")

    active = set(MODEL_GPU_MAP.keys())
    while active:
        model_name, line = line_queue.get()
        if line is None:
            active.discard(model_name)
            continue
        print(f"{tags[model_name]}{line}")

    for th in threads:
        th.join()

    failed = []
    for model_name, (proc, log_f) in procs.items():
        ret = proc.wait()
        log_f.close()
        if ret == 0:
            print(f"[{model_name}] OK")
        else:
            failed.append(model_name)
            print(f"[{model_name}] FAILED (exit code {ret}) — check log_{model_name}.txt")

    if failed:
        raise RuntimeError(f"Worker process(es) failed: {failed}. "
                            f"Inspect log files in {out_dir}.")

    # ---- aggregate outputs written by the two workers ----
    summary_rows = []
    lu_dfs = []
    fold_rows = []
    repeat_rows = []
    feat_imp_dfs = {}

    for model_name in MODEL_GPU_MAP:
        with open(os.path.join(out_dir, f'summary_{model_name}.json')) as f:
            summary_rows.append(json.load(f))

        lu_dfs.append(pd.read_csv(os.path.join(out_dir, f'per_LUv2_metrics_{model_name}.csv')))

        fm = pd.read_csv(os.path.join(out_dir, f'fold_details_{model_name}.csv'))
        fm.insert(0, 'Model', model_name)
        fold_rows.append(fm)

        rs = pd.read_csv(os.path.join(out_dir, f'repeat_summaries_{model_name}.csv'))
        rs.insert(0, 'Model', model_name)
        repeat_rows.append(rs)

        fi_path = os.path.join(out_dir, f'feat_importance_{model_name}.csv')
        if os.path.exists(fi_path):
            feat_imp_dfs[model_name] = pd.read_csv(fi_path)

    summary_df = (pd.DataFrame(summary_rows)
                    .sort_values(['Overall_R2', 'Overall_RMSE'], ascending=[False, True])
                    .reset_index(drop=True))
    lu_all_df = pd.concat(lu_dfs, ignore_index=True)
    fold_all_df = pd.concat(fold_rows, ignore_index=True)
    repeat_all_df = pd.concat(repeat_rows, ignore_index=True)
    best_name = summary_df.iloc[0]['Model']

    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(summary_df.drop(columns=['final_refit_best_params', 'luv2_classes']).to_string(index=False))
    print(f"\nBest model: {best_name}")

    print("\n" + "=" * 70)
    print("PER-LUv2 METRICS - ALL MODELS")
    print("=" * 70)
    print(lu_all_df.to_string(index=False))

    summary_df.to_csv(os.path.join(out_dir, 'summary.csv'), index=False)
    lu_all_df.to_csv(os.path.join(out_dir, 'per_LUv2_metrics_all_models.csv'), index=False)
    fold_all_df.to_csv(os.path.join(out_dir, 'fold_details.csv'), index=False)
    repeat_all_df.to_csv(os.path.join(out_dir, 'repeat_summaries.csv'), index=False)

    out_excel = os.path.join(out_dir, 'all_results_summary.xlsx')
    with pd.ExcelWriter(out_excel, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Model_summary', index=False)
        lu_all_df.to_excel(writer, sheet_name='PerLUv2_all', index=False)
        fold_all_df.to_excel(writer, sheet_name='Fold_details', index=False)
        repeat_all_df.to_excel(writer, sheet_name='Repeat_summaries', index=False)
        for model_name in MODEL_GPU_MAP:
            preds = pd.read_csv(os.path.join(out_dir, f'cv_predictions_{model_name}.csv'))
            preds.to_excel(writer, sheet_name=f'Preds_{model_name[:10]}', index=False)
            lu = pd.read_csv(os.path.join(out_dir, f'per_LUv2_metrics_{model_name}.csv'))
            lu.to_excel(writer, sheet_name=f'LUv2_{model_name[:10]}', index=False)
            params = pd.read_csv(os.path.join(out_dir, f'best_params_per_outer_fold_{model_name}.csv'))
            params.to_excel(writer, sheet_name=f'Params_{model_name[:9]}', index=False)
            if model_name in feat_imp_dfs:
                feat_imp_dfs[model_name].to_excel(writer, sheet_name=f'FeatImp_{model_name[:8]}', index=False)

    best_row = [r for r in summary_rows if r['Model'] == best_name][0]
    with open(os.path.join(out_dir, 'best_model_info.json'), 'w') as f:
        json.dump({
            'best_model': best_name,
            'cv_scheme': f'nested {N_OUTER_FOLDS}-fold x {N_OUTER_REPEATS}-repeat outer, {N_INNER_FOLDS}-fold inner',
            'total_outer_fold_runs': N_OUTER_FOLDS * N_OUTER_REPEATS,
            'optuna_trials_per_inner_search': OPTUNA_TRIALS,
            'overall_r2': best_row['Overall_R2'],
            'overall_rmse': best_row['Overall_RMSE'],
            'overall_mae': best_row['Overall_MAE'],
            'overall_bias': best_row['Overall_Bias'],
            'overall_rpd': best_row['Overall_RPD'],
            'overall_rpiq': best_row['Overall_RPIQ'],
            'overall_ccc': best_row['Overall_CCC'],
            'mean_cv_r2': best_row['Mean_CV_R2'],
            'std_cv_r2': best_row['Std_CV_R2'],
            'mean_cv_rmse': best_row['Mean_CV_RMSE'],
            'std_cv_rmse': best_row['Std_CV_RMSE'],
            'final_refit_best_params': best_row['final_refit_best_params'],
            'response_variable': RESPONSE_VARIABLE,
            'n_outer_folds': N_OUTER_FOLDS,
            'n_outer_repeats': N_OUTER_REPEATS,
            'n_inner_folds': N_INNER_FOLDS,
            'n_features': best_row['n_features'],
            'luv2_classes': best_row['luv2_classes'],
        }, f, indent=4)

    print(f"\nAll outputs saved to : {out_dir}")
    print(f"Excel summary        : {out_excel}")
    return summary_df, lu_all_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(MODEL_GPU_MAP.keys()), default=None,
                         help='Internal: run a single model on a single GPU (used by the driver).')
    parser.add_argument('--gpu', type=int, default=None,
                         help='Internal: physical GPU index for --model.')
    args = parser.parse_args()

    if args.model is not None:
        if args.gpu is None:
            raise ValueError('--gpu is required when --model is passed')
        out_dir = os.path.join(OUTPUT_BASE_DIR, f'results_{RESPONSE_VARIABLE}')
        os.makedirs(out_dir, exist_ok=True)
        run_single_model_pipeline(args.model, args.gpu, out_dir)
    else:
        summary, lu_metrics = _driver()
