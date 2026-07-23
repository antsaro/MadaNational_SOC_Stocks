import os
import gc
import traceback
import numpy as np
import joblib


def gpu_worker(gpu_id, bootstrap_pkl, features, work_q, result_q):
    """
    1. X_raw_np (numpy, unscaled) → X_raw (cupy, GPU)     [one cudaMemcpy]
    2. For b in range(B):
         X_sc = (X_raw − μ_b) / σ_b              [cupy subtract+divide, GPU]
         pred = booster_b.inplace_predict(X_sc)   [XGBoost GPU tree eval]
         Welford update on GPU (float64 accumulators)
    3. std  = sqrt(M2 / (B−1))                   [GPU]
    4. mean, std → numpy                          [one cudaMemcpy per array]
    5. Send (tile_idx, mean_np, std_np) → result_q
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    import cupy as cp
    import xgboost as xgb

    print(f'[GPU {gpu_id}] Initialising...', flush=True)

    # ── Load and prepare all B bootstrap models ───────────────────────────────
    save_data  = joblib.load(bootstrap_pkl)
    model_list = save_data['models']
    n_models   = len(model_list)

    boosters, sc_means_gpu, sc_scales_gpu = [], [], []
    for md in model_list:
        bst = xgb.Booster()
        bst.load_model(bytearray(md['booster_bytes']))
        try:
            bst.set_param({'device': 'cuda:0'})
        except Exception:
            try:
                bst.set_param({'predictor': 'gpu_predictor', 'gpu_id': 0})
            except Exception:
                pass  

        boosters.append(bst)
        # Store scaler params as cupy arrays for broadcasting
        sc_means_gpu .append(cp.asarray(md['sc_mean'],  dtype=cp.float32))
        sc_scales_gpu.append(cp.asarray(md['sc_scale'], dtype=cp.float32))

    print(f'[GPU {gpu_id}] {n_models} boosters ready. Waiting for tiles...', flush=True)

    # ── Main tile loop ────────────────────────────────────────────────────────
    while True:
        item = work_q.get()
        if item is None:
            break  # graceful shutdown sentinel

        tile_idx, X_raw_np = item  # X_raw_np: float32 numpy, unscaled
        n_valid = X_raw_np.shape[0]

        try:
            # Host → device (one transfer per tile)
            X_raw = cp.asarray(X_raw_np, dtype=cp.float32)
            del X_raw_np

            # Welford accumulators in float64 for numerical stability
            w_mean = cp.zeros(n_valid, dtype=cp.float64)
            w_M2   = cp.zeros(n_valid, dtype=cp.float64)

            for b in range(n_models):
                # ── Per-replicate scaling on GPU ──────────────────────────────
                # Shape: (n_valid, n_feat) broadcast with (n_feat,)
                X_sc = (X_raw - sc_means_gpu[b]) / sc_scales_gpu[b]
                # Ensure C-contiguous memory (required by inplace_predict)
                if not X_sc.flags['C_CONTIGUOUS']:
                    X_sc = cp.ascontiguousarray(X_sc)

                # ── GPU prediction (no DMatrix allocation) ────────────────────
                raw_pred = boosters[b].inplace_predict(X_sc)
                pred     = cp.asarray(raw_pred, dtype=cp.float64)  # stays on GPU

                # ── Welford online update (numerically stable) ─────────────────
                delta   = pred - w_mean
                w_mean += delta / (b + 1)
                w_M2   += delta * (pred - w_mean)   # uses updated mean

            # ── Finalise ──────────────────────────────────────────────────────
            std_gpu  = cp.sqrt(w_M2 / max(n_models - 1, 1))

            # Device → host (two small transfers per tile)
            mean_out = cp.asnumpy(w_mean.astype(cp.float32))
            std_out  = cp.asnumpy(std_gpu.astype(cp.float32))

            del X_raw, w_mean, w_M2, std_gpu
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

            result_q.put((tile_idx, mean_out, std_out))

        except Exception as exc:
            # Send exception back to main process for logging
            result_q.put((tile_idx, RuntimeError(str(exc)), traceback.format_exc()))

    result_q.put(None)  # signal: this worker is done
