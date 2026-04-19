"""Backend functions for the public Stage-6 final K=3 fit workflow.

This module holds the active runtime behind step60_fit_final_k3_fusion_hmm.ipynb.
It keeps the preserved final-fit logic in normal Python code so the public notebook
can stay focused on inputs, outputs, and the scientific role of the step."""

from __future__ import annotations


def _display_fallback(obj):
    try:
        from IPython.display import display as ipy_display
        ipy_display(obj)
    except Exception:
        print(obj)


def run_final_fit_backend(
    *,
    segments_root,
    final_model_root,
    data_variant: str = "intermediate",
    feature_mode: str = "nolags",
    minlen: int = 15,
    k_final: int = 3,
    manifest_tsv=None,
    gpu_memory_limit_mb=None,
    save_gamma: bool = True,
    do_viterbi: bool = True,
):
    """Run the final full-data K=3 fit using the public Stage-6 contract."""
    from pathlib import Path
    SEGMENTS_ROOT = Path(segments_root)
    FINAL_MODEL_ROOT = Path(final_model_root)
    DATA_VARIANT = str(data_variant)
    FEATURE_MODE = str(feature_mode)
    MINLEN = int(minlen)
    K_FINAL = int(k_final)
    MANIFEST_TSV = Path(manifest_tsv) if manifest_tsv is not None else None
    GPU_MEMORY_LIMIT_MB = gpu_memory_limit_mb
    SAVE_GAMMA = bool(save_gamma)
    DO_VITERBI = bool(do_viterbi)
    
    display = _display_fallback
    
    SEGMENT_BRANCH_ROOT = SEGMENTS_ROOT / DATA_VARIANT
    FINAL_MODEL_OUTPUT_ROOT = FINAL_MODEL_ROOT / f"PipelineE_final_K{K_FINAL:02d}_{DATA_VARIANT}_{FEATURE_MODE}_minlen{MINLEN}"
    
    # ---- cleaned public configuration bridge ----
    from pathlib import Path
    import os, json
    
    FINAL_ROOT = SEGMENT_BRANCH_ROOT
    OUT_ROOT = FINAL_MODEL_OUTPUT_ROOT
    N_PARCELS = 200
    TR_SEC = 2.1
    
    if FEATURE_MODE.lower() == "lags":
        LAGS_TR = [-1, 0, 1]
    elif FEATURE_MODE.lower() == "nolags":
        LAGS_TR = [0]
    else:
        raise ValueError("FEATURE_MODE must be 'lags' or 'nolags'")
    
    D_BOLD = N_PARCELS
    D_EEG = N_PARCELS * len(LAGS_TR)
    D_TOTAL = D_BOLD + D_EEG
    
    SEQ_LEN = 10
    STEP_SIZE = 1
    BATCH_SIZE = 16
    REBATCH_DROP_REMAINDER_DEFAULT = True
    SHUFFLE_BUFFER = 2048
    
    N_BOLD_PCS = 40
    N_EEG_PCS = 40
    
    LEARNING_RATE = 1e-3
    N_EPOCHS = 60
    COV_EPS = 1e-6
    DIAGONAL_COVS = False
    INIT_TAKE = 0.30
    INIT_EPOCHS = 5
    SEEDS = list(range(11, 11 + 2*30, 2))
    USE_RUNWISE_ZSCORE = True
    TWO_STAGE_TRAIN_TRANS = True
    STAGE1_EPOCH_FRAC = 0.60
    TOPM_REFINES = 5
    FO_MAX_THRESH = 0.95
    FO_ACTIVE_THRESH = 0.01
    MIN_ACTIVE_STATES_BASE = 3
    FE_BETTER = "lower"
    DISABLE_XLA_AT_IMPORT = True
    if DISABLE_XLA_AT_IMPORT:
        os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=-1"
        os.environ["TF_XLA_ENABLE_XLA_DEVICES"] = "0"
        os.environ["XLA_FLAGS"] = ""
    FORCE_EAGER = False
    DISABLE_PREFETCH = True
    DISABLE_CALLBACKS = True
    
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    print("OUT_ROOT:", OUT_ROOT)
    print("K_FINAL:", K_FINAL, "| MODE:", FEATURE_MODE, "| MINLEN:", MINLEN)
    print("D_TOTAL:", D_TOTAL, "| LAGS_TR:", LAGS_TR)
    print("SEQ_LEN/STEP/BATCH:", SEQ_LEN, STEP_SIZE, BATCH_SIZE)
    print("SEEDS:", SEEDS[:5], "...", f"(n={len(SEEDS)})")
    print("Two-stage:", TWO_STAGE_TRAIN_TRANS, "| TOPM_REFINES:", TOPM_REFINES)
    # ---- notebook cell 2 ----
    # =========================
    # Cell 1 — Imports + manifest resolution + basic QC + provenance
    # =========================
    import gc, math, json, re
    from pathlib import Path
    
    import numpy as np
    import pandas as pd
    
    import tensorflow as tf
    from osl_dynamics.data import Data
    from osl_dynamics.models.hmm import Config, Model
    
    PIPELINE_TAG  = "PipelineE_FINAL"
    PIPELINE_DATE = str(pd.Timestamp.now().date())  # YYYY-MM-DD
    
    def auto_find_manifest(final_root: Path, feature_mode: str, minlen: int) -> Path:
        mode = feature_mode.lower()
        candidates = [
            final_root / f"hmm_segments_minlen{minlen}_{mode}" / "segments_manifest.tsv",
            final_root / f"hmm_segments_minlen{minlen}" / "segments_manifest.tsv",
        ]
        for m in candidates:
            if m.exists():
                return m
        hits = list(final_root.rglob("segments_manifest.tsv"))
        if hits:
            def score(p: Path):
                s = str(p)
                sc = 0
                if f"minlen{minlen}" in s: sc += 10
                if mode in s: sc += 5
                return sc
            hits = sorted(hits, key=score, reverse=True)
            return hits[0]
        raise FileNotFoundError(f"Could not find segments_manifest.tsv under {final_root}")
    
    if MANIFEST_TSV is None:
        MANIFEST_TSV = auto_find_manifest(FINAL_ROOT, FEATURE_MODE, MINLEN)
    
    print("MANIFEST_TSV:", MANIFEST_TSV)
    manifest = pd.read_csv(MANIFEST_TSV, sep="\t")
    
    if "run" not in manifest.columns or "seg_path" not in manifest.columns:
        raise ValueError("Expected manifest columns: 'run', 'seg_path'")
    
    def parse_subject(run: str) -> str:
        parts = str(run).split("_")
        for p in parts:
            if p.startswith("sub-"):
                return p
        return parts[0]
    
    manifest["subject"] = manifest["run"].apply(parse_subject)
    
    # deterministic ordering
    sort_cols = ["subject", "run"]
    if "seg_id" in manifest.columns:
        sort_cols.append("seg_id")
    manifest = manifest.sort_values(sort_cols).reset_index(drop=True)
    
    SEG_ROOT = Path(MANIFEST_TSV).parent
    def resolve_seg_path(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else (SEG_ROOT / pp)
    
    manifest["seg_abs"] = [resolve_seg_path(p) for p in manifest["seg_path"].tolist()]
    missing = manifest.loc[~manifest["seg_abs"].apply(lambda p: p.exists())]
    if len(missing):
        display(missing.head(10))
        raise FileNotFoundError("Missing seg files referenced by manifest.")
    
    subjects = sorted(manifest["subject"].unique().tolist())
    runs = manifest["run"].tolist()
    
    print("n_subjects:", len(subjects))
    print("n_runs:", len(pd.unique(runs)))
    print("n_segments:", len(manifest))
    
    run_meta = dict(
        pipeline=PIPELINE_TAG,
        date=PIPELINE_DATE,
        K_FINAL=K_FINAL,
        DATA_VARIANT=DATA_VARIANT,
        FEATURE_MODE=FEATURE_MODE,
        MINLEN=MINLEN,
        FINAL_ROOT=str(FINAL_ROOT),
        OUT_ROOT=str(OUT_ROOT),
        N_PARCELS=N_PARCELS,
        TR_SEC=TR_SEC,
        LAGS_TR=LAGS_TR,
        D_TOTAL=D_TOTAL,
        SEQ_LEN=SEQ_LEN,
        STEP_SIZE=STEP_SIZE,
        BATCH_SIZE=BATCH_SIZE,
        REBATCH_DROP_REMAINDER_DEFAULT=REBATCH_DROP_REMAINDER_DEFAULT,
        SHUFFLE_BUFFER=SHUFFLE_BUFFER,
        N_BOLD_PCS=N_BOLD_PCS,
        N_EEG_PCS=N_EEG_PCS,
        LEARNING_RATE=LEARNING_RATE,
        N_EPOCHS=N_EPOCHS,
        DIAGONAL_COVS=DIAGONAL_COVS,
        COV_EPS=COV_EPS,
        INIT_TAKE=INIT_TAKE,
        INIT_EPOCHS=INIT_EPOCHS,
        SEEDS=SEEDS,
        USE_RUNWISE_ZSCORE=USE_RUNWISE_ZSCORE,
        TWO_STAGE_TRAIN_TRANS=TWO_STAGE_TRAIN_TRANS,
        STAGE1_EPOCH_FRAC=STAGE1_EPOCH_FRAC,
        TOPM_REFINES=TOPM_REFINES,
        FO_MAX_THRESH=FO_MAX_THRESH,
        FO_ACTIVE_THRESH=FO_ACTIVE_THRESH,
        MIN_ACTIVE_STATES_BASE=MIN_ACTIVE_STATES_BASE,
        FE_BETTER=FE_BETTER,
        DISABLE_XLA_AT_IMPORT=DISABLE_XLA_AT_IMPORT,
        FORCE_EAGER=FORCE_EAGER,
        DISABLE_PREFETCH=DISABLE_PREFETCH,
        DISABLE_CALLBACKS=DISABLE_CALLBACKS,
        GPU_MEMORY_LIMIT_MB=GPU_MEMORY_LIMIT_MB,
        SAVE_GAMMA=SAVE_GAMMA,
        DO_VITERBI=DO_VITERBI,
    )
    (OUT_ROOT / "run_meta.json").write_text(json.dumps(run_meta, indent=2))
    print("Wrote:", OUT_ROOT / "run_meta.json")
    
    # ---- notebook cell 3 ----
    # =========================
    # Cell 2 — TF GPU config (PipelineD-style hardening)
    # =========================
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass
    
    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs:", gpus)
    
    if gpus:
        try:
            if GPU_MEMORY_LIMIT_MB is not None:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=int(GPU_MEMORY_LIMIT_MB))]
                )
                print("[INFO] GPU memory capped:", GPU_MEMORY_LIMIT_MB, "MB")
            else:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("[INFO] memory_growth enabled")
        except Exception as e:
            print("[WARN] GPU config:", e)
    else:
        print("[INFO] CPU-only")
    
    try:
        tf.config.optimizer.set_jit(False)
    except Exception:
        pass
    
    if FORCE_EAGER:
        tf.config.run_functions_eagerly(True)
        print("[INFO] FORCE_EAGER=True")
    else:
        print("[INFO] FORCE_EAGER=False")
    
    _ = tf.matmul(tf.random.normal((64,64)), tf.random.normal((64,64)))
    print("TF OK")
    
    # ---- notebook cell 4 ----
    # =========================
    # Cell 3 — Helpers (global preproc + datasets + signatures + collapse + matching)
    # =========================
    from scipy.optimize import linear_sum_assignment
    
    def gc_now():
        gc.collect()
    
    def stable_clear():
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass
        gc_now()
    
    def log_append(path: Path, line: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line.rstrip("\n") + "\n")
    
    def fe_key(fe: float):
        return fe if str(FE_BETTER).lower() == "lower" else -fe
    
    def load_segments_from_manifest(manifest_df: pd.DataFrame):
        Xs = []
        for p in manifest_df["seg_abs"].tolist():
            x = np.load(p).astype(np.float32)
            if x.ndim != 2 or x.shape[1] != D_TOTAL:
                raise ValueError(f"Bad segment shape {x.shape} for {p}")
            if not np.isfinite(x).all():
                raise ValueError(f"Non-finite values (NaN/Inf) found in {p}")
            Xs.append(x)
        return Xs
    
    def runwise_zscore_segments(X_list, run_ids, sl: slice):
        run_ids = np.asarray(run_ids, dtype=object)
        uniq = pd.unique(run_ids)
        mu, sd = {}, {}
        for r in uniq:
            idx = np.where(run_ids == r)[0]
            X = np.concatenate([X_list[i][:, sl] for i in idx], axis=0)
            m = X.mean(axis=0)
            s = X.std(axis=0, ddof=0)
            s[s < 1e-12] = 1.0
            mu[r] = m
            sd[r] = s
        out = []
        for X, r in zip(X_list, run_ids):
            Z = X.copy()
            Z[:, sl] = (Z[:, sl] - mu[r]) / sd[r]
            out.append(Z.astype(np.float32))
        return out
    
    # ---- Global standardize + PCA via SVD ----
    def fit_standardizer(X):
        mu = X.mean(axis=0)
        sd = X.std(axis=0, ddof=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        return mu.astype(np.float32), sd.astype(np.float32)
    
    def apply_standardizer(X, mu, sd):
        return ((X - mu) / sd).astype(np.float32)
    
    def fit_pca_svd(X, n_fixed):
        mu = X.mean(axis=0, keepdims=True)
        Xc = X - mu
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        n_comp = int(min(n_fixed, Vt.shape[0]))
        V = Vt[:n_comp].T.astype(np.float32)
        var = (S**2) / max(Xc.shape[0] - 1, 1)
        pve = float(np.sum(var[:n_comp]) / np.sum(var)) if np.sum(var) > 0 else np.nan
        return mu.ravel().astype(np.float32), V, pve
    
    def apply_pca(X, mu, V):
        return ((X - mu) @ V).astype(np.float32)
    
    def make_global_preproc(X_all_raw_list):
        Xall = np.concatenate(X_all_raw_list, axis=0)
        Xb, Xe = Xall[:, :D_BOLD], Xall[:, D_BOLD:]
    
        mu_b, sd_b = fit_standardizer(Xb)
        mu_e, sd_e = fit_standardizer(Xe)
    
        Xb_z = apply_standardizer(Xb, mu_b, sd_b)
        Xe_z = apply_standardizer(Xe, mu_e, sd_e)
    
        mu_pb, Vb, pve_b = fit_pca_svd(Xb_z, N_BOLD_PCS)
        mu_pe, Ve, pve_e = fit_pca_svd(Xe_z, N_EEG_PCS)
    
        params = dict(
            mu_b=mu_b, sd_b=sd_b,
            mu_e=mu_e, sd_e=sd_e,
            mu_pb=mu_pb, Vb=Vb,
            mu_pe=mu_pe, Ve=Ve,
        )
        meta = dict(
            pve_bold=float(pve_b),
            pve_eeg=float(pve_e),
            n_bold_pcs=int(Vb.shape[1]),
            n_eeg_pcs=int(Ve.shape[1]),
            D_pca=int(Vb.shape[1] + Ve.shape[1]),
        )
        return params, meta
    
    def apply_global_preproc(X, params):
        Xb = apply_standardizer(X[:, :D_BOLD], params["mu_b"], params["sd_b"])
        Xe = apply_standardizer(X[:, D_BOLD:], params["mu_e"], params["sd_e"])
        Xb_p = apply_pca(Xb, params["mu_pb"], params["Vb"])
        Xe_p = apply_pca(Xe, params["mu_pe"], params["Ve"])
        return np.concatenate([Xb_p, Xe_p], axis=1).astype(np.float32)
    
    def make_config(K, D):
        cfg = Config(
            n_states=int(K),
            n_channels=int(D),
            sequence_length=int(SEQ_LEN),
            learn_means=True,
            learn_covariances=True,
            learn_trans_prob=True,
            batch_size=int(BATCH_SIZE),
            learning_rate=float(LEARNING_RATE),
            n_epochs=int(N_EPOCHS),
            covariances_epsilon=float(COV_EPS),
        )
        try:
            cfg.covariance_matrix_type = "diag" if DIAGONAL_COVS else "full"
        except Exception:
            pass
        try:
            cfg.n_init = 1
        except Exception:
            pass
        return cfg
    
    def callbacks():
        return [] if DISABLE_CALLBACKS else []
    
    def as_tf_dataset(data: Data, shuffle: bool, drop_remainder: bool, repeat: bool = False):
        ds_obj = data.dataset(
            sequence_length=SEQ_LEN,
            step_size=STEP_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=False,
            concatenate=False,
        )
        if isinstance(ds_obj, (list, tuple)):
            ds_list = list(ds_obj)
            if len(ds_list) == 0:
                raise ValueError("Empty dataset list: no windows. Check SEQ_LEN/STEP_SIZE.")
            ds = ds_list[0]
            for d in ds_list[1:]:
                ds = ds.concatenate(d)
        else:
            ds = ds_obj
    
        if drop_remainder:
            try:
                ds = ds.unbatch().batch(int(BATCH_SIZE), drop_remainder=True)
            except Exception:
                pass
    
        if shuffle:
            try:
                ds = ds.shuffle(buffer_size=int(SHUFFLE_BUFFER), reshuffle_each_iteration=True)
            except Exception:
                pass
    
        if repeat:
            ds = ds.repeat()
    
        if not DISABLE_PREFETCH:
            try:
                ds = ds.prefetch(1)
            except Exception:
                pass
    
        return ds
    
    def free_energy(model, data: Data, drop_remainder: bool):
        ds = as_tf_dataset(data, shuffle=False, drop_remainder=drop_remainder, repeat=False)
        fe = model.free_energy(ds)
        if isinstance(fe, (list, tuple, np.ndarray)):
            fe = float(np.asarray(fe).ravel()[0])
        return float(fe)
    
    def normalize_alpha_list(alpha_like):
        if isinstance(alpha_like, (list, tuple)):
            out = []
            for a in alpha_like:
                a = np.asarray(a)
                if a.ndim == 2:
                    out.append(a)
                elif a.ndim == 3:
                    out.extend([a[i] for i in range(a.shape[0])])
                else:
                    raise ValueError(f"Unexpected alpha element shape {a.shape}")
            return out
        a = np.asarray(alpha_like)
        if a.ndim == 2: return [a]
        if a.ndim == 3: return [a[i] for i in range(a.shape[0])]
        raise ValueError(f"Unexpected alpha shape {a.shape}")
    
    def get_alpha_list(model, data: Data):
        if hasattr(model, "get_alpha"):
            return normalize_alpha_list(model.get_alpha(data, concatenate=False, verbose=0))
        if hasattr(model, "get_gamma"):
            return normalize_alpha_list(model.get_gamma(data, concatenate=False, verbose=0))
        raise AttributeError("Model lacks get_alpha/get_gamma")
    
    def summarize_alpha(alpha_list, K, eps=1e-12):
        alpha_list = normalize_alpha_list(alpha_list)
        tot_T = 0
        fo_num = np.zeros(int(K), dtype=np.float64)
        ent_sum_norm = 0.0
    
        for a in alpha_list:
            a = np.asarray(a, dtype=np.float64)
            tot_T += a.shape[0]
            fo_num += a.sum(axis=0)
    
            a_clip = np.clip(a, eps, 1.0)
            Ht = -(a_clip * np.log(a_clip)).sum(axis=1)
            ent_sum_norm += (Ht / np.log(K)).sum()
    
        fo = (fo_num / max(tot_T, 1))
        fo_max = float(np.max(fo)) if fo.size else np.nan
        ent_norm = float(ent_sum_norm / max(tot_T, 1))
        n_active = int(np.sum(fo > FO_ACTIVE_THRESH)) if fo.size else 0
        return fo.astype(np.float32), fo_max, ent_norm, n_active, int(tot_T)
    
    def fo_entropy_and_neff(fo, K, eps=1e-12):
        fo = np.asarray(fo, dtype=np.float64)
        fo = np.clip(fo, eps, 1.0)
        fo = fo / fo.sum()
        H = float(-(fo * np.log(fo)).sum())
        fo_entropy_norm = float(H / np.log(K))
        neff = float(np.exp(H))
        return fo_entropy_norm, neff
    
    def is_collapsed(fo_max, n_active, K):
        min_active = min(int(MIN_ACTIVE_STATES_BASE), int(K))
        if not np.isfinite(fo_max):
            return True
        if fo_max > FO_MAX_THRESH:
            return True
        if int(n_active) < int(min_active):
            return True
        return False
    
    def dwell_from_A(A, eps=1e-12):
        A = np.asarray(A, dtype=np.float64)
        Akk = np.clip(np.diag(A), 0.0, 1.0 - eps)
        return (1.0 / (1.0 - Akk)).astype(np.float32)
    
    # ---- Signatures + matching (PipelineD definition) ----
    def ensure_cov_3d(covs):
        covs = np.asarray(covs)
        if covs.ndim == 2:
            K, P = covs.shape
            out = np.zeros((K, P, P), dtype=covs.dtype)
            for k in range(K):
                np.fill_diagonal(out[k], covs[k])
            return out
        return covs
    
    def cov_to_corr_ut(C, eps=1e-12):
        C = np.asarray(C, dtype=np.float64)
        d = np.sqrt(np.clip(np.diag(C), eps, None))
        corr = C / (d[:, None] * d[None, :])
        iu = np.triu_indices(corr.shape[0], k=1)
        return corr[iu].astype(np.float32)
    
    def backproject_cov_bold(covs_pca, Vb, nbpc):
        covs_pca = ensure_cov_3d(covs_pca)
        K = covs_pca.shape[0]
        out = []
        for k in range(K):
            Cbb = covs_pca[k, :nbpc, :nbpc]
            out.append((Vb @ Cbb @ Vb.T).astype(np.float32))
        return np.stack(out, axis=0)
    
    def compute_signature_ut_boldcorr(covs_pca, Vb, nbpc):
        cov_bold = backproject_cov_bold(covs_pca, Vb, nbpc)
        sig_ut = np.stack([cov_to_corr_ut(cov_bold[k]) for k in range(cov_bold.shape[0])], axis=0)
        return sig_ut.astype(np.float32)
    
    def match_states_to_reference(sig, ref_sig):
        sig = np.asarray(sig, dtype=np.float64)
        ref_sig = np.asarray(ref_sig, dtype=np.float64)
        K = sig.shape[0]
        corr = np.corrcoef(sig, ref_sig)[:K, K:]
        corr = np.nan_to_num(corr, nan=-1.0, posinf=-1.0, neginf=-1.0)
        corr = np.clip(corr, -1.0, 1.0)
        row_ind, col_ind = linear_sum_assignment(-corr)
        inv = np.zeros(K, dtype=int)
        for r, c in zip(row_ind, col_ind):
            inv[c] = r
        per_state_corr = np.array([corr[inv[j], j] for j in range(K)], dtype=float)
        return inv, per_state_corr
    
    def safe_corr(a, b):
        a = np.asarray(a).ravel()
        b = np.asarray(b).ravel()
        if a.size == 0 or b.size == 0:
            return np.nan
        if np.all(a == a[0]) or np.all(b == b[0]):
            return np.nan
        return float(np.corrcoef(a, b)[0,1])
    
    # ---- Two-stage helpers (direct carryover structure from PipelineD) ----
    def get_keras_model_from_osl(m):
        for attr in ["model", "_model", "keras_model"]:
            if hasattr(m, attr):
                km = getattr(m, attr)
                if hasattr(km, "get_weights") and hasattr(km, "set_weights"):
                    return km
        if hasattr(m, "get_weights") and hasattr(m, "set_weights"):
            return m
        raise AttributeError("Could not locate underlying Keras model on osl-dynamics Model wrapper.")
    
    def safe_set_weights(dst_m, src_m):
        dst = get_keras_model_from_osl(dst_m)
        src = get_keras_model_from_osl(src_m)
        w_src = src.get_weights()
        try:
            dst.set_weights(w_src)
            return "full"
        except Exception:
            w_dst = dst.get_weights()
            w_new = []
            for a, b in zip(w_dst, w_src):
                w_new.append(b if (hasattr(a, "shape") and hasattr(b, "shape") and a.shape == b.shape) else a)
            try:
                dst.set_weights(w_new)
                return "shape_matched"
            except Exception:
                return "failed"
    
    def fit_one_stage(m, train_ds, steps_tr, callbacks_list):
        try:
            m.fit(train_ds, steps_per_epoch=int(steps_tr), callbacks=callbacks_list)
        except TypeError:
            m.fit(train_ds, steps_per_epoch=int(steps_tr))
    
    def fit_two_stage_full(K, D_pca, full_data, train_ds, steps_tr, init_take, outdir: Path):
        e1 = max(1, int(round(N_EPOCHS * float(STAGE1_EPOCH_FRAC))))
        e2 = max(1, int(N_EPOCHS - e1))
    
        # stage1: transitions frozen
        cfg1 = make_config(K, D_pca)
        cfg1.learn_trans_prob = False
        try: cfg1.n_epochs = int(e1)
        except Exception: pass
    
        m1 = Model(cfg1)
        try:
            try:
                m1.random_subset_initialization(full_data, take=float(init_take), n_epochs=int(INIT_EPOCHS), n_init=1)
            except TypeError:
                m1.random_subset_initialization(full_data, take=float(init_take), n_epochs=int(INIT_EPOCHS))
        except Exception as e:
            raise RuntimeError(f"stage1_init: {repr(e)}")
    
        fit_one_stage(m1, train_ds, steps_tr, callbacks())
    
        # stage2: transitions learnable (warm start)
        cfg2 = make_config(K, D_pca)
        cfg2.learn_trans_prob = True
        try: cfg2.n_epochs = int(e2)
        except Exception: pass
    
        m2 = Model(cfg2)
        try:
            try:
                m2.random_subset_initialization(full_data, take=float(init_take), n_epochs=1, n_init=1)
            except TypeError:
                m2.random_subset_initialization(full_data, take=float(init_take), n_epochs=1)
        except Exception as e:
            raise RuntimeError(f"stage2_build: {repr(e)}")
    
        copied = safe_set_weights(m2, m1)
        log_append(outdir / "two_stage_log.txt", f"weight_copy={copied}")
    
        fit_one_stage(m2, train_ds, steps_tr, callbacks())
        return m2
    
    # ---- notebook cell 5 ----
    # =========================
    # Cell 4 — Build FULL dataset (load + runwise zscore + global preproc + Data)
    # =========================
    X_all_raw = load_segments_from_manifest(manifest)
    
    if USE_RUNWISE_ZSCORE:
        run_ids = manifest["run"].tolist()
        X_all_raw = runwise_zscore_segments(X_all_raw, run_ids, slice(0, D_BOLD))
        X_all_raw = runwise_zscore_segments(X_all_raw, run_ids, slice(D_BOLD, D_TOTAL))
    
    params, meta = make_global_preproc(X_all_raw)
    (OUT_ROOT / "preproc_meta.json").write_text(json.dumps(meta, indent=2))
    
    X_all = [apply_global_preproc(x, params) for x in X_all_raw]
    
    Xcat = np.concatenate(X_all, axis=0)
    if not np.isfinite(Xcat).all():
        raise RuntimeError("Non-finite values after global preprocessing.")
    del Xcat
    gc_now()
    
    np.savez_compressed(
        OUT_ROOT / "preproc_params.npz",
        mu_b=params["mu_b"], sd_b=params["sd_b"],
        mu_e=params["mu_e"], sd_e=params["sd_e"],
        mu_pb=params["mu_pb"], Vb=params["Vb"],
        mu_pe=params["mu_pe"], Ve=params["Ve"],
    )
    
    full_data = Data(X_all)
    
    def count_windows(X_list):
        n = 0
        for x in X_list:
            T = int(x.shape[0])
            if T >= SEQ_LEN:
                n += 1 + (T - SEQ_LEN) // STEP_SIZE
        return int(n)
    
    def steps_from_windows(n_windows: int, drop_remainder: bool):
        if n_windows <= 0: return 0
        if drop_remainder: return int(n_windows // int(BATCH_SIZE))
        return int(math.ceil(n_windows / float(BATCH_SIZE)))
    
    nwin = count_windows(X_all)
    drop_rem = bool(REBATCH_DROP_REMAINDER_DEFAULT)
    steps = steps_from_windows(nwin, drop_remainder=drop_rem)
    if steps == 0:
        drop_rem = False
        steps = steps_from_windows(nwin, drop_remainder=drop_rem)
    
    print("Global PCA meta:", meta)
    print("windows:", nwin, "| steps_per_epoch:", steps, "| drop_remainder:", drop_rem)
    
    # ---- notebook cell 6 ----
    # =========================
    # Cell 5 — Seed screening (one-stage; fast) + per-seed artifact save
    # =========================
    K = int(K_FINAL)
    cfg = make_config(K, int(meta["D_pca"]))
    init_take = float(INIT_TAKE)
    nbpc = int(meta.get("n_bold_pcs", N_BOLD_PCS))
    
    seed_root = OUT_ROOT / "seeds"
    seed_root.mkdir(exist_ok=True)
    
    cand_log = OUT_ROOT / "seed_candidates.tsv"
    fail_log = OUT_ROOT / "seed_failures.tsv"
    cand_log.write_text("seed\tfe\tfo_max\tentropy_norm\tn_active\tfo_entropy\tneff\tcollapsed\n", encoding="utf-8")
    fail_log.write_text("seed\tstage\terr\n", encoding="utf-8")
    
    train_ds = as_tf_dataset(full_data, shuffle=True, drop_remainder=drop_rem, repeat=True)
    
    cands = []
    for seed in SEEDS:
        stable_clear()
        np.random.seed(int(seed))
        tf.random.set_seed(int(seed))
    
        sd = seed_root / f"seed_{seed:03d}"
        sd.mkdir(exist_ok=True)
    
        m = Model(cfg)
    
        try:
            try:
                m.random_subset_initialization(full_data, take=init_take, n_epochs=int(INIT_EPOCHS), n_init=1)
            except TypeError:
                m.random_subset_initialization(full_data, take=init_take, n_epochs=int(INIT_EPOCHS))
        except Exception as e:
            log_append(fail_log, f"{seed}\tinit\t{repr(e)}")
            continue
    
        try:
            fit_one_stage(m, train_ds, steps, callbacks())
        except Exception as e:
            log_append(fail_log, f"{seed}\tfit\t{repr(e)}")
            continue
    
        try:
            fe = free_energy(m, full_data, drop_remainder=drop_rem)
            alpha = get_alpha_list(m, full_data)
            fo, fo_max, ent_norm, n_active, totT = summarize_alpha(alpha, K)
            fo_ent, neff = fo_entropy_and_neff(fo, K)
            collapsed = is_collapsed(float(fo_max), int(n_active), K)
        except Exception as e:
            log_append(fail_log, f"{seed}\tmetrics\t{repr(e)}")
            continue
    
        c = dict(seed=int(seed), fe=float(fe), fo_max=float(fo_max), entropy_norm=float(ent_norm),
                 n_active=int(n_active), fo_entropy=float(fo_ent), neff=float(neff),
                 collapsed=bool(collapsed), total_T=int(totT))
        cands.append(c)
        log_append(cand_log, f"{seed}\t{fe}\t{fo_max}\t{ent_norm}\t{n_active}\t{fo_ent}\t{neff}\t{collapsed}")
    
        # Save per-seed artifacts needed for identifiability
        try:
            A = np.asarray(m.get_trans_prob(), dtype=np.float32)
            covs = ensure_cov_3d(np.asarray(m.get_covariances(), dtype=np.float32))
            sig = compute_signature_ut_boldcorr(covs, params["Vb"], nbpc)
    
            if not np.isfinite(A).all(): raise ValueError("A_nonfinite")
            if not np.isfinite(covs).all(): raise ValueError("covs_nonfinite")
            if not np.isfinite(sig).all(): raise ValueError("sig_nonfinite")
    
            np.save(sd / "trans_prob.npy", A)
            np.save(sd / "covs_pca.npy", covs)
            try:
                means = np.asarray(m.get_means(), dtype=np.float32)
                np.save(sd / "means_pca.npy", means)
            except Exception:
                pass
            np.save(sd / "state_signature_ut_boldcorr.npy", sig)
            (sd / "seed_metrics.json").write_text(json.dumps(c, indent=2))
        except Exception as e:
            log_append(fail_log, f"{seed}\tsave\t{repr(e)}")
    
    if len(cands) == 0:
        raise RuntimeError("No successful seeds. See seed_failures.tsv")
    
    non = [c for c in cands if not c["collapsed"]]
    pool = non if len(non) else cands
    
    ranked = sorted(pool, key=lambda c: (fe_key(c["fe"]), c["fo_max"], -c["n_active"], -c["neff"]))
    
    (OUT_ROOT / "candidates_index.json").write_text(json.dumps(ranked, indent=2))
    
    topM = ranked[:max(1, int(TOPM_REFINES))]
    (OUT_ROOT / "topM_seeds.json").write_text(json.dumps(topM, indent=2))
    
    print("Top 10 seeds:")
    for r in ranked[:10]:
        print(r)
    
    print("TOPM:", [t["seed"] for t in topM])
    
    # ---- notebook cell 7 ----
    # =========================
    # Cell 6 — Two-stage refit on TOP-M seeds + numeric guards + choose final best seed
    # =========================
    topM = json.loads((OUT_ROOT / "topM_seeds.json").read_text())
    
    refit_log = OUT_ROOT / "refit_attempts.tsv"
    refit_log.write_text("rank\tseed\tstatus\treason\tfe\tfo_max\tn_active\tneff\n", encoding="utf-8")
    
    final_dir = OUT_ROOT / "final"
    final_dir.mkdir(exist_ok=True)
    
    best_model = None
    best_info  = None
    refit_results = []
    
    for rank_i, cand in enumerate(topM, start=1):
        seed = int(cand["seed"])
        try_dir = final_dir / f"refit_try_seed{seed:03d}"
        try_dir.mkdir(parents=True, exist_ok=True)
    
        stable_clear()
        np.random.seed(seed)
        tf.random.set_seed(seed)
    
        try:
            if TWO_STAGE_TRAIN_TRANS:
                m = fit_two_stage_full(
                    K=int(K_FINAL),
                    D_pca=int(meta["D_pca"]),
                    full_data=full_data,
                    train_ds=train_ds,
                    steps_tr=int(steps),
                    init_take=float(INIT_TAKE),
                    outdir=try_dir,
                )
            else:
                m = Model(cfg)
                try:
                    try:
                        m.random_subset_initialization(full_data, take=float(INIT_TAKE), n_epochs=int(INIT_EPOCHS), n_init=1)
                    except TypeError:
                        m.random_subset_initialization(full_data, take=float(INIT_TAKE), n_epochs=int(INIT_EPOCHS))
                except Exception as e:
                    raise RuntimeError(f"init: {repr(e)}")
                fit_one_stage(m, train_ds, int(steps), callbacks())
    
            A = np.asarray(m.get_trans_prob(), dtype=np.float32)
            covs = ensure_cov_3d(np.asarray(m.get_covariances(), dtype=np.float32))
            sig = compute_signature_ut_boldcorr(covs, params["Vb"], nbpc)
    
            if not np.isfinite(A).all():   raise ValueError("bad_A_nonfinite")
            if not np.isfinite(covs).all():raise ValueError("bad_covs_nonfinite")
            if not np.isfinite(sig).all(): raise ValueError("bad_sig_nonfinite")
    
            fe = free_energy(m, full_data, drop_remainder=drop_rem)
            alpha = get_alpha_list(m, full_data)
            fo, fo_max, ent_norm, n_active, totT = summarize_alpha(alpha, K_FINAL)
            fo_ent, neff = fo_entropy_and_neff(fo, K_FINAL)
            collapsed = is_collapsed(float(fo_max), int(n_active), K_FINAL)
    
            info = dict(
                seed=seed, rank=rank_i,
                fe=float(fe),
                fo=fo.tolist(),
                fo_max=float(fo_max),
                entropy_norm=float(ent_norm),
                n_active=int(n_active),
                fo_entropy=float(fo_ent),
                neff=float(neff),
                collapsed=bool(collapsed),
                total_T=int(totT),
            )
            refit_results.append(info)
    
            log_append(refit_log, f"{rank_i}\t{seed}\tok\t-\t{fe}\t{fo_max}\t{n_active}\t{neff}")
    
            # save trial artifacts
            np.save(try_dir / "trans_prob.npy", A)
            np.save(try_dir / "covs_pca.npy", covs)
            np.save(try_dir / "state_signature_ut_boldcorr.npy", sig)
            try:
                means = np.asarray(m.get_means(), dtype=np.float32)
                np.save(try_dir / "means_pca.npy", means)
            except Exception:
                pass
            (try_dir / "refit_metrics.json").write_text(json.dumps(info, indent=2))
    
            # selection among refits: best FE among non-collapsed
            if best_info is None:
                best_info, best_model = info, m
            else:
                if (best_info["collapsed"] and not info["collapsed"]):
                    best_info, best_model = info, m
                elif (best_info["collapsed"] == info["collapsed"]):
                    if fe_key(info["fe"]) < fe_key(best_info["fe"]):
                        best_info, best_model = info, m
    
        except Exception as e:
            log_append(refit_log, f"{rank_i}\t{seed}\tfail\t{repr(e)}\t\t\t\t")
            continue
    
    if best_info is None:
        raise RuntimeError("All TOP-M refits failed. Inspect refit_attempts.tsv and seed_failures.tsv")
    
    (final_dir / "refit_results.json").write_text(json.dumps(refit_results, indent=2))
    (final_dir / "best_seed.json").write_text(json.dumps(best_info, indent=2))
    print("BEST (post-refit):", best_info)
    
    # Save final artifacts (best model still in memory)
    A = np.asarray(best_model.get_trans_prob(), dtype=np.float32)
    covs = ensure_cov_3d(np.asarray(best_model.get_covariances(), dtype=np.float32))
    sig = compute_signature_ut_boldcorr(covs, params["Vb"], nbpc)
    
    np.save(final_dir / "trans_prob.npy", A)
    np.save(final_dir / "covs_pca.npy", covs)
    np.save(final_dir / "state_signature_ut_boldcorr.npy", sig)
    try:
        means = np.asarray(best_model.get_means(), dtype=np.float32)
        np.save(final_dir / "means_pca.npy", means)
    except Exception:
        pass
    
    print("Saved final artifacts to:", final_dir)
    
    # ---- notebook cell 8 ----
    # =========================
    # Cell 7 — Decode Gamma/Viterbi per run + compute run/subject metrics
    # =========================
    from collections import defaultdict
    
    gamma_root = OUT_ROOT / "gamma"
    viterbi_root = OUT_ROOT / "viterbi"
    gamma_root.mkdir(exist_ok=True)
    viterbi_root.mkdir(exist_ok=True)
    
    run_to_rows = defaultdict(list)
    for i, row in manifest.iterrows():
        run_to_rows[str(row["run"])].append(i)
    
    def make_run_data(run_name: str):
        idxs = run_to_rows[run_name]
        Xr = [X_all[i] for i in idxs]  # already preprocessed
        return Data(Xr), idxs
    
    def try_get_viterbi(model, data: Data):
        for fn in ["get_viterbi_path", "get_viterbi", "viterbi_path"]:
            if hasattr(model, fn):
                try:
                    return getattr(model, fn)(data, concatenate=False)
                except TypeError:
                    return getattr(model, fn)(data)
                except Exception:
                    pass
        return None
    
    run_rows = []
    
    for run_name in sorted(run_to_rows.keys()):
        rd, idxs = make_run_data(run_name)
    
        alpha_list = get_alpha_list(best_model, rd)
        fo, fo_max, ent_norm, n_active, totT = summarize_alpha(alpha_list, K_FINAL)
        fo_ent, neff = fo_entropy_and_neff(fo, K_FINAL)
    
        if SAVE_GAMMA:
            rdir = gamma_root / run_name
            rdir.mkdir(parents=True, exist_ok=True)
            for seg_local_i, a in enumerate(alpha_list):
                np.save(rdir / f"gamma_seg{seg_local_i:04d}.npy", np.asarray(a, dtype=np.float32))
    
        if DO_VITERBI:
            vpath_list = try_get_viterbi(best_model, rd)
            if vpath_list is not None:
                vdir = viterbi_root / run_name
                vdir.mkdir(parents=True, exist_ok=True)
                vpath_list = normalize_alpha_list(vpath_list) if not isinstance(vpath_list, list) else vpath_list
                for seg_local_i, vp in enumerate(vpath_list):
                    np.save(vdir / f"viterbi_seg{seg_local_i:04d}.npy", np.asarray(vp, dtype=np.int16))
    
        subject = parse_subject(run_name)
        run_rows.append(dict(
            subject=subject,
            run=run_name,
            total_T=int(totT),
            FO_max=float(fo_max),
            n_active=int(n_active),
            neff=float(neff),
            entropy_mean_norm=float(ent_norm),
            fo_entropy=float(fo_ent),
            **{f"FO_s{i+1:02d}": float(fo[i]) for i in range(K_FINAL)},
        ))
    
    run_metrics = pd.DataFrame(run_rows).sort_values(["subject","run"]).reset_index(drop=True)
    run_metrics.to_csv(OUT_ROOT / "run_metrics.tsv", sep="\t", index=False)
    print("Wrote:", OUT_ROOT / "run_metrics.tsv")
    
    # subject-level (weighted by total_T)
    subj_metrics = []
    for subject, df in run_metrics.groupby("subject"):
        w = df["total_T"].values.astype(float)
        w = w / max(w.sum(), 1.0)
        fo_cols = [f"FO_s{i+1:02d}" for i in range(K_FINAL)]
        FO = (df[fo_cols].values * w[:, None]).sum(axis=0)
    
        FO_max = float(np.max(FO))
        n_active = int(np.sum(FO > FO_ACTIVE_THRESH))
        fo_ent, neff = fo_entropy_and_neff(FO, K_FINAL)
    
        subj_metrics.append(dict(
            subject=subject,
            n_runs=int(len(df)),
            total_T=int(df["total_T"].sum()),
            FO_max=float(FO_max),
            n_active=int(n_active),
            neff=float(neff),
            fo_entropy=float(fo_ent),
            **{f"FO_s{i+1:02d}": float(FO[i]) for i in range(K_FINAL)},
        ))
    
    subject_metrics = pd.DataFrame(subj_metrics).sort_values("subject").reset_index(drop=True)
    subject_metrics.to_csv(OUT_ROOT / "subject_metrics.tsv", sep="\t", index=False)
    print("Wrote:", OUT_ROOT / "subject_metrics.tsv")
    
    dwellA_TR = dwell_from_A(np.load(final_dir / "trans_prob.npy"))
    pd.DataFrame({
        "state":[f"s{i+1:02d}" for i in range(K_FINAL)],
        "dwell_A_TR":dwellA_TR,
        "dwell_A_sec":dwellA_TR*TR_SEC
    }).to_csv(OUT_ROOT / "dwell_from_A.tsv", sep="\t", index=False)
    print("Wrote:", OUT_ROOT / "dwell_from_A.tsv")
    
    # ---- notebook cell 9 ----
    # =========================
    # Cell 8 — Seed-to-seed identifiability (signatures + A)
    # =========================
    seed_dirs = sorted([p for p in (OUT_ROOT / "seeds").glob("seed_*") if p.is_dir()])
    ok_seeds, sigs, As = [], [], []
    
    for sd in seed_dirs:
        try:
            s = int(sd.name.split("_")[1])
            sig = np.load(sd / "state_signature_ut_boldcorr.npy")
            A = np.load(sd / "trans_prob.npy")
            if sig.shape[0] != K_FINAL or A.shape != (K_FINAL, K_FINAL):
                continue
            if not (np.isfinite(sig).all() and np.isfinite(A).all()):
                continue
            ok_seeds.append(s)
            sigs.append(sig)
            As.append(A)
        except Exception:
            continue
    
    print("Seeds with valid artifacts:", len(ok_seeds))
    
    if len(ok_seeds) >= 2:
        ref_sig = np.load(final_dir / "state_signature_ut_boldcorr.npy")
    
        sigs_m, As_m = [], []
        rows = []
        for s, sig_i, A_i in zip(ok_seeds, sigs, As):
            inv, per_state_corr = match_states_to_reference(sig_i, ref_sig)
            sigs_m.append(sig_i[inv, :])
            As_m.append(A_i[np.ix_(inv, inv)])
            rows.append({
                "seed": s,
                "mean_state_corr": float(np.nanmean(per_state_corr)),
                **{f"state_corr_s{i+1:02d}": float(per_state_corr[i]) for i in range(K_FINAL)}
            })
    
        pd.DataFrame(rows).to_csv(OUT_ROOT / "seed_matching_scores.tsv", sep="\t", index=False)
    
        N = len(ok_seeds)
        sim_sig = np.full((N, N), np.nan)
        sim_A = np.full((N, N), np.nan)
    
        for i in range(N):
            for j in range(N):
                cs = [safe_corr(sigs_m[i][k], sigs_m[j][k]) for k in range(K_FINAL)]
                sim_sig[i, j] = float(np.nanmean(cs))
                sim_A[i, j] = safe_corr(As_m[i].ravel(), As_m[j].ravel())
    
        pd.DataFrame(sim_sig, index=ok_seeds, columns=ok_seeds).to_csv(OUT_ROOT / "seed_sim_signature.tsv", sep="\t")
        pd.DataFrame(sim_A, index=ok_seeds, columns=ok_seeds).to_csv(OUT_ROOT / "seed_sim_A.tsv", sep="\t")
    
        A_stack = np.stack(As_m, axis=0).astype(float)
        pd.DataFrame(np.nanmean(A_stack, axis=0)).to_csv(OUT_ROOT / "seed_A_mean.tsv", sep="\t", index=False)
        pd.DataFrame(np.nanstd(A_stack, axis=0, ddof=1)).to_csv(OUT_ROOT / "seed_A_std.tsv", sep="\t", index=False)
    
        print("Wrote seed stability tables to OUT_ROOT.")
    else:
        print("Not enough seeds for seed-to-seed stability.")
    
    final_dir = locals().get("final_dir", OUT_ROOT / "final")
    gamma_dir = locals().get("gamma_root", OUT_ROOT / "gamma")
    viterbi_dir = locals().get("viterbi_root", OUT_ROOT / "viterbi")
    return {
        "out_root": str(OUT_ROOT),
        "final_dir": str(final_dir),
        "gamma_dir": str(gamma_dir),
        "viterbi_dir": str(viterbi_dir),
        "manifest_tsv": str(MANIFEST_TSV),
        "qc_summary_json": str(OUT_ROOT / "qc_summary.json"),
        "run_metrics_tsv": str(OUT_ROOT / "run_metrics.tsv"),
        "subject_metrics_tsv": str(OUT_ROOT / "subject_metrics.tsv"),
    }
