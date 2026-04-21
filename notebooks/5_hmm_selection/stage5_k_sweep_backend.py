"""Backend functions for the public Stage-5 broad K-sweep workflow.

This module holds the active Python implementation behind
`step50_run_loso_k_sweep_model_selection.ipynb`.

It preserves the original LOSO K-sweep behavior while moving the dense
TensorFlow and `osl_dynamics` machinery out of the public notebook and into a
normal Python backend module.

The main outputs are the per-fold free-energy tables, candidate summaries, and
the manuscript-facing K-screening recommendation files that feed Step 52.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stage5_backend_common import auto_find_manifest, parse_subject_from_run, resolve_segment_path


@dataclass
class KSweepBackendConfig:
    """User-facing and frozen settings for the Stage-5 broad K sweep."""

    segments_root: str | Path
    model_selection_root: str | Path
    data_variant: str = "intermediate"
    feature_mode: str = "nolags"
    minlen: int = 15
    manifest_tsv: str | Path | None = None
    k_grid: list[int] | None = None
    max_new_pairs_per_run: int | None = 15
    gpu_memory_limit_mb: int | None = 4096
    debug_max_folds: int | None = None
    debug_k_grid: list[int] | None = None
    debug_seeds: list[int] | None = None
    n_parcels: int = 200
    tr_sec: float = 2.1
    seq_len: int = 10
    step_size: int = 1
    batch_size: int = 16
    shuffle_buffer: int = 2048
    rebatch_drop_remainder: bool = True
    n_bold_pcs: int = 40
    n_eeg_pcs: int = 40
    learning_rate: float = 1e-3
    n_epochs_cv: int = 60
    cv_learn_means: bool = True
    cv_learn_covs: bool = True
    cv_learn_trans: bool = True
    cv_diagonal_covs: bool = False
    cov_eps: float = 1e-6
    init_take: float = 0.30
    init_epochs: int = 5
    bigk_thresh: int = 6
    init_take_bigk: float = 0.20
    seeds: list[int] = field(default_factory=lambda: [11, 23, 37, 53, 71])
    val_subject_policy: str = "max_segments"
    use_runwise_zscore: bool = True
    fo_max_thresh: float = 0.95
    entropy_norm_min: float = 0.05
    fo_active_thresh: float = 0.01
    min_active_states_base: int = 3
    disable_xla_at_import: bool = True
    force_eager: bool = False
    disable_prefetch: bool = True
    disable_callbacks: bool = True
    resume_if_results_exist: bool = True


def run_loso_k_sweep_backend(config: KSweepBackendConfig) -> dict[str, Any]:
    """Run the active Stage-5 broad K-sweep backend without notebook patching."""

    import gc
    import json
    import math
    import os

    if config.disable_xla_at_import:
        os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=-1"
        os.environ["TF_XLA_ENABLE_XLA_DEVICES"] = "0"
        os.environ["XLA_FLAGS"] = ""

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from scipy.stats import ttest_rel, wilcoxon
    from osl_dynamics.data import Data
    from osl_dynamics.models.hmm import Config as HMMConfig, Model

    DATA_VARIANT = str(config.data_variant)
    FEATURE_MODE = str(config.feature_mode)
    MINLEN = int(config.minlen)
    K_GRID = list(config.k_grid if config.k_grid is not None else range(2, 13))
    FINAL_ROOT = Path(config.segments_root) / DATA_VARIANT
    MANIFEST_TSV = Path(config.manifest_tsv) if config.manifest_tsv is not None else None
    OUT_ROOT = Path(config.model_selection_root) / f"{DATA_VARIANT}_{FEATURE_MODE}_minlen{MINLEN}"

    N_PARCELS = int(config.n_parcels)
    TR_SEC = float(config.tr_sec)
    if FEATURE_MODE.lower() == "lags":
        LAGS_TR = [-1, 0, 1]
    elif FEATURE_MODE.lower() == "nolags":
        LAGS_TR = [0]
    else:
        raise ValueError("FEATURE_MODE must be 'lags' or 'nolags'")

    D_BOLD = N_PARCELS
    D_EEG = N_PARCELS * len(LAGS_TR)
    D_TOTAL = D_BOLD + D_EEG

    SEQ_LEN = int(config.seq_len)
    STEP_SIZE = int(config.step_size)
    BATCH_SIZE = int(config.batch_size)
    SHUFFLE_BUFFER = int(config.shuffle_buffer)
    REBATCH_DROP_REMAINDER = bool(config.rebatch_drop_remainder)

    N_BOLD_PCS = int(config.n_bold_pcs)
    N_EEG_PCS = int(config.n_eeg_pcs)

    LEARNING_RATE = float(config.learning_rate)
    N_EPOCHS_CV = int(config.n_epochs_cv)
    CV_LEARN_MEANS = bool(config.cv_learn_means)
    CV_LEARN_COVS = bool(config.cv_learn_covs)
    CV_LEARN_TRANS = bool(config.cv_learn_trans)
    CV_DIAGONAL_COVS = bool(config.cv_diagonal_covs)
    COV_EPS = float(config.cov_eps)

    INIT_TAKE = float(config.init_take)
    INIT_EPOCHS = int(config.init_epochs)
    BIGK_THRESH = int(config.bigk_thresh)
    INIT_TAKE_BIGK = float(config.init_take_bigk)

    SEEDS = list(config.seeds)
    VAL_SUBJECT_POLICY = str(config.val_subject_policy)
    USE_RUNWISE_ZSCORE = bool(config.use_runwise_zscore)

    FO_MAX_THRESH = float(config.fo_max_thresh)
    ENTROPY_NORM_MIN = float(config.entropy_norm_min)
    FO_ACTIVE_THRESH = float(config.fo_active_thresh)
    MIN_ACTIVE_STATES_BASE = int(config.min_active_states_base)

    FORCE_EAGER = bool(config.force_eager)
    DISABLE_PREFETCH = bool(config.disable_prefetch)
    DISABLE_CALLBACKS = bool(config.disable_callbacks)
    GPU_MEMORY_LIMIT_MB = config.gpu_memory_limit_mb

    RESUME_IF_RESULTS_EXIST = bool(config.resume_if_results_exist)
    MAX_NEW_PAIRS_PER_RUN = config.max_new_pairs_per_run
    DEBUG_MAX_FOLDS = config.debug_max_folds
    DEBUG_K_GRID = config.debug_k_grid
    DEBUG_SEEDS = config.debug_seeds

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CV_TSV = OUT_ROOT / "cv_results.tsv"
    CAND_TSV = OUT_ROOT / "cv_candidates_long.tsv"
    FOLD_META_TSV = OUT_ROOT / "fold_meta.tsv"

    if MANIFEST_TSV is None:
        MANIFEST_TSV = auto_find_manifest(FINAL_ROOT, FEATURE_MODE, MINLEN)

    print("FINAL_ROOT:", FINAL_ROOT)
    print("MANIFEST_TSV:", MANIFEST_TSV)
    print("OUT_ROOT:", OUT_ROOT)
    print(f"D_TOTAL={D_TOTAL} (BOLD={D_BOLD}, EEG={D_EEG})")
    print("K_GRID:", K_GRID)
    print("SEQ_LEN:", SEQ_LEN, "STEP_SIZE:", STEP_SIZE, "BATCH_SIZE:", BATCH_SIZE)
    print("SEEDS:", SEEDS)
    print("MAX_NEW_PAIRS_PER_RUN:", MAX_NEW_PAIRS_PER_RUN)

    manifest = pd.read_csv(MANIFEST_TSV, sep="\t")
    print("Rows:", len(manifest))
    print("Columns:", list(manifest.columns))

    if "run" not in manifest.columns or "seg_path" not in manifest.columns:
        raise ValueError("Expected manifest columns: 'run', 'seg_path'. Please confirm header.")

    manifest["subject"] = manifest["run"].apply(parse_subject_from_run)
    sort_cols = ["subject", "run"]
    if "seg_id" in manifest.columns:
        sort_cols.append("seg_id")
    manifest = manifest.sort_values(sort_cols).reset_index(drop=True)

    seg_root = MANIFEST_TSV.parent
    seg_paths = [resolve_segment_path(seg_root, seg_path) for seg_path in manifest["seg_path"].tolist()]
    missing = [path for path in seg_paths if not path.exists()]
    if missing:
        print("Missing seg files (first 10):", missing[:10])
        raise FileNotFoundError("Some segment files do not exist.")

    all_X = [np.load(path).astype(np.float32) for path in seg_paths]
    manifest["X_index"] = np.arange(len(manifest), dtype=int)

    example_segment = all_X[0]
    print("Example segment shape:", example_segment.shape)
    if example_segment.shape[1] != D_TOTAL:
        raise ValueError(f"Expected D={D_TOTAL}, got {example_segment.shape[1]}")
    print(f"Loaded {len(all_X)} segments | total TR = {int(np.sum([x.shape[0] for x in all_X]))}")

    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass

    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs visible to TF:", gpus)

    if gpus:
        try:
            if GPU_MEMORY_LIMIT_MB is not None:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=int(GPU_MEMORY_LIMIT_MB))],
                )
                print(f"[INFO] Capped GPU memory to {int(GPU_MEMORY_LIMIT_MB)} MB.")
            else:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("[INFO] Enabled memory_growth (no explicit cap).")
        except Exception as exc:
            print("[WARN] GPU config:", exc)
    else:
        print("[INFO] Running CPU-only.")

    try:
        tf.config.optimizer.set_jit(False)
        print("[INFO] tf.config.optimizer.set_jit(False)")
    except Exception as exc:
        print("[WARN] set_jit:", exc)

    if FORCE_EAGER:
        tf.config.run_functions_eagerly(True)
        print("[INFO] FORCE_EAGER=True")
    else:
        print("[INFO] FORCE_EAGER=False (recommended for memory efficiency)")

    sanity = tf.random.normal((64, 64))
    _ = tf.matmul(sanity, sanity)
    print("TF OK")

    def gc_now() -> None:
        gc.collect()

    def count_windows_for_segments(X_list: list[np.ndarray]) -> int:
        n_windows = 0
        for X in X_list:
            T = int(X.shape[0])
            if T >= SEQ_LEN:
                n_windows += 1 + (T - SEQ_LEN) // STEP_SIZE
        return int(n_windows)

    def steps_from_windows(n_windows: int) -> int:
        if n_windows <= 0:
            return 0
        if REBATCH_DROP_REMAINDER:
            return int(n_windows // int(BATCH_SIZE))
        return int(math.ceil(n_windows / float(BATCH_SIZE)))

    def runwise_zscore_segments(X_list: list[np.ndarray], run_ids: list[str], sl: slice) -> list[np.ndarray]:
        run_ids_arr = np.asarray(run_ids, dtype=object)
        uniq = pd.unique(run_ids_arr)
        mu: dict[str, np.ndarray] = {}
        sd: dict[str, np.ndarray] = {}
        for run_id in uniq:
            idx = np.where(run_ids_arr == run_id)[0]
            X = np.concatenate([X_list[i][:, sl] for i in idx], axis=0)
            mean = X.mean(axis=0)
            std = X.std(axis=0, ddof=0)
            std[std < 1e-12] = 1.0
            mu[str(run_id)] = mean
            sd[str(run_id)] = std

        out = []
        for X, run_id in zip(X_list, run_ids_arr):
            Z = X.copy()
            Z[:, sl] = (Z[:, sl] - mu[str(run_id)]) / sd[str(run_id)]
            out.append(Z.astype(np.float32))
        return out

    def fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = X.mean(axis=0)
        sd = X.std(axis=0, ddof=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        return mu.astype(np.float32), sd.astype(np.float32)

    def apply_standardizer(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
        return ((X - mu) / sd).astype(np.float32)

    def fit_pca(X: np.ndarray, n_fixed: int) -> tuple[np.ndarray, np.ndarray, float]:
        mu = X.mean(axis=0, keepdims=True)
        Xc = X - mu
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        n_comp = int(min(n_fixed, Vt.shape[0]))
        V = Vt[:n_comp].T.astype(np.float32)
        var = (S**2) / max(Xc.shape[0] - 1, 1)
        pve = float(np.sum(var[:n_comp]) / np.sum(var)) if np.sum(var) > 0 else np.nan
        return mu.ravel().astype(np.float32), V, pve

    def apply_pca(X: np.ndarray, mu: np.ndarray, V: np.ndarray) -> np.ndarray:
        return ((X - mu) @ V).astype(np.float32)

    def make_fold_preproc(X_tr_raw_list: list[np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        Xtr = np.concatenate(X_tr_raw_list, axis=0)
        Xb, Xe = Xtr[:, :D_BOLD], Xtr[:, D_BOLD:]

        mu_b, sd_b = fit_standardizer(Xb)
        mu_e, sd_e = fit_standardizer(Xe)
        Xb_z = apply_standardizer(Xb, mu_b, sd_b)
        Xe_z = apply_standardizer(Xe, mu_e, sd_e)

        mu_pb, Vb, pve_b = fit_pca(Xb_z, N_BOLD_PCS)
        mu_pe, Ve, pve_e = fit_pca(Xe_z, N_EEG_PCS)

        params = dict(mu_b=mu_b, sd_b=sd_b, mu_e=mu_e, sd_e=sd_e, mu_pb=mu_pb, Vb=Vb, mu_pe=mu_pe, Ve=Ve)
        meta = dict(
            pve_bold=pve_b,
            pve_eeg=pve_e,
            n_bold_pcs=int(Vb.shape[1]),
            n_eeg_pcs=int(Ve.shape[1]),
            D_pca=int(Vb.shape[1] + Ve.shape[1]),
        )
        return params, meta

    def apply_fold_preproc(X: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
        Xb = apply_standardizer(X[:, :D_BOLD], params["mu_b"], params["sd_b"])
        Xe = apply_standardizer(X[:, D_BOLD:], params["mu_e"], params["sd_e"])
        Xb_p = apply_pca(Xb, params["mu_pb"], params["Vb"])
        Xe_p = apply_pca(Xe, params["mu_pe"], params["Ve"])
        return np.concatenate([Xb_p, Xe_p], axis=1).astype(np.float32)

    def choose_val_subject(train_df: pd.DataFrame) -> str:
        if VAL_SUBJECT_POLICY == "max_segments":
            return str(train_df.groupby("subject").size().sort_values(ascending=False).index[0])
        return str(sorted(train_df["subject"].unique().tolist())[0])

    def make_config(K: int, D: int) -> HMMConfig:
        hmm_config = HMMConfig(
            n_states=K,
            n_channels=D,
            sequence_length=SEQ_LEN,
            learn_means=CV_LEARN_MEANS,
            learn_covariances=CV_LEARN_COVS,
            learn_trans_prob=CV_LEARN_TRANS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            n_epochs=N_EPOCHS_CV,
            covariances_epsilon=COV_EPS,
        )
        try:
            hmm_config.covariance_matrix_type = "diag" if CV_DIAGONAL_COVS else "full"
        except Exception:
            pass
        try:
            hmm_config.n_init = 1
        except Exception:
            pass
        return hmm_config

    def callbacks() -> list[Any]:
        if DISABLE_CALLBACKS:
            return []
        return []

    def as_tf_dataset(data: Data, shuffle: bool):
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
                raise ValueError("Data.dataset returned empty list (no windows). Check SEQ_LEN/STEP_SIZE vs segment lengths.")
            ds = ds_list[0]
            for dataset in ds_list[1:]:
                ds = ds.concatenate(dataset)
        else:
            ds = ds_obj

        if REBATCH_DROP_REMAINDER:
            try:
                ds = ds.unbatch().batch(int(BATCH_SIZE), drop_remainder=True)
            except Exception:
                pass

        if shuffle:
            try:
                ds = ds.shuffle(buffer_size=int(SHUFFLE_BUFFER), reshuffle_each_iteration=True)
            except Exception:
                pass

        if not DISABLE_PREFETCH:
            try:
                ds = ds.prefetch(tf.data.AUTOTUNE)
            except Exception:
                pass
        return ds

    def free_energy(model: Model, data: Data) -> float:
        ds = as_tf_dataset(data, shuffle=False)
        fe = model.free_energy(ds)
        if isinstance(fe, (list, tuple, np.ndarray)):
            fe = float(np.asarray(fe).ravel()[0])
        return float(fe)

    def normalize_alpha_list(alpha_like: Any) -> list[np.ndarray]:
        if isinstance(alpha_like, (list, tuple)):
            out = []
            for alpha in alpha_like:
                alpha = np.asarray(alpha)
                if alpha.ndim == 2:
                    out.append(alpha)
                elif alpha.ndim == 3:
                    out.extend([alpha[i] for i in range(alpha.shape[0])])
                else:
                    raise ValueError(f"Unexpected alpha element ndim={alpha.ndim}, shape={alpha.shape}")
            return out
        alpha = np.asarray(alpha_like)
        if alpha.ndim == 2:
            return [alpha]
        if alpha.ndim == 3:
            return [alpha[i] for i in range(alpha.shape[0])]
        raise ValueError(f"Unexpected alpha ndim={alpha.ndim}, shape={alpha.shape}")

    def get_alpha_list(model: Model, data: Data) -> list[np.ndarray]:
        if hasattr(model, "get_alpha"):
            out = model.get_alpha(data, concatenate=False, verbose=0)
            return normalize_alpha_list(out)
        if hasattr(model, "get_gamma"):
            out = model.get_gamma(data, concatenate=False, verbose=0)
            return normalize_alpha_list(out)
        raise AttributeError("Model does not expose get_alpha/get_gamma.")

    def summarize_alpha(alpha_list: Any, K: int, eps: float = 1e-12) -> tuple[np.ndarray, float, float, int]:
        alpha_list = normalize_alpha_list(alpha_list)
        total_T = 0
        fo_num = np.zeros(K, dtype=np.float64)
        ent_sum_norm = 0.0
        for alpha in alpha_list:
            alpha = np.asarray(alpha, dtype=np.float64)
            if alpha.ndim != 2 or alpha.shape[1] != K:
                raise ValueError(f"Expected alpha shape (T,{K}); got {alpha.shape}")
            total_T += alpha.shape[0]
            fo_num += alpha.sum(axis=0)
            alpha_clip = np.clip(alpha, eps, 1.0)
            Ht = -(alpha_clip * np.log(alpha_clip)).sum(axis=1)
            ent_sum_norm += (Ht / np.log(K)).sum()
        fo = fo_num / max(total_T, 1)
        fo_max = float(np.max(fo)) if fo.size else np.nan
        ent_norm = float(ent_sum_norm / max(total_T, 1))
        n_active = int(np.sum(fo > FO_ACTIVE_THRESH)) if fo.size else 0
        return fo.astype(np.float32), fo_max, ent_norm, n_active

    def fo_entropy_and_neff(fo: np.ndarray, K: int, eps: float = 1e-12) -> tuple[float, float]:
        fo = np.asarray(fo, dtype=np.float64)
        if fo.ndim != 1 or fo.size != K:
            raise ValueError(f"Expected fo shape ({K},), got {fo.shape}")
        fo = np.clip(fo, eps, 1.0)
        fo = fo / fo.sum()
        H_nat = float(-(fo * np.log(fo)).sum())
        fo_entropy_norm = float(H_nat / np.log(K))
        neff = float(np.exp(H_nat))
        return fo_entropy_norm, neff

    def is_collapsed(fo_max: float, n_active: int, K: int) -> bool:
        min_active = min(int(MIN_ACTIVE_STATES_BASE), int(K))
        if not np.isfinite(fo_max):
            return True
        if fo_max > FO_MAX_THRESH:
            return True
        if int(n_active) < int(min_active):
            return True
        return False

    def choose_best_candidate(cands: list[dict[str, Any]]) -> dict[str, Any]:
        non = [candidate for candidate in cands if not candidate["collapsed"]]
        pool = non if len(non) else cands

        def key(candidate: dict[str, Any]) -> tuple[float, float, int, float]:
            fe = candidate.get("fe_val", np.nan)
            fe = fe if np.isfinite(fe) else np.inf
            fo_max = candidate.get("fo_max", np.nan)
            fo_max = fo_max if np.isfinite(fo_max) else np.inf
            n_active = int(candidate.get("n_active", 0))
            fo_entropy = candidate.get("fo_entropy", np.nan)
            fo_entropy = fo_entropy if np.isfinite(fo_entropy) else -np.inf
            return (fe, fo_max, -n_active, -fo_entropy)

        return sorted(pool, key=key)[0]

    def clear_session() -> None:
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass
        gc_now()

    def split_loso(df: pd.DataFrame):
        subs = sorted(df["subject"].unique().tolist())
        for fold_i, test_sub in enumerate(subs):
            train_idx = df.index[df["subject"] != test_sub].to_numpy()
            test_idx = df.index[df["subject"] == test_sub].to_numpy()
            yield fold_i, test_sub, train_idx, test_idx

    class _ChunkComplete(Exception):
        pass

    K_GRID_RUN = DEBUG_K_GRID if DEBUG_K_GRID is not None else K_GRID
    SEEDS_RUN = DEBUG_SEEDS if DEBUG_SEEDS is not None else SEEDS

    done: set[tuple[int, int]] = set()
    if RESUME_IF_RESULTS_EXIST and CV_TSV.exists():
        prev = pd.read_csv(CV_TSV, sep="\t")
        for _, row in prev.iterrows():
            done.add((int(row["fold"]), int(row["K"])))
        print(f"[RESUME] existing rows={len(prev)} | done pairs={len(done)}")
    else:
        print("[START] no previous results (or resume off)")

    cv_rows: list[dict[str, Any]] = []
    cand_rows: list[dict[str, Any]] = []
    if RESUME_IF_RESULTS_EXIST and CV_TSV.exists():
        cv_rows = pd.read_csv(CV_TSV, sep="\t").to_dict("records")
    if RESUME_IF_RESULTS_EXIST and CAND_TSV.exists():
        cand_rows = pd.read_csv(CAND_TSV, sep="\t").to_dict("records")

    fold_meta: list[dict[str, Any]] = []
    new_pairs_done = 0
    chunk_message = ""

    try:
        for fold_i, test_sub, train_idx, test_idx in split_loso(manifest):
            if DEBUG_MAX_FOLDS is not None and fold_i >= int(DEBUG_MAX_FOLDS):
                print(f"[DEBUG] stopping after fold {fold_i}")
                break

            print(f"===== Fold {fold_i + 1}/{manifest['subject'].nunique()} | test subject = {test_sub} =====")

            train_df = manifest.loc[train_idx].copy()
            test_df = manifest.loc[test_idx].copy()

            val_sub = choose_val_subject(train_df)
            val_df = train_df.loc[train_df["subject"] == val_sub].copy()
            trn_df = train_df.loc[train_df["subject"] != val_sub].copy()

            X_trn_raw = [all_X[i] for i in trn_df["X_index"].tolist()]
            X_val_raw = [all_X[i] for i in val_df["X_index"].tolist()]
            X_tst_raw = [all_X[i] for i in test_df["X_index"].tolist()]

            trn_runs = trn_df["run"].tolist()
            val_runs = val_df["run"].tolist()
            tst_runs = test_df["run"].tolist()

            if USE_RUNWISE_ZSCORE:
                X_trn_raw = runwise_zscore_segments(X_trn_raw, trn_runs, slice(0, D_BOLD))
                X_trn_raw = runwise_zscore_segments(X_trn_raw, trn_runs, slice(D_BOLD, D_TOTAL))
                X_val_raw = runwise_zscore_segments(X_val_raw, val_runs, slice(0, D_BOLD))
                X_val_raw = runwise_zscore_segments(X_val_raw, val_runs, slice(D_BOLD, D_TOTAL))
                X_tst_raw = runwise_zscore_segments(X_tst_raw, tst_runs, slice(0, D_BOLD))
                X_tst_raw = runwise_zscore_segments(X_tst_raw, tst_runs, slice(D_BOLD, D_TOTAL))

            params, meta = make_fold_preproc(X_trn_raw)
            print(f"  PCA PVE(train): BOLD {100 * meta['pve_bold']:.1f}% ({meta['n_bold_pcs']} PCs) | EEG {100 * meta['pve_eeg']:.1f}% ({meta['n_eeg_pcs']} PCs)")
            print(f"  Fold D_pca={meta['D_pca']} | train segs={len(X_trn_raw)} | val segs={len(X_val_raw)} | test segs={len(X_tst_raw)}")

            X_trn = [apply_fold_preproc(X, params) for X in X_trn_raw]
            X_val = [apply_fold_preproc(X, params) for X in X_val_raw]
            X_tst = [apply_fold_preproc(X, params) for X in X_tst_raw]

            nwin_tr = count_windows_for_segments(X_trn)
            nwin_va = count_windows_for_segments(X_val)
            steps_tr = steps_from_windows(nwin_tr)
            steps_va = steps_from_windows(nwin_va)
            if steps_tr == 0 or steps_va == 0:
                if REBATCH_DROP_REMAINDER:
                    print("[WARN] steps==0 under drop_remainder rebatching; disabling rebatch for this fold.")
                    REBATCH_DROP_REMAINDER = False
                    steps_tr = steps_from_windows(nwin_tr)
                    steps_va = steps_from_windows(nwin_va)
            if steps_tr == 0 or steps_va == 0:
                raise RuntimeError(f"Zero steps in fold {fold_i}. Check SEQ_LEN/STEP_SIZE and segment lengths.")

            train_data = Data(X_trn)
            val_data = Data(X_val)
            test_data = Data(X_tst)

            fold_meta.append(
                dict(
                    fold=fold_i,
                    test_subject=test_sub,
                    val_subject=val_sub,
                    D_pca=meta["D_pca"],
                    pve_bold=meta["pve_bold"],
                    pve_eeg=meta["pve_eeg"],
                    n_bold_pcs=meta["n_bold_pcs"],
                    n_eeg_pcs=meta["n_eeg_pcs"],
                    n_train_segments=len(X_trn),
                    n_val_segments=len(X_val),
                    n_test_segments=len(X_tst),
                    nwin_train=int(nwin_tr),
                    nwin_val=int(nwin_va),
                    steps_per_epoch=int(steps_tr),
                    validation_steps=int(steps_va),
                    seq_len=int(SEQ_LEN),
                    step_size=int(STEP_SIZE),
                    batch_size=int(BATCH_SIZE),
                    runwise_zscore=bool(USE_RUNWISE_ZSCORE),
                )
            )

            for K in K_GRID_RUN:
                if (fold_i, K) in done:
                    continue

                cfg = make_config(int(K), int(meta["D_pca"]))
                init_take = INIT_TAKE_BIGK if int(K) >= BIGK_THRESH else INIT_TAKE

                print(f"  [K={K}] steps_per_epoch={steps_tr} | val_steps={steps_va} | seeds={SEEDS_RUN}")
                candidates: list[dict[str, Any]] = []

                for seed in SEEDS_RUN:
                    print(f"    seed={seed} …", end=" ")
                    clear_session()
                    np.random.seed(int(seed))
                    tf.random.set_seed(int(seed))

                    train_ds = as_tf_dataset(train_data, shuffle=True)
                    val_ds = as_tf_dataset(val_data, shuffle=False)

                    model = Model(cfg)
                    try:
                        try:
                            model.random_subset_initialization(train_data, take=float(init_take), n_epochs=int(INIT_EPOCHS), n_init=1)
                        except TypeError:
                            model.random_subset_initialization(train_data, take=float(init_take), n_epochs=int(INIT_EPOCHS))
                    except Exception:
                        print("INIT_FAIL")
                        continue

                    try:
                        model.fit(
                            train_ds,
                            validation_data=val_ds,
                            steps_per_epoch=int(steps_tr),
                            validation_steps=int(steps_va),
                            callbacks=callbacks(),
                        )
                    except TypeError:
                        model.fit(
                            train_ds,
                            validation_data=val_ds,
                            steps_per_epoch=int(steps_tr),
                            validation_steps=int(steps_va),
                        )
                    except Exception:
                        print("FIT_FAIL")
                        continue

                    try:
                        fe_val = free_energy(model, val_data)
                    except Exception:
                        fe_val = np.nan

                    try:
                        alpha_val = get_alpha_list(model, val_data)
                        fo, fo_max, ent_norm, n_active = summarize_alpha(alpha_val, int(K))
                        fo_entropy, neff = fo_entropy_and_neff(fo, int(K))
                        collapsed = is_collapsed(fo_max, n_active, int(K))
                    except Exception:
                        fo_max, ent_norm, fo_entropy, neff, n_active, collapsed = np.nan, np.nan, np.nan, np.nan, 0, True

                    cand = dict(
                        fold=fold_i,
                        test_subject=test_sub,
                        val_subject=val_sub,
                        K=int(K),
                        seed=int(seed),
                        init_take=float(init_take),
                        fe_val=float(fe_val) if np.isfinite(fe_val) else np.nan,
                        fo_max=float(fo_max) if np.isfinite(fo_max) else np.nan,
                        entropy_norm=float(ent_norm) if np.isfinite(ent_norm) else np.nan,
                        n_active=int(n_active),
                        fo_entropy=float(fo_entropy) if np.isfinite(fo_entropy) else np.nan,
                        neff=float(neff) if np.isfinite(neff) else np.nan,
                        collapsed=bool(collapsed),
                    )
                    candidates.append(cand)
                    cand_rows.append(cand)
                    print(f"FE_val={cand['fe_val']:.3f} | collapsed={cand['collapsed']}")

                    try:
                        del model, train_ds, val_ds, alpha_val
                    except Exception:
                        pass
                    gc_now()

                if len(candidates) == 0:
                    row = dict(
                        fold=fold_i,
                        test_subject=test_sub,
                        val_subject=val_sub,
                        K=int(K),
                        seed_selected=np.nan,
                        collapsed=True,
                        fe_train=np.nan,
                        fe_val=np.nan,
                        fe_test=np.nan,
                        fo_max=np.nan,
                        entropy_norm=np.nan,
                        n_active=0,
                        n_seeds_tried=0,
                        init_take=float(init_take),
                        D_pca=int(meta["D_pca"]),
                        feature_mode=FEATURE_MODE.lower(),
                        seq_len=int(SEQ_LEN),
                        step_size=int(STEP_SIZE),
                        batch_size=int(BATCH_SIZE),
                        runwise_zscore=bool(USE_RUNWISE_ZSCORE),
                    )
                    cv_rows.append(row)
                    done.add((fold_i, int(K)))
                else:
                    best = choose_best_candidate(candidates)
                    print(
                        f"    BEST seed={best['seed']} | collapsed={best['collapsed']} | FE_val={best['fe_val']:.3f} "
                        f"| FOmax={best['fo_max']:.3f} | Hnorm={best['entropy_norm']:.3f} | n_active={best['n_active']}"
                    )

                    clear_session()
                    np.random.seed(int(best["seed"]))
                    tf.random.set_seed(int(best["seed"]))

                    train_ds = as_tf_dataset(train_data, shuffle=True)
                    val_ds = as_tf_dataset(val_data, shuffle=False)

                    model = Model(cfg)
                    try:
                        try:
                            model.random_subset_initialization(train_data, take=float(init_take), n_epochs=int(INIT_EPOCHS), n_init=1)
                        except TypeError:
                            model.random_subset_initialization(train_data, take=float(init_take), n_epochs=int(INIT_EPOCHS))
                        try:
                            model.fit(train_ds, validation_data=val_ds, steps_per_epoch=int(steps_tr), validation_steps=int(steps_va), callbacks=callbacks())
                        except TypeError:
                            model.fit(train_ds, validation_data=val_ds, steps_per_epoch=int(steps_tr), validation_steps=int(steps_va))
                        fe_tr = free_energy(model, train_data)
                        fe_te = free_energy(model, test_data)
                    except Exception as exc:
                        print("    [WARN] refit(best) failed:", repr(exc))
                        fe_tr = fe_te = np.nan

                    row = dict(
                        fold=fold_i,
                        test_subject=test_sub,
                        val_subject=val_sub,
                        K=int(K),
                        seed_selected=int(best["seed"]),
                        collapsed=bool(best["collapsed"]),
                        fe_train=float(fe_tr) if np.isfinite(fe_tr) else np.nan,
                        fe_val=float(best["fe_val"]) if np.isfinite(best["fe_val"]) else np.nan,
                        fe_test=float(fe_te) if np.isfinite(fe_te) else np.nan,
                        fo_max=float(best["fo_max"]),
                        entropy_norm=float(best["entropy_norm"]),
                        fo_entropy=float(best["fo_entropy"]) if np.isfinite(best["fo_entropy"]) else np.nan,
                        neff=float(best["neff"]) if np.isfinite(best["neff"]) else np.nan,
                        n_active=int(best["n_active"]),
                        n_seeds_tried=int(len(candidates)),
                        init_take=float(init_take),
                        D_pca=int(meta["D_pca"]),
                        feature_mode=FEATURE_MODE.lower(),
                        seq_len=int(SEQ_LEN),
                        step_size=int(STEP_SIZE),
                        batch_size=int(BATCH_SIZE),
                        runwise_zscore=bool(USE_RUNWISE_ZSCORE),
                    )
                    cv_rows.append(row)
                    done.add((fold_i, int(K)))

                    try:
                        del model, train_ds, val_ds
                    except Exception:
                        pass
                    gc_now()

                pd.DataFrame(cv_rows).to_csv(CV_TSV, sep="\t", index=False)
                pd.DataFrame(cand_rows).to_csv(CAND_TSV, sep="\t", index=False)
                pd.DataFrame(fold_meta).to_csv(FOLD_META_TSV, sep="\t", index=False)

                new_pairs_done += 1
                if MAX_NEW_PAIRS_PER_RUN is not None and new_pairs_done >= int(MAX_NEW_PAIRS_PER_RUN):
                    print(f"[CHUNK DONE] processed {new_pairs_done} new (fold,K) pairs. Saved and stopping to avoid OOM.")
                    raise _ChunkComplete("Chunk complete. Rerun Step 50 to continue from the saved CV tables.")

    except _ChunkComplete as exc:
        chunk_message = str(exc)
        print(chunk_message)

    if not chunk_message and CV_TSV.exists() and CAND_TSV.exists():
        cv = pd.read_csv(CV_TSV, sep="\t")
        print("cv rows:", len(cv))

        feasible = cv.groupby("K")["collapsed"].apply(lambda x: 1.0 - np.mean(x.astype(float))).rename("feasible_frac").reset_index()
        fe_mean = cv.groupby("K")["fe_test"].mean().rename("fe_test_mean").reset_index()
        quick_summary = feasible.merge(fe_mean, on="K", how="left").sort_values("K")

        plt.figure(figsize=(6, 4))
        plt.plot(quick_summary["K"], quick_summary["feasible_frac"], marker="o")
        plt.ylim(-0.05, 1.05)
        plt.xlabel("K")
        plt.ylabel("Fraction non-collapsed folds (selected model)")
        plt.title("C2-K-sweep feasibility vs K")
        plt.tight_layout()
        plt.savefig(OUT_ROOT / "feasibility_vs_K.png", dpi=220)
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.plot(quick_summary["K"], quick_summary["fe_test_mean"], marker="o")
        plt.xlabel("K")
        plt.ylabel("Mean test free energy (selected model)")
        plt.title("C2-K-sweep mean test FE vs K")
        plt.tight_layout()
        plt.savefig(OUT_ROOT / "mean_testFE_vs_K.png", dpi=220)
        plt.close()

        cand = pd.read_csv(CAND_TSV, sep="\t")
        summary_rows = []
        for K in sorted(cv["K"].unique()):
            d = cv.loc[cv["K"] == K].sort_values("fold")
            fe = d["fe_test"].astype(float).to_numpy()
            mean = float(np.mean(fe))
            std = float(np.std(fe, ddof=1))
            sem = std / math.sqrt(len(fe))
            feasible_frac = float(np.mean(~d["collapsed"].astype(bool)))
            summary_rows.append(
                dict(
                    K=int(K),
                    n_folds=int(len(fe)),
                    feasible_frac=feasible_frac,
                    fe_test_mean=mean,
                    fe_test_std=std,
                    fe_test_sem=sem,
                    fo_max_median=float(np.median(d["fo_max"])),
                    n_active_median=float(np.median(d["n_active"])),
                    neff_median=float(np.median(d["neff"])),
                )
            )
        summary = pd.DataFrame(summary_rows).sort_values("K")
        summary.to_csv(OUT_ROOT / "summary_byK_selected.tsv", sep="\t", index=False)

        cand["collapsed"] = cand["collapsed"].astype(bool)
        cand_summ = cand.groupby("K").agg(
            cand_noncollapsed_frac=("collapsed", lambda x: float(np.mean(~x))),
            cand_collapsed_frac=("collapsed", lambda x: float(np.mean(x))),
            cand_mean_fo_max=("fo_max", "mean"),
            cand_mean_entropy=("entropy_norm", "mean"),
            cand_mean_n_active=("n_active", "mean"),
        ).reset_index().sort_values("K")
        cand_summ.to_csv(OUT_ROOT / "summary_byK_candidates.tsv", sep="\t", index=False)

        best_row = summary.loc[summary["fe_test_mean"].idxmin()]
        K_best = int(best_row["K"])
        best_mu = float(best_row["fe_test_mean"])
        best_sem = float(best_row["fe_test_sem"])
        threshold = best_mu + best_sem
        K_1se = int(summary.loc[summary["fe_test_mean"] <= threshold, "K"].min())

        best = cv.loc[cv["K"] == K_best].sort_values("fold").set_index("fold")["fe_test"].astype(float)
        tests = []
        for K in sorted(cv["K"].unique()):
            if int(K) == K_best:
                continue
            x = cv.loc[cv["K"] == K].sort_values("fold").set_index("fold")["fe_test"].astype(float)
            common = best.index.intersection(x.index)
            diff = (x.loc[common] - best.loc[common]).to_numpy()

            try:
                p_w = float(wilcoxon(diff).pvalue)
            except Exception:
                p_w = np.nan
            p_t = float(ttest_rel(x.loc[common].to_numpy(), best.loc[common].to_numpy()).pvalue)

            tests.append(
                dict(
                    K=int(K),
                    mean_diff_vs_best=float(np.mean(diff)),
                    wilcoxon_p=p_w,
                    paired_t_p=p_t,
                )
            )
        tests_df = pd.DataFrame(tests).sort_values("K")
        tests_df.to_csv(OUT_ROOT / "paired_tests_vs_bestK.tsv", sep="\t", index=False)

        local_minima = []
        s = summary.reset_index(drop=True)
        for i in range(len(s)):
            left = s.loc[i - 1, "fe_test_mean"] if i > 0 else np.inf
            right = s.loc[i + 1, "fe_test_mean"] if i < len(s) - 1 else np.inf
            if s.loc[i, "fe_test_mean"] < left and s.loc[i, "fe_test_mean"] < right:
                local_minima.append(int(s.loc[i, "K"]))

        shortlist_primary = [5, 9, 12]
        shortlist_optional = [K_1se] if K_1se not in shortlist_primary else []

        recommendation = dict(
            K_best=K_best,
            best_mean_testFE=best_mu,
            best_sem_testFE=best_sem,
            oneSE_threshold=threshold,
            K_1se=K_1se,
            local_minima=local_minima,
            shortlist_primary=shortlist_primary,
            shortlist_optional=shortlist_optional,
        )
        (OUT_ROOT / "K_selection_recommendation.json").write_text(json.dumps(recommendation, indent=2), encoding="utf-8")

        print("K_best:", K_best, "mean:", best_mu, "SEM:", best_sem)
        print("1-SE threshold:", threshold, "=> K_1se:", K_1se)
        print("Local minima:", local_minima)
        print("Recommended shortlist (primary):", shortlist_primary, "| optional:", shortlist_optional)

        plt.figure(figsize=(6, 4))
        plt.errorbar(summary["K"], summary["fe_test_mean"], yerr=summary["fe_test_sem"], marker="o", capsize=3)
        plt.axhline(best_mu, linestyle="--")
        plt.axhline(threshold, linestyle=":")
        plt.xlabel("K")
        plt.ylabel("Mean test free energy (selected) ± SEM")
        plt.title("C2 K-sweep: mean test FE with SEM + 1-SE threshold")
        plt.tight_layout()
        plt.savefig(OUT_ROOT / "mean_testFE_with_SEM.png", dpi=220)
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.plot(summary["K"], summary["feasible_frac"], marker="o")
        plt.ylim(-0.05, 1.05)
        plt.xlabel("K")
        plt.ylabel("Feasible fraction (selected model)")
        plt.title("C2 K-sweep: feasibility vs K")
        plt.tight_layout()
        plt.savefig(OUT_ROOT / "feasibility_vs_K.png", dpi=220)
        plt.close()

    return {
        "out_root": str(OUT_ROOT),
        "manifest_tsv": str(MANIFEST_TSV),
        "status": "chunk_complete" if chunk_message else "finished",
        "chunk_message": chunk_message,
        "cv_results_tsv": str(CV_TSV),
        "cv_candidates_long_tsv": str(CAND_TSV),
        "fold_meta_tsv": str(FOLD_META_TSV),
    }
