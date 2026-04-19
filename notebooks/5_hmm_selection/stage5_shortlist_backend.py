"""Backend functions for the public Stage-5 shortlist stability workflow.

This module holds the active Python implementation behind
`step51_run_loso_shortlist_stability_checks.ipynb`.

It preserves the original shortlist training, refit, and cross-fold matching
logic while removing the old dependence on executing preserved provenance
notebooks at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ShortlistBackendConfig:
    """User-facing and frozen settings for the Stage-5 shortlist run."""

    segments_root: str | Path
    shortlist_output_root: str | Path
    k_list: list[int] = field(default_factory=lambda: [3, 5])
    data_variant: str = "intermediate"
    feature_mode: str = "nolags"
    minlen: int = 15
    manifest_tsv: str | Path | None = None
    max_new_folds_per_run: int | None = 1
    gpu_memory_limit_mb: int | None = None
    force_rerun_heldouts: list[str] = field(default_factory=list)
    debug_max_folds: int | None = None
    debug_subjects: list[str] | None = None
    debug_seeds: list[int] | None = None
    n_parcels: int = 200
    tr_sec: float = 2.1
    seq_len: int = 10
    step_size: int = 1
    batch_size: int = 16
    rebatch_drop_remainder_default: bool = True
    shuffle_buffer: int = 2048
    n_bold_pcs: int = 40
    n_eeg_pcs: int = 40
    learning_rate: float = 1e-3
    n_epochs: int = 60
    cov_eps: float = 1e-6
    diagonal_covs: bool = False
    init_take: float = 0.30
    init_epochs: int = 5
    bigk_thresh: int = 6
    init_take_bigk: float = 0.20
    seeds: list[int] = field(default_factory=lambda: list(range(11, 11 + 2 * 30, 2)))
    val_subject_policy: str = "max_segments"
    use_runwise_zscore: bool = True
    two_stage_refit_trans: bool = True
    stage1_epoch_frac: float = 0.60
    log_test_metrics_per_seed: bool = True
    fo_max_thresh: float = 0.95
    fo_active_thresh: float = 0.01
    min_active_states_base: int = 4
    fe_better: str = "lower"
    disable_xla_at_import: bool = True
    force_eager: bool = False
    disable_prefetch: bool = True
    disable_callbacks: bool = True
    resume_if_exists: bool = True
    do_contiguity_qc_if_possible: bool = True
    check_numerics: bool = True


def _auto_find_manifest(final_root: Path, feature_mode: str, minlen: int) -> Path:
    mode = feature_mode.lower()
    candidates = [
        final_root / f"hmm_segments_minlen{minlen}_{mode}" / "segments_manifest.tsv",
        final_root / f"hmm_segments_minlen{minlen}" / "segments_manifest.tsv",
    ]
    for manifest_tsv in candidates:
        if manifest_tsv.exists():
            return manifest_tsv

    hits = list(final_root.rglob("segments_manifest.tsv"))
    if hits:
        def score(path: Path) -> int:
            path_str = str(path)
            score_value = 0
            if f"minlen{minlen}" in path_str:
                score_value += 10
            if mode in path_str:
                score_value += 5
            return score_value

        hits = sorted(hits, key=score, reverse=True)
        return hits[0]

    raise FileNotFoundError(f"Could not find segments_manifest.tsv under {final_root}")


def _parse_subject(run: str) -> str:
    for part in str(run).split("_"):
        if part.startswith("sub-"):
            return part
    return str(run).split("_")[0]


def _resolve_seg_path(seg_root: Path, path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (seg_root / path)


def run_loso_shortlist_backend(config: ShortlistBackendConfig) -> dict[str, Any]:
    """Run the active Stage-5 shortlist backend without notebook patching."""

    import gc
    import json
    import math
    import os
    import shutil

    if config.disable_xla_at_import:
        os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=-1"
        os.environ["TF_XLA_ENABLE_XLA_DEVICES"] = "0"
        os.environ["XLA_FLAGS"] = ""

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from scipy.optimize import linear_sum_assignment
    from osl_dynamics.data import Data
    from osl_dynamics.models.hmm import Config as HMMConfig, Model

    K_LIST = [int(K) for K in config.k_list]
    DATA_VARIANT = str(config.data_variant)
    FEATURE_MODE = str(config.feature_mode)
    MINLEN = int(config.minlen)
    FINAL_ROOT = Path(config.segments_root) / DATA_VARIANT
    MANIFEST_TSV = Path(config.manifest_tsv) if config.manifest_tsv is not None else None
    OUT_ROOT = Path(config.shortlist_output_root) / f"PipelineD_C2_{DATA_VARIANT}_{FEATURE_MODE}_minlen{MINLEN}"

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
    REBATCH_DROP_REMAINDER_DEFAULT = bool(config.rebatch_drop_remainder_default)
    SHUFFLE_BUFFER = int(config.shuffle_buffer)

    N_BOLD_PCS = int(config.n_bold_pcs)
    N_EEG_PCS = int(config.n_eeg_pcs)

    LEARNING_RATE = float(config.learning_rate)
    N_EPOCHS = int(config.n_epochs)
    COV_EPS = float(config.cov_eps)
    DIAGONAL_COVS = bool(config.diagonal_covs)

    INIT_TAKE = float(config.init_take)
    INIT_EPOCHS = int(config.init_epochs)
    BIGK_THRESH = int(config.bigk_thresh)
    INIT_TAKE_BIGK = float(config.init_take_bigk)

    SEEDS = list(config.seeds)
    VAL_SUBJECT_POLICY = str(config.val_subject_policy)
    USE_RUNWISE_ZSCORE = bool(config.use_runwise_zscore)

    TWO_STAGE_REFIT_TRANS = bool(config.two_stage_refit_trans)
    STAGE1_EPOCH_FRAC = float(config.stage1_epoch_frac)
    LOG_TEST_METRICS_PER_SEED = bool(config.log_test_metrics_per_seed)

    FORCE_RERUN_HELDOUTS = list(config.force_rerun_heldouts)
    FO_MAX_THRESH = float(config.fo_max_thresh)
    FO_ACTIVE_THRESH = float(config.fo_active_thresh)
    MIN_ACTIVE_STATES_BASE = int(config.min_active_states_base)
    FE_BETTER = str(config.fe_better)

    FORCE_EAGER = bool(config.force_eager)
    DISABLE_PREFETCH = bool(config.disable_prefetch)
    DISABLE_CALLBACKS = bool(config.disable_callbacks)
    GPU_MEMORY_LIMIT_MB = config.gpu_memory_limit_mb

    RESUME_IF_EXISTS = bool(config.resume_if_exists)
    MAX_NEW_FOLDS_PER_RUN = config.max_new_folds_per_run
    DO_CONTIGUITY_QC_IF_POSSIBLE = bool(config.do_contiguity_qc_if_possible)
    CHECK_NUMERICS = bool(config.check_numerics)

    DEBUG_MAX_FOLDS = config.debug_max_folds
    DEBUG_SUBJECTS = config.debug_subjects
    DEBUG_SEEDS = config.debug_seeds

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    PIPELINE_TAG = "PipelineD_C2_LOSO_stability_shortlist"
    PIPELINE_DATE = "2026-02-25"

    run_meta = dict(
        pipeline=PIPELINE_TAG,
        date=PIPELINE_DATE,
        K_LIST=K_LIST,
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
        N_BOLD_PCS=N_BOLD_PCS,
        N_EEG_PCS=N_EEG_PCS,
        LEARNING_RATE=LEARNING_RATE,
        N_EPOCHS=N_EPOCHS,
        DIAGONAL_COVS=DIAGONAL_COVS,
        COV_EPS=COV_EPS,
        INIT_TAKE=INIT_TAKE,
        INIT_EPOCHS=INIT_EPOCHS,
        BIGK_THRESH=BIGK_THRESH,
        INIT_TAKE_BIGK=INIT_TAKE_BIGK,
        SEEDS=SEEDS,
        VAL_SUBJECT_POLICY=VAL_SUBJECT_POLICY,
        USE_RUNWISE_ZSCORE=USE_RUNWISE_ZSCORE,
        FO_MAX_THRESH=FO_MAX_THRESH,
        FO_ACTIVE_THRESH=FO_ACTIVE_THRESH,
        MIN_ACTIVE_STATES_BASE=MIN_ACTIVE_STATES_BASE,
        FE_BETTER=FE_BETTER,
        DISABLE_XLA_AT_IMPORT=bool(config.disable_xla_at_import),
        FORCE_EAGER=FORCE_EAGER,
        DISABLE_PREFETCH=DISABLE_PREFETCH,
        DISABLE_CALLBACKS=DISABLE_CALLBACKS,
        GPU_MEMORY_LIMIT_MB=GPU_MEMORY_LIMIT_MB,
        DO_CONTIGUITY_QC_IF_POSSIBLE=DO_CONTIGUITY_QC_IF_POSSIBLE,
        CHECK_NUMERICS=CHECK_NUMERICS,
    )
    (OUT_ROOT / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    if MANIFEST_TSV is None:
        MANIFEST_TSV = _auto_find_manifest(FINAL_ROOT, FEATURE_MODE, MINLEN)

    print("K_LIST:", K_LIST)
    print("OUT_ROOT:", OUT_ROOT)
    print("FINAL_ROOT:", FINAL_ROOT)
    print("MANIFEST_TSV:", MANIFEST_TSV)
    print("SEQ_LEN/STEP/BATCH:", SEQ_LEN, STEP_SIZE, BATCH_SIZE)
    print("REBATCH_DROP_REMAINDER_DEFAULT:", REBATCH_DROP_REMAINDER_DEFAULT)
    print("SEEDS:", SEEDS)
    print("MAX_NEW_FOLDS_PER_RUN:", MAX_NEW_FOLDS_PER_RUN)
    print("TWO_STAGE_REFIT_TRANS:", TWO_STAGE_REFIT_TRANS)
    print("LOG_TEST_METRICS_PER_SEED:", LOG_TEST_METRICS_PER_SEED)
    print("FORCE_RERUN_HELDOUTS:", FORCE_RERUN_HELDOUTS)

    manifest = pd.read_csv(MANIFEST_TSV, sep="\t")
    if "run" not in manifest.columns or "seg_path" not in manifest.columns:
        raise ValueError("Expected manifest columns: 'run', 'seg_path'")

    manifest["subject"] = manifest["run"].apply(_parse_subject)
    sort_cols = ["subject", "run"]
    if "seg_id" in manifest.columns:
        sort_cols.append("seg_id")
    manifest = manifest.sort_values(sort_cols).reset_index(drop=True)

    seg_root = MANIFEST_TSV.parent
    manifest["seg_abs"] = [_resolve_seg_path(seg_root, seg_path) for seg_path in manifest["seg_path"].tolist()]
    missing = manifest.loc[~manifest["seg_abs"].apply(lambda path: path.exists())]
    if len(missing):
        print("Missing referenced seg files (showing first 10):")
        print(missing.head(10))
        raise FileNotFoundError("Missing seg files referenced by manifest.")

    subjects = sorted(manifest["subject"].unique().tolist())
    if DEBUG_SUBJECTS is not None:
        subjects = [subject for subject in subjects if subject in list(DEBUG_SUBJECTS)]

    print("n_subjects:", len(subjects))
    print(subjects)

    def fold_dir(K: int, heldout: str) -> Path:
        directory = OUT_ROOT / f"K{K:02d}" / f"fold_holdout-{heldout}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def detect_start_end_cols(df: pd.DataFrame) -> tuple[str | None, str | None]:
        cols = set(df.columns)
        start_candidates = ["start_tr", "start_TR", "tr_start", "TR_start", "start_idx", "t_start", "t0", "start"]
        end_candidates = ["end_tr", "end_TR", "tr_end", "TR_end", "end_idx", "t_end", "t1", "end", "stop"]
        start_col = next((c for c in start_candidates if c in cols), None)
        end_col = next((c for c in end_candidates if c in cols), None)
        return start_col, end_col

    def contiguity_qc(df: pd.DataFrame) -> None:
        start_col, end_col = detect_start_end_cols(df)
        if start_col is None or end_col is None:
            print("[QC] Contiguity QC skipped: manifest lacks start/end columns.")
            return
        print(f"[QC] Contiguity QC using columns: start={start_col}, end={end_col}")
        bad = []
        for run, group in df.groupby("run"):
            group = group.sort_values(start_col)
            starts = group[start_col].to_numpy()
            ends = group[end_col].to_numpy()
            if np.any(ends < starts):
                bad.append((run, "end<start"))
                continue
            if np.any(starts[1:] <= ends[:-1]):
                bad.append((run, "overlap_or_touch"))
        if bad:
            print("[QC] Potentially problematic runs (overlap):")
            for run, msg in bad[:20]:
                print("  ", run, msg)
            print("[QC] If overlaps are unexpected, fix upstream manifest segmentation.")
        else:
            print("[QC] No overlaps detected across segments per run.")

    if DO_CONTIGUITY_QC_IF_POSSIBLE:
        contiguity_qc(manifest)

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
                    [tf.config.LogicalDeviceConfiguration(memory_limit=int(GPU_MEMORY_LIMIT_MB))],
                )
                print("[INFO] GPU memory capped:", GPU_MEMORY_LIMIT_MB, "MB")
            else:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("[INFO] memory_growth enabled")
        except Exception as exc:
            print("[WARN] GPU config:", exc)
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

    _ = tf.matmul(tf.random.normal((64, 64)), tf.random.normal((64, 64)))
    print("TF OK")

    def gc_now() -> None:
        gc.collect()

    def stable_clear() -> None:
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass
        gc_now()

    def log_append(path: Path, line: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line.rstrip("\n") + "\n")

    def load_segments(df: pd.DataFrame) -> list[np.ndarray]:
        Xs = []
        for path in df["seg_abs"].tolist():
            x = np.load(path).astype(np.float32)
            if x.ndim != 2 or x.shape[1] != D_TOTAL:
                raise ValueError(f"Bad segment shape {x.shape} for {path}")
            if CHECK_NUMERICS and (not np.isfinite(x).all()):
                raise ValueError(f"Non-finite values (NaN/Inf) found in {path}")
            Xs.append(x)
        return Xs

    def count_windows(X_list: list[np.ndarray]) -> int:
        n_windows = 0
        for X in X_list:
            T = int(X.shape[0])
            if T >= SEQ_LEN:
                n_windows += 1 + (T - SEQ_LEN) // STEP_SIZE
        return int(n_windows)

    def steps_from_windows(n_windows: int, drop_remainder: bool) -> int:
        if n_windows <= 0:
            return 0
        if drop_remainder:
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

    def make_config(K: int, D: int) -> HMMConfig:
        hmm_config = HMMConfig(
            n_states=K,
            n_channels=D,
            sequence_length=SEQ_LEN,
            learn_means=True,
            learn_covariances=True,
            learn_trans_prob=True,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            n_epochs=N_EPOCHS,
            covariances_epsilon=COV_EPS,
        )
        try:
            hmm_config.covariance_matrix_type = "diag" if DIAGONAL_COVS else "full"
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

    def get_keras_model_from_osl(model: Model):
        for attr in ["model", "_model", "keras_model"]:
            if hasattr(model, attr):
                keras_model = getattr(model, attr)
                if hasattr(keras_model, "get_weights") and hasattr(keras_model, "set_weights"):
                    return keras_model
        if hasattr(model, "get_weights") and hasattr(model, "set_weights"):
            return model
        raise AttributeError(
            "Could not locate underlying Keras model on osl-dynamics Model. "
            "Try inspecting dir(m) to find the attribute that holds the tf.keras.Model."
        )

    def safe_set_weights(dst_m: Model, src_m: Model) -> str:
        dst = get_keras_model_from_osl(dst_m)
        src = get_keras_model_from_osl(src_m)
        src_weights = src.get_weights()
        try:
            dst.set_weights(src_weights)
            return "full"
        except Exception:
            dst_weights = dst.get_weights()
            new_weights = []
            for dst_weight, src_weight in zip(dst_weights, src_weights):
                if hasattr(dst_weight, "shape") and hasattr(src_weight, "shape") and dst_weight.shape == src_weight.shape:
                    new_weights.append(src_weight)
                else:
                    new_weights.append(dst_weight)
            try:
                dst.set_weights(new_weights)
                return "shape_matched"
            except Exception:
                return "failed"

    def fit_one_stage(model: Model, train_ds, val_ds, steps_tr: int, steps_va: int, callbacks_list: list[Any]) -> None:
        try:
            model.fit(
                train_ds,
                validation_data=val_ds,
                steps_per_epoch=int(steps_tr),
                validation_steps=int(steps_va),
                callbacks=callbacks_list,
            )
        except TypeError:
            model.fit(
                train_ds,
                validation_data=val_ds,
                steps_per_epoch=int(steps_tr),
                validation_steps=int(steps_va),
            )

    def fit_two_stage_refit(K: int, D_pca: int, train_data: Data, train_ds, val_ds, steps_tr: int, steps_va: int, init_take: float, heldout_outdir: Path) -> Model:
        stage1_epochs = max(1, int(round(N_EPOCHS * float(STAGE1_EPOCH_FRAC))))
        stage2_epochs = max(1, int(N_EPOCHS - stage1_epochs))

        cfg1 = make_config(K, D_pca)
        cfg1.learn_trans_prob = False
        try:
            cfg1.n_epochs = int(stage1_epochs)
        except Exception:
            pass

        model_stage1 = Model(cfg1)
        try:
            try:
                model_stage1.random_subset_initialization(train_data, take=float(init_take), n_epochs=int(INIT_EPOCHS), n_init=1)
            except TypeError:
                model_stage1.random_subset_initialization(train_data, take=float(init_take), n_epochs=int(INIT_EPOCHS))
        except Exception as exc:
            raise RuntimeError(f"Stage1 init failed: {repr(exc)}")

        fit_one_stage(model_stage1, train_ds, val_ds, steps_tr, steps_va, callbacks())

        cfg2 = make_config(K, D_pca)
        cfg2.learn_trans_prob = True
        try:
            cfg2.n_epochs = int(stage2_epochs)
        except Exception:
            pass

        model_stage2 = Model(cfg2)
        try:
            try:
                model_stage2.random_subset_initialization(train_data, take=float(init_take), n_epochs=1, n_init=1)
            except TypeError:
                model_stage2.random_subset_initialization(train_data, take=float(init_take), n_epochs=1)
        except Exception as exc:
            raise RuntimeError(f"Stage2 build/init failed: {repr(exc)}")

        copy_mode = safe_set_weights(model_stage2, model_stage1)
        log_append(heldout_outdir / "two_stage_info.txt", f"stage1_epochs={stage1_epochs}\tstage2_epochs={stage2_epochs}\tcopy_mode={copy_mode}")
        fit_one_stage(model_stage2, train_ds, val_ds, steps_tr, steps_va, callbacks())

        try:
            del model_stage1
        except Exception:
            pass
        gc_now()
        return model_stage2

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
            for dataset in ds_list[1:]:
                ds = ds.concatenate(dataset)
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
                ds = ds.prefetch(tf.data.AUTOTUNE)
            except Exception:
                pass
        return ds

    def free_energy(model: Model, data: Data, drop_remainder: bool) -> float:
        ds = as_tf_dataset(data, shuffle=False, drop_remainder=drop_remainder)
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
                    raise ValueError(f"Unexpected alpha element shape {alpha.shape}")
            return out
        alpha = np.asarray(alpha_like)
        if alpha.ndim == 2:
            return [alpha]
        if alpha.ndim == 3:
            return [alpha[i] for i in range(alpha.shape[0])]
        raise ValueError(f"Unexpected alpha shape {alpha.shape}")

    def get_alpha_list(model: Model, data: Data) -> list[np.ndarray]:
        if hasattr(model, "get_alpha"):
            return normalize_alpha_list(model.get_alpha(data, concatenate=False, verbose=0))
        if hasattr(model, "get_gamma"):
            return normalize_alpha_list(model.get_gamma(data, concatenate=False, verbose=0))
        raise AttributeError("Model lacks get_alpha/get_gamma")

    def summarize_alpha(alpha_list: Any, K: int, eps: float = 1e-12) -> tuple[np.ndarray, float, float, int, np.ndarray, int]:
        alpha_list = normalize_alpha_list(alpha_list)
        total_T = 0
        fo_num = np.zeros(K, dtype=np.float64)
        ent_sum_norm = 0.0
        dwell_lengths = [[] for _ in range(K)]

        for alpha in alpha_list:
            alpha = np.asarray(alpha, dtype=np.float64)
            total_T += alpha.shape[0]
            fo_num += alpha.sum(axis=0)
            alpha_clip = np.clip(alpha, eps, 1.0)
            Ht = -(alpha_clip * np.log(alpha_clip)).sum(axis=1)
            ent_sum_norm += (Ht / np.log(K)).sum()

            states = np.argmax(alpha, axis=1)
            if len(states) > 0:
                cur = states[0]
                run = 1
                for t in range(1, len(states)):
                    if states[t] == cur:
                        run += 1
                    else:
                        dwell_lengths[cur].append(run)
                        cur = states[t]
                        run = 1
                dwell_lengths[cur].append(run)

        fo = fo_num / max(total_T, 1)
        fo_max = float(np.max(fo)) if fo.size else np.nan
        ent_norm = float(ent_sum_norm / max(total_T, 1))
        n_active = int(np.sum(fo > FO_ACTIVE_THRESH)) if fo.size else 0
        dwell_map_mean = np.array([np.mean(d) if len(d) else np.nan for d in dwell_lengths], dtype=np.float32)
        return fo.astype(np.float32), fo_max, ent_norm, n_active, dwell_map_mean.astype(np.float32), int(total_T)

    def fo_entropy_and_neff(fo: np.ndarray, K: int, eps: float = 1e-12) -> tuple[float, float]:
        fo = np.asarray(fo, dtype=np.float64)
        fo = np.clip(fo, eps, 1.0)
        fo = fo / fo.sum()
        H = float(-(fo * np.log(fo)).sum())
        fo_entropy_norm = float(H / np.log(K))
        neff = float(np.exp(H))
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

    def choose_best_candidate(cands: list[dict[str, Any]], fe_window: float = 2.0) -> dict[str, Any]:
        pool = [candidate for candidate in cands if not candidate["collapsed"]]
        if not pool:
            pool = cands

        def fe_key(fe: float) -> float:
            return fe if FE_BETTER.lower() == "lower" else -fe

        best_fe = min(pool, key=lambda candidate: fe_key(candidate["fe_val"]))["fe_val"]
        best_fe_key = fe_key(best_fe)
        near = [candidate for candidate in pool if fe_key(candidate["fe_val"]) <= best_fe_key + fe_window]
        near_sorted = sorted(
            near,
            key=lambda candidate: (
                -candidate["n_active"],
                -candidate["neff"],
                candidate["fo_max"],
                fe_key(candidate["fe_val"]),
            ),
        )
        return near_sorted[0]

    def dwell_from_A(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        A = np.asarray(A, dtype=np.float64)
        Akk = np.clip(np.diag(A), 0.0, 1.0 - eps)
        return (1.0 / (1.0 - Akk)).astype(np.float32)

    def cov_to_corr_ut(C: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        C = np.asarray(C, dtype=np.float64)
        d = np.sqrt(np.clip(np.diag(C), eps, None))
        corr = C / (d[:, None] * d[None, :])
        iu = np.triu_indices(corr.shape[0], k=1)
        return corr[iu].astype(np.float32)

    def ensure_cov_3d(covs: np.ndarray) -> np.ndarray:
        covs = np.asarray(covs)
        if covs.ndim == 2:
            K, P = covs.shape
            out = np.zeros((K, P, P), dtype=covs.dtype)
            for k in range(K):
                np.fill_diagonal(out[k], covs[k])
            return out
        return covs

    def backproject_cov_bold(covs_pca: np.ndarray, Vb: np.ndarray, nbpc: int) -> np.ndarray:
        covs_pca = ensure_cov_3d(covs_pca)
        K = covs_pca.shape[0]
        out = []
        for k in range(K):
            Cbb = covs_pca[k, :nbpc, :nbpc]
            out.append((Vb @ Cbb @ Vb.T).astype(np.float32))
        return np.stack(out, axis=0)

    def compute_signature_ut_boldcorr(covs_pca: np.ndarray, Vb: np.ndarray, nbpc: int) -> np.ndarray:
        cov_bold = backproject_cov_bold(covs_pca, Vb, nbpc)
        sig_ut = np.stack([cov_to_corr_ut(cov_bold[k]) for k in range(cov_bold.shape[0])], axis=0)
        return sig_ut.astype(np.float32)

    def match_states_to_reference(sig: np.ndarray, ref_sig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sig = np.asarray(sig, dtype=np.float64)
        ref_sig = np.asarray(ref_sig, dtype=np.float64)
        K = sig.shape[0]
        corr = np.corrcoef(sig, ref_sig)[:K, K:]
        corr = np.nan_to_num(corr, nan=-1.0, posinf=-1.0, neginf=-1.0)
        corr = np.clip(corr, -1.0, 1.0)
        row_ind, col_ind = linear_sum_assignment(-corr)
        inv = np.zeros(K, dtype=int)
        for row_i, col_i in zip(row_ind, col_ind):
            inv[col_i] = row_i
        per_state_corr = np.array([corr[inv[j], j] for j in range(K)], dtype=float)
        return inv, per_state_corr

    def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a).ravel()
        b = np.asarray(b).ravel()
        if a.size == 0 or b.size == 0:
            return np.nan
        if np.all(a == a[0]) or np.all(b == b[0]):
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    def assert_finite_array(name: str, X_list: list[np.ndarray], heldout: str) -> None:
        bad = 0
        for X in X_list:
            bad += int(np.sum(~np.isfinite(X)))
        if bad > 0:
            raise RuntimeError(f"[{heldout}] {name} contains {bad} non-finite values (NaN/Inf).")

    def fe_key(fe: float) -> float:
        return fe if str(FE_BETTER).lower() == "lower" else -fe

    class _ChunkComplete(Exception):
        pass

    new_folds_done = 0
    chunk_message = ""

    SIG_FNAME_NEW = "state_signature_ut_boldcorr.npy"
    SIG_FNAME_OLD = "state_signature_corr_ut_bold.npy"
    SIG_FNAME = SIG_FNAME_NEW

    try:
        for K in K_LIST:
            print(f"\n==================== K={K} ====================")
            (OUT_ROOT / f"K{K:02d}").mkdir(parents=True, exist_ok=True)

            for fi, heldout in enumerate(subjects):
                if DEBUG_MAX_FOLDS is not None and fi >= int(DEBUG_MAX_FOLDS):
                    break

                outdir = fold_dir(int(K), heldout)
                sentinel_new = outdir / SIG_FNAME_NEW
                sentinel_old = outdir / SIG_FNAME_OLD

                force_set = set(FORCE_RERUN_HELDOUTS or [])
                if RESUME_IF_EXISTS and (sentinel_new.exists() or sentinel_old.exists()) and (heldout not in force_set):
                    if sentinel_old.exists() and not sentinel_new.exists():
                        try:
                            shutil.copy2(sentinel_old, sentinel_new)
                        except Exception:
                            np.save(sentinel_new, np.load(sentinel_old))
                    print("[SKIP]", heldout, "exists")
                    continue

                fail_log = outdir / "seed_failures.tsv"
                cand_log = outdir / "seed_candidates.tsv"
                (outdir / "fold_stdout.txt").write_text("", encoding="utf-8")

                def fold_print(*args) -> None:
                    msg = " ".join(str(a) for a in args)
                    print(msg)
                    log_append(outdir / "fold_stdout.txt", msg)

                test_df = manifest.loc[manifest["subject"] == heldout].copy()
                train_df = manifest.loc[manifest["subject"] != heldout].copy()

                if VAL_SUBJECT_POLICY == "max_segments":
                    val_sub = str(train_df.groupby("subject").size().sort_values(ascending=False).index[0])
                elif VAL_SUBJECT_POLICY == "first_subject":
                    val_sub = str(sorted(train_df["subject"].unique().tolist())[0])
                else:
                    raise ValueError("VAL_SUBJECT_POLICY must be 'max_segments' or 'first_subject'")

                val_df = train_df.loc[train_df["subject"] == val_sub].copy()
                trn_df = train_df.loc[train_df["subject"] != val_sub].copy()

                X_trn_raw = load_segments(trn_df)
                X_val_raw = load_segments(val_df)
                X_tst_raw = load_segments(test_df)

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
                X_trn = [apply_fold_preproc(x, params) for x in X_trn_raw]
                X_val = [apply_fold_preproc(x, params) for x in X_val_raw]
                X_tst = [apply_fold_preproc(x, params) for x in X_tst_raw]

                assert_finite_array("X_trn", X_trn, heldout)
                assert_finite_array("X_val", X_val, heldout)
                assert_finite_array("X_tst", X_tst, heldout)

                nwin_tr = count_windows(X_trn)
                nwin_va = count_windows(X_val)
                drop_rem = bool(REBATCH_DROP_REMAINDER_DEFAULT)
                steps_tr = steps_from_windows(nwin_tr, drop_remainder=drop_rem)
                steps_va = steps_from_windows(nwin_va, drop_remainder=drop_rem)

                if steps_tr == 0 or steps_va == 0:
                    fold_print("[WARN] steps==0 under drop_remainder; using drop_remainder=False for this fold.")
                    drop_rem = False
                    steps_tr = steps_from_windows(nwin_tr, drop_remainder=drop_rem)
                    steps_va = steps_from_windows(nwin_va, drop_remainder=drop_rem)

                train_data = Data(X_trn)
                val_data = Data(X_val)
                test_data = Data(X_tst)

                cfg = make_config(int(K), int(meta["D_pca"]))
                init_take = INIT_TAKE_BIGK if int(K) >= BIGK_THRESH else INIT_TAKE
                seeds_run = DEBUG_SEEDS if DEBUG_SEEDS is not None else SEEDS

                fold_print(
                    f"--- fold={fi + 1}/{len(subjects)} holdout={heldout} | val={val_sub} | "
                    f"train_segs={len(X_trn)} val_segs={len(X_val)} test_segs={len(X_tst)} | "
                    f"D_pca={meta['D_pca']} | windows(tr,val)=({nwin_tr},{nwin_va}) | steps={steps_tr} ---"
                )

                fail_log.parent.mkdir(parents=True, exist_ok=True)
                cand_log.parent.mkdir(parents=True, exist_ok=True)
                fail_log.write_text("seed\tstage\terr\n", encoding="utf-8")

                cand_header = "seed\tfe_val\tfe_val_init\tfo_max\tentropy_norm\tn_active\tfo_entropy\tneff\tcollapsed"
                if LOG_TEST_METRICS_PER_SEED:
                    cand_header += "\tfo_max_test\tentropy_norm_test\tn_active_test\tneff_test\tcollapsed_test"
                cand_log.write_text(cand_header + "\n", encoding="utf-8")

                candidates: list[dict[str, Any]] = []
                for seed in seeds_run:
                    stable_clear()
                    np.random.seed(int(seed))
                    tf.random.set_seed(int(seed))

                    try:
                        train_ds = as_tf_dataset(train_data, shuffle=True, drop_remainder=drop_rem, repeat=True)
                        val_ds = as_tf_dataset(val_data, shuffle=False, drop_remainder=drop_rem, repeat=True)
                    except Exception as exc:
                        log_append(fail_log, f"{seed}\tdataset\t{repr(exc)}")
                        continue

                    model = Model(cfg)
                    try:
                        try:
                            model.random_subset_initialization(train_data, take=float(init_take), n_epochs=int(INIT_EPOCHS), n_init=1)
                        except TypeError:
                            model.random_subset_initialization(train_data, take=float(init_take), n_epochs=int(INIT_EPOCHS))
                    except Exception as exc:
                        log_append(fail_log, f"{seed}\tinit\t{repr(exc)}")
                        continue

                    fe_val_init = np.nan
                    try:
                        fe_val_init = free_energy(model, val_data, drop_remainder=drop_rem)
                    except Exception as exc:
                        log_append(fail_log, f"{seed}\tfe_init\t{repr(exc)}")

                    try:
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
                    except Exception as exc:
                        log_append(fail_log, f"{seed}\tfit\t{repr(exc)}")
                        continue

                    fo_max_test = None
                    ent_norm_test = None
                    n_active_test = None
                    neff_test = None
                    collapsed_test = None

                    try:
                        fe_val = free_energy(model, val_data, drop_remainder=drop_rem)
                        alpha_val = get_alpha_list(model, val_data)
                        fo, fo_max, ent_norm, n_active, _, _ = summarize_alpha(alpha_val, int(K))
                        fo_entropy, neff = fo_entropy_and_neff(fo, int(K))
                        collapsed = is_collapsed(float(fo_max), int(n_active), int(K))

                        if LOG_TEST_METRICS_PER_SEED:
                            try:
                                alpha_test = get_alpha_list(model, test_data)
                                fo_t, fo_max_test, ent_norm_test, n_active_test, _, _ = summarize_alpha(alpha_test, int(K))
                                _, neff_test = fo_entropy_and_neff(fo_t, int(K))
                                collapsed_test = is_collapsed(float(fo_max_test), int(n_active_test), int(K))
                            except Exception as exc:
                                log_append(fail_log, f"{seed}\ttest_metrics\t{repr(exc)}")
                    except Exception as exc:
                        log_append(fail_log, f"{seed}\tmetrics\t{repr(exc)}")
                        continue

                    cand = dict(
                        seed=int(seed),
                        fe_val=float(fe_val),
                        fe_val_init=float(fe_val_init) if np.isfinite(fe_val_init) else None,
                        fo_max=float(fo_max),
                        entropy_norm=float(ent_norm),
                        n_active=int(n_active),
                        fo_entropy=float(fo_entropy),
                        neff=float(neff),
                        collapsed=bool(collapsed),
                        fo_max_test=float(fo_max_test) if fo_max_test is not None else None,
                        entropy_norm_test=float(ent_norm_test) if ent_norm_test is not None else None,
                        n_active_test=int(n_active_test) if n_active_test is not None else None,
                        neff_test=float(neff_test) if neff_test is not None else None,
                        collapsed_test=bool(collapsed_test) if collapsed_test is not None else None,
                    )
                    candidates.append(cand)

                    if LOG_TEST_METRICS_PER_SEED:
                        log_append(
                            cand_log,
                            f"{seed}\t{fe_val}\t{fe_val_init}\t{fo_max}\t{ent_norm}\t{n_active}\t{fo_entropy}\t{neff}\t{collapsed}"
                            f"\t{fo_max_test}\t{ent_norm_test}\t{n_active_test}\t{neff_test}\t{collapsed_test}",
                        )
                    else:
                        log_append(
                            cand_log,
                            f"{seed}\t{fe_val}\t{fe_val_init}\t{fo_max}\t{ent_norm}\t{n_active}\t{fo_entropy}\t{neff}\t{collapsed}",
                        )

                    try:
                        del model, train_ds, val_ds
                    except Exception:
                        pass
                    try:
                        del alpha_val
                    except Exception:
                        pass
                    gc_now()

                if len(candidates) == 0:
                    raise RuntimeError(f"No candidates trained for fold holdout={heldout}, K={K}. See {fail_log}")

                diffs = [candidate["fe_val"] - (candidate["fe_val_init"] if candidate["fe_val_init"] is not None else np.nan) for candidate in candidates]
                diffs = [diff for diff in diffs if np.isfinite(diff)]
                if len(diffs):
                    fold_print(f"[Diag] mean(fe_val - fe_val_init) over seeds = {float(np.mean(diffs)):.6f} (FE_BETTER='{FE_BETTER}')")

                (outdir / "candidates_index.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")

                pool = [candidate for candidate in candidates if not candidate["collapsed"]]
                if not pool:
                    pool = candidates
                ranked = sorted(
                    pool,
                    key=lambda candidate: (
                        fe_key(candidate["fe_val"]),
                        candidate["fo_max"],
                        -candidate["n_active"],
                        -candidate["neff"],
                    ),
                )

                refit_attempts = []
                selected = None
                A = None
                covs = None
                sig_ut = None
                model = None
                nbpc = int(meta.get("n_bold_pcs", N_BOLD_PCS))

                for rank_i, candidate in enumerate(ranked, start=1):
                    seed = int(candidate["seed"])
                    stable_clear()
                    np.random.seed(seed)
                    tf.random.set_seed(seed)

                    try:
                        train_ds = as_tf_dataset(train_data, shuffle=True, drop_remainder=drop_rem, repeat=True)
                        val_ds = as_tf_dataset(val_data, shuffle=False, drop_remainder=drop_rem, repeat=True)
                    except Exception as exc:
                        refit_attempts.append({"seed": seed, "rank": rank_i, "status": "dataset_fail", "err": repr(exc)})
                        log_append(fail_log, f"{seed}\trefit_dataset\t{repr(exc)}")
                        continue

                    try_dir = outdir / f"refit_try_seed{seed:03d}"
                    try_dir.mkdir(parents=True, exist_ok=True)

                    try:
                        if TWO_STAGE_REFIT_TRANS:
                            model = fit_two_stage_refit(
                                K=int(K),
                                D_pca=int(meta["D_pca"]),
                                train_data=train_data,
                                train_ds=train_ds,
                                val_ds=val_ds,
                                steps_tr=int(steps_tr),
                                steps_va=int(steps_va),
                                init_take=float(init_take),
                                heldout_outdir=try_dir,
                            )
                        else:
                            model = Model(cfg)
                            try:
                                model.random_subset_initialization(train_data, take=float(init_take), n_epochs=int(INIT_EPOCHS), n_init=1)
                            except TypeError:
                                model.random_subset_initialization(train_data, take=float(init_take), n_epochs=int(INIT_EPOCHS))
                            fit_one_stage(model, train_ds, val_ds, steps_tr, steps_va, callbacks())

                        A_try = np.asarray(model.get_trans_prob(), dtype=np.float32)
                        covs_try = ensure_cov_3d(np.asarray(model.get_covariances(), dtype=np.float32))
                        if not np.isfinite(A_try).all():
                            raise ValueError("bad_A_nonfinite")
                        if not np.isfinite(covs_try).all():
                            raise ValueError("bad_covs_nonfinite")

                        sig_try = compute_signature_ut_boldcorr(covs_try, params["Vb"], nbpc)
                        if not np.isfinite(sig_try).all():
                            raise ValueError("bad_sig_nonfinite")

                        selected = dict(candidate)
                        selected["refit_selected_rank"] = int(rank_i)
                        A, covs, sig_ut = A_try, covs_try, sig_try
                        refit_attempts.append({"seed": seed, "rank": rank_i, "status": "ok"})
                        break
                    except Exception as exc:
                        refit_attempts.append({"seed": seed, "rank": rank_i, "status": "refit_bad_numeric", "err": repr(exc)})
                        log_append(fail_log, f"{seed}\trefit_bad_numeric\t{repr(exc)}")
                        try:
                            del model
                        except Exception:
                            pass
                        continue

                (outdir / "refit_attempts.json").write_text(json.dumps(refit_attempts, indent=2), encoding="utf-8")

                if selected is None:
                    last_err = refit_attempts[-1]["err"] if refit_attempts else "no_attempts"
                    raise RuntimeError(f"[{heldout}] All refit attempts failed or produced non-finite artifacts. Last err: {last_err}")

                (outdir / "best_candidate.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
                np.save(outdir / "trans_prob.npy", A)
                np.save(outdir / "covs_pca.npy", covs)

                try:
                    means = np.asarray(model.get_means(), dtype=np.float32)
                    np.save(outdir / "means_pca.npy", means)
                except Exception:
                    pass

                np.save(outdir / SIG_FNAME, sig_ut)

                fe_train = float(free_energy(model, train_data, drop_remainder=drop_rem))
                fe_test = float(free_energy(model, test_data, drop_remainder=drop_rem))

                alpha_test = get_alpha_list(model, test_data)
                fo_t, fo_max_t, ent_t, n_active_t, dwell_map_mean_t, total_T_test = summarize_alpha(alpha_test, int(K))
                fo_entropy_t, neff_t = fo_entropy_and_neff(fo_t, int(K))
                dwellA = dwell_from_A(A)

                fold_summ = dict(
                    K=int(K),
                    heldout_subject=str(heldout),
                    val_subject=str(val_sub),
                    total_T_test=int(total_T_test),
                    FO=fo_t.tolist(),
                    FO_max=float(fo_max_t),
                    entropy_mean_norm=float(ent_t),
                    n_active=int(n_active_t),
                    fo_entropy=float(fo_entropy_t),
                    neff=float(neff_t),
                    dwell_map_mean_TR=dwell_map_mean_t.tolist(),
                    dwell_A_TR=dwellA.tolist(),
                    dwell_A_sec=(dwellA * TR_SEC).tolist(),
                    fe_train=float(fe_train),
                    fe_test=float(fe_test),
                )
                (outdir / "fold_summaries.json").write_text(json.dumps(fold_summ, indent=2), encoding="utf-8")

                fold_info = dict(
                    pipeline=PIPELINE_TAG,
                    K=int(K),
                    heldout_subject=str(heldout),
                    val_subject=str(val_sub),
                    seq_len=int(SEQ_LEN),
                    step_size=int(STEP_SIZE),
                    batch_size=int(BATCH_SIZE),
                    n_train_segments=int(len(X_trn)),
                    n_val_segments=int(len(X_val)),
                    n_test_segments=int(len(X_tst)),
                    nwin_train=int(nwin_tr),
                    nwin_val=int(nwin_va),
                    steps_per_epoch=int(steps_tr),
                    validation_steps=int(steps_va),
                    NBPC=int(N_BOLD_PCS),
                    NEPC=int(N_EEG_PCS),
                    D_pca=int(meta["D_pca"]),
                    pve_bold=float(meta["pve_bold"]),
                    pve_eeg=float(meta["pve_eeg"]),
                    seeds=list(map(int, seeds_run)),
                    init_take=float(init_take),
                    best_candidate=selected,
                    runwise_zscore=bool(USE_RUNWISE_ZSCORE),
                    drop_remainder_fold=bool(drop_rem),
                    FE_BETTER=str(FE_BETTER),
                    xla_disabled_at_import=bool(config.disable_xla_at_import),
                )
                (outdir / "fold_info.json").write_text(json.dumps(fold_info, indent=2), encoding="utf-8")

                np.savez_compressed(
                    outdir / "preproc_params.npz",
                    mu_b=params["mu_b"], sd_b=params["sd_b"],
                    mu_e=params["mu_e"], sd_e=params["sd_e"],
                    mu_pb=params["mu_pb"], Vb=params["Vb"],
                    mu_pe=params["mu_pe"], Ve=params["Ve"],
                )

                try:
                    del model, train_ds, val_ds, alpha_test
                except Exception:
                    pass
                stable_clear()

                new_folds_done += 1
                if MAX_NEW_FOLDS_PER_RUN is not None and new_folds_done >= int(MAX_NEW_FOLDS_PER_RUN):
                    raise _ChunkComplete("Chunk complete. Rerun Step 51 to continue from the saved fold outputs.")

    except _ChunkComplete as exc:
        chunk_message = str(exc)
        print(chunk_message)

    def plot_matrix(M: np.ndarray, title: str, out_png: Path, xlabels=None, ylabels=None) -> None:
        M = np.asarray(M, dtype=float)
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="white")
        plt.figure(figsize=(6, 5))
        plt.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cmap)
        plt.title(title)
        plt.colorbar()
        if xlabels is not None:
            plt.xticks(np.arange(len(xlabels)), xlabels)
        if ylabels is not None:
            plt.yticks(np.arange(len(ylabels)), ylabels)
        plt.tight_layout()
        plt.savefig(out_png, dpi=220)
        plt.close()

    def _load_fold_artifacts(fd: Path):
        sig_path = fd / SIG_FNAME_NEW
        if not sig_path.exists():
            sig_path = fd / SIG_FNAME_OLD
        A_path = fd / "trans_prob.npy"
        summary_path = fd / "fold_summaries.json"

        if not (sig_path.exists() and A_path.exists() and summary_path.exists()):
            return None, None, None, f"missing_artifacts sig={sig_path.exists()} A={A_path.exists()} summ={summary_path.exists()}"

        sig = np.load(sig_path)
        A = np.load(A_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not np.isfinite(sig).all():
            return None, None, None, "bad_signature_nonfinite"
        if not np.isfinite(A).all():
            return None, None, None, "bad_A_nonfinite"
        fo = np.asarray(summary.get("FO", []), dtype=float)
        if fo.size == 0 or (not np.isfinite(fo).all()):
            return None, None, None, "bad_summary_FO"
        return sig, A, summary, None

    def stability_for_K(K: int) -> None:
        K_out = OUT_ROOT / f"K{K:02d}"
        folds_all = sorted([path for path in K_out.glob("fold_holdout-*") if path.is_dir()])
        if len(folds_all) == 0:
            raise RuntimeError(f"No folds found for K={K} under {K_out}")

        valid = []
        invalid_rows = []
        for fd in folds_all:
            sig, A, summary, err = _load_fold_artifacts(fd)
            if err is not None:
                invalid_rows.append({"fold": fd.name, "reason": err})
                continue
            if sig.shape[0] != K or A.shape != (K, K):
                invalid_rows.append({"fold": fd.name, "reason": f"shape_mismatch sig={sig.shape} A={A.shape}"})
                continue
            valid.append((fd, sig, A, summary))

        pd.DataFrame(invalid_rows).to_csv(K_out / "invalid_folds.tsv", sep="\t", index=False)

        if len(valid) < 2:
            raise RuntimeError(f"[K={K}] Need >=2 valid folds for stability. Valid={len(valid)}, invalid={len(invalid_rows)}. See invalid_folds.tsv")

        folds = [v[0] for v in valid]
        sigs = [v[1] for v in valid]
        As = [v[2] for v in valid]
        summaries = [v[3] for v in valid]
        F = len(folds)

        ref0 = sigs[0]
        sigs0 = []
        for sig in sigs:
            inv, _ = match_states_to_reference(sig, ref0)
            sigs0.append(sig[inv, :])

        sim0 = np.full((F, F), np.nan, dtype=float)
        for i in range(F):
            for j in range(F):
                corrs = [safe_corr(sigs0[i][k], sigs0[j][k]) for k in range(K)]
                sim0[i, j] = float(np.nanmean(corrs))

        medoid_idx = int(np.nanargmax(np.nanmean(sim0, axis=1)))
        ref_sig = sigs[medoid_idx]
        ref_name = folds[medoid_idx].name
        print(f"[K={K}] Reference fold (medoid among VALID folds): {ref_name}")

        matched_dir = K_out / "matched_folds"
        matched_dir.mkdir(exist_ok=True)

        sigs_m, As_m = [], []
        match_rows = []
        for fd, sig, A in zip(folds, sigs, As):
            inv, per_state_corr = match_states_to_reference(sig, ref_sig)
            sig_m = sig[inv, :]
            A_m = A[np.ix_(inv, inv)]
            outd = matched_dir / fd.name
            outd.mkdir(parents=True, exist_ok=True)
            np.save(outd / "match_reorder_idx.npy", inv.astype(int))
            np.save(outd / "match_per_state_corr.npy", per_state_corr.astype(np.float32))

            sigs_m.append(sig_m)
            As_m.append(A_m)

            row = {"fold": fd.name, "mean_state_corr": float(np.nanmean(per_state_corr))}
            for i, corr_value in enumerate(per_state_corr, start=1):
                row[f"state_corr_s{i:02d}"] = float(corr_value)
            match_rows.append(row)

        pd.DataFrame(match_rows).to_csv(K_out / "state_matching_scores.tsv", sep="\t", index=False)
        (K_out / "reference_fold.txt").write_text(ref_name + "\n", encoding="utf-8")

        sim_sig = np.full((F, F), np.nan, dtype=float)
        sim_A = np.full((F, F), np.nan, dtype=float)
        for i in range(F):
            for j in range(F):
                corrs = [safe_corr(sigs_m[i][k], sigs_m[j][k]) for k in range(K)]
                sim_sig[i, j] = float(np.nanmean(corrs))
                sim_A[i, j] = safe_corr(As_m[i].ravel(), As_m[j].ravel())

        fold_labels = [str(i) for i in range(1, F + 1)]
        pd.DataFrame(sim_sig, index=fold_labels, columns=fold_labels).to_csv(K_out / "sim_matrix_signature.tsv", sep="\t")
        pd.DataFrame(sim_A, index=fold_labels, columns=fold_labels).to_csv(K_out / "sim_matrix_A.tsv", sep="\t")

        A_stack = np.stack(As_m, axis=0).astype(float)
        A_mean = np.nanmean(A_stack, axis=0)
        if A_stack.shape[0] >= 2:
            A_std = np.nanstd(A_stack, axis=0, ddof=1)
        else:
            A_std = np.full_like(A_mean, np.nan)

        state_labels = [str(i) for i in range(1, K + 1)]
        pd.DataFrame(A_mean, index=state_labels, columns=state_labels).to_csv(K_out / "A_mean.tsv", sep="\t")
        pd.DataFrame(A_std, index=state_labels, columns=state_labels).to_csv(K_out / "A_std.tsv", sep="\t")

        rows = []
        for fd, summary in zip(folds, summaries):
            inv = np.load(matched_dir / fd.name / "match_reorder_idx.npy").astype(int)
            fo = np.asarray(summary["FO"], dtype=float)[inv]
            dwellA = np.asarray(summary["dwell_A_TR"], dtype=float)[inv]

            row = dict(
                fold=fd.name,
                heldout_subject=summary.get("heldout_subject", ""),
                val_subject=summary.get("val_subject", ""),
                total_T_test=int(summary.get("total_T_test", -1)),
                FO_max=float(np.nanmax(fo)),
                n_active=int(np.sum(fo > FO_ACTIVE_THRESH)),
                neff=float(summary.get("neff", np.nan)),
                fe_test=float(summary.get("fe_test", np.nan)),
            )
            for i, value in enumerate(fo, start=1):
                row[f"FO_s{i:02d}"] = float(value)
            for i, value in enumerate(dwellA, start=1):
                row[f"dwellA_TR_s{i:02d}"] = float(value)
            rows.append(row)

        pd.DataFrame(rows).to_csv(K_out / "fold_summaries_table_matched.tsv", sep="\t", index=False)
        plot_matrix(sim_sig, f"K={K} Fold Similarity (Signature) [valid folds]", K_out / "plot_sim_signature.png", fold_labels, fold_labels)
        plot_matrix(sim_A, f"K={K} Fold Similarity (A) [valid folds]", K_out / "plot_sim_A.png", fold_labels, fold_labels)
        plot_matrix(A_std, f"K={K} Transition Std (A_std) [valid folds]", K_out / "plot_A_std.png", state_labels, state_labels)

        print(f"[K={K}] stability outputs written to {K_out}")
        if invalid_rows:
            print(f"[K={K}] WARNING: {len(invalid_rows)} invalid folds excluded. See invalid_folds.tsv")

    if not chunk_message:
        for K in K_LIST:
            stability_for_K(int(K))

    return {
        "out_root": str(OUT_ROOT),
        "manifest_tsv": str(MANIFEST_TSV),
        "status": "chunk_complete" if chunk_message else "finished",
        "chunk_message": chunk_message,
    }
