# R01_PipelineD_C2_LOSO_stability_shortlist.ipynb

This notebook runs **LOSO stability tests** for a shortlisted set of **K** values and writes the “stability proof” bundle per K:

- **State matchability** across LOSO folds (Hungarian matching on BOLD-correlation signatures)
- **Fold similarity matrices** (signature similarity, transition matrix similarity)
- **A_std** (transition stability)
- **Matched FO + dwell(A)** consistency tables

**PipelineD = manuscript-grade K justification step** (after the C2 LOSO-CV K sweep; before final model training).



```python
# =========================
# Cell 0 — USER INPUTS (edit here only)
# =========================
from pathlib import Path
import os, json

# -------------------------
# Shortlist for stability
# -------------------------
K_LIST = [5]  #[3, 5, 9, 12]

# -------------------------
# Dataset identity
# -------------------------
DATA_VARIANT = "intermediate"
FEATURE_MODE = "nolags"   # "lags" or "nolags"
MINLEN       = 15

# -------------------------
# Paths (WSL format)
# -------------------------
FINAL_ROOT   = Path("/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/align_trmask_lags/FINAL_v3_gnorm_allTR") / DATA_VARIANT
MANIFEST_TSV = None  # auto-find if None

OUT_ROOT = Path("/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability") / f"PipelineD_C2_{DATA_VARIANT}_{FEATURE_MODE}_minlen{MINLEN}"

# -------------------------
# Feature layout (Pipeline B builds X = [BOLD | EEG])
# -------------------------
N_PARCELS = 200
TR_SEC    = 2.1

if FEATURE_MODE.lower() == "lags":
    LAGS_TR = [-1, 0, 1]
elif FEATURE_MODE.lower() == "nolags":
    LAGS_TR = [0]
else:
    raise ValueError("FEATURE_MODE must be 'lags' or 'nolags'")

D_BOLD  = N_PARCELS
D_EEG   = N_PARCELS * len(LAGS_TR)
D_TOTAL = D_BOLD + D_EEG

# -------------------------
# Windowing (C2 paradigm)
# -------------------------
SEQ_LEN    = 10
STEP_SIZE  = 1
BATCH_SIZE = 16

# IMPORTANT: shape-stable batching to prevent retracing/recompiles.
# We keep this as a *default*, but we do NOT mutate it globally inside folds.
REBATCH_DROP_REMAINDER_DEFAULT = True
SHUFFLE_BUFFER = 2048

# -------------------------
# Fold-wise PCA (leakage-safe; fit on TRAIN only)
# -------------------------
N_BOLD_PCS = 40
N_EEG_PCS  = 40

# -------------------------
# HMM training hyperparameters
# -------------------------
LEARNING_RATE = 1e-3
N_EPOCHS      = 60
COV_EPS       = 1e-6
DIAGONAL_COVS = False   # recommended: FULL covariances

# Initialization per seed
INIT_TAKE      = 0.30
INIT_EPOCHS    = 5
BIGK_THRESH    = 6
INIT_TAKE_BIGK = 0.20

# Multi-seed restarts + selection
SEEDS = list(range(11, 11 + 2*30, 2))  # 30 seeds   #[11, 23, 37, 53, 71]
VAL_SUBJECT_POLICY = "max_segments"  # or "first_subject"

# Run-wise normalization within each split (train/val/test)
USE_RUNWISE_ZSCORE = True

# -------------------------
# Two-stage refit (freeze transitions then unfreeze)
# -------------------------
TWO_STAGE_REFIT_TRANS = True
STAGE1_EPOCH_FRAC     = 0.60   # fraction of N_EPOCHS in stage-1 (trans frozen)

# -------------------------
# Diagnostics (do NOT use test metrics for seed selection)
# -------------------------
LOG_TEST_METRICS_PER_SEED = True

# -------------------------
# Rerun control
# -------------------------
# If you want to rerun specific heldout folds even when RESUME_IF_EXISTS=True, list them here.
FORCE_RERUN_HELDOUTS = []  # e.g., ['sub-01','sub-18']

# -------------------------
# FO-based anti-collapse screen
# -------------------------
FO_MAX_THRESH           = 0.95   # collapse if one state dominates overall
FO_ACTIVE_THRESH        = 0.01   # state counts as "active" if FO > this
MIN_ACTIVE_STATES_BASE  = 4      # require at least min(3,K) active states

# -------------------------
# Free-energy selection convention
# -------------------------
# NOTE: OSL typically uses "lower free energy is better" (more negative).
# We KEEP the default, but we also record a quick per-fold sanity diagnostic.
FE_BETTER = "lower"   # "lower" or "higher"

# -------------------------
# OOM hardening (WSL)
# -------------------------
DISABLE_XLA_AT_IMPORT = True
if DISABLE_XLA_AT_IMPORT:
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=-1"
    os.environ["TF_XLA_ENABLE_XLA_DEVICES"] = "0"
    os.environ["XLA_FLAGS"] = ""

FORCE_EAGER = False
DISABLE_PREFETCH  = True
DISABLE_CALLBACKS = True
GPU_MEMORY_LIMIT_MB = None       # None => memory_growth

# -------------------------
# Chunk + resume (recommended on WSL)
# -------------------------
RESUME_IF_EXISTS = True
MAX_NEW_FOLDS_PER_RUN = 1        # do a couple folds then restart kernel (safety)

# -------------------------
# Optional QC
# -------------------------
DO_CONTIGUITY_QC_IF_POSSIBLE = True   # only runs if manifest has suitable start/end columns
CHECK_NUMERICS = True                # check for NaN/Inf in loaded segments

# Optional debug controls
DEBUG_MAX_FOLDS = None       # e.g., 1
DEBUG_SUBJECTS = None        # e.g., ["sub-01","sub-02"] or None (if all folds)
DEBUG_SEEDS     = None       # e.g., [11]

print("K_LIST:", K_LIST)
print("OUT_ROOT:", OUT_ROOT)
print("FINAL_ROOT:", FINAL_ROOT)
print("SEQ_LEN/STEP/BATCH:", SEQ_LEN, STEP_SIZE, BATCH_SIZE)
print("REBATCH_DROP_REMAINDER_DEFAULT:", REBATCH_DROP_REMAINDER_DEFAULT)
print("SEEDS:", SEEDS)
print("MAX_NEW_FOLDS_PER_RUN:", MAX_NEW_FOLDS_PER_RUN)

# IMPORTANT:
# If you change DISABLE_XLA_AT_IMPORT, restart the kernel and run from Cell 0 again.
print('TWO_STAGE_REFIT_TRANS:', TWO_STAGE_REFIT_TRANS)
print('LOG_TEST_METRICS_PER_SEED:', LOG_TEST_METRICS_PER_SEED)
print('FORCE_RERUN_HELDOUTS:', FORCE_RERUN_HELDOUTS)

```


```python
# =========================
# Cell 1 — Imports + manifest resolution + subject list + basic QC
# =========================
import gc, math, json, re
from pathlib import Path

import numpy as np
import pandas as pd

import tensorflow as tf
from osl_dynamics.data import Data
from osl_dynamics.models.hmm import Config, Model

OUT_ROOT.mkdir(parents=True, exist_ok=True)

PIPELINE_TAG = "PipelineD_C2_LOSO_stability_shortlist"
PIPELINE_DATE = "2026-02-25"

# Save run meta for reproducibility
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
    DISABLE_XLA_AT_IMPORT=DISABLE_XLA_AT_IMPORT,
    FORCE_EAGER=FORCE_EAGER,
    DISABLE_PREFETCH=DISABLE_PREFETCH,
    DISABLE_CALLBACKS=DISABLE_CALLBACKS,
    GPU_MEMORY_LIMIT_MB=GPU_MEMORY_LIMIT_MB,
    DO_CONTIGUITY_QC_IF_POSSIBLE=DO_CONTIGUITY_QC_IF_POSSIBLE,
    CHECK_NUMERICS=CHECK_NUMERICS,
)
(OUT_ROOT / "run_meta.json").write_text(json.dumps(run_meta, indent=2))

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

# Sort deterministically
sort_cols = ["subject", "run"]
if "seg_id" in manifest.columns:
    sort_cols.append("seg_id")
manifest = manifest.sort_values(sort_cols).reset_index(drop=True)

SEG_ROOT = MANIFEST_TSV.parent
def resolve_seg_path(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (SEG_ROOT / pp)

manifest["seg_abs"] = [resolve_seg_path(p) for p in manifest["seg_path"].tolist()]
missing = manifest.loc[~manifest["seg_abs"].apply(lambda p: p.exists())]
if len(missing):
    print("Missing referenced seg files (showing first 10):")
    display(missing.head(10))
    raise FileNotFoundError("Missing seg files referenced by manifest.")

subjects = sorted(manifest["subject"].unique().tolist())
if DEBUG_SUBJECTS is not None:
    subjects = [s for s in subjects if s in list(DEBUG_SUBJECTS)]

print("n_subjects:", len(subjects))
print(subjects)
display(manifest.head())

def fold_dir(K, heldout):
    d = OUT_ROOT / f"K{K:02d}" / f"fold_holdout-{heldout}"
    d.mkdir(parents=True, exist_ok=True)
    return d

# ---- Optional contiguity QC if manifest provides start/end indices ----
def detect_start_end_cols(df: pd.DataFrame):
    cols = set(df.columns)
    start_candidates = ["start_tr","start_TR","tr_start","TR_start","start_idx","t_start","t0","start"]
    end_candidates   = ["end_tr","end_TR","tr_end","TR_end","end_idx","t_end","t1","end","stop"]
    start = next((c for c in start_candidates if c in cols), None)
    end   = next((c for c in end_candidates   if c in cols), None)
    return start, end
    
def contiguity_qc(df: pd.DataFrame):
    start_col, end_col = detect_start_end_cols(df)
    if start_col is None or end_col is None:
        print("[QC] Contiguity QC skipped: manifest lacks start/end columns.")
        return
    print(f"[QC] Contiguity QC using columns: start={start_col}, end={end_col}")
    bad = []
    for run, g in df.groupby("run"):
        gg = g.sort_values(start_col)
        starts = gg[start_col].to_numpy()
        ends   = gg[end_col].to_numpy()
        if np.any(ends < starts):
            bad.append((run, "end<start"))
            continue
        if np.any(starts[1:] <= ends[:-1]):
            bad.append((run, "overlap_or_touch"))
    if bad:
        print("[QC] Potentially problematic runs (overlap):")
        for r, msg in bad[:20]:
            print("  ", r, msg)
        print("[QC] If overlaps are unexpected, fix upstream manifest segmentation.")
    else:
        print("[QC] No overlaps detected across segments per run.")

if DO_CONTIGUITY_QC_IF_POSSIBLE:
    contiguity_qc(manifest)

```


```python
# =========================
# Cell 2 — TF GPU config (cap 4GB) + sanity
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

```


```python
# =========================
# Cell 3 — Helpers (preproc + dataset + signatures + collapse screen + logging)
# =========================
from scipy.optimize import linear_sum_assignment

def gc_now():
    gc.collect()

def stable_clear():
    # WSL-safe teardown
    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc_now()

def log_append(path: Path, line: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")

def load_segments(df: pd.DataFrame):
    # Lazy load for RAM safety
    Xs = []
    for p in df["seg_abs"].tolist():
        x = np.load(p).astype(np.float32)
        if x.ndim != 2 or x.shape[1] != D_TOTAL:
            raise ValueError(f"Bad segment shape {x.shape} for {p}")
        if CHECK_NUMERICS and (not np.isfinite(x).all()):
            raise ValueError(f"Non-finite values (NaN/Inf) found in {p}")
        Xs.append(x)
    return Xs

def count_windows(X_list):
    n = 0
    for x in X_list:
        T = int(x.shape[0])
        if T >= SEQ_LEN:
            n += 1 + (T - SEQ_LEN) // STEP_SIZE
    return int(n)

def steps_from_windows(n_windows: int, drop_remainder: bool):
    if n_windows <= 0:
        return 0
    if drop_remainder:
        return int(n_windows // int(BATCH_SIZE))
    return int(math.ceil(n_windows / float(BATCH_SIZE)))

def runwise_zscore_segments(X_list, run_ids, sl: slice):
    # compute mean/std per run_id within this split only
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

def fit_standardizer(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)

def apply_standardizer(X, mu, sd):
    return ((X - mu) / sd).astype(np.float32)

def fit_pca(X, n_fixed):
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

def make_fold_preproc(X_tr_raw_list):
    Xtr = np.concatenate(X_tr_raw_list, axis=0)
    Xb, Xe = Xtr[:, :D_BOLD], Xtr[:, D_BOLD:]

    mu_b, sd_b = fit_standardizer(Xb)
    mu_e, sd_e = fit_standardizer(Xe)

    Xb_z = apply_standardizer(Xb, mu_b, sd_b)
    Xe_z = apply_standardizer(Xe, mu_e, sd_e)

    mu_pb, Vb, pve_b = fit_pca(Xb_z, N_BOLD_PCS)
    mu_pe, Ve, pve_e = fit_pca(Xe_z, N_EEG_PCS)

    params = dict(mu_b=mu_b, sd_b=sd_b, mu_e=mu_e, sd_e=sd_e,
                  mu_pb=mu_pb, Vb=Vb, mu_pe=mu_pe, Ve=Ve)
    meta = dict(pve_bold=pve_b, pve_eeg=pve_e,
                n_bold_pcs=int(Vb.shape[1]), n_eeg_pcs=int(Ve.shape[1]),
                D_pca=int(Vb.shape[1] + Ve.shape[1]))
    return params, meta

def apply_fold_preproc(X, params):
    Xb = apply_standardizer(X[:, :D_BOLD], params["mu_b"], params["sd_b"])
    Xe = apply_standardizer(X[:, D_BOLD:], params["mu_e"], params["sd_e"])
    Xb_p = apply_pca(Xb, params["mu_pb"], params["Vb"])
    Xe_p = apply_pca(Xe, params["mu_pe"], params["Ve"])
    return np.concatenate([Xb_p, Xe_p], axis=1).astype(np.float32)

def make_config(K, D):
    cfg = Config(
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
        cfg.covariance_matrix_type = "diag" if DIAGONAL_COVS else "full"
    except Exception:
        pass
    try:
        cfg.n_init = 1
    except Exception:
        pass
    return cfg

def callbacks():
    # Keep callbacks disabled by default on WSL for stability.
    # Always return a list (keras expects an iterable).
    if DISABLE_CALLBACKS:
        return []
    return []

# -------------------------
# Two-stage training helpers (freeze/unfreeze transitions)
# -------------------------
def get_keras_model_from_osl(m):
    """Return the underlying tf.keras.Model for an osl-dynamics Model wrapper."""
    for attr in ["model", "_model", "keras_model"]:
        if hasattr(m, attr):
            km = getattr(m, attr)
            if hasattr(km, "get_weights") and hasattr(km, "set_weights"):
                return km
    # Some versions expose keras methods directly
    if hasattr(m, "get_weights") and hasattr(m, "set_weights"):
        return m
    raise AttributeError("Could not locate underlying Keras model on osl-dynamics Model. "
                         "Try inspecting dir(m) to find the attribute that holds the tf.keras.Model.")

def safe_set_weights(dst_m, src_m):
    """Try to copy all weights from src->dst; fallback to shape-matched copy."""
    dst = get_keras_model_from_osl(dst_m)
    src = get_keras_model_from_osl(src_m)
    w_src = src.get_weights()
    try:
        dst.set_weights(w_src)
        return "full"
    except Exception:
        # Fallback: shape-matched copy (in case of version mismatch)
        w_dst = dst.get_weights()
        w_new = []
        for a, b in zip(w_dst, w_src):
            w_new.append(b if (hasattr(a, "shape") and hasattr(b, "shape") and a.shape == b.shape) else a)
        try:
            dst.set_weights(w_new)
            return "shape_matched"
        except Exception:
            return "failed"

def fit_one_stage(m, train_ds, val_ds, steps_tr, steps_va, callbacks_list):
    """Fit using whatever signature is available (OSL wraps keras fit)."""
    try:
        m.fit(
            train_ds,
            validation_data=val_ds,
            steps_per_epoch=int(steps_tr),
            validation_steps=int(steps_va),
            callbacks=callbacks_list,
        )
    except TypeError:
        m.fit(
            train_ds,
            validation_data=val_ds,
            steps_per_epoch=int(steps_tr),
            validation_steps=int(steps_va),
        )

def fit_two_stage_refit(K, D_pca, train_data, train_ds, val_ds, steps_tr, steps_va, init_take, heldout_outdir):
    """Two-stage refit: stage1 (trans frozen) then stage2 (trans learnable), warm-started."""
    # Stage split
    e1 = max(1, int(round(N_EPOCHS * float(STAGE1_EPOCH_FRAC))))
    e2 = max(1, int(N_EPOCHS - e1))

    # Stage 1 config
    cfg1 = make_config(K, D_pca)
    cfg1.learn_trans_prob = False
    try:
        cfg1.n_epochs = int(e1)
    except Exception:
        pass

    m1 = Model(cfg1)
    try:
        try:
            m1.random_subset_initialization(train_data, take=float(init_take), n_epochs=int(INIT_EPOCHS), n_init=1)
        except TypeError:
            m1.random_subset_initialization(train_data, take=float(init_take), n_epochs=int(INIT_EPOCHS))
    except Exception as e:
        raise RuntimeError(f"Stage1 init failed: {repr(e)}")

    fit_one_stage(m1, train_ds, val_ds, steps_tr, steps_va, callbacks())

    # Stage 2 config
    cfg2 = make_config(K, D_pca)
    cfg2.learn_trans_prob = True
    try:
        cfg2.n_epochs = int(e2)
    except Exception:
        pass

    m2 = Model(cfg2)
    # Build variables via the same initialization call
    try:
        try:
            m2.random_subset_initialization(train_data, take=float(init_take), n_epochs=1, n_init=1)
        except TypeError:
            m2.random_subset_initialization(train_data, take=float(init_take), n_epochs=1)
    except Exception as e:
        raise RuntimeError(f"Stage2 build/init failed: {repr(e)}")

    mode = safe_set_weights(m2, m1)
    log_append(heldout_outdir / "two_stage_info.txt", f"stage1_epochs={e1}\tstage2_epochs={e2}\tcopy_mode={mode}")

    fit_one_stage(m2, train_ds, val_ds, steps_tr, steps_va, callbacks())

    # Cleanup stage1
    try:
        del m1
    except Exception:
        pass
    gc_now()

    return m2
    


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
            ds = ds.prefetch(tf.data.AUTOTUNE)
        except Exception:
            pass

    return ds

def free_energy(model, data: Data, drop_remainder: bool):
    ds = as_tf_dataset(data, shuffle=False, drop_remainder=drop_remainder)
    fe = model.free_energy(ds)
    if isinstance(fe, (list, tuple, np.ndarray)):
        fe = float(np.asarray(fe).ravel()[0])
    return float(fe)

def normalize_alpha_list(alpha_like):
    # Normalize to list of (T,K) arrays
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
    if a.ndim == 2:
        return [a]
    if a.ndim == 3:
        return [a[i] for i in range(a.shape[0])]
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
    fo_num = np.zeros(K, dtype=np.float64)
    ent_sum_norm = 0.0
    dwell_lengths = [[] for _ in range(K)]

    for a in alpha_list:
        a = np.asarray(a, dtype=np.float64)
        tot_T += a.shape[0]
        fo_num += a.sum(axis=0)

        a_clip = np.clip(a, eps, 1.0)
        Ht = -(a_clip * np.log(a_clip)).sum(axis=1)
        ent_sum_norm += (Ht / np.log(K)).sum()

        s = np.argmax(a, axis=1)
        if len(s) > 0:
            cur = s[0]
            run = 1
            for t in range(1, len(s)):
                if s[t] == cur:
                    run += 1
                else:
                    dwell_lengths[cur].append(run)
                    cur = s[t]
                    run = 1
            dwell_lengths[cur].append(run)

    fo = (fo_num / max(tot_T, 1))
    fo_max = float(np.max(fo)) if fo.size else np.nan
    ent_norm = float(ent_sum_norm / max(tot_T, 1))
    n_active = int(np.sum(fo > FO_ACTIVE_THRESH)) if fo.size else 0
    dwell_map_mean = np.array([np.mean(d) if len(d) else np.nan for d in dwell_lengths], dtype=np.float32)

    return fo.astype(np.float32), fo_max, ent_norm, n_active, dwell_map_mean.astype(np.float32), int(tot_T)

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

def choose_best_candidate(cands, fe_window=2.0):
    # Prefer validation-noncollapsed candidates
    pool = [c for c in cands if not c["collapsed"]]
    if not pool:
        pool = cands

    # FE direction
    def fe_key(fe):
        return fe if FE_BETTER.lower() == "lower" else -fe

    # Best FE among pool
    best_fe = min(pool, key=lambda c: fe_key(c["fe_val"]))["fe_val"]
    best_fe_k = fe_key(best_fe)

    # Keep candidates within FE window of the best
    near = [c for c in pool if (fe_key(c["fe_val"]) <= best_fe_k + fe_window)]

    # Among near-best FE, prefer: more active states, higher neff, lower FO_max
    near_sorted = sorted(
        near,
        key=lambda c: (
            -c["n_active"],
            -c["neff"],
            c["fo_max"],
            fe_key(c["fe_val"]),
        )
    )
    return near_sorted[0]

def dwell_from_A(A, eps=1e-12):
    A = np.asarray(A, dtype=np.float64)
    Akk = np.clip(np.diag(A), 0.0, 1.0 - eps)
    return (1.0 / (1.0 - Akk)).astype(np.float32)

def cov_to_corr_ut(C, eps=1e-12):
    C = np.asarray(C, dtype=np.float64)
    d = np.sqrt(np.clip(np.diag(C), eps, None))
    corr = C / (d[:, None] * d[None, :])
    iu = np.triu_indices(corr.shape[0], k=1)
    return corr[iu].astype(np.float32)

def ensure_cov_3d(covs):
    covs = np.asarray(covs)
    if covs.ndim == 2:
        K, P = covs.shape
        out = np.zeros((K, P, P), dtype=covs.dtype)
        for k in range(K):
            np.fill_diagonal(out[k], covs[k])
        return out
    return covs

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
    """
    Robust Hungarian matching even if some correlations are NaN/Inf due to
    constant/degenerate signatures.
    """
    sig = np.asarray(sig, dtype=np.float64)
    ref_sig = np.asarray(ref_sig, dtype=np.float64)
    K = sig.shape[0]

    # corr matrix: rows=fold states, cols=ref states
    corr = np.corrcoef(sig, ref_sig)[:K, K:]

    # IMPORTANT: Hungarian solver requires all-finite entries
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

```


```python
# =========================
# Cell 4 — Train LOSO folds for shortlisted K (resume + chunk safe)
#   - Overwrites logs per fold run
#   - Candidate pass (multi-seed) on validation metrics
#   - Refit with FALLBACK (try next candidate) if A/covs/sig are non-finite
# =========================
new_folds_done = 0

import shutil

SIG_FNAME_NEW = "state_signature_ut_boldcorr.npy"   # (K, E) upper-triangle BOLD correlation signature
SIG_FNAME_OLD = "state_signature_corr_ut_bold.npy"  # legacy name
SIG_FNAME = SIG_FNAME_NEW

def assert_finite_array(name: str, X_list, heldout: str):
    """Assert every segment is finite; raise if any NaN/Inf."""
    bad = 0
    for x in X_list:
        bad += int(np.sum(~np.isfinite(x)))
    if bad > 0:
        raise RuntimeError(f"[{heldout}] {name} contains {bad} non-finite values (NaN/Inf).")

def fe_key(fe: float):
    """Return FE ordering key based on FE_BETTER."""
    return fe if str(FE_BETTER).lower() == "lower" else -fe

for K in K_LIST:
    print(f"\n==================== K={K} ====================")
    (OUT_ROOT / f"K{K:02d}").mkdir(parents=True, exist_ok=True)

    for fi, heldout in enumerate(subjects):
        if DEBUG_MAX_FOLDS is not None and fi >= int(DEBUG_MAX_FOLDS):
            break

        outdir = fold_dir(K, heldout)
        sentinel_new = outdir / SIG_FNAME_NEW
        sentinel_old = outdir / SIG_FNAME_OLD

        # Robust resume guard (FORCE_RERUN_HELDOUTS may be None)
        force_set = set(FORCE_RERUN_HELDOUTS or [])
        if RESUME_IF_EXISTS and (sentinel_new.exists() or sentinel_old.exists()) and (heldout not in force_set):
            if sentinel_old.exists() and not sentinel_new.exists():
                try:
                    shutil.copy2(sentinel_old, sentinel_new)
                except Exception:
                    np.save(sentinel_new, np.load(sentinel_old))
            print("[SKIP]", heldout, "exists")
            continue

        # Fold logs
        fail_log = outdir / "seed_failures.tsv"
        cand_log = outdir / "seed_candidates.tsv"
        (outdir / "fold_stdout.txt").write_text("")  # reset for this run

        def fold_print(*args):
            msg = " ".join(str(a) for a in args)
            print(msg)
            log_append(outdir / "fold_stdout.txt", msg)

        # LOSO split
        test_df  = manifest.loc[manifest["subject"] == heldout].copy()
        train_df = manifest.loc[manifest["subject"] != heldout].copy()

        # inner validation subject selection
        if VAL_SUBJECT_POLICY == "max_segments":
            val_sub = train_df.groupby("subject").size().sort_values(ascending=False).index[0]
        elif VAL_SUBJECT_POLICY == "first_subject":
            val_sub = sorted(train_df["subject"].unique().tolist())[0]
        else:
            raise ValueError("VAL_SUBJECT_POLICY must be 'max_segments' or 'first_subject'")

        val_df = train_df.loc[train_df["subject"] == val_sub].copy()
        trn_df = train_df.loc[train_df["subject"] != val_sub].copy()

        # load raw segments
        X_trn_raw = load_segments(trn_df)
        X_val_raw = load_segments(val_df)
        X_tst_raw = load_segments(test_df)

        trn_runs = trn_df["run"].tolist()
        val_runs = val_df["run"].tolist()
        tst_runs = test_df["run"].tolist()

        # run-wise zscore within each split (train/val/test), BOLD and EEG separately
        if USE_RUNWISE_ZSCORE:
            X_trn_raw = runwise_zscore_segments(X_trn_raw, trn_runs, slice(0, D_BOLD))
            X_trn_raw = runwise_zscore_segments(X_trn_raw, trn_runs, slice(D_BOLD, D_TOTAL))
            X_val_raw = runwise_zscore_segments(X_val_raw, val_runs, slice(0, D_BOLD))
            X_val_raw = runwise_zscore_segments(X_val_raw, val_runs, slice(D_BOLD, D_TOTAL))
            X_tst_raw = runwise_zscore_segments(X_tst_raw, tst_runs, slice(0, D_BOLD))
            X_tst_raw = runwise_zscore_segments(X_tst_raw, tst_runs, slice(D_BOLD, D_TOTAL))

        # leakage-safe fold preproc (fit on TRAIN only)
        params, meta = make_fold_preproc(X_trn_raw)

        X_trn = [apply_fold_preproc(x, params) for x in X_trn_raw]
        X_val = [apply_fold_preproc(x, params) for x in X_val_raw]
        X_tst = [apply_fold_preproc(x, params) for x in X_tst_raw]

        # FINITE CHECKS (catch upstream NaN/Inf early)
        assert_finite_array("X_trn", X_trn, heldout)
        assert_finite_array("X_val", X_val, heldout)
        assert_finite_array("X_tst", X_tst, heldout)

        # Determine fold-local drop_remainder
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
        val_data   = Data(X_val)
        test_data  = Data(X_tst)

        cfg = make_config(K, meta["D_pca"])
        init_take = INIT_TAKE_BIGK if K >= BIGK_THRESH else INIT_TAKE
        seeds_run = DEBUG_SEEDS if DEBUG_SEEDS is not None else SEEDS

        fold_print(
            f"--- fold={fi+1}/{len(subjects)} holdout={heldout} | val={val_sub} | "
            f"train_segs={len(X_trn)} val_segs={len(X_val)} test_segs={len(X_tst)} | "
            f"D_pca={meta['D_pca']} | windows(tr,val)=({nwin_tr},{nwin_va}) | steps={steps_tr} ---"
        )

        # -------------------------
        # Reset logs (OVERWRITE) for this fold run
        # -------------------------
        fail_log.parent.mkdir(parents=True, exist_ok=True)
        cand_log.parent.mkdir(parents=True, exist_ok=True)

        fail_log.write_text("seed\tstage\terr\n", encoding="utf-8")

        cand_header = "seed\tfe_val\tfe_val_init\tfo_max\tentropy_norm\tn_active\tfo_entropy\tneff\tcollapsed"
        if LOG_TEST_METRICS_PER_SEED:
            cand_header += "\tfo_max_test\tentropy_norm_test\tn_active_test\tneff_test\tcollapsed_test"
        cand_log.write_text(cand_header + "\n", encoding="utf-8")

        # -------------------------
        # Multi-seed candidate pass (val FE + FO metrics)
        # -------------------------
        candidates = []
        for seed in seeds_run:
            stable_clear()
            np.random.seed(int(seed))
            tf.random.set_seed(int(seed))

            try:
                train_ds = as_tf_dataset(train_data, shuffle=True,  drop_remainder=drop_rem, repeat=True)
                val_ds   = as_tf_dataset(val_data,   shuffle=False, drop_remainder=drop_rem, repeat=True)
            except Exception as e:
                log_append(fail_log, f"{seed}\tdataset\t{repr(e)}")
                continue

            m = Model(cfg)

            # init
            try:
                try:
                    m.random_subset_initialization(train_data, take=float(init_take), n_epochs=int(INIT_EPOCHS), n_init=1)
                except TypeError:
                    m.random_subset_initialization(train_data, take=float(init_take), n_epochs=int(INIT_EPOCHS))
            except Exception as e:
                log_append(fail_log, f"{seed}\tinit\t{repr(e)}")
                continue

            # FE before fit (diagnostic only)
            fe_val_init = np.nan
            try:
                fe_val_init = free_energy(m, val_data, drop_remainder=drop_rem)
            except Exception as e:
                log_append(fail_log, f"{seed}\tfe_init\t{repr(e)}")

            # fit
            try:
                try:
                    m.fit(
                        train_ds,
                        validation_data=val_ds,
                        steps_per_epoch=int(steps_tr),
                        validation_steps=int(steps_va),
                        callbacks=callbacks(),
                    )
                except TypeError:
                    m.fit(
                        train_ds,
                        validation_data=val_ds,
                        steps_per_epoch=int(steps_tr),
                        validation_steps=int(steps_va),
                    )
            except Exception as e:
                log_append(fail_log, f"{seed}\tfit\t{repr(e)}")
                continue

            # metrics
            fo_max_test = None
            ent_norm_test = None
            n_active_test = None
            neff_test = None
            collapsed_test = None

            try:
                fe_val = free_energy(m, val_data, drop_remainder=drop_rem)
                alpha_val = get_alpha_list(m, val_data)
                fo, fo_max, ent_norm, n_active, _, _ = summarize_alpha(alpha_val, K)
                fo_entropy, neff = fo_entropy_and_neff(fo, K)
                collapsed = is_collapsed(float(fo_max), int(n_active), K)

                if LOG_TEST_METRICS_PER_SEED:
                    try:
                        alpha_test = get_alpha_list(m, test_data)
                        fo_t, fo_max_test, ent_norm_test, n_active_test, _, _ = summarize_alpha(alpha_test, K)
                        _, neff_test = fo_entropy_and_neff(fo_t, K)
                        collapsed_test = is_collapsed(float(fo_max_test), int(n_active_test), K)
                    except Exception as e:
                        log_append(fail_log, f"{seed}\ttest_metrics\t{repr(e)}")

            except Exception as e:
                log_append(fail_log, f"{seed}\tmetrics\t{repr(e)}")
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

            # log candidate row
            if LOG_TEST_METRICS_PER_SEED:
                log_append(
                    cand_log,
                    f"{seed}\t{fe_val}\t{fe_val_init}\t{fo_max}\t{ent_norm}\t{n_active}\t{fo_entropy}\t{neff}\t{collapsed}"
                    f"\t{fo_max_test}\t{ent_norm_test}\t{n_active_test}\t{neff_test}\t{collapsed_test}"
                )
            else:
                log_append(
                    cand_log,
                    f"{seed}\t{fe_val}\t{fe_val_init}\t{fo_max}\t{ent_norm}\t{n_active}\t{fo_entropy}\t{neff}\t{collapsed}"
                )

            # cleanup per seed (FIXED indentation)
            try:
                del m, train_ds, val_ds
            except Exception:
                pass
            try:
                del alpha_val
            except Exception:
                pass
            gc_now()

        if len(candidates) == 0:
            raise RuntimeError(f"No candidates trained for fold holdout={heldout}, K={K}. See {fail_log}")

        # Quick FE direction sanity diagnostic (informative)
        diffs = [c["fe_val"] - (c["fe_val_init"] if c["fe_val_init"] is not None else np.nan) for c in candidates]
        diffs = [d for d in diffs if np.isfinite(d)]
        if len(diffs):
            fold_print(f"[Diag] mean(fe_val - fe_val_init) over seeds = {float(np.mean(diffs)):.6f} (FE_BETTER='{FE_BETTER}')")

        # Save candidates list (always)
        (outdir / "candidates_index.json").write_text(json.dumps(candidates, indent=2))

        # -------------------------
        # Refit with FALLBACK:
        # try candidates in ranked order until we get finite A/covs/sig
        # -------------------------
        pool = [c for c in candidates if not c["collapsed"]]
        if not pool:
            pool = candidates

        ranked = sorted(
            pool,
            key=lambda c: (
                fe_key(c["fe_val"]),  # primary: best FE on validation
                c["fo_max"],          # secondary: lower dominance
                -c["n_active"],       # more active states
                -c["neff"],           # more balanced FO
            )
        )

        refit_attempts = []
        selected = None
        A = None
        covs = None
        sig_ut = None
        m = None

        nbpc = int(meta.get("n_bold_pcs", N_BOLD_PCS))

        for rank_i, cand in enumerate(ranked, start=1):
            seed = int(cand["seed"])
            stable_clear()
            np.random.seed(seed)
            tf.random.set_seed(seed)

            # Create fresh datasets (repeat=True for fit)
            try:
                train_ds = as_tf_dataset(train_data, shuffle=True,  drop_remainder=drop_rem, repeat=True)
                val_ds   = as_tf_dataset(val_data,   shuffle=False, drop_remainder=drop_rem, repeat=True)
            except Exception as e:
                refit_attempts.append({"seed": seed, "rank": rank_i, "status": "dataset_fail", "err": repr(e)})
                log_append(fail_log, f"{seed}\trefit_dataset\t{repr(e)}")
                continue

            # Put refit traces in a per-seed directory to avoid overwrites
            try_dir = outdir / f"refit_try_seed{seed:03d}"
            try_dir.mkdir(parents=True, exist_ok=True)

            try:
                if TWO_STAGE_REFIT_TRANS:
                    m = fit_two_stage_refit(
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
                    m = Model(cfg)
                    try:
                        m.random_subset_initialization(train_data, take=float(init_take), n_epochs=int(INIT_EPOCHS), n_init=1)
                    except TypeError:
                        m.random_subset_initialization(train_data, take=float(init_take), n_epochs=int(INIT_EPOCHS))
                    fit_one_stage(m, train_ds, val_ds, steps_tr, steps_va, callbacks())

                A_try = np.asarray(m.get_trans_prob(), dtype=np.float32)
                covs_try = ensure_cov_3d(np.asarray(m.get_covariances(), dtype=np.float32))

                if not np.isfinite(A_try).all():
                    raise ValueError("bad_A_nonfinite")
                if not np.isfinite(covs_try).all():
                    raise ValueError("bad_covs_nonfinite")

                sig_try = compute_signature_ut_boldcorr(covs_try, params["Vb"], nbpc)
                if not np.isfinite(sig_try).all():
                    raise ValueError("bad_sig_nonfinite")

                # SUCCESS
                selected = dict(cand)
                selected["refit_selected_rank"] = int(rank_i)
                A, covs, sig_ut = A_try, covs_try, sig_try
                refit_attempts.append({"seed": seed, "rank": rank_i, "status": "ok"})
                break

            except Exception as e:
                refit_attempts.append({"seed": seed, "rank": rank_i, "status": "refit_bad_numeric", "err": repr(e)})
                log_append(fail_log, f"{seed}\trefit_bad_numeric\t{repr(e)}")
                try:
                    del m
                except Exception:
                    pass
                continue

        (outdir / "refit_attempts.json").write_text(json.dumps(refit_attempts, indent=2))

        if selected is None:
            last_err = refit_attempts[-1]["err"] if refit_attempts else "no_attempts"
            raise RuntimeError(f"[{heldout}] All refit attempts failed or produced non-finite artifacts. Last err: {last_err}")

        # Record selected candidate (may differ from the single best if fallback was needed)
        (outdir / "best_candidate.json").write_text(json.dumps(selected, indent=2))

        # Save fold artifacts from selected refit
        np.save(outdir / "trans_prob.npy", A)
        np.save(outdir / "covs_pca.npy", covs)

        # Means (optional)
        try:
            means = np.asarray(m.get_means(), dtype=np.float32)
            np.save(outdir / "means_pca.npy", means)
        except Exception:
            pass

        np.save(outdir / SIG_FNAME, sig_ut)

        # Train/test FE
        fe_train = float(free_energy(m, train_data, drop_remainder=drop_rem))
        fe_test  = float(free_energy(m, test_data,  drop_remainder=drop_rem))

        # Test summaries
        alpha_test = get_alpha_list(m, test_data)
        fo_t, fo_max_t, ent_t, n_active_t, dwell_map_mean_t, totT_test = summarize_alpha(alpha_test, K)
        fo_entropy_t, neff_t = fo_entropy_and_neff(fo_t, K)
        dwellA = dwell_from_A(A)

        fold_summ = dict(
            K=int(K),
            heldout_subject=str(heldout),
            val_subject=str(val_sub),
            total_T_test=int(totT_test),
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
        (outdir / "fold_summaries.json").write_text(json.dumps(fold_summ, indent=2))

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
            xla_disabled_at_import=bool(DISABLE_XLA_AT_IMPORT),
        )
        (outdir / "fold_info.json").write_text(json.dumps(fold_info, indent=2))

        np.savez_compressed(
            outdir / "preproc_params.npz",
            mu_b=params["mu_b"], sd_b=params["sd_b"],
            mu_e=params["mu_e"], sd_e=params["sd_e"],
            mu_pb=params["mu_pb"], Vb=params["Vb"],
            mu_pe=params["mu_pe"], Ve=params["Ve"],
        )

        # Cleanup fold
        try:
            del m, train_ds, val_ds, alpha_test
        except Exception:
            pass
        stable_clear()

        new_folds_done += 1
        if MAX_NEW_FOLDS_PER_RUN is not None and new_folds_done >= int(MAX_NEW_FOLDS_PER_RUN):
            raise SystemExit("Chunk complete. Restart kernel and run Cell 4 again to resume.")
```


```python
# =========================
# Cell 5 — Stability analysis per K (ROBUST to invalid folds)
# =========================
import matplotlib.pyplot as plt

SIG_FNAME_NEW = "state_signature_ut_boldcorr.npy"
SIG_FNAME_OLD = "state_signature_corr_ut_bold.npy"

def plot_matrix(M, title, out_png, xlabels=None, ylabels=None):
    M = np.asarray(M, dtype=float)
    # Make NaNs show as white
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")

    plt.figure(figsize=(6,5))
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
    s_path = fd / "fold_summaries.json"

    if not (sig_path.exists() and A_path.exists() and s_path.exists()):
        return None, None, None, f"missing_artifacts sig={sig_path.exists()} A={A_path.exists()} summ={s_path.exists()}"

    sig = np.load(sig_path)
    A   = np.load(A_path)
    summ = json.loads(s_path.read_text())

    # Validate numeric integrity
    if (not np.isfinite(sig).all()):
        return None, None, None, "bad_signature_nonfinite"
    if (not np.isfinite(A).all()):
        return None, None, None, "bad_A_nonfinite"

    # Validate summary FO presence
    fo = np.asarray(summ.get("FO", []), dtype=float)
    if fo.size == 0 or (not np.isfinite(fo).all()):
        return None, None, None, "bad_summary_FO"

    return sig, A, summ, None

def stability_for_K(K: int):
    K_out = OUT_ROOT / f"K{K:02d}"
    folds_all = sorted([p for p in K_out.glob("fold_holdout-*") if p.is_dir()])
    if len(folds_all) == 0:
        raise RuntimeError(f"No folds found for K={K} under {K_out}")

    # ---- Load + filter invalid folds ----
    valid = []
    invalid_rows = []
    for fd in folds_all:
        sig, A, summ, err = _load_fold_artifacts(fd)
        if err is not None:
            invalid_rows.append({"fold": fd.name, "reason": err})
            continue
        # shape sanity
        if sig.shape[0] != K or A.shape != (K, K):
            invalid_rows.append({"fold": fd.name, "reason": f"shape_mismatch sig={sig.shape} A={A.shape}"})
            continue
        valid.append((fd, sig, A, summ))

    pd.DataFrame(invalid_rows).to_csv(K_out / "invalid_folds.tsv", sep="\t", index=False)

    if len(valid) < 2:
        raise RuntimeError(f"[K={K}] Need >=2 valid folds for stability. Valid={len(valid)}, invalid={len(invalid_rows)}. See invalid_folds.tsv")

    folds = [v[0] for v in valid]
    sigs  = [v[1] for v in valid]
    As    = [v[2] for v in valid]
    sums  = [v[3] for v in valid]
    F = len(folds)

    # ---------- Choose a robust reference fold (medoid) ----------
    ref0 = sigs[0]
    sigs0 = []
    for sig in sigs:
        inv, _ = match_states_to_reference(sig, ref0)
        sigs0.append(sig[inv, :])

    sim0 = np.full((F, F), np.nan, dtype=float)
    for i in range(F):
        for j in range(F):
            cs = [safe_corr(sigs0[i][k], sigs0[j][k]) for k in range(K)]
            sim0[i, j] = float(np.nanmean(cs))

    medoid_idx = int(np.nanargmax(np.nanmean(sim0, axis=1)))
    ref_sig = sigs[medoid_idx]
    ref_name = folds[medoid_idx].name
    print(f"[K={K}] Reference fold (medoid among VALID folds): {ref_name}")

    # ---------- Final matching to medoid reference ----------
    matched_dir = K_out / "matched_folds"
    matched_dir.mkdir(exist_ok=True)

    sigs_m, As_m = [], []
    match_rows = []

    for fd, sig, A in zip(folds, sigs, As):
        inv, per_state_corr = match_states_to_reference(sig, ref_sig)
        sig_m = sig[inv, :]
        A_m   = A[np.ix_(inv, inv)]

        outd = matched_dir / fd.name
        outd.mkdir(parents=True, exist_ok=True)
        np.save(outd / "match_reorder_idx.npy", inv.astype(int))
        np.save(outd / "match_per_state_corr.npy", per_state_corr.astype(np.float32))

        sigs_m.append(sig_m)
        As_m.append(A_m)

        row = {"fold": fd.name, "mean_state_corr": float(np.nanmean(per_state_corr))}
        for i, c in enumerate(per_state_corr, start=1):
            row[f"state_corr_s{i:02d}"] = float(c)
        match_rows.append(row)

    pd.DataFrame(match_rows).to_csv(K_out / "state_matching_scores.tsv", sep="\t", index=False)
    (K_out / "reference_fold.txt").write_text(ref_name + "\n")

    # ---------- Fold similarity matrices (VALID folds only) ----------
    sim_sig = np.full((F, F), np.nan, dtype=float)
    sim_A   = np.full((F, F), np.nan, dtype=float)

    for i in range(F):
        for j in range(F):
            cs = [safe_corr(sigs_m[i][k], sigs_m[j][k]) for k in range(K)]
            sim_sig[i, j] = float(np.nanmean(cs))
            sim_A[i, j]   = safe_corr(As_m[i].ravel(), As_m[j].ravel())

    fold_labels = [str(i) for i in range(1, F+1)]
    pd.DataFrame(sim_sig, index=fold_labels, columns=fold_labels).to_csv(K_out / "sim_matrix_signature.tsv", sep="\t")
    pd.DataFrame(sim_A,   index=fold_labels, columns=fold_labels).to_csv(K_out / "sim_matrix_A.tsv", sep="\t")

    # ---------- A mean/std (VALID folds only; nan-safe) ----------
    A_stack = np.stack(As_m, axis=0).astype(float)
    A_mean = np.nanmean(A_stack, axis=0)
    # ddof=1 only if >=2 folds
    if A_stack.shape[0] >= 2:
        A_std = np.nanstd(A_stack, axis=0, ddof=1)
    else:
        A_std = np.full_like(A_mean, np.nan)

    state_labels = [str(i) for i in range(1, K+1)]
    pd.DataFrame(A_mean, index=state_labels, columns=state_labels).to_csv(K_out / "A_mean.tsv", sep="\t")
    pd.DataFrame(A_std,  index=state_labels, columns=state_labels).to_csv(K_out / "A_std.tsv", sep="\t")

    # ---------- Matched fold summaries table (VALID folds only) ----------
    rows = []
    for fd, summ in zip(folds, sums):
        inv = np.load(matched_dir / fd.name / "match_reorder_idx.npy").astype(int)
        fo = np.asarray(summ["FO"], dtype=float)[inv]
        dwellA = np.asarray(summ["dwell_A_TR"], dtype=float)[inv]

        r = dict(
            fold=fd.name,
            heldout_subject=summ.get("heldout_subject", ""),
            val_subject=summ.get("val_subject", ""),
            total_T_test=int(summ.get("total_T_test", -1)),
            FO_max=float(np.nanmax(fo)),
            n_active=int(np.sum(fo > FO_ACTIVE_THRESH)),
            neff=float(summ.get("neff", np.nan)),
            fe_test=float(summ.get("fe_test", np.nan)),
        )
        for i, v in enumerate(fo, start=1):
            r[f"FO_s{i:02d}"] = float(v)
        for i, v in enumerate(dwellA, start=1):
            r[f"dwellA_TR_s{i:02d}"] = float(v)
        rows.append(r)

    pd.DataFrame(rows).to_csv(K_out / "fold_summaries_table_matched.tsv", sep="\t", index=False)

    # ---------- Plots ----------
    plot_matrix(sim_sig, f"K={K} Fold Similarity (Signature) [valid folds]", K_out / "plot_sim_signature.png", fold_labels, fold_labels)
    plot_matrix(sim_A,   f"K={K} Fold Similarity (A) [valid folds]",         K_out / "plot_sim_A.png",         fold_labels, fold_labels)
    plot_matrix(A_std,   f"K={K} Transition Std (A_std) [valid folds]",      K_out / "plot_A_std.png",         state_labels, state_labels)

    print(f"[K={K}] stability outputs written to {K_out}")
    if invalid_rows:
        print(f"[K={K}] WARNING: {len(invalid_rows)} invalid folds excluded. See invalid_folds.tsv")

for K in K_LIST:
    stability_for_K(int(K))
```


```python

```
