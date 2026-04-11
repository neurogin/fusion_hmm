# R01 — Pipeline C (UNIFIED): Fusion HMM LOSO-CV (K=2..8) — lags vs no-lags (RESUME + XLA OFF)

This unified notebook supports:

- `FEATURE_MODE = "lags"`: X = [BOLD PC1 (200) | EEG lags (-1,0,+1) (600)] → **D=800**
- `FEATURE_MODE = "nolags"`: X = [BOLD PC1 (200) | EEG (0 lag) (200)] → **D=400**

It enforces a **common LOSO-CV contract** (harmonized hyperparameters) for comparability across lenient/strict/intermediate runs by changing only `FINAL_ROOT`.

**Patch note:** Fixed a compatibility issue where `osl_dynamics.data.Data` may not implement `__len__()` in some versions. We now record segment counts from the underlying lists, and/or use `data.n_sessions` if available.


```python
# Cell 0 — USER INPUTS (only 2 things to edit)

from pathlib import Path

FINAL_ROOT = Path("/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/align_trmask_lags/FINAL_v3_gnorm_allTR/intermediate")
FEATURE_MODE = "lags"

print("FINAL_ROOT:", FINAL_ROOT)
print("FEATURE_MODE:", FEATURE_MODE)
```

    FINAL_ROOT: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/align_trmask_lags/FINAL_v3_gnorm_allTR/intermediate
    FEATURE_MODE: lags



```python
# Cell 1 — Common CV contract (harmonized across lenient/strict/intermediate)

import os
import numpy as np
import pandas as pd

FORCE_EAGER = True
DISABLE_XLA = True

N_PARCELS = 200
TR_SEC = 2.1

if FEATURE_MODE.lower() == "lags":
    LAGS_TR = [-1, 0, 1]
elif FEATURE_MODE.lower() == "nolags":
    LAGS_TR = [0]
else:
    raise ValueError("FEATURE_MODE must be 'lags' or 'nolags'")

D_BOLD  = N_PARCELS
D_EEG   = N_PARCELS * len(LAGS_TR)
D_TOTAL = D_BOLD + D_EEG

PCA_MODE = "fixed"
PCA_VAR_PCT = 0.95
PCA_CAP_BOLD = min(200, D_BOLD)
PCA_CAP_EEG  = min(600, D_EEG)
N_BOLD_PCS = min(40, D_BOLD)
N_EEG_PCS  = min(40, D_EEG)

K_GRID = list(range(2, 13))  # extended to K=12
SEED = 42

SEQ_LEN    = 10
STEP_SIZE  = None
BATCH_SIZE = 32

LEARNING_RATE = 1e-3
N_EPOCHS_CV   = 60

CV_LEARN_MEANS   = True
CV_LEARN_COVS    = True
CV_DIAGONAL_COVS = True
CV_LEARN_TRANS   = True
COV_EPS          = 1e-6

INIT_METHOD = "random_subset"
N_INITS     = 10
INIT_EPOCHS = 5
INIT_TAKE   = 0.30
BIGK_THRESH    = 5
N_INITS_BIGK   = 5
INIT_TAKE_BIGK = 0.20

USE_EARLY_STOPPING = True
VAL_MODE = "subject"
VAL_SUBJECT_POLICY = "max_segments"
ES_PATIENCE  = 8
ES_MIN_DELTA = 0.0

USE_REDUCE_LR = True
LR_FACTOR   = 0.5
LR_PATIENCE = 4
LR_MIN      = 1e-5

RESUME_IF_RESULTS_EXIST = True
SAVE_MODELS = False
ERRORBAR_MODE = "sem"

# GPU memory cap (optional)
GPU_MEMORY_LIMIT_MB = None  # set None to disable; e.g., 4096 for ~4GB

import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


print(f"D_TOTAL={D_TOTAL} (BOLD={D_BOLD}, EEG={D_EEG}) | K_GRID={K_GRID}")
```

    D_TOTAL=800 (BOLD=200, EEG=600) | K_GRID=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]



```python
# Cell 2 — Resolve manifest + output folders (auto-detect naming)

from pathlib import Path
import numpy as np

def auto_find_manifest(final_root: Path, feature_mode: str) -> Path:
    mode = feature_mode.lower()
    candidates = [
        final_root / f"hmm_segments_minlen15_{mode}",
        final_root / "hmm_segments_minlen15",
    ]
    for seg_root in candidates:
        m = seg_root / "segments_manifest.tsv"
        if m.exists():
            return m
    hits = list(final_root.rglob("segments_manifest.tsv"))
    if hits:
        def score(p: Path):
            s = str(p)
            sc = 0
            if "minlen10" in s: sc += 10
            if mode in s: sc += 5
            return sc
        hits = sorted(hits, key=score, reverse=True)
        return hits[0]
    raise FileNotFoundError(f"Could not find segments_manifest.tsv under {final_root}")

MANIFEST_TSV = FINAL_ROOT / "hmm_segments_minlen15_lags" / "segments_manifest.tsv"
SEG_ROOT = MANIFEST_TSV.parent

OUT_ROOT = FINAL_ROOT / "hmm_models_fusion_LOSO_lags_minlen15"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
CV_TSV = OUT_ROOT / "cv_results.tsv"


CV_FIG_FE = OUT_ROOT / "cv_testFE_vs_K.png"
CV_FIG_DFE = OUT_ROOT / "cv_deltaTestFE_vs_K.png"
CV_FIG_DFE_ELBOW = OUT_ROOT / "cv_deltaFE_elbow_vs_K.png"
PAIRED_TSV = OUT_ROOT / "paired_K_vs_Kplus1_tests.tsv"

print("MANIFEST_TSV:", MANIFEST_TSV)
print("OUT_ROOT:", OUT_ROOT)

manifest = pd.read_csv(MANIFEST_TSV, sep="\t")
print("Rows:", len(manifest))
print("Columns:", list(manifest.columns))

def parse_subject(run: str) -> str:
    parts = str(run).split("_")
    for p in parts:
        if p.startswith("sub-"):
            return p
    return parts[0]

manifest["subject"] = manifest["run"].apply(parse_subject)
manifest = manifest.sort_values(["subject","run","seg_id"]).reset_index(drop=True)

x0 = np.load(manifest.loc[0, "seg_path"])
print("Example segment shape:", x0.shape)
assert x0.shape[1] == D_TOTAL, f"Expected D={D_TOTAL}, got {x0.shape[1]}"
```

    MANIFEST_TSV: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/align_trmask_lags/FINAL_v3_gnorm_allTR/intermediate/hmm_segments_minlen15_lags/segments_manifest.tsv
    OUT_ROOT: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/align_trmask_lags/FINAL_v3_gnorm_allTR/intermediate/hmm_models_fusion_LOSO_lags_minlen15
    Rows: 64
    Columns: ['run', 'feature_mode', 'lags_tr', 'seg_id', 'start_TR', 'end_TR', 'len_TR', 'start_sec', 'end_sec', 'dur_sec', 'n_features', 'seg_path']
    Example segment shape: (18, 800)



```python
# Cell 3 — Load segments + NaN hole check

from pathlib import Path
import numpy as np

seg_paths = [Path(p) for p in manifest["seg_path"].tolist()]
all_X = [np.load(p).astype(np.float32) for p in seg_paths]

manifest["X_index"] = np.arange(len(manifest), dtype=int)

nan_any = any(np.isnan(x).any() for x in all_X)
print("Any NaNs across all segments?", nan_any)

total_tr = int(np.sum([x.shape[0] for x in all_X]))
print(f"Loaded {len(all_X)} segments | total TR across segments = {total_tr}")
```

    Any NaNs across all segments? False
    Loaded 64 segments | total TR across segments = 3312



```python
# Cell 4 — TensorFlow + osl-dynamics imports + anti-stall config  (GPU cap + stability)

import os
import tensorflow as tf
from osl_dynamics.data import Data
from osl_dynamics.models.hmm import Config, Model  # explicit imports tend to be a bit more stable

gpus = tf.config.list_physical_devices("GPU")
print("GPUs visible to TF:", gpus)

# -------------------------------------------------------------------
# GPU memory policy:
#   - If GPU_MEMORY_LIMIT_MB is set (e.g., 4096), we cap GPU:0 to that.
#   - Else we enable memory_growth (recommended default).
#
# IMPORTANT:
#   - This cell must run BEFORE creating any TF tensors/models that use the GPU.
#   - If you change GPU_MEMORY_LIMIT_MB, RESTART KERNEL then run from Cell 0.
# -------------------------------------------------------------------
if gpus:
    try:
        # Optional hard cap (MB). Must be defined in Cell 1; if not, default to None.
        GPU_MEMORY_LIMIT_MB = globals().get("GPU_MEMORY_LIMIT_MB", None)

        if GPU_MEMORY_LIMIT_MB is not None:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=int(GPU_MEMORY_LIMIT_MB))]
            )
            logical = tf.config.list_logical_devices("GPU")
            print(f"[INFO] Capped GPU memory to {int(GPU_MEMORY_LIMIT_MB)} MB (GPU:0). Logical GPUs:", logical)
        else:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("[INFO] Enabled memory_growth for all GPUs (no explicit cap).")

    except Exception as e:
        print("[WARN] GPU config:", e)
else:
    print("[INFO] Running CPU-only (no GPUs visible).")

# -------------------------------------------------------------------
# Disable XLA/JIT (stability)
# -------------------------------------------------------------------
if DISABLE_XLA:
    try:
        tf.config.optimizer.set_jit(False)
        os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
        os.environ["XLA_FLAGS"] = "--xla_gpu_disable_async_compilation=true"
        print("[INFO] DISABLE_XLA=True (JIT off).")
    except Exception as e:
        print("[WARN] disable XLA:", e)

# -------------------------------------------------------------------
# Force eager (stability; slower)
# -------------------------------------------------------------------
if FORCE_EAGER:
    try:
        tf.config.run_functions_eagerly(True)
        print("[INFO] FORCE_EAGER=True")
    except Exception as e:
        print("[WARN] FORCE_EAGER:", e)

# Quick sanity matmul
a = tf.random.normal((64, 64))
_ = tf.matmul(a, a)
print("TF OK")

```

    2026-02-11 10:38:09.958450: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2026-02-11 10:38:09.988608: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    E0000 00:00:1770777490.018573  482597 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    E0000 00:00:1770777490.027593  482597 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    W0000 00:00:1770777490.049795  482597 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1770777490.049828  482597 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1770777490.049830  482597 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1770777490.049831  482597 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    2026-02-11 10:38:10.057893: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    /home/gincru/miniforge3/envs/osl_gpu/lib/python3.12/site-packages/osl_dynamics/__init__.py:2: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
      from pkg_resources import DistributionNotFound, get_distribution


    GPUs visible to TF: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
    [INFO] Enabled memory_growth for all GPUs (no explicit cap).
    [INFO] DISABLE_XLA=True (JIT off).
    [INFO] FORCE_EAGER=True
    TF OK


    I0000 00:00:1770777499.320499  482597 gpu_process_state.cc:208] Using CUDA malloc Async allocator for GPU: 0
    I0000 00:00:1770777499.326494  482597 gpu_device.cc:2019] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 5563 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 4060 Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.9



```python
# Cell 5 — Fold split + train-only standardization + train-only PCA + model helpers

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

import gc

def clear_session():
    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()


rng = np.random.default_rng(SEED)

def split_loso(df: pd.DataFrame):
    subs = sorted(df["subject"].unique().tolist())
    for fi, test_sub in enumerate(subs):
        train_idx = df.index[df["subject"] != test_sub].to_numpy()
        test_idx  = df.index[df["subject"] == test_sub].to_numpy()
        yield fi, test_sub, train_idx, test_idx

def stack_rows(idxs: np.ndarray) -> np.ndarray:
    return np.concatenate([all_X[manifest.loc[i, "X_index"]] for i in idxs], axis=0)

def fit_standardizer(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)

def apply_standardizer(X, mu, sd):
    return ((X - mu) / sd).astype(np.float32)

def fit_pca(X, mode, n_fixed, var_pct, cap):
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (S**2) / (Xc.shape[0] - 1)
    pve = var / np.sum(var)
    cum = np.cumsum(pve)
    if mode == "fixed":
        n_comp = int(min(n_fixed, cap, Vt.shape[0]))
    else:
        n_comp = int(np.searchsorted(cum, var_pct) + 1)
        n_comp = int(min(n_comp, cap, Vt.shape[0]))
    V = Vt[:n_comp].T.astype(np.float32)
    return mu.ravel().astype(np.float32), V, float(cum[n_comp-1])

def apply_pca(X, mu, V):
    return ((X - mu) @ V).astype(np.float32)

def blockwise_fit(train_idxs):
    Xtr = stack_rows(train_idxs)
    Xb, Xe = Xtr[:, :D_BOLD], Xtr[:, D_BOLD:]

    mu_b, sd_b = fit_standardizer(Xb)
    mu_e, sd_e = fit_standardizer(Xe)

    Xb_z = apply_standardizer(Xb, mu_b, sd_b)
    Xe_z = apply_standardizer(Xe, mu_e, sd_e)

    mu_pb, Vb, pve_b = fit_pca(Xb_z, PCA_MODE, N_BOLD_PCS, PCA_VAR_PCT, PCA_CAP_BOLD)
    mu_pe, Ve, pve_e = fit_pca(Xe_z, PCA_MODE, N_EEG_PCS,  PCA_VAR_PCT, PCA_CAP_EEG)

    params = dict(mu_b=mu_b, sd_b=sd_b, mu_e=mu_e, sd_e=sd_e,
                  mu_pb=mu_pb, Vb=Vb, mu_pe=mu_pe, Ve=Ve)
    meta = dict(pve_bold=pve_b, pve_eeg=pve_e,
                n_bold_pcs=Vb.shape[1], n_eeg_pcs=Ve.shape[1],
                D_pca=Vb.shape[1] + Ve.shape[1])
    return params, meta

def blockwise_apply(X, params):
    Xb = apply_standardizer(X[:, :D_BOLD], params["mu_b"], params["sd_b"])
    Xe = apply_standardizer(X[:, D_BOLD:], params["mu_e"], params["sd_e"])
    Xb_p = apply_pca(Xb, params["mu_pb"], params["Vb"])
    Xe_p = apply_pca(Xe, params["mu_pe"], params["Ve"])
    return np.concatenate([Xb_p, Xe_p], axis=1).astype(np.float32)

def choose_val_subject(train_df):
    return train_df.groupby("subject").size().sort_values(ascending=False).index[0]

def make_config(K, D):
    cfg = Config(
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
        cfg.covariance_matrix_type = "diag" if CV_DIAGONAL_COVS else "full"
    except Exception:
        pass
    if STEP_SIZE is not None:
        try:
            cfg.step_size = STEP_SIZE
        except Exception:
            pass
    return cfg

def callbacks():
    cbs = []
    if USE_EARLY_STOPPING:
        cbs.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=ES_PATIENCE,
            min_delta=ES_MIN_DELTA, restore_best_weights=True, verbose=0
        ))
    if USE_REDUCE_LR:
        cbs.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=LR_FACTOR,
            patience=LR_PATIENCE, min_lr=LR_MIN, verbose=0
        ))
    return cbs

def _as_tf_dataset(data, shuffle=False):
    """Convert an osl-dynamics Data object -> tf.data.Dataset (Keras/TF2.16+ safe)."""
    if data is None:
        return None
    if hasattr(data, "dataset"):
        kwargs = dict(
            sequence_length=SEQ_LEN,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            concatenate=False,   # <-- IMPORTANT: keep segments separate
        )
        if STEP_SIZE is not None:
            kwargs["step_size"] = STEP_SIZE
        ds = data.dataset(**kwargs)
        try:
            ds = ds.prefetch(tf.data.AUTOTUNE)
        except Exception:
            pass
        return ds
    return data

def free_energy(model, data):
    ds = _as_tf_dataset(data, shuffle=False)
    fe = model.free_energy(ds)
    if isinstance(fe, (list, tuple, np.ndarray)):
        fe = float(np.asarray(fe).ravel()[0])
    return float(fe)

def train_multiinit(K, train_data, val_data):
    """Stable multi-init: loop in Python, init from Data (not dataset), fit on tf.data.Dataset."""
    cfg = make_config(K, train_data.n_channels)

    train_ds = _as_tf_dataset(train_data, shuffle=True)
    val_ds   = _as_tf_dataset(val_data, shuffle=False) if val_data is not None else None

    if K >= BIGK_THRESH:
        n_inits, take = N_INITS_BIGK, INIT_TAKE_BIGK
    else:
        n_inits, take = N_INITS, INIT_TAKE

    best_model = None
    best_metric = np.inf

    for ii in range(n_inits):
        # hard-reset TF state between inits (helps prevent GPU/kernel crashes)
        clear_session()

        m = Model(cfg)

        # IMPORTANT: init from Data object (not tf.data.Dataset)
        try:
            m.random_subset_initialization(train_data, take=take, n_epochs=INIT_EPOCHS)
        except TypeError:
            # fallback for older signatures
            m.random_subset_initialization(train_data)

        fit_kwargs = {}
        if val_ds is not None:
            fit_kwargs["validation_data"] = val_ds
            fit_kwargs["callbacks"] = callbacks()

        m.fit(train_ds, **fit_kwargs)

        # pick best init by validation FE if available, otherwise train FE
        try:
            metric = free_energy(m, val_data if val_data is not None else train_data)
        except Exception:
            metric = free_energy(m, train_data)

        if metric < best_metric:
            # discard old best to reduce GPU memory buildup
            if best_model is not None:
                try:
                    del best_model
                except Exception:
                    pass
            best_model = m
            best_metric = metric
        else:
            try:
                del m
            except Exception:
                pass

    info = dict(best_metric=float(best_metric), n_inits=int(n_inits), init_take=float(take))
    return best_model, info

```


```python
# Cell 6 — Run LOSO-CV (resume-safe incremental writes)  [PATCHED]

cv_rows = []
done = set()

if RESUME_IF_RESULTS_EXIST and CV_TSV.exists():
    prev = pd.read_csv(CV_TSV, sep="\t")
    cv_rows = prev.to_dict("records")
    for _, r in prev.iterrows():
        done.add((int(r["fold"]), int(r["K"])))
    print(f"[RESUME] loaded {len(prev)} rows; done pairs={len(done)}")
else:
    print("[START] no previous results (or resume off)")

fold_meta = []

for fold_i, test_sub, train_idx, test_idx in split_loso(manifest):
    print(f"===== Fold {fold_i+1}/{manifest['subject'].nunique()} | test subject = {test_sub} =====")

    train_df = manifest.loc[train_idx].copy()
    test_df  = manifest.loc[test_idx].copy()

    params, meta = blockwise_fit(train_idx)
    print(f"  PCA PVE(train): BOLD {100*meta['pve_bold']:.1f}% ({meta['n_bold_pcs']} PCs) | EEG {100*meta['pve_eeg']:.1f}% ({meta['n_eeg_pcs']} PCs)")
    print(f"  Fold D_pca: {meta['D_pca']}")

    X_train = [blockwise_apply(all_X[i], params) for i in train_df["X_index"].tolist()]
    X_test  = [blockwise_apply(all_X[i], params) for i in test_df["X_index"].tolist()]

    if USE_EARLY_STOPPING and VAL_MODE == "subject":
        val_sub = choose_val_subject(train_df)
        val_mask = (train_df["subject"] == val_sub).to_numpy()
        X_val = [X_train[j] for j in np.where(val_mask)[0]]
        X_tr  = [X_train[j] for j in np.where(~val_mask)[0]]
        train_data = Data(X_tr)
        val_data   = Data(X_val)
        n_train_segments = len(X_tr)
        n_val_segments   = len(X_val)
    else:
        train_data = Data(X_train)
        val_data = None
        n_train_segments = len(X_train)
        n_val_segments   = 0

    test_data = Data(X_test)
    n_test_segments = len(X_test)

    # Data may not implement __len__ in some versions, so do NOT call len(train_data)
    fold_meta.append(dict(
        fold=fold_i, test_subject=test_sub,
        D_pca=meta["D_pca"],
        pve_bold=meta["pve_bold"], pve_eeg=meta["pve_eeg"],
        n_bold_pcs=meta["n_bold_pcs"], n_eeg_pcs=meta["n_eeg_pcs"],
        n_train_segments=n_train_segments,
        n_val_segments=n_val_segments,
        n_test_segments=n_test_segments,
    ))

    for K in K_GRID:
        if (fold_i, K) in done:
            continue

        print(f"  [K={K}] training…")
        model, info = train_multiinit(K, train_data, val_data)

        fe_tr = free_energy(model, train_data)
        fe_te = free_energy(model, test_data)

        # Free TF resources aggressively (helps avoid hard kernel crashes)
        try:
            del model
        except Exception:
            pass
        clear_session()

        row = dict(
            fold=fold_i, test_subject=test_sub, K=K,
            fe_train=fe_tr, fe_test=fe_te,
            best_init_metric=info["best_metric"],
            n_inits=info["n_inits"], init_take=info["init_take"],
            D_pca=meta["D_pca"],
            pve_bold=meta["pve_bold"], pve_eeg=meta["pve_eeg"],
            n_bold_pcs=meta["n_bold_pcs"], n_eeg_pcs=meta["n_eeg_pcs"],
            feature_mode=FEATURE_MODE.lower(),
        )
        cv_rows.append(row)
        done.add((fold_i, K))

        pd.DataFrame(cv_rows).to_csv(CV_TSV, sep="\t", index=False)

pd.DataFrame(fold_meta).to_csv(OUT_ROOT / "fold_meta.tsv", sep="\t", index=False)
print("Saved:", CV_TSV)
```

    [RESUME] loaded 84 rows; done pairs=84
    ===== Fold 1/12 | test subject = sub-01 =====
      PCA PVE(train): BOLD 68.7% (40 PCs) | EEG 88.2% (40 PCs)
      Fold D_pca: 80



    Loading files:   0%|          | 0/49 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/9 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/6 [00:00<?, ?it/s]


      [K=9] training…



```python
# Cell 7 — Summaries + manuscript-ready plots (SEM/SD) + paired K vs K+1 tests with correction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cv = pd.read_csv(CV_TSV, sep="\t")
assert len(cv) > 0, "cv_results.tsv is empty."

pv = cv.pivot_table(index="fold", columns="K", values="fe_test", aggfunc="mean")
pv = pv.reindex(sorted(pv.columns), axis=1)  # ensure K columns are sorted for diff/elbow
Ks = sorted([k for k in K_GRID if k in pv.columns])
K0 = min(Ks)

def mean_err(x, mode="sem"):
    x = np.asarray(x, float)
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=1)
    if mode == "sd":
        return mu, sd
    n = np.sum(np.isfinite(x))
    return mu, sd / np.sqrt(max(n, 1))

# RAW FE plot
m = []; e = []
for k in Ks:
    mu, er = mean_err(pv[k].values, ERRORBAR_MODE)
    m.append(mu); e.append(er)

plt.figure(figsize=(6,4))
plt.errorbar(Ks, m, yerr=e, marker="o", capsize=3)
plt.xlabel("Number of states K")
plt.ylabel("Test free energy (lower is better)")
plt.title(f"LOSO-CV Test FE vs K (mean ± {ERRORBAR_MODE.upper()})")
plt.tight_layout()
plt.savefig(CV_FIG_FE, dpi=220)
plt.show()

# ΔFE vs baseline
delta = pv.subtract(pv[K0], axis=0)
m = []; e = []
for k in Ks:
    mu, er = mean_err(delta[k].values, ERRORBAR_MODE)
    m.append(mu); e.append(er)

plt.figure(figsize=(6,4))
plt.errorbar(Ks, m, yerr=e, marker="o", capsize=3)
plt.axhline(0, linewidth=1)
plt.xlabel("Number of states K")
plt.ylabel(f"Δ Test FE = FE(K) - FE(K={K0})")
plt.title(f"Δ Test FE vs K (baseline K={K0}; mean ± {ERRORBAR_MODE.upper()})")
plt.tight_layout()
plt.savefig(CV_FIG_DFE, dpi=220)
plt.show()

# Elbow increment FE(K)-FE(K-1)
d_elbow = pv.diff(axis=1)
Ks_elbow = [k for k in Ks if k != Ks[0]]
m = []; e = []
for k in Ks_elbow:
    mu, er = mean_err(d_elbow[k].values, ERRORBAR_MODE)
    m.append(mu); e.append(er)

plt.figure(figsize=(6,4))
plt.errorbar(Ks_elbow, m, yerr=e, marker="o", capsize=3)
plt.axhline(0, linewidth=1)
plt.xlabel("K (increment)")
plt.ylabel("Increment = FE(K) - FE(K-1)")
plt.title(f"Elbow increment (mean ± {ERRORBAR_MODE.upper()})")
plt.tight_layout()
plt.savefig(CV_FIG_DFE_ELBOW, dpi=220)
plt.show()

print("Saved plots:")
print(" ", CV_FIG_FE)
print(" ", CV_FIG_DFE)
print(" ", CV_FIG_DFE_ELBOW)

from scipy.stats import ttest_rel, wilcoxon

def holm(p):
    p = np.asarray(p, float); m = len(p)
    o = np.argsort(p); adj = np.empty_like(p)
    for i, idx in enumerate(o):
        adj[idx] = min(1.0, (m-i)*p[idx])
    s = adj[o]
    for i in range(1, m):
        s[i] = max(s[i], s[i-1])
    adj[o] = s
    return adj

def bh(p):
    p = np.asarray(p, float); m = len(p)
    o = np.argsort(p); r = p[o]
    q = r*m/(np.arange(m)+1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(p); out[o] = np.clip(q, 0, 1)
    return out

rows = []
for k in range(min(K_GRID), max(K_GRID)):
    if k not in pv.columns or (k+1) not in pv.columns:
        continue
    x = pv[k].values
    y = pv[k+1].values
    d = (y-x)
    d = d[np.isfinite(d)]
    if d.size < 3:
        continue
    t = ttest_rel(y, x, nan_policy="omit").pvalue
    try:
        w = wilcoxon(d).pvalue
    except Exception:
        w = np.nan
    rows.append(dict(K=k, K_plus_1=k+1, n=int(d.size),
                     mean_diff=float(np.nanmean(d)),
                     t_p=float(t), wilcoxon_p=float(w)))

tests = pd.DataFrame(rows)
if len(tests):
    tests["wilcoxon_p_holm"] = holm(tests["wilcoxon_p"].values)
    tests["wilcoxon_p_fdr"]  = bh(tests["wilcoxon_p"].values)
    tests.to_csv(PAIRED_TSV, sep="\t", index=False)
    display(tests)
    print("Saved:", PAIRED_TSV)

methods_blurb = f"""Model order selection (K) was assessed using leave-one-subject-out cross-validation (LOSO-CV).
For each fold, the held-out subject’s segments formed the test set and all remaining subjects formed the training set.
All preprocessing (feature standardization and PCA) was fit on training data only and applied to the held-out subject.
For each K, an HMM was fit to the training set and test free energy was computed on the held-out subject.
Performance is summarized as mean ± {ERRORBAR_MODE.upper()} across folds, alongside Δ free energy relative to baseline K={K0}.
Adjacent K values were compared using fold-wise paired tests (Wilcoxon signed-rank by default) with Holm–Bonferroni and BH-FDR correction across K comparisons."""

supp_caption = """Supplementary Table X. Fold-wise paired comparisons of adjacent model orders (K vs K+1) based on LOSO-CV test free energy.
Wilcoxon signed-rank tests were used and p-values were corrected using Holm–Bonferroni and Benjamini–Hochberg FDR procedures."""

print("\n--- Methods blurb ---\n")
print(methods_blurb)
print("\n--- Supplementary Table caption ---\n")
print(supp_caption)
```
