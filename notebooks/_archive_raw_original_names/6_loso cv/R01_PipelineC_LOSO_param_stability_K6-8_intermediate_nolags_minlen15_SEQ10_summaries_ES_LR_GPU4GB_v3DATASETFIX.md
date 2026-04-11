# R01 Pipeline C — Fusion HMM LOSO (K=6/7/8) — Stability Test (Intermediate / Nolags / MinLen15 / SEQ_LEN=10)

This notebook performs the **fold-wise parameter saving** needed for your **LOSO stability/robustness test** on the primary dataset:

- **Variant:** intermediate  
- **Feature mode:** nolags  
- **Min segment length:** 15 TR  
- **Sequence length for training batches:** 10  
- **K candidates:** 6, 7, 8  
- **concatenate=False** (avoid artificial transitions across censored boundaries)

## What gets saved (per fold, per K)
**Core model parameters (for state matching + similarity analyses):**
- `covs_pca.npy` : state covariances in PCA space, shape `(K, P, P)`
- `trans_prob.npy` : transition matrix `A`, shape `(K, K)`
- `means_pca.npy` : (if learned/available), shape `(K, P)`
- `state_signature_corr_ut_bold.npy` : per-state BOLD correlation **upper-triangle** signature in parcel space, shape `(K, n_edges)`  
  (Primary object used for **state matching across folds**.)

**Lightweight per-fold summaries (no saving of full Γ):**
- `fo_test.npy` : fractional occupancy (held-out subject), shape `(K,)`
- `fold_summaries.json` : FO, entropy summaries, dwell times (A-based), MAP-dwell summaries

**Fold preprocessing (for backprojection reproducibility):**
- `preproc_params.npz` : z-scoring params + PCA loadings (`Vb`, `Ve`) + slice definitions

## Global outputs
- `run_meta.json` : environment + run configuration
- `triu_idx_200.npy` : exact upper-triangle indices used for signatures
- `cv_fold_summary.tsv` : one-line summary per fold/K for downstream scripts

---



```python
# =========================
# Cell 0 — USER INPUTS
# =========================
from pathlib import Path

# Dataset identity
DATA_VARIANT   = "intermediate"
FEATURE_MODE   = "nolags"
MINLEN         = 15
K_LIST         = [6, 7, 8]

# Paths (Windows paths OK; notebook auto-maps to /mnt/c/... in WSL/Linux)
MANIFEST_TSV = r"/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/align_trmask_lags/FINAL_v3_gnorm_allTR/intermediate/hmm_segments_minlen15_nolags/segments_manifest.tsv"
QC_CSV       = r"/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/align_trmask_lags/FINAL_v3_gnorm_allTR/intermediate/hmm_segments_minlen15_nolags/qc/per_run_segments_minlen15.csv"

# Output root for this stability test
OUT_ROOT = r"/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15"

# Feature layout (Pipeline B builds X = [BOLD | EEG])
N_PARCELS = 200
TR_SEC    = 2.1
N_BOLD    = N_PARCELS  # BOLD parcels

# PCA (fold-specific; leakage-safe)
NBPC = 40   # from earlier contract defaults
NEPC = 40

# Training data loader
SEQ_LEN    = 10     # keep 10 for this stability test
STEP_SIZE  = None   # None -> default behavior
BATCH_SIZE = 32

# Optim/training (from earlier contract cell)
LEARNING_RATE = 1e-3
N_EPOCHS      = 60

# HMM options (from earlier contract cell)
LEARN_MEANS   = True
LEARN_COVS    = True
DIAGONAL_COVS = True  # diagonal covariances in PCA space
COV_EPS       = 1e-6

# Init robustness (from earlier contract cell; use random_subset init)
INIT_METHOD = "random_subset"
N_INITS     = 10
INIT_EPOCHS = 5
INIT_TAKE   = 0.30

# Early stopping + ReduceLROnPlateau (best-effort; applied if fit() supports callbacks + validation_data)
USE_EARLY_STOPPING = True
ES_PATIENCE  = 8
ES_MIN_DELTA = 0.0

USE_REDUCE_LR = True
LR_FACTOR   = 0.5
LR_PATIENCE = 4
LR_MIN      = 1e-5

# Validation split strategy (inner split inside each LOSO training fold)
VAL_MODE = "subject"
VAL_SUBJECT_POLICY = "max_segments"

# Reproducibility / resume
SEED                  = 42
CONCATENATE_SESSIONS  = False  # MUST stay False
RESUME_SKIP_IF_EXISTS = True

# GPU behavior (your machine: ~4GB GPU memory cap)
USE_GPU               = True
GPU_INDEX             = 0
GPU_MEMORY_LIMIT_MB   = 4096   # set None to disable; 4096 ~ 4GB cap

# TensorFlow execution controls
FORCE_EAGER = True
DISABLE_XLA = True

# Save lightweight summaries (no saving of full Gamma/alpha)
SAVE_FO_TEST         = True
SAVE_ENTROPY_TEST    = True
SAVE_DWELL_A_TEST    = True
SAVE_DWELL_MAP_TEST  = True

print("MANIFEST_TSV:", MANIFEST_TSV)
print("QC_CSV:", QC_CSV)
print("OUT_ROOT:", OUT_ROOT)

```

    MANIFEST_TSV: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/align_trmask_lags/FINAL_v3_gnorm_allTR/intermediate/hmm_segments_minlen15_nolags/segments_manifest.tsv
    QC_CSV: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/align_trmask_lags/FINAL_v3_gnorm_allTR/intermediate/hmm_segments_minlen15_nolags/qc/per_run_segments_minlen15.csv
    OUT_ROOT: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15



```python
# =========================
# Cell 1 — Imports + TF/GPU config + provenance helpers
# =========================
import os, re, json, platform, sys, inspect
from pathlib import Path
from datetime import datetime

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import numpy as np
import pandas as pd

import tensorflow as tf
from osl_dynamics.data import Data
from osl_dynamics.models.hmm import Config as HMMConfig, Model as HMMModel

def normalize_path(p: str) -> Path:
    """
    Accept Windows paths or Linux/WSL paths.

    - If running on Linux/WSL and given a Windows path like:
        C:\\EEGFMRI\\hmm\\...
      it is mapped to:
        /mnt/c/EEGFMRI/hmm/...

    - Otherwise returns Path(p) unchanged.
    """
    p = str(p)

    # Only remap when we're NOT on native Windows
    if os.name != "nt":
        # Match "C:\..." or "C:/..."
        m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
        if m:
            drive = m.group(1).lower()
            rest = m.group(2).replace("\\", "/")
            return Path(f"/mnt/{drive}/{rest}")

    return Path(p)

def set_global_seeds(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def setup_tf_execution(force_eager=True, disable_xla=True):
    if force_eager:
        try:
            tf.config.run_functions_eagerly(True)
            print("[TF] Eager execution forced.")
        except Exception as e:
            print("[TF] Could not force eager:", repr(e))
    if disable_xla:
        try:
            tf.config.optimizer.set_jit(False)
            print("[TF] XLA/JIT disabled.")
        except Exception as e:
            print("[TF] Could not disable XLA:", repr(e))

def setup_gpu(use_gpu=True, gpu_index=0, mem_limit_mb=None):
    if not use_gpu:
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        print("[GPU] disabled (forcing CPU).")
        return

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[GPU] no GPUs detected. Using CPU.")
        return

    gpu = gpus[gpu_index]
    try:
        tf.config.set_visible_devices(gpu, "GPU")
        if mem_limit_mb is not None:
            tf.config.set_logical_device_configuration(
                gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=mem_limit_mb)]
            )
            print(f"[GPU] Using GPU index {gpu_index} with memory cap {mem_limit_mb} MB.")
        else:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[GPU] Using GPU index {gpu_index} (memory growth enabled).")
    except Exception as e:
        print("[GPU] Could not configure GPU:", repr(e))

def env_fingerprint():
    return dict(
        timestamp=str(datetime.now()),
        python=sys.version.replace("\\n"," "),
        platform=platform.platform(),
        numpy=np.__version__,
        pandas=pd.__version__,
        tensorflow=tf.__version__,
        osl_dynamics=getattr(__import__("osl_dynamics"), "__version__", "unknown"),
    )

set_global_seeds(SEED)
setup_tf_execution(FORCE_EAGER, DISABLE_XLA)
setup_gpu(USE_GPU, GPU_INDEX, GPU_MEMORY_LIMIT_MB)

MANIFEST_TSV = normalize_path(MANIFEST_TSV)
QC_CSV       = normalize_path(QC_CSV)
OUT_ROOT     = normalize_path(OUT_ROOT)
OUT_ROOT.mkdir(parents=True, exist_ok=True)

print("Resolved MANIFEST_TSV:", MANIFEST_TSV)
print("Resolved QC_CSV:", QC_CSV)
print("Resolved OUT_ROOT:", OUT_ROOT)

```

    2026-02-18 22:16:41.636712: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2026-02-18 22:16:41.656699: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    E0000 00:00:1771424201.682572    4098 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    E0000 00:00:1771424201.688544    4098 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    W0000 00:00:1771424201.705615    4098 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1771424201.705647    4098 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1771424201.705649    4098 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1771424201.705650    4098 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    2026-02-18 22:16:41.711884: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    /home/gincru/miniforge3/envs/osl_gpu/lib/python3.12/site-packages/osl_dynamics/__init__.py:2: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
      from pkg_resources import DistributionNotFound, get_distribution


    [TF] Eager execution forced.
    [TF] XLA/JIT disabled.
    [GPU] Using GPU index 0 with memory cap 4096 MB.
    Resolved MANIFEST_TSV: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/align_trmask_lags/FINAL_v3_gnorm_allTR/intermediate/hmm_segments_minlen15_nolags/segments_manifest.tsv
    Resolved QC_CSV: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/align_trmask_lags/FINAL_v3_gnorm_allTR/intermediate/hmm_segments_minlen15_nolags/qc/per_run_segments_minlen15.csv
    Resolved OUT_ROOT: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15



```python
# =========================
# Cell 2 — Load manifest and validate columns/paths
# =========================
if not MANIFEST_TSV.exists():
    raise FileNotFoundError(f"Manifest not found: {MANIFEST_TSV}")

manifest = pd.read_csv(MANIFEST_TSV, sep="\t")
print("Manifest columns:", list(manifest.columns))
display(manifest.head())

path_candidates = [c for c in manifest.columns if c.lower() in ("seg_path","segment_path","path","relpath","file","segment_file")]
if not path_candidates:
    for c in manifest.columns:
        if "path" in c.lower() or "file" in c.lower():
            path_candidates = [c]
            break
if not path_candidates:
    raise ValueError("Could not identify segment path column in manifest.")
SEG_PATH_COL = path_candidates[0]
print("Using segment path column:", SEG_PATH_COL)

SEG_ROOT = MANIFEST_TSV.parent

def resolve_seg_path(p) -> Path:
    p = normalize_path(p)
    return p if p.is_absolute() else (SEG_ROOT / p)

manifest["__seg_abs"] = manifest[SEG_PATH_COL].apply(resolve_seg_path)

missing = manifest.loc[~manifest["__seg_abs"].apply(lambda p: p.exists())]
if len(missing) > 0:
    print("[ERROR] Missing segment files (first 10):")
    display(missing[[SEG_PATH_COL, "__seg_abs"]].head(10))
    raise FileNotFoundError("Some segment files listed in the manifest do not exist.")
else:
    print("All segment paths exist. n_segments =", len(manifest))

```

    Manifest columns: ['run', 'feature_mode', 'lags_tr', 'seg_id', 'start_TR', 'end_TR', 'len_TR', 'start_sec', 'end_sec', 'dur_sec', 'n_features', 'seg_path']



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>run</th>
      <th>feature_mode</th>
      <th>lags_tr</th>
      <th>seg_id</th>
      <th>start_TR</th>
      <th>end_TR</th>
      <th>len_TR</th>
      <th>start_sec</th>
      <th>end_sec</th>
      <th>dur_sec</th>
      <th>n_features</th>
      <th>seg_path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sub-01_ses-01</td>
      <td>nolags</td>
      <td>0</td>
      <td>sub-01_ses-01__seg0000</td>
      <td>1</td>
      <td>21</td>
      <td>20</td>
      <td>2.1</td>
      <td>44.1</td>
      <td>42.0</td>
      <td>400</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fu...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sub-01_ses-01</td>
      <td>nolags</td>
      <td>0</td>
      <td>sub-01_ses-01__seg0001</td>
      <td>22</td>
      <td>42</td>
      <td>20</td>
      <td>46.2</td>
      <td>88.2</td>
      <td>42.0</td>
      <td>400</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fu...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sub-01_ses-01</td>
      <td>nolags</td>
      <td>0</td>
      <td>sub-01_ses-01__seg0002</td>
      <td>43</td>
      <td>83</td>
      <td>40</td>
      <td>90.3</td>
      <td>174.3</td>
      <td>84.0</td>
      <td>400</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fu...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sub-01_ses-01</td>
      <td>nolags</td>
      <td>0</td>
      <td>sub-01_ses-01__seg0003</td>
      <td>84</td>
      <td>159</td>
      <td>75</td>
      <td>176.4</td>
      <td>333.9</td>
      <td>157.5</td>
      <td>400</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fu...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sub-01_ses-01</td>
      <td>nolags</td>
      <td>0</td>
      <td>sub-01_ses-01__seg0004</td>
      <td>160</td>
      <td>259</td>
      <td>99</td>
      <td>336.0</td>
      <td>543.9</td>
      <td>207.9</td>
      <td>400</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fu...</td>
    </tr>
  </tbody>
</table>
</div>


    Using segment path column: seg_path
    All segment paths exist. n_segments = 71



```python
# =========================
# Cell 3 — Derive subject_id from run IDs (Pipeline B uses 'run' like sub-01_ses-01)
# =========================
if "run" in manifest.columns:
    manifest["run_id"] = manifest["run"].astype(str)
elif "seg_id" in manifest.columns:
    manifest["run_id"] = manifest["seg_id"].astype(str).str.split("__seg").str[0]
else:
    raise ValueError("Manifest must have a 'run' or 'seg_id' column to derive run IDs.")

def parse_subject(run_id: str) -> str:
    m = re.search(r"(sub-\\d+)", run_id)
    if m:
        return m.group(1)
    tok = run_id.split("_")[0]
    return tok if tok.startswith("sub-") else run_id

manifest["subject_id"] = manifest["run_id"].apply(parse_subject)
subjects = sorted(manifest["subject_id"].unique().tolist())

print(f"N subjects/folds: {len(subjects)}")
print("Subjects:", subjects)

seg_counts = manifest.groupby("subject_id").size().sort_values(ascending=False)
print("\\nSegments per subject (descending):")
display(seg_counts.to_frame("n_segments").head(10))

```

    N subjects/folds: 12
    Subjects: ['sub-01', 'sub-02', 'sub-03', 'sub-08', 'sub-09', 'sub-13', 'sub-14', 'sub-16', 'sub-17', 'sub-18', 'sub-20', 'sub-21']
    \nSegments per subject (descending):



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>n_segments</th>
    </tr>
    <tr>
      <th>subject_id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sub-08</th>
      <td>10</td>
    </tr>
    <tr>
      <th>sub-17</th>
      <td>10</td>
    </tr>
    <tr>
      <th>sub-13</th>
      <td>9</td>
    </tr>
    <tr>
      <th>sub-03</th>
      <td>8</td>
    </tr>
    <tr>
      <th>sub-18</th>
      <td>6</td>
    </tr>
    <tr>
      <th>sub-01</th>
      <td>6</td>
    </tr>
    <tr>
      <th>sub-21</th>
      <td>6</td>
    </tr>
    <tr>
      <th>sub-16</th>
      <td>5</td>
    </tr>
    <tr>
      <th>sub-14</th>
      <td>4</td>
    </tr>
    <tr>
      <th>sub-20</th>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



```python
# =========================
# Cell 4 — QC summary
# =========================
# NOTE:
# - per_run_segments_minlen15.csv is **run-level**, not segment-level.
# - Segment-level lengths are taken from the manifest column `len_TR`.

# Segment-length distribution (ground truth: manifest)
if "len_TR" in manifest.columns:
    seg_lens = manifest["len_TR"].astype(int).values
    print("[Segments] n =", len(seg_lens))
    print("[Segments] min/median/max =", int(seg_lens.min()), int(np.median(seg_lens)), int(seg_lens.max()))
    print(f"[Segments] shorter than SEQ_LEN={SEQ_LEN}:", int((seg_lens < SEQ_LEN).sum()))
else:
    print("[WARN] manifest has no 'len_TR' column; cannot report segment length distribution.")

# Run-level QC table (optional)
if not QC_CSV.exists():
    print("[WARN] QC CSV not found at:", QC_CSV)
    print("       Skipping run-level QC display. (Does not block training.)")
else:
    qc = pd.read_csv(QC_CSV)
    print("\n[Run-level QC] columns:", list(qc.columns))
    display(qc.head())

    # Helpful totals if present
    for col in ["n_segments_minlen15", "kept_TR_minlen15", "usable_min_minlen15", "maxSeg_TR_minlen15"]:
        if col in qc.columns:
            print(f"[Run-level QC] {col}: min/median/max =",
                  float(qc[col].min()), float(qc[col].median()), float(qc[col].max()))

```

    [Segments] n = 71
    [Segments] min/median/max = 15 33 198
    [Segments] shorter than SEQ_LEN=10: 0
    
    [Run-level QC] columns: ['run', 'feature_mode', 'lags_tr', 'T_total_TR', 'n_segments_minlen15', 'kept_TR_minlen15', 'usable_min_minlen15', 'maxSeg_TR_minlen15']



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>run</th>
      <th>feature_mode</th>
      <th>lags_tr</th>
      <th>T_total_TR</th>
      <th>n_segments_minlen15</th>
      <th>kept_TR_minlen15</th>
      <th>usable_min_minlen15</th>
      <th>maxSeg_TR_minlen15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sub-20_ses-01</td>
      <td>nolags</td>
      <td>0</td>
      <td>288</td>
      <td>3</td>
      <td>283</td>
      <td>9.905</td>
      <td>187</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sub-01_ses-01</td>
      <td>nolags</td>
      <td>0</td>
      <td>288</td>
      <td>6</td>
      <td>277</td>
      <td>9.695</td>
      <td>99</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sub-02_ses-01</td>
      <td>nolags</td>
      <td>0</td>
      <td>288</td>
      <td>2</td>
      <td>276</td>
      <td>9.660</td>
      <td>194</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sub-14_ses-01</td>
      <td>nolags</td>
      <td>0</td>
      <td>288</td>
      <td>4</td>
      <td>270</td>
      <td>9.450</td>
      <td>102</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sub-17_ses-02</td>
      <td>nolags</td>
      <td>0</td>
      <td>288</td>
      <td>5</td>
      <td>265</td>
      <td>9.275</td>
      <td>130</td>
    </tr>
  </tbody>
</table>
</div>


    [Run-level QC] n_segments_minlen15: min/median/max = 2.0 5.0 8.0
    [Run-level QC] kept_TR_minlen15: min/median/max = 111.0 240.0 283.0
    [Run-level QC] usable_min_minlen15: min/median/max = 3.885 8.4 9.905
    [Run-level QC] maxSeg_TR_minlen15: min/median/max = 28.0 102.0 198.0



```python
# =========================
# Cell 5 — Helpers: loading, preprocessing, signatures, summaries
# =========================
def load_segments(paths):
    out, lens = [], []
    for p in paths:
        X = np.load(p)
        if X.ndim != 2:
            raise ValueError(f"Bad segment shape {X.shape} for {p}")
        out.append(X.astype(np.float32))
        lens.append(X.shape[0])
    return out, np.asarray(lens, dtype=int)

def fit_standardizer(train_list, sl):
    X = np.concatenate([x[:, sl] for x in train_list], axis=0)
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd[sd < 1e-8] = 1.0
    return mu, sd

def fit_pca(train_list, sl, n_comp):
    X = np.concatenate([x[:, sl] for x in train_list], axis=0)
    mu = X.mean(axis=0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt[:n_comp].T
    return mu, V

def build_fusion_pca_features(x_list, bold_sl, eeg_sl,
                              bold_mu, bold_sd, eeg_mu, eeg_sd,
                              bold_mu_pca, Vb, eeg_mu_pca, Ve):
    out = []
    for x in x_list:
        xz = x.copy()
        xz[:, bold_sl] = (xz[:, bold_sl] - bold_mu) / bold_sd
        xz[:, eeg_sl]  = (xz[:, eeg_sl]  - eeg_mu)  / eeg_sd
        zb = (xz[:, bold_sl] - bold_mu_pca) @ Vb
        ze = (xz[:, eeg_sl]  - eeg_mu_pca)  @ Ve
        out.append(np.concatenate([zb, ze], axis=1).astype(np.float32))
    return out

def ensure_cov_3d(covs):
    covs = np.asarray(covs)
    if covs.ndim == 2:
        K, D_ = covs.shape
        out = np.zeros((K, D_, D_), dtype=covs.dtype)
        for k in range(K):
            out[k] = np.diag(covs[k])
        return out
    return covs

def cov_to_corr_ut(cov, eps=1e-12):
    d = np.sqrt(np.clip(np.diag(cov), eps, None))
    corr = cov / (d[:, None] * d[None, :])
    iu = np.triu_indices(corr.shape[0], k=1)
    return corr[iu].astype(np.float32)

def backproject_cov_bold_only(covs_pca, Vb, nbpc):
    covs_pca = np.asarray(covs_pca)
    if covs_pca.ndim == 2:
        covs_pca = covs_pca[None, :, :]
    K = covs_pca.shape[0]
    out = []
    for k in range(K):
        Cbb = covs_pca[k, :nbpc, :nbpc]
        out.append((Vb @ Cbb @ Vb.T).astype(np.float32))
    return np.stack(out, axis=0)

def safe_get_trans_prob(model):
    for name in ["get_trans_prob", "get_trans_probs", "get_transition_matrix"]:
        if hasattr(model, name):
            return np.asarray(getattr(model, name)())
    if hasattr(model, "trans_prob"):
        return np.asarray(model.trans_prob)
    raise AttributeError("Could not extract transition probabilities from model.")

def safe_get_means(model):
    if hasattr(model, "get_means"):
        try:
            return np.asarray(model.get_means())
        except Exception:
            return None
    if hasattr(model, "means"):
        return np.asarray(model.means)
    return None

def safe_get_covariances(model):
    if hasattr(model, "get_covariances"):
        return np.asarray(model.get_covariances())
    if hasattr(model, "covariances"):
        return np.asarray(model.covariances)
    raise AttributeError("Could not extract covariances from model.")

def get_alpha_list(model, data_obj):
    if hasattr(model, "get_alpha"):
        return model.get_alpha(data_obj, concatenate=False, verbose=0)
    if hasattr(model, "get_gamma"):
        return model.get_gamma(data_obj, concatenate=False, verbose=0)
    raise AttributeError("Model does not expose get_alpha/get_gamma in this osl-dynamics version.")

def summarize_from_alpha(alpha_list, K, eps=1e-12):
    tot_T = 0
    fo_num = np.zeros(K, dtype=np.float64)
    ent_sum = 0.0
    ent_sum_norm = 0.0
    dwell_lengths = [[] for _ in range(K)]

    for a in alpha_list:
        a = np.asarray(a, dtype=np.float64)
        T = a.shape[0]
        tot_T += T
        fo_num += a.sum(axis=0)

        a_clip = np.clip(a, eps, 1.0)
        Ht = -(a_clip * np.log(a_clip)).sum(axis=1)
        ent_sum += Ht.sum()
        ent_sum_norm += (Ht / np.log(K)).sum()

        s = np.argmax(a, axis=1)
        if T > 0:
            cur = s[0]
            run = 1
            for t in range(1, T):
                if s[t] == cur:
                    run += 1
                else:
                    dwell_lengths[cur].append(run)
                    cur = s[t]
                    run = 1
            dwell_lengths[cur].append(run)

    fo = (fo_num / max(tot_T, 1)).astype(np.float32)
    ent_mean = float(ent_sum / max(tot_T, 1))
    ent_mean_norm = float(ent_sum_norm / max(tot_T, 1))
    dwell_map_mean = np.array([np.mean(d) if len(d) else np.nan for d in dwell_lengths], dtype=np.float32)
    dwell_map_median = np.array([np.median(d) if len(d) else np.nan for d in dwell_lengths], dtype=np.float32)

    return {
        "FO": fo,
        "entropy_mean": ent_mean,
        "entropy_mean_norm": ent_mean_norm,
        "dwell_map_mean_TR": dwell_map_mean,
        "dwell_map_median_TR": dwell_map_median,
        "total_T": int(tot_T),
    }

def dwell_from_A(A, eps=1e-12):
    A = np.asarray(A, dtype=np.float64)
    Akk = np.clip(np.diag(A), 0.0, 1.0 - eps)
    return (1.0 / (1.0 - Akk)).astype(np.float32)

def fold_outdir(out_root: Path, K: int, heldout_sub: str) -> Path:
    d = out_root / f"K{K:02d}" / f"fold_holdout-{heldout_sub}"
    d.mkdir(parents=True, exist_ok=True)
    return d

```


```python
# =========================
# Cell 6 — Determine feature slices from first segment (assumes X=[BOLD|EEG])
# =========================
x0 = np.load(manifest["__seg_abs"].iloc[0])
D = x0.shape[1]
if D < N_BOLD:
    raise ValueError(f"D={D} < N_BOLD={N_BOLD}. Check N_BOLD and segment features.")

N_EEG = D - N_BOLD
BOLD_SL = slice(0, N_BOLD)
EEG_SL  = slice(N_BOLD, N_BOLD + N_EEG)

print("First segment shape:", x0.shape)
print("D:", D, "| N_BOLD:", N_BOLD, "| N_EEG:", N_EEG)
print("BOLD_SL:", BOLD_SL, "EEG_SL:", EEG_SL)

feature_slices_json = SEG_ROOT / "feature_slices.json"
feature_slices_json.write_text(json.dumps({"bold": [0, N_BOLD], "eeg": [N_BOLD, N_BOLD + N_EEG]}, indent=2))
print("Wrote:", feature_slices_json)

```

    First segment shape: (20, 400)
    D: 400 | N_BOLD: 200 | N_EEG: 200
    BOLD_SL: slice(0, 200, None) EEG_SL: slice(200, 400, None)
    Wrote: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/align_trmask_lags/FINAL_v3_gnorm_allTR/intermediate/hmm_segments_minlen15_nolags/feature_slices.json



```python
# =========================
# Cell 7 — Global run meta + signature index meta
# =========================
run_meta = {
    "env": env_fingerprint(),
    "inputs": {
        "DATA_VARIANT": DATA_VARIANT,
        "FEATURE_MODE": FEATURE_MODE,
        "MINLEN": MINLEN,
        "MANIFEST_TSV": str(MANIFEST_TSV),
        "QC_CSV": str(QC_CSV),
        "OUT_ROOT": str(OUT_ROOT),
        "N_PARCELS": int(N_PARCELS),
        "TR_SEC": float(TR_SEC),
        "N_BOLD": int(N_BOLD),
        "NBPC": int(NBPC),
        "NEPC": int(NEPC),
        "SEQ_LEN": int(SEQ_LEN),
        "STEP_SIZE": STEP_SIZE,
        "BATCH_SIZE": int(BATCH_SIZE),
        "LEARNING_RATE": float(LEARNING_RATE),
        "N_EPOCHS": int(N_EPOCHS),
        "LEARN_MEANS": bool(LEARN_MEANS),
        "LEARN_COVS": bool(LEARN_COVS),
        "DIAGONAL_COVS": bool(DIAGONAL_COVS),
        "COV_EPS": float(COV_EPS),
        "INIT_METHOD": INIT_METHOD,
        "N_INITS": int(N_INITS),
        "INIT_EPOCHS": int(INIT_EPOCHS),
        "INIT_TAKE": float(INIT_TAKE),
        "USE_EARLY_STOPPING": bool(USE_EARLY_STOPPING),
        "USE_REDUCE_LR": bool(USE_REDUCE_LR),
        "VAL_MODE": VAL_MODE,
        "VAL_SUBJECT_POLICY": VAL_SUBJECT_POLICY,
        "SEED": int(SEED),
        "CONCATENATE_SESSIONS": bool(CONCATENATE_SESSIONS),
        "GPU_MEMORY_LIMIT_MB": GPU_MEMORY_LIMIT_MB,
    }
}
(OUT_ROOT / "run_meta.json").write_text(json.dumps(run_meta, indent=2))
print("Saved:", OUT_ROOT / "run_meta.json")

iu = np.triu_indices(N_PARCELS, k=1)
np.save(OUT_ROOT / "triu_idx_200.npy", np.vstack(iu).astype(np.int32))
print("Saved:", OUT_ROOT / "triu_idx_200.npy")

```

    Saved: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/run_meta.json
    Saved: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/triu_idx_200.npy



```python
# =========================
# Cell 8 — LOSO training loop (K=6/7/8) with inner validation subject (best-effort early stopping/LR schedule)
# =========================
results_rows = []

callbacks = []
if USE_EARLY_STOPPING:
    callbacks.append(tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=ES_PATIENCE,
        min_delta=ES_MIN_DELTA,
        restore_best_weights=True,
        verbose=0
    ))
if USE_REDUCE_LR:
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=LR_MIN,
        verbose=0
    ))

def choose_val_subject(train_df):
    if VAL_SUBJECT_POLICY == "max_segments":
        counts = train_df.groupby("subject_id").size()
        return counts.sort_values(ascending=False).index[0]
    return sorted(train_df["subject_id"].unique().tolist())[0]

for K in K_LIST:
    print(f"\\n==================== K={K} ====================")

    for heldout_sub in subjects:
        outdir = fold_outdir(OUT_ROOT, K, heldout_sub)
        sentinel = outdir / "covs_pca.npy"

        if RESUME_SKIP_IF_EXISTS and sentinel.exists():
            print(f"[SKIP] {heldout_sub} (K={K}) exists:", outdir)
            continue

        test_df  = manifest.loc[manifest["subject_id"] == heldout_sub].copy()
        train_df = manifest.loc[manifest["subject_id"] != heldout_sub].copy()

        train_subjects = sorted(train_df["subject_id"].unique().tolist())
        if len(train_subjects) < 2:
            print("[SKIP] Not enough training subjects after holdout:", heldout_sub)
            continue

        val_sub = choose_val_subject(train_df)
        val_df  = train_df.loc[train_df["subject_id"] == val_sub].copy()
        trn_df  = train_df.loc[train_df["subject_id"] != val_sub].copy()

        trn_paths = trn_df["__seg_abs"].tolist()
        val_paths = val_df["__seg_abs"].tolist()
        tst_paths = test_df["__seg_abs"].tolist()

        if len(trn_paths) == 0 or len(val_paths) == 0 or len(tst_paths) == 0:
            print("[SKIP] empty split for", heldout_sub, "(val_sub=", val_sub, ")")
            continue

        trn_list, trn_lens = load_segments(trn_paths)
        val_list, val_lens = load_segments(val_paths)
        tst_list, tst_lens = load_segments(tst_paths)

        n_trn_short = int((trn_lens < SEQ_LEN).sum())
        n_val_short = int((val_lens < SEQ_LEN).sum())
        n_tst_short = int((tst_lens < SEQ_LEN).sum())

        print(f"\\n--- holdout={heldout_sub} | val={val_sub} | K={K} ---")
        print(f"train segs={len(trn_list)} short={n_trn_short} | val segs={len(val_list)} short={n_val_short} | test segs={len(tst_list)} short={n_tst_short}")

        # Preprocessing fit only on training (excluding val/test)
        bold_mu, bold_sd = fit_standardizer(trn_list, BOLD_SL)
        eeg_mu,  eeg_sd  = fit_standardizer(trn_list, EEG_SL)

        trn_z = []
        for x in trn_list:
            xz = x.copy()
            xz[:, BOLD_SL] = (xz[:, BOLD_SL] - bold_mu) / bold_sd
            xz[:, EEG_SL]  = (xz[:, EEG_SL]  - eeg_mu)  / eeg_sd
            trn_z.append(xz)

        bold_mu_pca, Vb = fit_pca(trn_z, BOLD_SL, NBPC)
        eeg_mu_pca,  Ve = fit_pca(trn_z, EEG_SL,  NEPC)

        trn_pca = build_fusion_pca_features(trn_list, BOLD_SL, EEG_SL, bold_mu, bold_sd, eeg_mu, eeg_sd, bold_mu_pca, Vb, eeg_mu_pca, Ve)
        val_pca = build_fusion_pca_features(val_list, BOLD_SL, EEG_SL, bold_mu, bold_sd, eeg_mu, eeg_sd, bold_mu_pca, Vb, eeg_mu_pca, Ve)
        tst_pca = build_fusion_pca_features(tst_list, BOLD_SL, EEG_SL, bold_mu, bold_sd, eeg_mu, eeg_sd, bold_mu_pca, Vb, eeg_mu_pca, Ve)

        P = NBPC + NEPC

        trn_data = Data(trn_pca, time_axis_first=True)
        val_data = Data(val_pca, time_axis_first=True)
        tst_data = Data(tst_pca, time_axis_first=True)

        ds_kwargs = dict(sequence_length=SEQ_LEN, batch_size=BATCH_SIZE, shuffle=True, concatenate=CONCATENATE_SESSIONS)
        if STEP_SIZE is not None:
            ds_kwargs["step_size"] = STEP_SIZE

        trn_ds = trn_data.dataset(**ds_kwargs)
        val_ds = val_data.dataset(**{**ds_kwargs, "shuffle": False})

        # IMPORTANT (osl-dynamics / TF compat):
        # If concatenate=False, Data.dataset(...) can return a *list* of tf.data.Datasets
        # (one per segment/session). Keras/osl-dynamics Model.fit cannot accept a list,
        # so we combine them into a single Dataset by concatenation. This preserves
        # segment boundaries (windows are still generated within each segment).
        def _dataset_to_single(ds):
            if isinstance(ds, (list, tuple)):
                if len(ds) == 0:
                    raise ValueError("Empty dataset list returned by Data.dataset().")
                d0 = ds[0]
                for di in ds[1:]:
                    d0 = d0.concatenate(di)
                return d0
            return ds

        trn_ds = _dataset_to_single(trn_ds)
        val_ds = _dataset_to_single(val_ds)

        config = HMMConfig(
            n_states=K,
            n_channels=P,
            sequence_length=SEQ_LEN,
            learn_means=LEARN_MEANS,
            learn_covariances=LEARN_COVS,
            diagonal_covariances=DIAGONAL_COVS,
            covariances_epsilon=COV_EPS,
            n_epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            init_method=INIT_METHOD,
            n_init=N_INITS,
            n_init_epochs=INIT_EPOCHS,
            init_take=INIT_TAKE,
            best_of=1,
        )

        model = HMMModel(config)

        # Best-effort: callbacks + validation if supported
        try:
            fit_sig = inspect.signature(model.fit)
            supports_kwargs = any(p.kind == p.VAR_KEYWORD for p in fit_sig.parameters.values())
            supports_val = ("validation_data" in fit_sig.parameters) or supports_kwargs
            supports_cb  = ("callbacks" in fit_sig.parameters) or supports_kwargs

            if supports_val and supports_cb and (len(callbacks) > 0):
                history = model.fit(trn_ds, validation_data=val_ds, callbacks=callbacks, verbose=1)
            else:
                history = model.fit(trn_ds, verbose=1)
        except Exception:
            print("[WARN] Could not inspect/enable callbacks; training without early stopping/LR schedule.")
            history = model.fit(trn_ds, verbose=1)

        means = safe_get_means(model)
        covs  = ensure_cov_3d(safe_get_covariances(model))
        A     = safe_get_trans_prob(model)

        cov_bold = backproject_cov_bold_only(covs, Vb, NBPC)
        sig_ut = np.stack([cov_to_corr_ut(cov_bold[k]) for k in range(K)], axis=0)

        # Lightweight summaries on held-out subject
        summaries = {}
        if SAVE_FO_TEST or SAVE_ENTROPY_TEST or SAVE_DWELL_MAP_TEST:
            alpha_list = get_alpha_list(model, tst_data)
            s = summarize_from_alpha(alpha_list, K)
            if SAVE_FO_TEST:
                np.save(outdir / "fo_test.npy", s["FO"])
            summaries.update({
                "FO": s["FO"].tolist(),
                "entropy_mean": s["entropy_mean"],
                "entropy_mean_norm": s["entropy_mean_norm"],
                "dwell_map_mean_TR": s["dwell_map_mean_TR"].tolist() if SAVE_DWELL_MAP_TEST else None,
                "dwell_map_median_TR": s["dwell_map_median_TR"].tolist() if SAVE_DWELL_MAP_TEST else None,
                "total_T_test": s["total_T"],
            })

        if SAVE_DWELL_A_TEST:
            dA = dwell_from_A(A)
            summaries.update({
                "dwell_A_TR": dA.tolist(),
                "dwell_A_sec": (dA * TR_SEC).tolist(),
            })

        # Save core params for stability + matching
        if means is not None:
            np.save(outdir / "means_pca.npy", means.astype(np.float32))
        np.save(outdir / "covs_pca.npy", covs.astype(np.float32))
        np.save(outdir / "trans_prob.npy", A.astype(np.float32))
        np.save(outdir / "state_signature_corr_ut_bold.npy", sig_ut.astype(np.float32))

        # Preproc params
        np.savez_compressed(outdir / "preproc_params.npz",
                            bold_mu=bold_mu.astype(np.float32),
                            bold_sd=bold_sd.astype(np.float32),
                            eeg_mu=eeg_mu.astype(np.float32),
                            eeg_sd=eeg_sd.astype(np.float32),
                            bold_mu_pca=bold_mu_pca.astype(np.float32),
                            eeg_mu_pca=eeg_mu_pca.astype(np.float32),
                            Vb=Vb.astype(np.float32),
                            Ve=Ve.astype(np.float32),
                            NBPC=int(NBPC),
                            NEPC=int(NEPC),
                            P=int(P),
                            BOLD_SL=np.array([BOLD_SL.start, BOLD_SL.stop], dtype=int),
                            EEG_SL=np.array([EEG_SL.start, EEG_SL.stop], dtype=int),
                            N_BOLD=int(N_BOLD),
                            N_EEG=int(N_EEG),
                            D=int(D))

        fold_info = dict(
            heldout_subject=heldout_sub,
            val_subject=val_sub,
            K=int(K),
            n_train_segments=int(len(trn_list)),
            n_val_segments=int(len(val_list)),
            n_test_segments=int(len(tst_list)),
            n_train_short=int(n_trn_short),
            n_val_short=int(n_val_short),
            n_test_short=int(n_tst_short),
            train_len_TR=trn_lens.tolist(),
            val_len_TR=val_lens.tolist(),
            test_len_TR=tst_lens.tolist(),
            seed=int(SEED),
        )
        (outdir / "fold_info.json").write_text(json.dumps(fold_info, indent=2))
        (outdir / "fold_summaries.json").write_text(json.dumps(summaries, indent=2))

        row = {
            "heldout_subject": heldout_sub,
            "val_subject": val_sub,
            "K": K,
            "n_train_segments": len(trn_list),
            "n_val_segments": len(val_list),
            "n_test_segments": len(tst_list),
            "n_train_short": n_trn_short,
            "n_val_short": n_val_short,
            "n_test_short": n_tst_short,
            "outdir": str(outdir),
            "entropy_mean": summaries.get("entropy_mean", np.nan),
            "entropy_mean_norm": summaries.get("entropy_mean_norm", np.nan),
        }
        if "FO" in summaries:
            for i, v in enumerate(summaries["FO"]):
                row[f"FO_s{i+1:02d}"] = v
        if "dwell_A_TR" in summaries:
            for i, v in enumerate(summaries["dwell_A_TR"]):
                row[f"dwellA_TR_s{i+1:02d}"] = v
        results_rows.append(row)

        print("  saved to:", outdir)

        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass

cv = pd.DataFrame(results_rows)
cv_path = OUT_ROOT / "cv_fold_summary.tsv"
cv.to_csv(cv_path, sep="\t", index=False)
print("\\nSaved:", cv_path)
display(cv.head())

```

    \n==================== K=6 ====================
    \n--- holdout=sub-01 | val=sub-08 | K=6 ---
    train segs=55 short=0 | val segs=10 short=0 | test segs=6 short=0



    Loading files:   0%|          | 0/55 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/6 [00:00<?, ?it/s]


    I0000 00:00:1771424212.000952    4098 gpu_process_state.cc:208] Using CUDA malloc Async allocator for GPU: 0
    I0000 00:00:1771424212.001354    4098 gpu_device.cc:2019] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 4096 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 4060 Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.9


    Epoch 1/60


    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1771424216.822777    4098 cuda_solvers.cc:175] Creating GpuSolver handles for stream 0x5bd2f8af04d0


    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m43s[0m 708ms/step - ll_loss: 271.0131 - loss: 224.0932 - val_ll_loss: 250.9406 - val_loss: 246.5321 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 676ms/step - ll_loss: 268.4718 - loss: 221.9359 - val_ll_loss: 248.3898 - val_loss: 244.1353 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m39s[0m 718ms/step - ll_loss: 265.5648 - loss: 219.6861 - val_ll_loss: 246.0311 - val_loss: 241.9524 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 753ms/step - ll_loss: 262.8051 - loss: 217.4855 - val_ll_loss: 244.0264 - val_loss: 240.1097 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 695ms/step - ll_loss: 260.4808 - loss: 215.5986 - val_ll_loss: 242.2700 - val_loss: 238.4867 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 731ms/step - ll_loss: 258.5008 - loss: 213.9897 - val_ll_loss: 240.7239 - val_loss: 237.0083 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 696ms/step - ll_loss: 256.7846 - loss: 212.5830 - val_ll_loss: 239.3365 - val_loss: 235.6762 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 660ms/step - ll_loss: 255.2680 - loss: 211.3553 - val_ll_loss: 238.1048 - val_loss: 234.4899 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 729ms/step - ll_loss: 253.9303 - loss: 210.2851 - val_ll_loss: 237.0365 - val_loss: 233.4620 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m39s[0m 711ms/step - ll_loss: 252.7259 - loss: 209.3420 - val_ll_loss: 236.0696 - val_loss: 232.5321 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 729ms/step - ll_loss: 251.6183 - loss: 208.4932 - val_ll_loss: 235.2119 - val_loss: 231.7019 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 736ms/step - ll_loss: 250.6590 - loss: 207.7542 - val_ll_loss: 234.4554 - val_loss: 230.9703 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 755ms/step - ll_loss: 249.7984 - loss: 207.1001 - val_ll_loss: 233.7931 - val_loss: 230.3353 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 685ms/step - ll_loss: 249.0353 - loss: 206.5229 - val_ll_loss: 233.2151 - val_loss: 229.7937 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m45s[0m 739ms/step - ll_loss: 248.3621 - loss: 206.0135 - val_ll_loss: 232.6976 - val_loss: 229.2969 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 744ms/step - ll_loss: 247.7664 - loss: 205.5625 - val_ll_loss: 232.2370 - val_loss: 228.8530 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 675ms/step - ll_loss: 247.2374 - loss: 205.1617 - val_ll_loss: 231.8266 - val_loss: 228.4577 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 722ms/step - ll_loss: 246.7662 - loss: 204.8046 - val_ll_loss: 231.4603 - val_loss: 228.1051 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 736ms/step - ll_loss: 246.3457 - loss: 204.4857 - val_ll_loss: 231.1326 - val_loss: 227.7899 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 663ms/step - ll_loss: 245.9695 - loss: 204.2005 - val_ll_loss: 230.8390 - val_loss: 227.5075 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 675ms/step - ll_loss: 245.6326 - loss: 203.9448 - val_ll_loss: 230.5755 - val_loss: 227.2543 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 751ms/step - ll_loss: 245.3304 - loss: 203.7154 - val_ll_loss: 230.3389 - val_loss: 227.0268 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 685ms/step - ll_loss: 245.0589 - loss: 203.5094 - val_ll_loss: 230.1261 - val_loss: 226.8224 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m39s[0m 711ms/step - ll_loss: 244.8148 - loss: 203.3241 - val_ll_loss: 229.9345 - val_loss: 226.6384 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 735ms/step - ll_loss: 244.5952 - loss: 203.1573 - val_ll_loss: 229.7620 - val_loss: 226.4728 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 687ms/step - ll_loss: 244.3974 - loss: 203.0071 - val_ll_loss: 229.6065 - val_loss: 226.3235 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 688ms/step - ll_loss: 244.2193 - loss: 202.8717 - val_ll_loss: 229.4663 - val_loss: 226.1889 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m44s[0m 799ms/step - ll_loss: 244.0586 - loss: 202.7496 - val_ll_loss: 229.3397 - val_loss: 226.0674 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m39s[0m 703ms/step - ll_loss: 243.9136 - loss: 202.6395 - val_ll_loss: 229.2255 - val_loss: 225.9578 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m39s[0m 706ms/step - ll_loss: 243.7828 - loss: 202.5401 - val_ll_loss: 229.1223 - val_loss: 225.8588 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 743ms/step - ll_loss: 243.6647 - loss: 202.4503 - val_ll_loss: 229.0291 - val_loss: 225.7694 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 684ms/step - ll_loss: 243.5580 - loss: 202.3691 - val_ll_loss: 228.9448 - val_loss: 225.6886 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 677ms/step - ll_loss: 243.4617 - loss: 202.2959 - val_ll_loss: 228.8687 - val_loss: 225.6156 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 724ms/step - ll_loss: 243.3746 - loss: 202.2297 - val_ll_loss: 228.7999 - val_loss: 225.5496 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m39s[0m 717ms/step - ll_loss: 243.2959 - loss: 202.1698 - val_ll_loss: 228.7376 - val_loss: 225.4899 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 701ms/step - ll_loss: 243.2247 - loss: 202.1156 - val_ll_loss: 228.6813 - val_loss: 225.4359 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 754ms/step - ll_loss: 243.1604 - loss: 202.0667 - val_ll_loss: 228.6304 - val_loss: 225.3871 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 695ms/step - ll_loss: 243.1022 - loss: 202.0224 - val_ll_loss: 228.5843 - val_loss: 225.3430 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 692ms/step - ll_loss: 243.0496 - loss: 201.9824 - val_ll_loss: 228.5426 - val_loss: 225.3030 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 738ms/step - ll_loss: 243.0020 - loss: 201.9461 - val_ll_loss: 228.5049 - val_loss: 225.2668 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 731ms/step - ll_loss: 242.9589 - loss: 201.9133 - val_ll_loss: 228.4707 - val_loss: 225.2341 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 701ms/step - ll_loss: 242.9199 - loss: 201.8836 - val_ll_loss: 228.4398 - val_loss: 225.2045 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m42s[0m 768ms/step - ll_loss: 242.8847 - loss: 201.8568 - val_ll_loss: 228.4119 - val_loss: 225.1777 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m39s[0m 705ms/step - ll_loss: 242.8528 - loss: 201.8325 - val_ll_loss: 228.3865 - val_loss: 225.1535 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 665ms/step - ll_loss: 242.8239 - loss: 201.8105 - val_ll_loss: 228.3636 - val_loss: 225.1315 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 685ms/step - ll_loss: 242.7977 - loss: 201.7906 - val_ll_loss: 228.3429 - val_loss: 225.1116 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 555ms/step - ll_loss: 242.7741 - loss: 201.7726 - val_ll_loss: 228.3241 - val_loss: 225.0937 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 501ms/step - ll_loss: 242.7527 - loss: 201.7563 - val_ll_loss: 228.3071 - val_loss: 225.0774 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 457ms/step - ll_loss: 242.7333 - loss: 201.7415 - val_ll_loss: 228.2917 - val_loss: 225.0626 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 450ms/step - ll_loss: 242.7158 - loss: 201.7281 - val_ll_loss: 228.2778 - val_loss: 225.0493 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 418ms/step - ll_loss: 242.7000 - loss: 201.7161 - val_ll_loss: 228.2652 - val_loss: 225.0372 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 469ms/step - ll_loss: 242.6856 - loss: 201.7051 - val_ll_loss: 228.2538 - val_loss: 225.0263 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 468ms/step - ll_loss: 242.6726 - loss: 201.6952 - val_ll_loss: 228.2434 - val_loss: 225.0164 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 446ms/step - ll_loss: 242.6608 - loss: 201.6862 - val_ll_loss: 228.2341 - val_loss: 225.0074 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m22s[0m 406ms/step - ll_loss: 242.6502 - loss: 201.6781 - val_ll_loss: 228.2256 - val_loss: 224.9993 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 427ms/step - ll_loss: 242.6406 - loss: 201.6708 - val_ll_loss: 228.2179 - val_loss: 224.9920 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 482ms/step - ll_loss: 242.6318 - loss: 201.6641 - val_ll_loss: 228.2110 - val_loss: 224.9853 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 447ms/step - ll_loss: 242.6239 - loss: 201.6581 - val_ll_loss: 228.2047 - val_loss: 224.9793 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 470ms/step - ll_loss: 242.6168 - loss: 201.6526 - val_ll_loss: 228.1990 - val_loss: 224.9739 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 473ms/step - ll_loss: 242.6104 - loss: 201.6477 - val_ll_loss: 228.1939 - val_loss: 224.9690 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/6 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K06/fold_holdout-sub-01
    \n--- holdout=sub-02 | val=sub-08 | K=6 ---
    train segs=59 short=0 | val segs=10 short=0 | test segs=2 short=0



    Loading files:   0%|          | 0/59 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/2 [00:00<?, ?it/s]


    Epoch 1/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 457ms/step - ll_loss: 261.7958 - loss: 238.7238 - val_ll_loss: 236.4815 - val_loss: 233.0987 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 492ms/step - ll_loss: 258.4641 - loss: 235.7344 - val_ll_loss: 233.6679 - val_loss: 230.3605 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 465ms/step - ll_loss: 255.2646 - loss: 232.9215 - val_ll_loss: 231.3721 - val_loss: 228.1577 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 444ms/step - ll_loss: 252.6572 - loss: 230.6302 - val_ll_loss: 229.4474 - val_loss: 226.3132 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 478ms/step - ll_loss: 250.4544 - loss: 228.6936 - val_ll_loss: 227.7993 - val_loss: 224.7348 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 476ms/step - ll_loss: 248.5590 - loss: 227.0278 - val_ll_loss: 226.3722 - val_loss: 223.3686 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 417ms/step - ll_loss: 246.9115 - loss: 225.5806 - val_ll_loss: 225.1274 - val_loss: 222.1772 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 446ms/step - ll_loss: 245.4698 - loss: 224.3147 - val_ll_loss: 224.0353 - val_loss: 221.1320 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 458ms/step - ll_loss: 244.2014 - loss: 223.2016 - val_ll_loss: 223.0732 - val_loss: 220.2113 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 482ms/step - ll_loss: 243.0811 - loss: 222.2189 - val_ll_loss: 222.2222 - val_loss: 219.3970 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 420ms/step - ll_loss: 242.0882 - loss: 221.3483 - val_ll_loss: 221.4672 - val_loss: 218.6747 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 471ms/step - ll_loss: 241.2058 - loss: 220.5748 - val_ll_loss: 220.7957 - val_loss: 218.0323 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 451ms/step - ll_loss: 240.4195 - loss: 219.8859 - val_ll_loss: 220.1970 - val_loss: 217.4595 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 457ms/step - ll_loss: 239.7175 - loss: 219.2709 - val_ll_loss: 219.6621 - val_loss: 216.9478 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 458ms/step - ll_loss: 239.0895 - loss: 218.7210 - val_ll_loss: 219.1834 - val_loss: 216.4898 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 414ms/step - ll_loss: 238.5269 - loss: 218.2285 - val_ll_loss: 218.7543 - val_loss: 216.0793 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 433ms/step - ll_loss: 238.0222 - loss: 217.7867 - val_ll_loss: 218.3691 - val_loss: 215.7109 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 460ms/step - ll_loss: 237.5689 - loss: 217.3900 - val_ll_loss: 218.0230 - val_loss: 215.3799 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 482ms/step - ll_loss: 237.1612 - loss: 217.0332 - val_ll_loss: 217.7117 - val_loss: 215.0822 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 398ms/step - ll_loss: 236.7943 - loss: 216.7122 - val_ll_loss: 217.4314 - val_loss: 214.8141 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 423ms/step - ll_loss: 236.4638 - loss: 216.4231 - val_ll_loss: 217.1789 - val_loss: 214.5725 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 418ms/step - ll_loss: 236.1659 - loss: 216.1625 - val_ll_loss: 216.9511 - val_loss: 214.3546 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 421ms/step - ll_loss: 235.8972 - loss: 215.9274 - val_ll_loss: 216.7456 - val_loss: 214.1581 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 437ms/step - ll_loss: 235.6546 - loss: 215.7153 - val_ll_loss: 216.5601 - val_loss: 213.9806 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 411ms/step - ll_loss: 235.4356 - loss: 215.5237 - val_ll_loss: 216.3924 - val_loss: 213.8203 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 430ms/step - ll_loss: 235.2378 - loss: 215.3506 - val_ll_loss: 216.2410 - val_loss: 213.6755 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 410ms/step - ll_loss: 235.0590 - loss: 215.1942 - val_ll_loss: 216.1041 - val_loss: 213.5446 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 434ms/step - ll_loss: 234.8973 - loss: 215.0528 - val_ll_loss: 215.9803 - val_loss: 213.4262 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 428ms/step - ll_loss: 234.7511 - loss: 214.9250 - val_ll_loss: 215.8683 - val_loss: 213.3190 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 394ms/step - ll_loss: 234.6189 - loss: 214.8093 - val_ll_loss: 215.7669 - val_loss: 213.2222 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 415ms/step - ll_loss: 234.4993 - loss: 214.7047 - val_ll_loss: 215.6752 - val_loss: 213.1345 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 412ms/step - ll_loss: 234.3910 - loss: 214.6100 - val_ll_loss: 215.5922 - val_loss: 213.0551 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 462ms/step - ll_loss: 234.2931 - loss: 214.5243 - val_ll_loss: 215.5171 - val_loss: 212.9832 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 388ms/step - ll_loss: 234.2044 - loss: 214.4467 - val_ll_loss: 215.4491 - val_loss: 212.9182 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 422ms/step - ll_loss: 234.1241 - loss: 214.3765 - val_ll_loss: 215.3875 - val_loss: 212.8593 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 419ms/step - ll_loss: 234.0514 - loss: 214.3129 - val_ll_loss: 215.3317 - val_loss: 212.8060 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 396ms/step - ll_loss: 233.9856 - loss: 214.2553 - val_ll_loss: 215.2812 - val_loss: 212.7577 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 453ms/step - ll_loss: 233.9260 - loss: 214.2032 - val_ll_loss: 215.2354 - val_loss: 212.7140 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 422ms/step - ll_loss: 233.8720 - loss: 214.1559 - val_ll_loss: 215.1940 - val_loss: 212.6744 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 422ms/step - ll_loss: 233.8232 - loss: 214.1132 - val_ll_loss: 215.1565 - val_loss: 212.6385 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 394ms/step - ll_loss: 233.7789 - loss: 214.0745 - val_ll_loss: 215.1225 - val_loss: 212.6060 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 435ms/step - ll_loss: 233.7388 - loss: 214.0394 - val_ll_loss: 215.0917 - val_loss: 212.5765 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 455ms/step - ll_loss: 233.7025 - loss: 214.0076 - val_ll_loss: 215.0638 - val_loss: 212.5499 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 415ms/step - ll_loss: 233.6697 - loss: 213.9789 - val_ll_loss: 215.0386 - val_loss: 212.5257 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 402ms/step - ll_loss: 233.6399 - loss: 213.9528 - val_ll_loss: 215.0157 - val_loss: 212.5039 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 415ms/step - ll_loss: 233.6129 - loss: 213.9292 - val_ll_loss: 214.9950 - val_loss: 212.4841 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 446ms/step - ll_loss: 233.5885 - loss: 213.9078 - val_ll_loss: 214.9762 - val_loss: 212.4662 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 447ms/step - ll_loss: 233.5664 - loss: 213.8885 - val_ll_loss: 214.9592 - val_loss: 212.4499 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 393ms/step - ll_loss: 233.5464 - loss: 213.8709 - val_ll_loss: 214.9438 - val_loss: 212.4352 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 420ms/step - ll_loss: 233.5282 - loss: 213.8550 - val_ll_loss: 214.9299 - val_loss: 212.4218 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 422ms/step - ll_loss: 233.5118 - loss: 213.8407 - val_ll_loss: 214.9172 - val_loss: 212.4098 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 455ms/step - ll_loss: 233.4969 - loss: 213.8277 - val_ll_loss: 214.9058 - val_loss: 212.3988 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 408ms/step - ll_loss: 233.4834 - loss: 213.8159 - val_ll_loss: 214.8955 - val_loss: 212.3889 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 422ms/step - ll_loss: 233.4712 - loss: 213.8052 - val_ll_loss: 214.8861 - val_loss: 212.3800 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 428ms/step - ll_loss: 233.4602 - loss: 213.7955 - val_ll_loss: 214.8776 - val_loss: 212.3719 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 382ms/step - ll_loss: 233.4502 - loss: 213.7867 - val_ll_loss: 214.8699 - val_loss: 212.3645 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 469ms/step - ll_loss: 233.4412 - loss: 213.7788 - val_ll_loss: 214.8629 - val_loss: 212.3578 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 420ms/step - ll_loss: 233.4330 - loss: 213.7717 - val_ll_loss: 214.8566 - val_loss: 212.3518 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 418ms/step - ll_loss: 233.4256 - loss: 213.7652 - val_ll_loss: 214.8509 - val_loss: 212.3463 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 386ms/step - ll_loss: 233.4188 - loss: 213.7593 - val_ll_loss: 214.8457 - val_loss: 212.3414 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/2 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K06/fold_holdout-sub-02
    \n--- holdout=sub-03 | val=sub-08 | K=6 ---
    train segs=53 short=0 | val segs=10 short=0 | test segs=8 short=0



    Loading files:   0%|          | 0/53 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 1/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m22s[0m 414ms/step - ll_loss: 232.9271 - loss: 235.8372 - val_ll_loss: 277.5252 - val_loss: 277.0999 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 473ms/step - ll_loss: 230.2672 - loss: 233.1552 - val_ll_loss: 274.0504 - val_loss: 273.6923 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m20s[0m 385ms/step - ll_loss: 227.6590 - loss: 230.4785 - val_ll_loss: 271.0694 - val_loss: 270.7579 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m22s[0m 420ms/step - ll_loss: 225.3062 - loss: 228.0792 - val_ll_loss: 268.5801 - val_loss: 268.3100 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 432ms/step - ll_loss: 223.3306 - loss: 226.0655 - val_ll_loss: 266.4555 - val_loss: 266.2220 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m20s[0m 383ms/step - ll_loss: 221.6377 - loss: 224.3406 - val_ll_loss: 264.6198 - val_loss: 264.4186 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 447ms/step - ll_loss: 220.1705 - loss: 222.8457 - val_ll_loss: 263.0208 - val_loss: 262.8480 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 456ms/step - ll_loss: 218.8892 - loss: 221.5401 - val_ll_loss: 261.6195 - val_loss: 261.4719 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m21s[0m 387ms/step - ll_loss: 217.7641 - loss: 220.3936 - val_ll_loss: 260.3859 - val_loss: 260.2607 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 434ms/step - ll_loss: 216.7716 - loss: 219.3823 - val_ll_loss: 259.2957 - val_loss: 259.1903 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m22s[0m 423ms/step - ll_loss: 215.8932 - loss: 218.4870 - val_ll_loss: 258.3289 - val_loss: 258.2413 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m22s[0m 407ms/step - ll_loss: 215.1132 - loss: 217.6921 - val_ll_loss: 257.4694 - val_loss: 257.3976 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 464ms/step - ll_loss: 214.4189 - loss: 216.9844 - val_ll_loss: 256.7034 - val_loss: 256.6456 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m22s[0m 416ms/step - ll_loss: 213.7995 - loss: 216.3530 - val_ll_loss: 256.0192 - val_loss: 255.9741 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m21s[0m 399ms/step - ll_loss: 213.2458 - loss: 215.7886 - val_ll_loss: 255.4071 - val_loss: 255.3733 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 433ms/step - ll_loss: 212.7501 - loss: 215.2832 - val_ll_loss: 254.8585 - val_loss: 254.8349 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m22s[0m 422ms/step - ll_loss: 212.3056 - loss: 214.8300 - val_ll_loss: 254.3662 - val_loss: 254.3517 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 450ms/step - ll_loss: 211.9065 - loss: 214.4230 - val_ll_loss: 253.9239 - val_loss: 253.9176 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 428ms/step - ll_loss: 211.5478 - loss: 214.0573 - val_ll_loss: 253.5260 - val_loss: 253.5272 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 429ms/step - ll_loss: 211.2250 - loss: 213.7282 - val_ll_loss: 253.1679 - val_loss: 253.1757 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m21s[0m 401ms/step - ll_loss: 210.9344 - loss: 213.4318 - val_ll_loss: 252.8451 - val_loss: 252.8590 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m22s[0m 424ms/step - ll_loss: 210.6725 - loss: 213.1647 - val_ll_loss: 252.5541 - val_loss: 252.5733 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 460ms/step - ll_loss: 210.4363 - loss: 212.9238 - val_ll_loss: 252.2915 - val_loss: 252.3156 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m22s[0m 412ms/step - ll_loss: 210.2231 - loss: 212.7064 - val_ll_loss: 252.0544 - val_loss: 252.0830 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m22s[0m 419ms/step - ll_loss: 210.0307 - loss: 212.5102 - val_ll_loss: 251.8402 - val_loss: 251.8728 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 436ms/step - ll_loss: 209.8568 - loss: 212.3328 - val_ll_loss: 251.6467 - val_loss: 251.6829 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m21s[0m 393ms/step - ll_loss: 209.6997 - loss: 212.1727 - val_ll_loss: 251.4718 - val_loss: 251.5112 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 435ms/step - ll_loss: 209.5577 - loss: 212.0278 - val_ll_loss: 251.3136 - val_loss: 251.3559 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 475ms/step - ll_loss: 209.4293 - loss: 211.8968 - val_ll_loss: 251.1704 - val_loss: 251.2155 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m21s[0m 395ms/step - ll_loss: 209.3132 - loss: 211.7784 - val_ll_loss: 251.0409 - val_loss: 251.0884 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 430ms/step - ll_loss: 209.2081 - loss: 211.6712 - val_ll_loss: 250.9237 - val_loss: 250.9734 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 426ms/step - ll_loss: 209.1130 - loss: 211.5742 - val_ll_loss: 250.8176 - val_loss: 250.8693 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 467ms/step - ll_loss: 209.0270 - loss: 211.4864 - val_ll_loss: 250.7216 - val_loss: 250.7751 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 477ms/step - ll_loss: 208.9491 - loss: 211.4070 - val_ll_loss: 250.6347 - val_loss: 250.6897 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 466ms/step - ll_loss: 208.8785 - loss: 211.3350 - val_ll_loss: 250.5559 - val_loss: 250.6125 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 465ms/step - ll_loss: 208.8147 - loss: 211.2699 - val_ll_loss: 250.4846 - val_loss: 250.5425 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 429ms/step - ll_loss: 208.7569 - loss: 211.2109 - val_ll_loss: 250.4200 - val_loss: 250.4791 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 465ms/step - ll_loss: 208.7045 - loss: 211.1575 - val_ll_loss: 250.3615 - val_loss: 250.4217 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 500ms/step - ll_loss: 208.6571 - loss: 211.1091 - val_ll_loss: 250.3085 - val_loss: 250.3697 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 463ms/step - ll_loss: 208.6142 - loss: 211.0653 - val_ll_loss: 250.2605 - val_loss: 250.3226 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 433ms/step - ll_loss: 208.5753 - loss: 211.0257 - val_ll_loss: 250.2170 - val_loss: 250.2799 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 467ms/step - ll_loss: 208.5401 - loss: 210.9897 - val_ll_loss: 250.1777 - val_loss: 250.2413 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 480ms/step - ll_loss: 208.5082 - loss: 210.9572 - val_ll_loss: 250.1420 - val_loss: 250.2062 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 487ms/step - ll_loss: 208.4793 - loss: 210.9277 - val_ll_loss: 250.1097 - val_loss: 250.1745 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 438ms/step - ll_loss: 208.4531 - loss: 210.9010 - val_ll_loss: 250.0804 - val_loss: 250.1458 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 467ms/step - ll_loss: 208.4294 - loss: 210.8768 - val_ll_loss: 250.0539 - val_loss: 250.1198 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m43s[0m 439ms/step - ll_loss: 208.4079 - loss: 210.8549 - val_ll_loss: 250.0298 - val_loss: 250.0962 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 490ms/step - ll_loss: 208.3885 - loss: 210.8351 - val_ll_loss: 250.0080 - val_loss: 250.0748 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 471ms/step - ll_loss: 208.3709 - loss: 210.8171 - val_ll_loss: 249.9883 - val_loss: 250.0555 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m22s[0m 415ms/step - ll_loss: 208.3549 - loss: 210.8008 - val_ll_loss: 249.9705 - val_loss: 250.0379 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 451ms/step - ll_loss: 208.3405 - loss: 210.7861 - val_ll_loss: 249.9543 - val_loss: 250.0221 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 475ms/step - ll_loss: 208.3274 - loss: 210.7728 - val_ll_loss: 249.9397 - val_loss: 250.0077 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 503ms/step - ll_loss: 208.3155 - loss: 210.7607 - val_ll_loss: 249.9264 - val_loss: 249.9947 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 432ms/step - ll_loss: 208.3048 - loss: 210.7497 - val_ll_loss: 249.9144 - val_loss: 249.9829 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 470ms/step - ll_loss: 208.2951 - loss: 210.7398 - val_ll_loss: 249.9035 - val_loss: 249.9722 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 458ms/step - ll_loss: 208.2863 - loss: 210.7308 - val_ll_loss: 249.8936 - val_loss: 249.9625 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 472ms/step - ll_loss: 208.2783 - loss: 210.7227 - val_ll_loss: 249.8847 - val_loss: 249.9538 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 456ms/step - ll_loss: 208.2711 - loss: 210.7153 - val_ll_loss: 249.8766 - val_loss: 249.9458 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 475ms/step - ll_loss: 208.2645 - loss: 210.7086 - val_ll_loss: 249.8693 - val_loss: 249.9386 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 458ms/step - ll_loss: 208.2586 - loss: 210.7026 - val_ll_loss: 249.8627 - val_loss: 249.9322 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/8 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K06/fold_holdout-sub-03
    \n--- holdout=sub-08 | val=sub-17 | K=6 ---
    train segs=51 short=0 | val segs=10 short=0 | test segs=10 short=0



    Loading files:   0%|          | 0/51 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]


    Epoch 1/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 468ms/step - ll_loss: 257.6648 - loss: 231.4084 - val_ll_loss: 278.9729 - val_loss: 268.8786 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 470ms/step - ll_loss: 255.0941 - loss: 229.0232 - val_ll_loss: 277.9122 - val_loss: 267.7920 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 497ms/step - ll_loss: 252.4924 - loss: 226.7417 - val_ll_loss: 276.8973 - val_loss: 266.8299 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 467ms/step - ll_loss: 249.9645 - loss: 224.5926 - val_ll_loss: 274.8091 - val_loss: 264.6497 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 445ms/step - ll_loss: 247.8858 - loss: 222.8177 - val_ll_loss: 272.5187 - val_loss: 262.4996 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 475ms/step - ll_loss: 246.1186 - loss: 221.3063 - val_ll_loss: 270.5569 - val_loss: 260.6574 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 493ms/step - ll_loss: 244.5932 - loss: 220.0011 - val_ll_loss: 268.8522 - val_loss: 259.0562 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 477ms/step - ll_loss: 243.2604 - loss: 218.8609 - val_ll_loss: 267.3581 - val_loss: 257.6525 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m22s[0m 432ms/step - ll_loss: 242.0856 - loss: 217.8561 - val_ll_loss: 266.0466 - val_loss: 256.4203 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 458ms/step - ll_loss: 241.0496 - loss: 216.9701 - val_ll_loss: 264.8902 - val_loss: 255.3337 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 472ms/step - ll_loss: 240.1325 - loss: 216.1859 - val_ll_loss: 263.8663 - val_loss: 254.3716 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 461ms/step - ll_loss: 239.3180 - loss: 215.4894 - val_ll_loss: 262.9569 - val_loss: 253.5172 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 491ms/step - ll_loss: 238.5927 - loss: 214.8693 - val_ll_loss: 262.1472 - val_loss: 252.7562 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 475ms/step - ll_loss: 237.9455 - loss: 214.3160 - val_ll_loss: 261.4243 - val_loss: 252.0770 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m22s[0m 434ms/step - ll_loss: 237.3668 - loss: 213.8213 - val_ll_loss: 260.7780 - val_loss: 251.4697 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 459ms/step - ll_loss: 236.8485 - loss: 213.3783 - val_ll_loss: 260.1991 - val_loss: 250.9256 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 497ms/step - ll_loss: 236.3837 - loss: 212.9810 - val_ll_loss: 259.6798 - val_loss: 250.4376 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 497ms/step - ll_loss: 235.9664 - loss: 212.6244 - val_ll_loss: 259.2133 - val_loss: 249.9993 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 449ms/step - ll_loss: 235.5912 - loss: 212.3037 - val_ll_loss: 258.7939 - val_loss: 249.6051 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 460ms/step - ll_loss: 235.2535 - loss: 212.0152 - val_ll_loss: 258.4165 - val_loss: 249.2505 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 468ms/step - ll_loss: 234.9495 - loss: 211.7554 - val_ll_loss: 258.0765 - val_loss: 248.9310 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 465ms/step - ll_loss: 234.6756 - loss: 211.5213 - val_ll_loss: 257.7701 - val_loss: 248.6430 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 491ms/step - ll_loss: 234.4285 - loss: 211.3102 - val_ll_loss: 257.4937 - val_loss: 248.3833 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 471ms/step - ll_loss: 234.2056 - loss: 211.1198 - val_ll_loss: 257.2442 - val_loss: 248.1488 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 466ms/step - ll_loss: 234.0044 - loss: 210.9478 - val_ll_loss: 257.0189 - val_loss: 247.9371 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m22s[0m 434ms/step - ll_loss: 233.8227 - loss: 210.7925 - val_ll_loss: 256.8153 - val_loss: 247.7458 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 495ms/step - ll_loss: 233.6585 - loss: 210.6522 - val_ll_loss: 256.6314 - val_loss: 247.5730 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 495ms/step - ll_loss: 233.5101 - loss: 210.5254 - val_ll_loss: 256.4652 - val_loss: 247.4169 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 448ms/step - ll_loss: 233.3760 - loss: 210.4108 - val_ll_loss: 256.3149 - val_loss: 247.2756 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 460ms/step - ll_loss: 233.2547 - loss: 210.3071 - val_ll_loss: 256.1789 - val_loss: 247.1478 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 472ms/step - ll_loss: 233.1450 - loss: 210.2134 - val_ll_loss: 256.0558 - val_loss: 247.0322 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 455ms/step - ll_loss: 233.0457 - loss: 210.1285 - val_ll_loss: 255.9446 - val_loss: 246.9276 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 502ms/step - ll_loss: 232.9559 - loss: 210.0518 - val_ll_loss: 255.8438 - val_loss: 246.8330 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 482ms/step - ll_loss: 232.8746 - loss: 209.9823 - val_ll_loss: 255.7526 - val_loss: 246.7472 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 476ms/step - ll_loss: 232.8011 - loss: 209.9194 - val_ll_loss: 255.6701 - val_loss: 246.6697 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m22s[0m 432ms/step - ll_loss: 232.7346 - loss: 209.8625 - val_ll_loss: 255.5954 - val_loss: 246.5995 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 490ms/step - ll_loss: 232.6743 - loss: 209.8110 - val_ll_loss: 255.5277 - val_loss: 246.5359 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 496ms/step - ll_loss: 232.6198 - loss: 209.7644 - val_ll_loss: 255.4665 - val_loss: 246.4783 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 445ms/step - ll_loss: 232.5704 - loss: 209.7222 - val_ll_loss: 255.4110 - val_loss: 246.4262 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 428ms/step - ll_loss: 232.5256 - loss: 209.6839 - val_ll_loss: 255.3607 - val_loss: 246.3790 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 470ms/step - ll_loss: 232.4852 - loss: 209.6493 - val_ll_loss: 255.3153 - val_loss: 246.3363 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 486ms/step - ll_loss: 232.4485 - loss: 209.6180 - val_ll_loss: 255.2740 - val_loss: 246.2976 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 466ms/step - ll_loss: 232.4153 - loss: 209.5896 - val_ll_loss: 255.2367 - val_loss: 246.2625 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 469ms/step - ll_loss: 232.3853 - loss: 209.5639 - val_ll_loss: 255.2029 - val_loss: 246.2307 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 471ms/step - ll_loss: 232.3581 - loss: 209.5406 - val_ll_loss: 255.1723 - val_loss: 246.2020 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m22s[0m 428ms/step - ll_loss: 232.3334 - loss: 209.5195 - val_ll_loss: 255.1446 - val_loss: 246.1759 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 485ms/step - ll_loss: 232.3111 - loss: 209.5005 - val_ll_loss: 255.1195 - val_loss: 246.1523 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 495ms/step - ll_loss: 232.2909 - loss: 209.4832 - val_ll_loss: 255.0968 - val_loss: 246.1310 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 474ms/step - ll_loss: 232.2726 - loss: 209.4675 - val_ll_loss: 255.0762 - val_loss: 246.1116 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m22s[0m 433ms/step - ll_loss: 232.2560 - loss: 209.4533 - val_ll_loss: 255.0575 - val_loss: 246.0941 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 469ms/step - ll_loss: 232.2410 - loss: 209.4405 - val_ll_loss: 255.0406 - val_loss: 246.0783 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 492ms/step - ll_loss: 232.2274 - loss: 209.4289 - val_ll_loss: 255.0253 - val_loss: 246.0639 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 477ms/step - ll_loss: 232.2151 - loss: 209.4184 - val_ll_loss: 255.0115 - val_loss: 246.0509 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 476ms/step - ll_loss: 232.2040 - loss: 209.4088 - val_ll_loss: 254.9990 - val_loss: 246.0391 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 472ms/step - ll_loss: 232.1939 - loss: 209.4002 - val_ll_loss: 254.9876 - val_loss: 246.0284 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m22s[0m 437ms/step - ll_loss: 232.1848 - loss: 209.3924 - val_ll_loss: 254.9773 - val_loss: 246.0188 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 487ms/step - ll_loss: 232.1765 - loss: 209.3853 - val_ll_loss: 254.9680 - val_loss: 246.0100 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m42s[0m 444ms/step - ll_loss: 232.1690 - loss: 209.3789 - val_ll_loss: 254.9596 - val_loss: 246.0021 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 481ms/step - ll_loss: 232.1623 - loss: 209.3731 - val_ll_loss: 254.9519 - val_loss: 245.9949 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 470ms/step - ll_loss: 232.1561 - loss: 209.3679 - val_ll_loss: 254.9450 - val_loss: 245.9884 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/10 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K06/fold_holdout-sub-08
    \n--- holdout=sub-09 | val=sub-08 | K=6 ---
    train segs=59 short=0 | val segs=10 short=0 | test segs=2 short=0



    Loading files:   0%|          | 0/59 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/2 [00:00<?, ?it/s]


    Epoch 1/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 481ms/step - ll_loss: 266.3570 - loss: 229.7255 - val_ll_loss: 250.7356 - val_loss: 246.5940 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 474ms/step - ll_loss: 263.4716 - loss: 227.0657 - val_ll_loss: 248.1597 - val_loss: 244.1732 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 438ms/step - ll_loss: 260.7481 - loss: 224.5593 - val_ll_loss: 245.6414 - val_loss: 241.7955 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 430ms/step - ll_loss: 257.8768 - loss: 222.0341 - val_ll_loss: 243.1030 - val_loss: 239.3866 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 466ms/step - ll_loss: 255.0317 - loss: 219.7073 - val_ll_loss: 240.9953 - val_loss: 237.3642 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 498ms/step - ll_loss: 252.7238 - loss: 217.8298 - val_ll_loss: 239.2050 - val_loss: 235.6489 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 451ms/step - ll_loss: 250.8025 - loss: 216.2654 - val_ll_loss: 237.6808 - val_loss: 234.1885 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 449ms/step - ll_loss: 249.1763 - loss: 214.9373 - val_ll_loss: 236.3709 - val_loss: 232.9293 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 434ms/step - ll_loss: 247.7795 - loss: 213.7954 - val_ll_loss: 235.2352 - val_loss: 231.8384 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 487ms/step - ll_loss: 246.5663 - loss: 212.8031 - val_ll_loss: 234.2426 - val_loss: 230.8858 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 475ms/step - ll_loss: 245.5045 - loss: 211.9344 - val_ll_loss: 233.3701 - val_loss: 230.0489 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 471ms/step - ll_loss: 244.5699 - loss: 211.1696 - val_ll_loss: 232.5997 - val_loss: 229.3102 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 462ms/step - ll_loss: 243.7435 - loss: 210.4935 - val_ll_loss: 231.9167 - val_loss: 228.6556 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 445ms/step - ll_loss: 243.0103 - loss: 209.8936 - val_ll_loss: 231.3095 - val_loss: 228.0737 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m43s[0m 462ms/step - ll_loss: 242.3578 - loss: 209.3598 - val_ll_loss: 230.7683 - val_loss: 227.5552 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 434ms/step - ll_loss: 241.7758 - loss: 208.8837 - val_ll_loss: 230.2849 - val_loss: 227.0922 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 457ms/step - ll_loss: 241.2557 - loss: 208.4583 - val_ll_loss: 229.8522 - val_loss: 226.6780 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 483ms/step - ll_loss: 240.7908 - loss: 208.0783 - val_ll_loss: 229.4640 - val_loss: 226.3062 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 498ms/step - ll_loss: 240.3745 - loss: 207.7383 - val_ll_loss: 229.1144 - val_loss: 225.9715 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 470ms/step - ll_loss: 239.9974 - loss: 207.4299 - val_ll_loss: 228.8002 - val_loss: 225.6707 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 467ms/step - ll_loss: 239.6582 - loss: 207.1525 - val_ll_loss: 228.5175 - val_loss: 225.4002 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 436ms/step - ll_loss: 239.3531 - loss: 206.9029 - val_ll_loss: 228.2632 - val_loss: 225.1568 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 491ms/step - ll_loss: 239.0784 - loss: 206.6783 - val_ll_loss: 228.0341 - val_loss: 224.9376 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 467ms/step - ll_loss: 238.8309 - loss: 206.4759 - val_ll_loss: 227.8275 - val_loss: 224.7399 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 457ms/step - ll_loss: 238.6079 - loss: 206.2935 - val_ll_loss: 227.6412 - val_loss: 224.5617 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 466ms/step - ll_loss: 238.4066 - loss: 206.1289 - val_ll_loss: 227.4731 - val_loss: 224.4008 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 471ms/step - ll_loss: 238.2250 - loss: 205.9803 - val_ll_loss: 227.3211 - val_loss: 224.2555 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 469ms/step - ll_loss: 238.0609 - loss: 205.8461 - val_ll_loss: 227.1840 - val_loss: 224.1243 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 464ms/step - ll_loss: 237.9127 - loss: 205.7249 - val_ll_loss: 227.0600 - val_loss: 224.0058 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 468ms/step - ll_loss: 237.7789 - loss: 205.6154 - val_ll_loss: 226.9480 - val_loss: 223.8986 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 467ms/step - ll_loss: 237.6578 - loss: 205.5164 - val_ll_loss: 226.8466 - val_loss: 223.8016 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 455ms/step - ll_loss: 237.5484 - loss: 205.4268 - val_ll_loss: 226.7549 - val_loss: 223.7140 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 444ms/step - ll_loss: 237.4494 - loss: 205.3458 - val_ll_loss: 226.6720 - val_loss: 223.6347 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 456ms/step - ll_loss: 237.3599 - loss: 205.2726 - val_ll_loss: 226.5970 - val_loss: 223.5629 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 467ms/step - ll_loss: 237.2789 - loss: 205.2063 - val_ll_loss: 226.5291 - val_loss: 223.4980 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 506ms/step - ll_loss: 237.2056 - loss: 205.1463 - val_ll_loss: 226.4676 - val_loss: 223.4392 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 471ms/step - ll_loss: 237.1393 - loss: 205.0920 - val_ll_loss: 226.4120 - val_loss: 223.3861 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 436ms/step - ll_loss: 237.0792 - loss: 205.0429 - val_ll_loss: 226.3616 - val_loss: 223.3379 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 466ms/step - ll_loss: 237.0249 - loss: 204.9985 - val_ll_loss: 226.3161 - val_loss: 223.2943 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 496ms/step - ll_loss: 236.9758 - loss: 204.9582 - val_ll_loss: 226.2747 - val_loss: 223.2549 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 468ms/step - ll_loss: 236.9313 - loss: 204.9217 - val_ll_loss: 226.2374 - val_loss: 223.2191 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 461ms/step - ll_loss: 236.8910 - loss: 204.8888 - val_ll_loss: 226.2035 - val_loss: 223.1868 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 428ms/step - ll_loss: 236.8545 - loss: 204.8589 - val_ll_loss: 226.1729 - val_loss: 223.1575 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 469ms/step - ll_loss: 236.8215 - loss: 204.8318 - val_ll_loss: 226.1452 - val_loss: 223.1310 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 486ms/step - ll_loss: 236.7916 - loss: 204.8073 - val_ll_loss: 226.1200 - val_loss: 223.1070 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 456ms/step - ll_loss: 236.7645 - loss: 204.7851 - val_ll_loss: 226.0973 - val_loss: 223.0852 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 470ms/step - ll_loss: 236.7400 - loss: 204.7651 - val_ll_loss: 226.0767 - val_loss: 223.0655 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 438ms/step - ll_loss: 236.7178 - loss: 204.7469 - val_ll_loss: 226.0580 - val_loss: 223.0477 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 497ms/step - ll_loss: 236.6977 - loss: 204.7304 - val_ll_loss: 226.0411 - val_loss: 223.0316 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 475ms/step - ll_loss: 236.6795 - loss: 204.7155 - val_ll_loss: 226.0258 - val_loss: 223.0169 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 461ms/step - ll_loss: 236.6630 - loss: 204.7020 - val_ll_loss: 226.0120 - val_loss: 223.0037 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 458ms/step - ll_loss: 236.6481 - loss: 204.6898 - val_ll_loss: 225.9994 - val_loss: 222.9917 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 494ms/step - ll_loss: 236.6346 - loss: 204.6787 - val_ll_loss: 225.9881 - val_loss: 222.9808 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 468ms/step - ll_loss: 236.6224 - loss: 204.6687 - val_ll_loss: 225.9778 - val_loss: 222.9711 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 468ms/step - ll_loss: 236.6113 - loss: 204.6596 - val_ll_loss: 225.9685 - val_loss: 222.9621 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 458ms/step - ll_loss: 236.6012 - loss: 204.6514 - val_ll_loss: 225.9600 - val_loss: 222.9541 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 464ms/step - ll_loss: 236.5922 - loss: 204.6440 - val_ll_loss: 225.9524 - val_loss: 222.9468 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 478ms/step - ll_loss: 236.5840 - loss: 204.6372 - val_ll_loss: 225.9455 - val_loss: 222.9402 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 439ms/step - ll_loss: 236.5765 - loss: 204.6312 - val_ll_loss: 225.9392 - val_loss: 222.9342 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 455ms/step - ll_loss: 236.5698 - loss: 204.6256 - val_ll_loss: 225.9335 - val_loss: 222.9288 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/2 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K06/fold_holdout-sub-09
    \n--- holdout=sub-13 | val=sub-08 | K=6 ---
    train segs=52 short=0 | val segs=10 short=0 | test segs=9 short=0



    Loading files:   0%|          | 0/52 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/9 [00:00<?, ?it/s]


    Epoch 1/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 476ms/step - ll_loss: 253.8124 - loss: 230.8637 - val_ll_loss: 231.0597 - val_loss: 227.3138 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 523ms/step - ll_loss: 251.3100 - loss: 228.4747 - val_ll_loss: 228.8920 - val_loss: 225.3358 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 476ms/step - ll_loss: 248.7423 - loss: 226.0952 - val_ll_loss: 226.7264 - val_loss: 223.2612 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 480ms/step - ll_loss: 246.1650 - loss: 223.8735 - val_ll_loss: 224.9279 - val_loss: 221.5181 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 441ms/step - ll_loss: 244.0516 - loss: 222.0414 - val_ll_loss: 223.4285 - val_loss: 220.0708 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 501ms/step - ll_loss: 242.2879 - loss: 220.5075 - val_ll_loss: 222.1479 - val_loss: 218.8375 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 502ms/step - ll_loss: 240.7793 - loss: 219.1931 - val_ll_loss: 221.0389 - val_loss: 217.7711 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 476ms/step - ll_loss: 239.4713 - loss: 218.0524 - val_ll_loss: 220.0704 - val_loss: 216.8407 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m22s[0m 433ms/step - ll_loss: 238.3274 - loss: 217.0543 - val_ll_loss: 219.2192 - val_loss: 216.0235 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 474ms/step - ll_loss: 237.3211 - loss: 216.1761 - val_ll_loss: 218.4679 - val_loss: 215.3027 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 505ms/step - ll_loss: 236.4319 - loss: 215.3998 - val_ll_loss: 217.8022 - val_loss: 214.6642 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 484ms/step - ll_loss: 235.6432 - loss: 214.7113 - val_ll_loss: 217.2106 - val_loss: 214.0970 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m22s[0m 425ms/step - ll_loss: 234.9417 - loss: 214.0989 - val_ll_loss: 216.6835 - val_loss: 213.5918 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 437ms/step - ll_loss: 234.3162 - loss: 213.5527 - val_ll_loss: 216.2129 - val_loss: 213.1408 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m22s[0m 428ms/step - ll_loss: 233.7574 - loss: 213.0648 - val_ll_loss: 215.7919 - val_loss: 212.7375 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m22s[0m 419ms/step - ll_loss: 233.2572 - loss: 212.6281 - val_ll_loss: 215.4148 - val_loss: 212.3763 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 456ms/step - ll_loss: 232.8089 - loss: 212.2367 - val_ll_loss: 215.0765 - val_loss: 212.0523 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m22s[0m 425ms/step - ll_loss: 232.4065 - loss: 211.8854 - val_ll_loss: 214.7726 - val_loss: 211.7613 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m21s[0m 404ms/step - ll_loss: 232.0450 - loss: 211.5697 - val_ll_loss: 214.4993 - val_loss: 211.4997 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 436ms/step - ll_loss: 231.7197 - loss: 211.2857 - val_ll_loss: 214.2534 - val_loss: 211.2642 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 444ms/step - ll_loss: 231.4270 - loss: 211.0301 - val_ll_loss: 214.0318 - val_loss: 211.0521 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m42s[0m 433ms/step - ll_loss: 231.1632 - loss: 210.7997 - val_ll_loss: 213.8322 - val_loss: 210.8610 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m21s[0m 400ms/step - ll_loss: 230.9254 - loss: 210.5921 - val_ll_loss: 213.6520 - val_loss: 210.6887 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 436ms/step - ll_loss: 230.7109 - loss: 210.4048 - val_ll_loss: 213.4895 - val_loss: 210.5331 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m22s[0m 428ms/step - ll_loss: 230.5174 - loss: 210.2357 - val_ll_loss: 213.3427 - val_loss: 210.3927 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m22s[0m 423ms/step - ll_loss: 230.3426 - loss: 210.0831 - val_ll_loss: 213.2101 - val_loss: 210.2658 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 455ms/step - ll_loss: 230.1846 - loss: 209.9451 - val_ll_loss: 213.0904 - val_loss: 210.1512 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 440ms/step - ll_loss: 230.0420 - loss: 209.8205 - val_ll_loss: 212.9821 - val_loss: 210.0476 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m21s[0m 402ms/step - ll_loss: 229.9130 - loss: 209.7079 - val_ll_loss: 212.8842 - val_loss: 209.9540 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 497ms/step - ll_loss: 229.7964 - loss: 209.6060 - val_ll_loss: 212.7956 - val_loss: 209.8693 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 519ms/step - ll_loss: 229.6910 - loss: 209.5139 - val_ll_loss: 212.7155 - val_loss: 209.7926 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 498ms/step - ll_loss: 229.5957 - loss: 209.4306 - val_ll_loss: 212.6431 - val_loss: 209.7233 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 459ms/step - ll_loss: 229.5094 - loss: 209.3552 - val_ll_loss: 212.5775 - val_loss: 209.6606 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 471ms/step - ll_loss: 229.4313 - loss: 209.2870 - val_ll_loss: 212.5181 - val_loss: 209.6039 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 474ms/step - ll_loss: 229.3607 - loss: 209.2253 - val_ll_loss: 212.4644 - val_loss: 209.5525 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 497ms/step - ll_loss: 229.2967 - loss: 209.1694 - val_ll_loss: 212.4158 - val_loss: 209.5060 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 458ms/step - ll_loss: 229.2389 - loss: 209.1188 - val_ll_loss: 212.3717 - val_loss: 209.4639 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 470ms/step - ll_loss: 229.1865 - loss: 209.0730 - val_ll_loss: 212.3319 - val_loss: 209.4258 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 479ms/step - ll_loss: 229.1391 - loss: 209.0316 - val_ll_loss: 212.2959 - val_loss: 209.3913 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 477ms/step - ll_loss: 229.0961 - loss: 208.9940 - val_ll_loss: 212.2631 - val_loss: 209.3600 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 485ms/step - ll_loss: 229.0573 - loss: 208.9601 - val_ll_loss: 212.2336 - val_loss: 209.3317 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 483ms/step - ll_loss: 229.0221 - loss: 208.9293 - val_ll_loss: 212.2067 - val_loss: 209.3061 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 481ms/step - ll_loss: 228.9902 - loss: 208.9014 - val_ll_loss: 212.1824 - val_loss: 209.2829 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 439ms/step - ll_loss: 228.9614 - loss: 208.8762 - val_ll_loss: 212.1605 - val_loss: 209.2619 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 476ms/step - ll_loss: 228.9352 - loss: 208.8534 - val_ll_loss: 212.1406 - val_loss: 209.2429 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 508ms/step - ll_loss: 228.9116 - loss: 208.8327 - val_ll_loss: 212.1226 - val_loss: 209.2257 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 494ms/step - ll_loss: 228.8902 - loss: 208.8139 - val_ll_loss: 212.1062 - val_loss: 209.2100 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 437ms/step - ll_loss: 228.8708 - loss: 208.7970 - val_ll_loss: 212.0915 - val_loss: 209.1959 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 471ms/step - ll_loss: 228.8532 - loss: 208.7816 - val_ll_loss: 212.0781 - val_loss: 209.1831 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 482ms/step - ll_loss: 228.8373 - loss: 208.7677 - val_ll_loss: 212.0660 - val_loss: 209.1715 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 498ms/step - ll_loss: 228.8229 - loss: 208.7551 - val_ll_loss: 212.0550 - val_loss: 209.1611 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 448ms/step - ll_loss: 228.8099 - loss: 208.7437 - val_ll_loss: 212.0451 - val_loss: 209.1516 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 477ms/step - ll_loss: 228.7981 - loss: 208.7334 - val_ll_loss: 212.0361 - val_loss: 209.1430 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 476ms/step - ll_loss: 228.7874 - loss: 208.7241 - val_ll_loss: 212.0279 - val_loss: 209.1352 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 442ms/step - ll_loss: 228.7777 - loss: 208.7156 - val_ll_loss: 212.0205 - val_loss: 209.1281 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 506ms/step - ll_loss: 228.7690 - loss: 208.7079 - val_ll_loss: 212.0139 - val_loss: 209.1217 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 475ms/step - ll_loss: 228.7611 - loss: 208.7010 - val_ll_loss: 212.0078 - val_loss: 209.1160 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 469ms/step - ll_loss: 228.7539 - loss: 208.6947 - val_ll_loss: 212.0023 - val_loss: 209.1107 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 436ms/step - ll_loss: 228.7474 - loss: 208.6891 - val_ll_loss: 211.9974 - val_loss: 209.1060 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 503ms/step - ll_loss: 228.7415 - loss: 208.6839 - val_ll_loss: 211.9929 - val_loss: 209.1017 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/9 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K06/fold_holdout-sub-13
    \n--- holdout=sub-14 | val=sub-08 | K=6 ---
    train segs=57 short=0 | val segs=10 short=0 | test segs=4 short=0



    Loading files:   0%|          | 0/57 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/4 [00:00<?, ?it/s]


    Epoch 1/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 489ms/step - ll_loss: 256.3831 - loss: 230.6763 - val_ll_loss: 238.7610 - val_loss: 235.3885 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m42s[0m 447ms/step - ll_loss: 253.5280 - loss: 228.1759 - val_ll_loss: 236.3841 - val_loss: 233.1155 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 469ms/step - ll_loss: 250.9244 - loss: 225.9123 - val_ll_loss: 234.3026 - val_loss: 231.1240 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 501ms/step - ll_loss: 248.6364 - loss: 223.9250 - val_ll_loss: 232.4962 - val_loss: 229.3946 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 491ms/step - ll_loss: 246.6397 - loss: 222.1944 - val_ll_loss: 230.9211 - val_loss: 227.8862 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 473ms/step - ll_loss: 244.8909 - loss: 220.6810 - val_ll_loss: 229.5417 - val_loss: 226.5648 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 439ms/step - ll_loss: 243.3535 - loss: 219.3521 - val_ll_loss: 228.3289 - val_loss: 225.4029 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 492ms/step - ll_loss: 241.9972 - loss: 218.1810 - val_ll_loss: 227.2590 - val_loss: 224.3775 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 509ms/step - ll_loss: 240.7972 - loss: 217.1456 - val_ll_loss: 226.3121 - val_loss: 223.4700 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 478ms/step - ll_loss: 239.7324 - loss: 216.2276 - val_ll_loss: 225.4717 - val_loss: 222.6645 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 473ms/step - ll_loss: 238.7853 - loss: 215.4116 - val_ll_loss: 224.7241 - val_loss: 221.9479 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 437ms/step - ll_loss: 237.9412 - loss: 214.6847 - val_ll_loss: 224.0576 - val_loss: 221.3089 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 490ms/step - ll_loss: 237.1873 - loss: 214.0357 - val_ll_loss: 223.4622 - val_loss: 220.7380 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 473ms/step - ll_loss: 236.5128 - loss: 213.4554 - val_ll_loss: 222.9294 - val_loss: 220.2272 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 478ms/step - ll_loss: 235.9085 - loss: 212.9356 - val_ll_loss: 222.4519 - val_loss: 219.7693 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 443ms/step - ll_loss: 235.3664 - loss: 212.4694 - val_ll_loss: 222.0234 - val_loss: 219.3584 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 466ms/step - ll_loss: 234.8794 - loss: 212.0508 - val_ll_loss: 221.6385 - val_loss: 218.9893 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 502ms/step - ll_loss: 234.4415 - loss: 211.6744 - val_ll_loss: 221.2922 - val_loss: 218.6573 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 472ms/step - ll_loss: 234.0474 - loss: 211.3358 - val_ll_loss: 220.9805 - val_loss: 218.3583 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 474ms/step - ll_loss: 233.6924 - loss: 211.0308 - val_ll_loss: 220.6996 - val_loss: 218.0890 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 473ms/step - ll_loss: 233.3723 - loss: 210.7559 - val_ll_loss: 220.4464 - val_loss: 217.8461 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 478ms/step - ll_loss: 233.0836 - loss: 210.5079 - val_ll_loss: 220.2179 - val_loss: 217.6270 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 476ms/step - ll_loss: 232.8231 - loss: 210.2841 - val_ll_loss: 220.0116 - val_loss: 217.4292 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 472ms/step - ll_loss: 232.5879 - loss: 210.0821 - val_ll_loss: 219.8253 - val_loss: 217.2505 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 471ms/step - ll_loss: 232.3753 - loss: 209.8996 - val_ll_loss: 219.6570 - val_loss: 217.0890 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 493ms/step - ll_loss: 232.1833 - loss: 209.7347 - val_ll_loss: 219.5048 - val_loss: 216.9431 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 459ms/step - ll_loss: 232.0096 - loss: 209.5856 - val_ll_loss: 219.3671 - val_loss: 216.8111 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 467ms/step - ll_loss: 231.8526 - loss: 209.4507 - val_ll_loss: 219.2427 - val_loss: 216.6917 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 480ms/step - ll_loss: 231.7105 - loss: 209.3288 - val_ll_loss: 219.1300 - val_loss: 216.5837 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 479ms/step - ll_loss: 231.5820 - loss: 209.2184 - val_ll_loss: 219.0281 - val_loss: 216.4859 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 515ms/step - ll_loss: 231.4657 - loss: 209.1186 - val_ll_loss: 218.9359 - val_loss: 216.3975 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 452ms/step - ll_loss: 231.3605 - loss: 209.0282 - val_ll_loss: 218.8523 - val_loss: 216.3174 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 464ms/step - ll_loss: 231.2652 - loss: 208.9464 - val_ll_loss: 218.7767 - val_loss: 216.2448 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 470ms/step - ll_loss: 231.1788 - loss: 208.8723 - val_ll_loss: 218.7083 - val_loss: 216.1792 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 506ms/step - ll_loss: 231.1008 - loss: 208.8052 - val_ll_loss: 218.6462 - val_loss: 216.1197 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 487ms/step - ll_loss: 231.0300 - loss: 208.7445 - val_ll_loss: 218.5901 - val_loss: 216.0659 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 448ms/step - ll_loss: 230.9660 - loss: 208.6895 - val_ll_loss: 218.5392 - val_loss: 216.0171 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 478ms/step - ll_loss: 230.9080 - loss: 208.6397 - val_ll_loss: 218.4932 - val_loss: 215.9729 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 486ms/step - ll_loss: 230.8555 - loss: 208.5946 - val_ll_loss: 218.4514 - val_loss: 215.9329 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 508ms/step - ll_loss: 230.8079 - loss: 208.5537 - val_ll_loss: 218.4136 - val_loss: 215.8966 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 474ms/step - ll_loss: 230.7648 - loss: 208.5167 - val_ll_loss: 218.3794 - val_loss: 215.8638 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 436ms/step - ll_loss: 230.7258 - loss: 208.4832 - val_ll_loss: 218.3483 - val_loss: 215.8340 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 478ms/step - ll_loss: 230.6904 - loss: 208.4528 - val_ll_loss: 218.3203 - val_loss: 215.8071 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 519ms/step - ll_loss: 230.6584 - loss: 208.4253 - val_ll_loss: 218.2948 - val_loss: 215.7827 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 484ms/step - ll_loss: 230.6294 - loss: 208.4004 - val_ll_loss: 218.2718 - val_loss: 215.7606 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 478ms/step - ll_loss: 230.6031 - loss: 208.3779 - val_ll_loss: 218.2509 - val_loss: 215.7406 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 443ms/step - ll_loss: 230.5794 - loss: 208.3575 - val_ll_loss: 218.2319 - val_loss: 215.7224 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 485ms/step - ll_loss: 230.5578 - loss: 208.3389 - val_ll_loss: 218.2148 - val_loss: 215.7060 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 500ms/step - ll_loss: 230.5383 - loss: 208.3222 - val_ll_loss: 218.1993 - val_loss: 215.6911 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 473ms/step - ll_loss: 230.5206 - loss: 208.3070 - val_ll_loss: 218.1852 - val_loss: 215.6776 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 471ms/step - ll_loss: 230.5046 - loss: 208.2932 - val_ll_loss: 218.1725 - val_loss: 215.6654 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 439ms/step - ll_loss: 230.4901 - loss: 208.2808 - val_ll_loss: 218.1610 - val_loss: 215.6543 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 509ms/step - ll_loss: 230.4770 - loss: 208.2695 - val_ll_loss: 218.1505 - val_loss: 215.6443 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 474ms/step - ll_loss: 230.4651 - loss: 208.2593 - val_ll_loss: 218.1411 - val_loss: 215.6353 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 470ms/step - ll_loss: 230.4543 - loss: 208.2500 - val_ll_loss: 218.1325 - val_loss: 215.6271 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 485ms/step - ll_loss: 230.4446 - loss: 208.2417 - val_ll_loss: 218.1248 - val_loss: 215.6196 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 460ms/step - ll_loss: 230.4357 - loss: 208.2341 - val_ll_loss: 218.1177 - val_loss: 215.6129 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 497ms/step - ll_loss: 230.4278 - loss: 208.2272 - val_ll_loss: 218.1114 - val_loss: 215.6068 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 445ms/step - ll_loss: 230.4205 - loss: 208.2210 - val_ll_loss: 218.1056 - val_loss: 215.6013 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 471ms/step - ll_loss: 230.4140 - loss: 208.2154 - val_ll_loss: 218.1004 - val_loss: 215.5963 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/4 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K06/fold_holdout-sub-14
    \n--- holdout=sub-16 | val=sub-08 | K=6 ---
    train segs=56 short=0 | val segs=10 short=0 | test segs=5 short=0



    Loading files:   0%|          | 0/56 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/5 [00:00<?, ?it/s]


    Epoch 1/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 523ms/step - ll_loss: 258.6620 - loss: 232.4547 - val_ll_loss: 236.8197 - val_loss: 233.3744 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 492ms/step - ll_loss: 255.9601 - loss: 230.1254 - val_ll_loss: 234.8119 - val_loss: 231.4631 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 474ms/step - ll_loss: 253.5431 - loss: 228.0238 - val_ll_loss: 233.0454 - val_loss: 229.7601 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 469ms/step - ll_loss: 251.4625 - loss: 226.2057 - val_ll_loss: 231.4517 - val_loss: 228.2150 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 459ms/step - ll_loss: 249.6301 - loss: 224.5964 - val_ll_loss: 229.9745 - val_loss: 226.7830 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 484ms/step - ll_loss: 247.9862 - loss: 223.1578 - val_ll_loss: 228.5631 - val_loss: 225.4113 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 472ms/step - ll_loss: 246.4557 - loss: 221.8384 - val_ll_loss: 227.2149 - val_loss: 224.0921 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 469ms/step - ll_loss: 245.0295 - loss: 220.6341 - val_ll_loss: 225.9900 - val_loss: 222.8930 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m44s[0m 501ms/step - ll_loss: 243.7470 - loss: 219.5573 - val_ll_loss: 224.8717 - val_loss: 221.8121 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 495ms/step - ll_loss: 242.5918 - loss: 218.5906 - val_ll_loss: 223.8418 - val_loss: 220.8308 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 468ms/step - ll_loss: 241.5498 - loss: 217.7198 - val_ll_loss: 222.9167 - val_loss: 219.9544 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 454ms/step - ll_loss: 240.6200 - loss: 216.9433 - val_ll_loss: 222.0969 - val_loss: 219.1786 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 480ms/step - ll_loss: 239.7884 - loss: 216.2528 - val_ll_loss: 221.3748 - val_loss: 218.4954 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 503ms/step - ll_loss: 239.0485 - loss: 215.6396 - val_ll_loss: 220.7332 - val_loss: 217.8891 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 480ms/step - ll_loss: 238.3905 - loss: 215.0935 - val_ll_loss: 220.1599 - val_loss: 217.3485 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 472ms/step - ll_loss: 237.8041 - loss: 214.6054 - val_ll_loss: 219.6502 - val_loss: 216.8684 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m42s[0m 486ms/step - ll_loss: 237.2807 - loss: 214.1685 - val_ll_loss: 219.1972 - val_loss: 216.4417 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 509ms/step - ll_loss: 236.8127 - loss: 213.7772 - val_ll_loss: 218.7941 - val_loss: 216.0620 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 450ms/step - ll_loss: 236.3938 - loss: 213.4262 - val_ll_loss: 218.4349 - val_loss: 215.7235 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 477ms/step - ll_loss: 236.0182 - loss: 213.1112 - val_ll_loss: 218.1145 - val_loss: 215.4213 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 471ms/step - ll_loss: 235.6814 - loss: 212.8281 - val_ll_loss: 217.8284 - val_loss: 215.1512 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 499ms/step - ll_loss: 235.3788 - loss: 212.5736 - val_ll_loss: 217.5724 - val_loss: 214.9093 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 496ms/step - ll_loss: 235.1069 - loss: 212.3446 - val_ll_loss: 217.3431 - val_loss: 214.6926 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 441ms/step - ll_loss: 234.8622 - loss: 212.1384 - val_ll_loss: 217.1375 - val_loss: 214.4980 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 467ms/step - ll_loss: 234.6420 - loss: 211.9524 - val_ll_loss: 216.9531 - val_loss: 214.3233 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 494ms/step - ll_loss: 234.4435 - loss: 211.7847 - val_ll_loss: 216.7874 - val_loss: 214.1663 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 502ms/step - ll_loss: 234.2646 - loss: 211.6334 - val_ll_loss: 216.6385 - val_loss: 214.0250 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 477ms/step - ll_loss: 234.1031 - loss: 211.4966 - val_ll_loss: 216.5045 - val_loss: 213.8978 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 450ms/step - ll_loss: 233.9575 - loss: 211.3731 - val_ll_loss: 216.3840 - val_loss: 213.7832 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m45s[0m 497ms/step - ll_loss: 233.8260 - loss: 211.2615 - val_ll_loss: 216.2753 - val_loss: 213.6800 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 490ms/step - ll_loss: 233.7073 - loss: 211.1605 - val_ll_loss: 216.1775 - val_loss: 213.5869 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 482ms/step - ll_loss: 233.5999 - loss: 211.0691 - val_ll_loss: 216.0892 - val_loss: 213.5028 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 448ms/step - ll_loss: 233.5029 - loss: 210.9864 - val_ll_loss: 216.0096 - val_loss: 213.4270 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 494ms/step - ll_loss: 233.4152 - loss: 210.9115 - val_ll_loss: 215.9378 - val_loss: 213.3585 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 512ms/step - ll_loss: 233.3358 - loss: 210.8438 - val_ll_loss: 215.8730 - val_loss: 213.2966 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 480ms/step - ll_loss: 233.2641 - loss: 210.7823 - val_ll_loss: 215.8145 - val_loss: 213.2407 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 479ms/step - ll_loss: 233.1992 - loss: 210.7267 - val_ll_loss: 215.7616 - val_loss: 213.1902 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 457ms/step - ll_loss: 233.1405 - loss: 210.6763 - val_ll_loss: 215.7138 - val_loss: 213.1445 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 518ms/step - ll_loss: 233.0873 - loss: 210.6306 - val_ll_loss: 215.6706 - val_loss: 213.1032 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 479ms/step - ll_loss: 233.0391 - loss: 210.5892 - val_ll_loss: 215.6316 - val_loss: 213.0658 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 473ms/step - ll_loss: 232.9956 - loss: 210.5517 - val_ll_loss: 215.5963 - val_loss: 213.0320 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 477ms/step - ll_loss: 232.9561 - loss: 210.5176 - val_ll_loss: 215.5643 - val_loss: 213.0013 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 476ms/step - ll_loss: 232.9203 - loss: 210.4867 - val_ll_loss: 215.5354 - val_loss: 212.9737 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 493ms/step - ll_loss: 232.8879 - loss: 210.4588 - val_ll_loss: 215.5093 - val_loss: 212.9486 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 485ms/step - ll_loss: 232.8586 - loss: 210.4334 - val_ll_loss: 215.4857 - val_loss: 212.9259 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 487ms/step - ll_loss: 232.8321 - loss: 210.4103 - val_ll_loss: 215.4643 - val_loss: 212.9053 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 481ms/step - ll_loss: 232.8080 - loss: 210.3895 - val_ll_loss: 215.4449 - val_loss: 212.8867 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 478ms/step - ll_loss: 232.7862 - loss: 210.3705 - val_ll_loss: 215.4274 - val_loss: 212.8698 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 486ms/step - ll_loss: 232.7665 - loss: 210.3533 - val_ll_loss: 215.4115 - val_loss: 212.8545 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 485ms/step - ll_loss: 232.7485 - loss: 210.3377 - val_ll_loss: 215.3972 - val_loss: 212.8407 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 491ms/step - ll_loss: 232.7324 - loss: 210.3236 - val_ll_loss: 215.3842 - val_loss: 212.8281 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 510ms/step - ll_loss: 232.7177 - loss: 210.3107 - val_ll_loss: 215.3724 - val_loss: 212.8167 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 474ms/step - ll_loss: 232.7043 - loss: 210.2991 - val_ll_loss: 215.3617 - val_loss: 212.8064 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 455ms/step - ll_loss: 232.6923 - loss: 210.2885 - val_ll_loss: 215.3520 - val_loss: 212.7971 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 483ms/step - ll_loss: 232.6814 - loss: 210.2789 - val_ll_loss: 215.3432 - val_loss: 212.7886 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 493ms/step - ll_loss: 232.6714 - loss: 210.2702 - val_ll_loss: 215.3353 - val_loss: 212.7809 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 508ms/step - ll_loss: 232.6625 - loss: 210.2623 - val_ll_loss: 215.3281 - val_loss: 212.7739 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 448ms/step - ll_loss: 232.6543 - loss: 210.2551 - val_ll_loss: 215.3216 - val_loss: 212.7676 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 477ms/step - ll_loss: 232.6470 - loss: 210.2486 - val_ll_loss: 215.3156 - val_loss: 212.7618 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 476ms/step - ll_loss: 232.6403 - loss: 210.2427 - val_ll_loss: 215.3102 - val_loss: 212.7566 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/5 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K06/fold_holdout-sub-16
    \n--- holdout=sub-17 | val=sub-08 | K=6 ---
    train segs=51 short=0 | val segs=10 short=0 | test segs=10 short=0



    Loading files:   0%|          | 0/51 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]


    Epoch 1/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 536ms/step - ll_loss: 257.2477 - loss: 231.0796 - val_ll_loss: 247.4448 - val_loss: 242.9587 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 493ms/step - ll_loss: 254.8247 - loss: 229.0049 - val_ll_loss: 244.9439 - val_loss: 240.5692 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 464ms/step - ll_loss: 252.2472 - loss: 226.8079 - val_ll_loss: 242.6800 - val_loss: 238.4100 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 491ms/step - ll_loss: 249.7872 - loss: 224.6749 - val_ll_loss: 240.8255 - val_loss: 236.6381 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 509ms/step - ll_loss: 247.7144 - loss: 222.8820 - val_ll_loss: 239.2635 - val_loss: 235.1435 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 538ms/step - ll_loss: 245.9500 - loss: 221.3596 - val_ll_loss: 237.9217 - val_loss: 233.8585 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 453ms/step - ll_loss: 244.4248 - loss: 220.0458 - val_ll_loss: 236.7565 - val_loss: 232.7419 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 492ms/step - ll_loss: 243.0939 - loss: 218.9009 - val_ll_loss: 235.7372 - val_loss: 231.7645 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 500ms/step - ll_loss: 241.9251 - loss: 217.8965 - val_ll_loss: 234.8404 - val_loss: 230.9044 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 521ms/step - ll_loss: 240.8938 - loss: 217.0110 - val_ll_loss: 234.0482 - val_loss: 230.1443 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 479ms/step - ll_loss: 239.9803 - loss: 216.2271 - val_ll_loss: 233.3459 - val_loss: 229.4703 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 484ms/step - ll_loss: 239.1686 - loss: 215.5311 - val_ll_loss: 232.7215 - val_loss: 228.8709 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 501ms/step - ll_loss: 238.4455 - loss: 214.9114 - val_ll_loss: 232.1649 - val_loss: 228.3365 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 497ms/step - ll_loss: 237.8002 - loss: 214.3584 - val_ll_loss: 231.6680 - val_loss: 227.8593 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 481ms/step - ll_loss: 237.2230 - loss: 213.8641 - val_ll_loss: 231.2233 - val_loss: 227.4323 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 500ms/step - ll_loss: 236.7059 - loss: 213.4215 - val_ll_loss: 230.8247 - val_loss: 227.0495 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 490ms/step - ll_loss: 236.2421 - loss: 213.0245 - val_ll_loss: 230.4671 - val_loss: 226.7060 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 495ms/step - ll_loss: 235.8256 - loss: 212.6681 - val_ll_loss: 230.1458 - val_loss: 226.3974 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 463ms/step - ll_loss: 235.4511 - loss: 212.3477 - val_ll_loss: 229.8569 - val_loss: 226.1198 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 544ms/step - ll_loss: 235.1142 - loss: 212.0594 - val_ll_loss: 229.5967 - val_loss: 225.8698 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 493ms/step - ll_loss: 234.8106 - loss: 211.7998 - val_ll_loss: 229.3624 - val_loss: 225.6447 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 490ms/step - ll_loss: 234.5371 - loss: 211.5659 - val_ll_loss: 229.1511 - val_loss: 225.4417 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 459ms/step - ll_loss: 234.2904 - loss: 211.3549 - val_ll_loss: 228.9605 - val_loss: 225.2585 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 500ms/step - ll_loss: 234.0678 - loss: 211.1646 - val_ll_loss: 228.7885 - val_loss: 225.0932 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 521ms/step - ll_loss: 233.8669 - loss: 210.9928 - val_ll_loss: 228.6331 - val_loss: 224.9440 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 459ms/step - ll_loss: 233.6854 - loss: 210.8376 - val_ll_loss: 228.4928 - val_loss: 224.8091 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 461ms/step - ll_loss: 233.5214 - loss: 210.6974 - val_ll_loss: 228.3659 - val_loss: 224.6872 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 499ms/step - ll_loss: 233.3732 - loss: 210.5708 - val_ll_loss: 228.2512 - val_loss: 224.5770 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 520ms/step - ll_loss: 233.2392 - loss: 210.4562 - val_ll_loss: 228.1475 - val_loss: 224.4774 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 486ms/step - ll_loss: 233.1180 - loss: 210.3527 - val_ll_loss: 228.0537 - val_loss: 224.3872 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 472ms/step - ll_loss: 233.0085 - loss: 210.2590 - val_ll_loss: 227.9688 - val_loss: 224.3057 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 489ms/step - ll_loss: 232.9093 - loss: 210.1742 - val_ll_loss: 227.8921 - val_loss: 224.2319 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 484ms/step - ll_loss: 232.8196 - loss: 210.0976 - val_ll_loss: 227.8225 - val_loss: 224.1651 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 483ms/step - ll_loss: 232.7384 - loss: 210.0282 - val_ll_loss: 227.7596 - val_loss: 224.1047 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 503ms/step - ll_loss: 232.6649 - loss: 209.9654 - val_ll_loss: 227.7027 - val_loss: 224.0500 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 489ms/step - ll_loss: 232.5984 - loss: 209.9085 - val_ll_loss: 227.6511 - val_loss: 224.0004 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 481ms/step - ll_loss: 232.5383 - loss: 209.8570 - val_ll_loss: 227.6044 - val_loss: 223.9555 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 458ms/step - ll_loss: 232.4838 - loss: 209.8104 - val_ll_loss: 227.5622 - val_loss: 223.9149 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 533ms/step - ll_loss: 232.4344 - loss: 209.7682 - val_ll_loss: 227.5239 - val_loss: 223.8781 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 492ms/step - ll_loss: 232.3897 - loss: 209.7301 - val_ll_loss: 227.4892 - val_loss: 223.8448 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 493ms/step - ll_loss: 232.3493 - loss: 209.6955 - val_ll_loss: 227.4578 - val_loss: 223.8147 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 467ms/step - ll_loss: 232.3127 - loss: 209.6641 - val_ll_loss: 227.4294 - val_loss: 223.7874 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 500ms/step - ll_loss: 232.2795 - loss: 209.6358 - val_ll_loss: 227.4037 - val_loss: 223.7626 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 532ms/step - ll_loss: 232.2494 - loss: 209.6101 - val_ll_loss: 227.3803 - val_loss: 223.7402 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 506ms/step - ll_loss: 232.2223 - loss: 209.5869 - val_ll_loss: 227.3592 - val_loss: 223.7199 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 452ms/step - ll_loss: 232.1976 - loss: 209.5658 - val_ll_loss: 227.3401 - val_loss: 223.7015 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 494ms/step - ll_loss: 232.1753 - loss: 209.5468 - val_ll_loss: 227.3228 - val_loss: 223.6849 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 510ms/step - ll_loss: 232.1551 - loss: 209.5295 - val_ll_loss: 227.3071 - val_loss: 223.6698 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 523ms/step - ll_loss: 232.1369 - loss: 209.5139 - val_ll_loss: 227.2929 - val_loss: 223.6562 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 450ms/step - ll_loss: 232.1203 - loss: 209.4997 - val_ll_loss: 227.2800 - val_loss: 223.6438 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 502ms/step - ll_loss: 232.1053 - loss: 209.4869 - val_ll_loss: 227.2684 - val_loss: 223.6326 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 494ms/step - ll_loss: 232.0917 - loss: 209.4753 - val_ll_loss: 227.2578 - val_loss: 223.6225 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 499ms/step - ll_loss: 232.0795 - loss: 209.4647 - val_ll_loss: 227.2482 - val_loss: 223.6133 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 474ms/step - ll_loss: 232.0683 - loss: 209.4552 - val_ll_loss: 227.2396 - val_loss: 223.6050 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 490ms/step - ll_loss: 232.0582 - loss: 209.4466 - val_ll_loss: 227.2318 - val_loss: 223.5975 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 482ms/step - ll_loss: 232.0491 - loss: 209.4388 - val_ll_loss: 227.2247 - val_loss: 223.5907 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 451ms/step - ll_loss: 232.0408 - loss: 209.4317 - val_ll_loss: 227.2182 - val_loss: 223.5845 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 509ms/step - ll_loss: 232.0334 - loss: 209.4253 - val_ll_loss: 227.2124 - val_loss: 223.5789 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 529ms/step - ll_loss: 232.0266 - loss: 209.4196 - val_ll_loss: 227.2072 - val_loss: 223.5739 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 494ms/step - ll_loss: 232.0205 - loss: 209.4143 - val_ll_loss: 227.2024 - val_loss: 223.5693 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/10 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K06/fold_holdout-sub-17
    \n--- holdout=sub-18 | val=sub-08 | K=6 ---
    train segs=55 short=0 | val segs=10 short=0 | test segs=6 short=0



    Loading files:   0%|          | 0/55 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/6 [00:00<?, ?it/s]


    Epoch 1/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 488ms/step - ll_loss: 261.5752 - loss: 233.1357 - val_ll_loss: 248.2537 - val_loss: 244.6532 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 462ms/step - ll_loss: 258.9502 - loss: 230.8394 - val_ll_loss: 245.8075 - val_loss: 242.3433 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 515ms/step - ll_loss: 256.3876 - loss: 228.5856 - val_ll_loss: 243.5151 - val_loss: 240.1825 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 483ms/step - ll_loss: 254.0148 - loss: 226.5178 - val_ll_loss: 241.6032 - val_loss: 238.3722 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 477ms/step - ll_loss: 251.9945 - loss: 224.7676 - val_ll_loss: 239.9655 - val_loss: 236.8180 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 489ms/step - ll_loss: 250.2452 - loss: 223.2577 - val_ll_loss: 238.5444 - val_loss: 235.4673 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 482ms/step - ll_loss: 248.7178 - loss: 221.9427 - val_ll_loss: 237.2993 - val_loss: 234.2827 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 516ms/step - ll_loss: 247.3766 - loss: 220.7906 - val_ll_loss: 236.1992 - val_loss: 233.2350 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 491ms/step - ll_loss: 246.1881 - loss: 219.7712 - val_ll_loss: 235.2295 - val_loss: 232.3111 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 498ms/step - ll_loss: 245.1354 - loss: 218.8694 - val_ll_loss: 234.3717 - val_loss: 231.4933 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 495ms/step - ll_loss: 244.2005 - loss: 218.0695 - val_ll_loss: 233.6102 - val_loss: 230.7673 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 486ms/step - ll_loss: 243.3681 - loss: 217.3579 - val_ll_loss: 232.9324 - val_loss: 230.1209 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 488ms/step - ll_loss: 242.6254 - loss: 216.7234 - val_ll_loss: 232.3279 - val_loss: 229.5442 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 484ms/step - ll_loss: 241.9615 - loss: 216.1566 - val_ll_loss: 231.7875 - val_loss: 229.0285 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 497ms/step - ll_loss: 241.3670 - loss: 215.6494 - val_ll_loss: 231.3036 - val_loss: 228.5668 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 513ms/step - ll_loss: 240.8339 - loss: 215.1948 - val_ll_loss: 230.8698 - val_loss: 228.1527 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 459ms/step - ll_loss: 240.3552 - loss: 214.7868 - val_ll_loss: 230.4803 - val_loss: 227.7809 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 495ms/step - ll_loss: 239.9250 - loss: 214.4202 - val_ll_loss: 230.1301 - val_loss: 227.4467 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 493ms/step - ll_loss: 239.5380 - loss: 214.0905 - val_ll_loss: 229.8151 - val_loss: 227.1459 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 508ms/step - ll_loss: 239.1895 - loss: 213.7937 - val_ll_loss: 229.5314 - val_loss: 226.8750 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 524ms/step - ll_loss: 238.8754 - loss: 213.5263 - val_ll_loss: 229.2756 - val_loss: 226.6308 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 465ms/step - ll_loss: 238.5921 - loss: 213.2852 - val_ll_loss: 229.0451 - val_loss: 226.4107 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 483ms/step - ll_loss: 238.3365 - loss: 213.0677 - val_ll_loss: 228.8370 - val_loss: 226.2119 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 482ms/step - ll_loss: 238.1059 - loss: 212.8713 - val_ll_loss: 228.6491 - val_loss: 226.0324 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 519ms/step - ll_loss: 237.8975 - loss: 212.6940 - val_ll_loss: 228.4794 - val_loss: 225.8704 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 493ms/step - ll_loss: 237.7092 - loss: 212.5339 - val_ll_loss: 228.3260 - val_loss: 225.7239 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 460ms/step - ll_loss: 237.5391 - loss: 212.3891 - val_ll_loss: 228.1874 - val_loss: 225.5915 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 481ms/step - ll_loss: 237.3853 - loss: 212.2583 - val_ll_loss: 228.0620 - val_loss: 225.4718 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 502ms/step - ll_loss: 237.2461 - loss: 212.1399 - val_ll_loss: 227.9486 - val_loss: 225.3635 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 510ms/step - ll_loss: 237.1203 - loss: 212.0329 - val_ll_loss: 227.8460 - val_loss: 225.2654 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 456ms/step - ll_loss: 237.0064 - loss: 211.9361 - val_ll_loss: 227.7532 - val_loss: 225.1768 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 466ms/step - ll_loss: 236.9034 - loss: 211.8484 - val_ll_loss: 227.6692 - val_loss: 225.0965 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 492ms/step - ll_loss: 236.8101 - loss: 211.7691 - val_ll_loss: 227.5931 - val_loss: 225.0239 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 523ms/step - ll_loss: 236.7257 - loss: 211.6973 - val_ll_loss: 227.5243 - val_loss: 224.9581 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 492ms/step - ll_loss: 236.6493 - loss: 211.6323 - val_ll_loss: 227.4620 - val_loss: 224.8986 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 448ms/step - ll_loss: 236.5801 - loss: 211.5735 - val_ll_loss: 227.4055 - val_loss: 224.8447 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 480ms/step - ll_loss: 236.5175 - loss: 211.5202 - val_ll_loss: 227.3544 - val_loss: 224.7959 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 508ms/step - ll_loss: 236.4608 - loss: 211.4720 - val_ll_loss: 227.3081 - val_loss: 224.7517 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 518ms/step - ll_loss: 236.4094 - loss: 211.4283 - val_ll_loss: 227.2662 - val_loss: 224.7116 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 493ms/step - ll_loss: 236.3629 - loss: 211.3887 - val_ll_loss: 227.2283 - val_loss: 224.6754 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 446ms/step - ll_loss: 236.3208 - loss: 211.3529 - val_ll_loss: 227.1939 - val_loss: 224.6425 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 497ms/step - ll_loss: 236.2827 - loss: 211.3205 - val_ll_loss: 227.1628 - val_loss: 224.6127 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 525ms/step - ll_loss: 236.2482 - loss: 211.2911 - val_ll_loss: 227.1346 - val_loss: 224.5859 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 499ms/step - ll_loss: 236.2169 - loss: 211.2645 - val_ll_loss: 227.1090 - val_loss: 224.5614 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 511ms/step - ll_loss: 236.1885 - loss: 211.2404 - val_ll_loss: 227.0859 - val_loss: 224.5394 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 448ms/step - ll_loss: 236.1629 - loss: 211.2186 - val_ll_loss: 227.0649 - val_loss: 224.5193 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 521ms/step - ll_loss: 236.1396 - loss: 211.1988 - val_ll_loss: 227.0460 - val_loss: 224.5012 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 499ms/step - ll_loss: 236.1186 - loss: 211.1809 - val_ll_loss: 227.0288 - val_loss: 224.4848 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 488ms/step - ll_loss: 236.0996 - loss: 211.1647 - val_ll_loss: 227.0132 - val_loss: 224.4699 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 492ms/step - ll_loss: 236.0824 - loss: 211.1501 - val_ll_loss: 226.9991 - val_loss: 224.4565 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 468ms/step - ll_loss: 236.0667 - loss: 211.1367 - val_ll_loss: 226.9863 - val_loss: 224.4443 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 510ms/step - ll_loss: 236.0526 - loss: 211.1247 - val_ll_loss: 226.9748 - val_loss: 224.4333 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m42s[0m 461ms/step - ll_loss: 236.0398 - loss: 211.1138 - val_ll_loss: 226.9643 - val_loss: 224.4232 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 497ms/step - ll_loss: 236.0282 - loss: 211.1040 - val_ll_loss: 226.9548 - val_loss: 224.4142 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 507ms/step - ll_loss: 236.0177 - loss: 211.0950 - val_ll_loss: 226.9463 - val_loss: 224.4060 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 501ms/step - ll_loss: 236.0082 - loss: 211.0869 - val_ll_loss: 226.9385 - val_loss: 224.3985 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 481ms/step - ll_loss: 235.9995 - loss: 211.0796 - val_ll_loss: 226.9314 - val_loss: 224.3918 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 444ms/step - ll_loss: 235.9917 - loss: 211.0730 - val_ll_loss: 226.9250 - val_loss: 224.3857 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 492ms/step - ll_loss: 235.9847 - loss: 211.0669 - val_ll_loss: 226.9193 - val_loss: 224.3802 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 518ms/step - ll_loss: 235.9783 - loss: 211.0615 - val_ll_loss: 226.9140 - val_loss: 224.3752 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/6 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K06/fold_holdout-sub-18
    \n--- holdout=sub-20 | val=sub-08 | K=6 ---
    train segs=58 short=0 | val segs=10 short=0 | test segs=3 short=0



    Loading files:   0%|          | 0/58 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/3 [00:00<?, ?it/s]


    Epoch 1/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 485ms/step - ll_loss: 260.2647 - loss: 232.4389 - val_ll_loss: 243.9364 - val_loss: 241.6085 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 488ms/step - ll_loss: 257.0450 - loss: 229.6292 - val_ll_loss: 241.2667 - val_loss: 239.0232 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 480ms/step - ll_loss: 253.9406 - loss: 226.9757 - val_ll_loss: 238.9169 - val_loss: 236.7409 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 519ms/step - ll_loss: 251.4035 - loss: 224.8064 - val_ll_loss: 236.9540 - val_loss: 234.8382 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 470ms/step - ll_loss: 249.2638 - loss: 222.9761 - val_ll_loss: 235.2751 - val_loss: 233.2122 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 492ms/step - ll_loss: 247.4231 - loss: 221.4019 - val_ll_loss: 233.8219 - val_loss: 231.8053 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 488ms/step - ll_loss: 245.8228 - loss: 220.0340 - val_ll_loss: 232.5543 - val_loss: 230.5784 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 515ms/step - ll_loss: 244.4217 - loss: 218.8371 - val_ll_loss: 231.4423 - val_loss: 229.5022 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 493ms/step - ll_loss: 243.1887 - loss: 217.7843 - val_ll_loss: 230.4622 - val_loss: 228.5537 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 477ms/step - ll_loss: 242.0991 - loss: 216.8544 - val_ll_loss: 229.5954 - val_loss: 227.7149 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 451ms/step - ll_loss: 241.1331 - loss: 216.0303 - val_ll_loss: 228.8262 - val_loss: 226.9705 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 489ms/step - ll_loss: 240.2741 - loss: 215.2980 - val_ll_loss: 228.1418 - val_loss: 226.3083 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 516ms/step - ll_loss: 239.5086 - loss: 214.6455 - val_ll_loss: 227.5316 - val_loss: 225.7178 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 477ms/step - ll_loss: 238.8250 - loss: 214.0631 - val_ll_loss: 226.9863 - val_loss: 225.1902 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 488ms/step - ll_loss: 238.2133 - loss: 213.5421 - val_ll_loss: 226.4983 - val_loss: 224.7179 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 492ms/step - ll_loss: 237.6653 - loss: 213.0754 - val_ll_loss: 226.0608 - val_loss: 224.2945 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 493ms/step - ll_loss: 237.1735 - loss: 212.6566 - val_ll_loss: 225.6681 - val_loss: 223.9145 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 482ms/step - ll_loss: 236.7317 - loss: 212.2806 - val_ll_loss: 225.3151 - val_loss: 223.5730 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 483ms/step - ll_loss: 236.3344 - loss: 211.9424 - val_ll_loss: 224.9976 - val_loss: 223.2657 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 481ms/step - ll_loss: 235.9767 - loss: 211.6381 - val_ll_loss: 224.7116 - val_loss: 222.9890 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 507ms/step - ll_loss: 235.6545 - loss: 211.3640 - val_ll_loss: 224.4539 - val_loss: 222.7397 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 495ms/step - ll_loss: 235.3640 - loss: 211.1168 - val_ll_loss: 224.2216 - val_loss: 222.5148 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 452ms/step - ll_loss: 235.1020 - loss: 210.8939 - val_ll_loss: 224.0118 - val_loss: 222.3119 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 491ms/step - ll_loss: 234.8655 - loss: 210.6927 - val_ll_loss: 223.8225 - val_loss: 222.1287 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 509ms/step - ll_loss: 234.6520 - loss: 210.5110 - val_ll_loss: 223.6515 - val_loss: 221.9633 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 496ms/step - ll_loss: 234.4590 - loss: 210.3470 - val_ll_loss: 223.4970 - val_loss: 221.8137 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 489ms/step - ll_loss: 234.2847 - loss: 210.1986 - val_ll_loss: 223.3573 - val_loss: 221.6786 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 487ms/step - ll_loss: 234.1271 - loss: 210.0645 - val_ll_loss: 223.2309 - val_loss: 221.5563 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 466ms/step - ll_loss: 233.9845 - loss: 209.9433 - val_ll_loss: 223.1166 - val_loss: 221.4457 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 523ms/step - ll_loss: 233.8556 - loss: 209.8336 - val_ll_loss: 223.0132 - val_loss: 221.3457 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 498ms/step - ll_loss: 233.7390 - loss: 209.7344 - val_ll_loss: 222.9196 - val_loss: 221.2552 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 485ms/step - ll_loss: 233.6334 - loss: 209.6446 - val_ll_loss: 222.8350 - val_loss: 221.1732 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 488ms/step - ll_loss: 233.5379 - loss: 209.5633 - val_ll_loss: 222.7583 - val_loss: 221.0991 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 517ms/step - ll_loss: 233.4514 - loss: 209.4898 - val_ll_loss: 222.6889 - val_loss: 221.0319 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 450ms/step - ll_loss: 233.3731 - loss: 209.4232 - val_ll_loss: 222.6260 - val_loss: 220.9711 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 493ms/step - ll_loss: 233.3023 - loss: 209.3629 - val_ll_loss: 222.5691 - val_loss: 220.9160 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 476ms/step - ll_loss: 233.2381 - loss: 209.3083 - val_ll_loss: 222.5175 - val_loss: 220.8662 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 517ms/step - ll_loss: 233.1800 - loss: 209.2589 - val_ll_loss: 222.4709 - val_loss: 220.8210 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 495ms/step - ll_loss: 233.1274 - loss: 209.2141 - val_ll_loss: 222.4286 - val_loss: 220.7801 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 478ms/step - ll_loss: 233.0798 - loss: 209.1736 - val_ll_loss: 222.3903 - val_loss: 220.7431 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 462ms/step - ll_loss: 233.0366 - loss: 209.1368 - val_ll_loss: 222.3556 - val_loss: 220.7095 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 512ms/step - ll_loss: 232.9976 - loss: 209.1036 - val_ll_loss: 222.3242 - val_loss: 220.6792 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 495ms/step - ll_loss: 232.9622 - loss: 209.0735 - val_ll_loss: 222.2957 - val_loss: 220.6516 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 496ms/step - ll_loss: 232.9301 - loss: 209.0462 - val_ll_loss: 222.2699 - val_loss: 220.6267 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 485ms/step - ll_loss: 232.9011 - loss: 209.0215 - val_ll_loss: 222.2466 - val_loss: 220.6041 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 498ms/step - ll_loss: 232.8748 - loss: 208.9991 - val_ll_loss: 222.2255 - val_loss: 220.5837 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 506ms/step - ll_loss: 232.8510 - loss: 208.9789 - val_ll_loss: 222.2063 - val_loss: 220.5652 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 456ms/step - ll_loss: 232.8295 - loss: 208.9605 - val_ll_loss: 222.1890 - val_loss: 220.5484 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 485ms/step - ll_loss: 232.8100 - loss: 208.9439 - val_ll_loss: 222.1733 - val_loss: 220.5332 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 507ms/step - ll_loss: 232.7923 - loss: 208.9289 - val_ll_loss: 222.1590 - val_loss: 220.5194 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 523ms/step - ll_loss: 232.7762 - loss: 208.9152 - val_ll_loss: 222.1461 - val_loss: 220.5070 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 489ms/step - ll_loss: 232.7617 - loss: 208.9029 - val_ll_loss: 222.1344 - val_loss: 220.4957 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 496ms/step - ll_loss: 232.7486 - loss: 208.8917 - val_ll_loss: 222.1239 - val_loss: 220.4854 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 469ms/step - ll_loss: 232.7368 - loss: 208.8816 - val_ll_loss: 222.1143 - val_loss: 220.4762 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 521ms/step - ll_loss: 232.7260 - loss: 208.8724 - val_ll_loss: 222.1056 - val_loss: 220.4678 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 490ms/step - ll_loss: 232.7162 - loss: 208.8641 - val_ll_loss: 222.0977 - val_loss: 220.4602 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 488ms/step - ll_loss: 232.7074 - loss: 208.8566 - val_ll_loss: 222.0907 - val_loss: 220.4533 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 498ms/step - ll_loss: 232.6994 - loss: 208.8498 - val_ll_loss: 222.0842 - val_loss: 220.4471 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 521ms/step - ll_loss: 232.6922 - loss: 208.8436 - val_ll_loss: 222.0784 - val_loss: 220.4415 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 449ms/step - ll_loss: 232.6856 - loss: 208.8380 - val_ll_loss: 222.0731 - val_loss: 220.4364 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/3 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K06/fold_holdout-sub-20
    \n--- holdout=sub-21 | val=sub-08 | K=6 ---
    train segs=55 short=0 | val segs=10 short=0 | test segs=6 short=0



    Loading files:   0%|          | 0/55 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/6 [00:00<?, ?it/s]


    Epoch 1/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 465ms/step - ll_loss: 260.6831 - loss: 232.7607 - val_ll_loss: 245.8920 - val_loss: 242.4009 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 487ms/step - ll_loss: 258.1633 - loss: 230.4531 - val_ll_loss: 243.6005 - val_loss: 240.2312 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 518ms/step - ll_loss: 255.3510 - loss: 227.7888 - val_ll_loss: 240.7982 - val_loss: 237.5882 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 495ms/step - ll_loss: 252.3761 - loss: 225.2494 - val_ll_loss: 238.6175 - val_loss: 235.5077 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 477ms/step - ll_loss: 250.0203 - loss: 223.2354 - val_ll_loss: 236.8288 - val_loss: 233.7975 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 461ms/step - ll_loss: 248.0634 - loss: 221.5615 - val_ll_loss: 235.3170 - val_loss: 232.3515 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 507ms/step - ll_loss: 246.3959 - loss: 220.1353 - val_ll_loss: 234.0178 - val_loss: 231.1086 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 535ms/step - ll_loss: 244.9547 - loss: 218.9030 - val_ll_loss: 232.8900 - val_loss: 230.0295 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 502ms/step - ll_loss: 243.6979 - loss: 217.8289 - val_ll_loss: 231.9036 - val_loss: 229.0855 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 476ms/step - ll_loss: 242.5946 - loss: 216.8862 - val_ll_loss: 231.0361 - val_loss: 228.2551 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 485ms/step - ll_loss: 241.6213 - loss: 216.0550 - val_ll_loss: 230.2700 - val_loss: 227.5217 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m39s[0m 710ms/step - ll_loss: 240.7595 - loss: 215.3192 - val_ll_loss: 229.5909 - val_loss: 226.8715 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m44s[0m 796ms/step - ll_loss: 239.9938 - loss: 214.6658 - val_ll_loss: 228.9871 - val_loss: 226.2934 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 754ms/step - ll_loss: 239.3118 - loss: 214.0839 - val_ll_loss: 228.4491 - val_loss: 225.7782 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m44s[0m 739ms/step - ll_loss: 238.7031 - loss: 213.5647 - val_ll_loss: 227.9685 - val_loss: 225.3181 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m44s[0m 798ms/step - ll_loss: 238.1586 - loss: 213.1003 - val_ll_loss: 227.5386 - val_loss: 224.9063 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 734ms/step - ll_loss: 237.6709 - loss: 212.6845 - val_ll_loss: 227.1533 - val_loss: 224.5374 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 738ms/step - ll_loss: 237.2334 - loss: 212.3115 - val_ll_loss: 226.8076 - val_loss: 224.2062 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m44s[0m 811ms/step - ll_loss: 236.8404 - loss: 211.9765 - val_ll_loss: 226.4970 - val_loss: 223.9087 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 738ms/step - ll_loss: 236.4871 - loss: 211.6754 - val_ll_loss: 226.2176 - val_loss: 223.6411 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 749ms/step - ll_loss: 236.1691 - loss: 211.4045 - val_ll_loss: 225.9661 - val_loss: 223.4002 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m44s[0m 800ms/step - ll_loss: 235.8827 - loss: 211.1604 - val_ll_loss: 225.7395 - val_loss: 223.1832 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 734ms/step - ll_loss: 235.6246 - loss: 210.9405 - val_ll_loss: 225.5352 - val_loss: 222.9876 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 413ms/step - ll_loss: 235.3918 - loss: 210.7421 - val_ll_loss: 225.3510 - val_loss: 222.8111 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 456ms/step - ll_loss: 235.1817 - loss: 210.5631 - val_ll_loss: 225.1847 - val_loss: 222.6517 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 453ms/step - ll_loss: 234.9921 - loss: 210.4015 - val_ll_loss: 225.0345 - val_loss: 222.5079 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 436ms/step - ll_loss: 234.8208 - loss: 210.2556 - val_ll_loss: 224.8988 - val_loss: 222.3779 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 488ms/step - ll_loss: 234.6660 - loss: 210.1237 - val_ll_loss: 224.7762 - val_loss: 222.2605 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 452ms/step - ll_loss: 234.5261 - loss: 210.0045 - val_ll_loss: 224.6653 - val_loss: 222.1543 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 458ms/step - ll_loss: 234.3996 - loss: 209.8968 - val_ll_loss: 224.5651 - val_loss: 222.0583 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 412ms/step - ll_loss: 234.2853 - loss: 209.7993 - val_ll_loss: 224.4744 - val_loss: 221.9714 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 473ms/step - ll_loss: 234.1818 - loss: 209.7112 - val_ll_loss: 224.3924 - val_loss: 221.8928 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 479ms/step - ll_loss: 234.0882 - loss: 209.6314 - val_ll_loss: 224.3182 - val_loss: 221.8217 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 447ms/step - ll_loss: 234.0036 - loss: 209.5592 - val_ll_loss: 224.2510 - val_loss: 221.7574 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 419ms/step - ll_loss: 233.9269 - loss: 209.4940 - val_ll_loss: 224.1902 - val_loss: 221.6992 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 447ms/step - ll_loss: 233.8576 - loss: 209.4349 - val_ll_loss: 224.1351 - val_loss: 221.6465 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 488ms/step - ll_loss: 233.7948 - loss: 209.3814 - val_ll_loss: 224.0853 - val_loss: 221.5987 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 460ms/step - ll_loss: 233.7380 - loss: 209.3329 - val_ll_loss: 224.0402 - val_loss: 221.5555 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 417ms/step - ll_loss: 233.6866 - loss: 209.2891 - val_ll_loss: 223.9994 - val_loss: 221.5164 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 466ms/step - ll_loss: 233.6400 - loss: 209.2494 - val_ll_loss: 223.9624 - val_loss: 221.4810 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 456ms/step - ll_loss: 233.5978 - loss: 209.2135 - val_ll_loss: 223.9289 - val_loss: 221.4489 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 458ms/step - ll_loss: 233.5597 - loss: 209.1809 - val_ll_loss: 223.8986 - val_loss: 221.4199 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 417ms/step - ll_loss: 233.5251 - loss: 209.1515 - val_ll_loss: 223.8711 - val_loss: 221.3936 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 451ms/step - ll_loss: 233.4938 - loss: 209.1248 - val_ll_loss: 223.8462 - val_loss: 221.3698 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 455ms/step - ll_loss: 233.4655 - loss: 209.1006 - val_ll_loss: 223.8237 - val_loss: 221.3482 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 415ms/step - ll_loss: 233.4398 - loss: 209.0788 - val_ll_loss: 223.8033 - val_loss: 221.3287 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 495ms/step - ll_loss: 233.4166 - loss: 209.0589 - val_ll_loss: 223.7849 - val_loss: 221.3110 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 454ms/step - ll_loss: 233.3955 - loss: 209.0410 - val_ll_loss: 223.7681 - val_loss: 221.2950 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 469ms/step - ll_loss: 233.3765 - loss: 209.0247 - val_ll_loss: 223.7530 - val_loss: 221.2805 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 423ms/step - ll_loss: 233.3592 - loss: 209.0101 - val_ll_loss: 223.7393 - val_loss: 221.2673 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 473ms/step - ll_loss: 233.3436 - loss: 208.9967 - val_ll_loss: 223.7269 - val_loss: 221.2554 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 485ms/step - ll_loss: 233.3295 - loss: 208.9846 - val_ll_loss: 223.7157 - val_loss: 221.2447 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 446ms/step - ll_loss: 233.3167 - loss: 208.9737 - val_ll_loss: 223.7055 - val_loss: 221.2349 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m23s[0m 418ms/step - ll_loss: 233.3051 - loss: 208.9638 - val_ll_loss: 223.6962 - val_loss: 221.2261 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 446ms/step - ll_loss: 233.2946 - loss: 208.9549 - val_ll_loss: 223.6879 - val_loss: 221.2181 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 485ms/step - ll_loss: 233.2851 - loss: 208.9468 - val_ll_loss: 223.6803 - val_loss: 221.2108 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 483ms/step - ll_loss: 233.2765 - loss: 208.9394 - val_ll_loss: 223.6735 - val_loss: 221.2043 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m22s[0m 407ms/step - ll_loss: 233.2687 - loss: 208.9328 - val_ll_loss: 223.6673 - val_loss: 221.1984 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 457ms/step - ll_loss: 233.2616 - loss: 208.9268 - val_ll_loss: 223.6617 - val_loss: 221.1930 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 450ms/step - ll_loss: 233.2552 - loss: 208.9213 - val_ll_loss: 223.6566 - val_loss: 221.1882 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/6 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K06/fold_holdout-sub-21
    \n==================== K=7 ====================
    \n--- holdout=sub-01 | val=sub-08 | K=7 ---
    train segs=55 short=0 | val segs=10 short=0 | test segs=6 short=0



    Loading files:   0%|          | 0/55 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/6 [00:00<?, ?it/s]


    Epoch 1/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 537ms/step - ll_loss: 270.7593 - loss: 223.5925 - val_ll_loss: 250.4399 - val_loss: 246.0867 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 457ms/step - ll_loss: 268.2196 - loss: 221.5045 - val_ll_loss: 247.8044 - val_loss: 243.5692 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 482ms/step - ll_loss: 265.5303 - loss: 219.3240 - val_ll_loss: 245.4148 - val_loss: 241.3125 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 489ms/step - ll_loss: 263.0348 - loss: 217.3472 - val_ll_loss: 243.4529 - val_loss: 239.4509 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 514ms/step - ll_loss: 260.9308 - loss: 215.6960 - val_ll_loss: 241.7838 - val_loss: 237.8636 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 524ms/step - ll_loss: 259.1164 - loss: 214.2799 - val_ll_loss: 240.3421 - val_loss: 236.4907 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 450ms/step - ll_loss: 257.5350 - loss: 213.0501 - val_ll_loss: 239.0856 - val_loss: 235.2932 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 497ms/step - ll_loss: 256.1472 - loss: 211.9740 - val_ll_loss: 237.9837 - val_loss: 234.2422 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 479ms/step - ll_loss: 254.9234 - loss: 211.0272 - val_ll_loss: 237.0126 - val_loss: 233.3156 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 543ms/step - ll_loss: 253.8400 - loss: 210.1906 - val_ll_loss: 236.1535 - val_loss: 232.4955 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 482ms/step - ll_loss: 252.8779 - loss: 209.4488 - val_ll_loss: 235.3912 - val_loss: 231.7674 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 459ms/step - ll_loss: 252.0213 - loss: 208.7892 - val_ll_loss: 234.7128 - val_loss: 231.1195 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 492ms/step - ll_loss: 251.2569 - loss: 208.2012 - val_ll_loss: 234.1077 - val_loss: 230.5414 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 517ms/step - ll_loss: 250.5737 - loss: 207.6762 - val_ll_loss: 233.5671 - val_loss: 230.0248 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 510ms/step - ll_loss: 249.9618 - loss: 207.2064 - val_ll_loss: 233.0831 - val_loss: 229.5623 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 485ms/step - ll_loss: 249.4131 - loss: 206.7854 - val_ll_loss: 232.6493 - val_loss: 229.1476 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 461ms/step - ll_loss: 248.9205 - loss: 206.4077 - val_ll_loss: 232.2598 - val_loss: 228.7753 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 488ms/step - ll_loss: 248.4777 - loss: 206.0684 - val_ll_loss: 231.9098 - val_loss: 228.4406 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 541ms/step - ll_loss: 248.0793 - loss: 205.7632 - val_ll_loss: 231.5949 - val_loss: 228.1395 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 481ms/step - ll_loss: 247.7205 - loss: 205.4885 - val_ll_loss: 231.3113 - val_loss: 227.8684 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 493ms/step - ll_loss: 247.3972 - loss: 205.2411 - val_ll_loss: 231.0558 - val_loss: 227.6241 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 457ms/step - ll_loss: 247.1057 - loss: 205.0180 - val_ll_loss: 230.8254 - val_loss: 227.4037 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 534ms/step - ll_loss: 246.8426 - loss: 204.8167 - val_ll_loss: 230.6174 - val_loss: 227.2048 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 499ms/step - ll_loss: 246.6051 - loss: 204.6351 - val_ll_loss: 230.4297 - val_loss: 227.0253 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 489ms/step - ll_loss: 246.3907 - loss: 204.4712 - val_ll_loss: 230.2602 - val_loss: 226.8631 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 544ms/step - ll_loss: 246.1969 - loss: 204.3230 - val_ll_loss: 230.1071 - val_loss: 226.7166 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 570ms/step - ll_loss: 246.0218 - loss: 204.1891 - val_ll_loss: 229.9685 - val_loss: 226.5842 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 518ms/step - ll_loss: 245.8635 - loss: 204.0681 - val_ll_loss: 229.8434 - val_loss: 226.4645 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 506ms/step - ll_loss: 245.7202 - loss: 203.9586 - val_ll_loss: 229.7301 - val_loss: 226.3561 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 546ms/step - ll_loss: 245.5908 - loss: 203.8597 - val_ll_loss: 229.6277 - val_loss: 226.2581 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 554ms/step - ll_loss: 245.4736 - loss: 203.7701 - val_ll_loss: 229.5350 - val_loss: 226.1695 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 570ms/step - ll_loss: 245.3676 - loss: 203.6891 - val_ll_loss: 229.4511 - val_loss: 226.0892 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 551ms/step - ll_loss: 245.2716 - loss: 203.6158 - val_ll_loss: 229.3752 - val_loss: 226.0166 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 540ms/step - ll_loss: 245.1848 - loss: 203.5494 - val_ll_loss: 229.3064 - val_loss: 225.9509 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 562ms/step - ll_loss: 245.1061 - loss: 203.4893 - val_ll_loss: 229.2442 - val_loss: 225.8913 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 571ms/step - ll_loss: 245.0350 - loss: 203.4349 - val_ll_loss: 229.1878 - val_loss: 225.8374 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 510ms/step - ll_loss: 244.9705 - loss: 203.3856 - val_ll_loss: 229.1368 - val_loss: 225.7886 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 533ms/step - ll_loss: 244.9122 - loss: 203.3411 - val_ll_loss: 229.0906 - val_loss: 225.7444 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 578ms/step - ll_loss: 244.8594 - loss: 203.3007 - val_ll_loss: 229.0488 - val_loss: 225.7044 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 567ms/step - ll_loss: 244.8115 - loss: 203.2641 - val_ll_loss: 229.0109 - val_loss: 225.6682 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 548ms/step - ll_loss: 244.7682 - loss: 203.2310 - val_ll_loss: 228.9766 - val_loss: 225.6353 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 549ms/step - ll_loss: 244.7290 - loss: 203.2010 - val_ll_loss: 228.9455 - val_loss: 225.6056 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 563ms/step - ll_loss: 244.6935 - loss: 203.1739 - val_ll_loss: 228.9173 - val_loss: 225.5787 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 552ms/step - ll_loss: 244.6613 - loss: 203.1493 - val_ll_loss: 228.8918 - val_loss: 225.5543 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 549ms/step - ll_loss: 244.6322 - loss: 203.1270 - val_ll_loss: 228.8687 - val_loss: 225.5322 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 516ms/step - ll_loss: 244.6058 - loss: 203.1069 - val_ll_loss: 228.8478 - val_loss: 225.5122 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 563ms/step - ll_loss: 244.5819 - loss: 203.0886 - val_ll_loss: 228.8289 - val_loss: 225.4941 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 549ms/step - ll_loss: 244.5603 - loss: 203.0721 - val_ll_loss: 228.8117 - val_loss: 225.4777 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 541ms/step - ll_loss: 244.5407 - loss: 203.0571 - val_ll_loss: 228.7962 - val_loss: 225.4628 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 540ms/step - ll_loss: 244.5230 - loss: 203.0435 - val_ll_loss: 228.7821 - val_loss: 225.4494 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 555ms/step - ll_loss: 244.5069 - loss: 203.0312 - val_ll_loss: 228.7694 - val_loss: 225.4372 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 563ms/step - ll_loss: 244.4923 - loss: 203.0201 - val_ll_loss: 228.7579 - val_loss: 225.4261 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 536ms/step - ll_loss: 244.4792 - loss: 203.0101 - val_ll_loss: 228.7474 - val_loss: 225.4162 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 541ms/step - ll_loss: 244.4672 - loss: 203.0009 - val_ll_loss: 228.7379 - val_loss: 225.4071 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 539ms/step - ll_loss: 244.4564 - loss: 202.9926 - val_ll_loss: 228.7293 - val_loss: 225.3988 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 565ms/step - ll_loss: 244.4466 - loss: 202.9852 - val_ll_loss: 228.7216 - val_loss: 225.3914 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m44s[0m 557ms/step - ll_loss: 244.4378 - loss: 202.9784 - val_ll_loss: 228.7146 - val_loss: 225.3847 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 524ms/step - ll_loss: 244.4298 - loss: 202.9723 - val_ll_loss: 228.7082 - val_loss: 225.3786 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 553ms/step - ll_loss: 244.4225 - loss: 202.9667 - val_ll_loss: 228.7024 - val_loss: 225.3731 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 552ms/step - ll_loss: 244.4159 - loss: 202.9617 - val_ll_loss: 228.6972 - val_loss: 225.3681 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/6 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K07/fold_holdout-sub-01
    \n--- holdout=sub-02 | val=sub-08 | K=7 ---
    train segs=59 short=0 | val segs=10 short=0 | test segs=2 short=0



    Loading files:   0%|          | 0/59 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/2 [00:00<?, ?it/s]


    Epoch 1/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 544ms/step - ll_loss: 261.4203 - loss: 238.3276 - val_ll_loss: 236.1850 - val_loss: 232.6509 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 546ms/step - ll_loss: 258.5851 - loss: 235.8236 - val_ll_loss: 233.9496 - val_loss: 230.5126 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 601ms/step - ll_loss: 256.1304 - loss: 233.5962 - val_ll_loss: 231.7697 - val_loss: 228.4077 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 538ms/step - ll_loss: 253.6132 - loss: 231.3474 - val_ll_loss: 229.5612 - val_loss: 226.3185 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 541ms/step - ll_loss: 251.0493 - loss: 229.0966 - val_ll_loss: 227.6945 - val_loss: 224.5448 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 582ms/step - ll_loss: 248.9090 - loss: 227.2161 - val_ll_loss: 226.1357 - val_loss: 223.0610 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 541ms/step - ll_loss: 247.1093 - loss: 225.6354 - val_ll_loss: 224.8060 - val_loss: 221.7947 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 537ms/step - ll_loss: 245.5660 - loss: 224.2809 - val_ll_loss: 223.6571 - val_loss: 220.7000 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 550ms/step - ll_loss: 244.2273 - loss: 223.1066 - val_ll_loss: 222.6557 - val_loss: 219.7455 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 580ms/step - ll_loss: 243.0566 - loss: 222.0802 - val_ll_loss: 221.7772 - val_loss: 218.9080 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 550ms/step - ll_loss: 242.0269 - loss: 221.1779 - val_ll_loss: 221.0028 - val_loss: 218.1696 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 582ms/step - ll_loss: 241.1171 - loss: 220.3811 - val_ll_loss: 220.3175 - val_loss: 217.5160 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 568ms/step - ll_loss: 240.3102 - loss: 219.6748 - val_ll_loss: 219.7089 - val_loss: 216.9355 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 545ms/step - ll_loss: 239.5927 - loss: 219.0468 - val_ll_loss: 219.1671 - val_loss: 216.4187 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 538ms/step - ll_loss: 238.9529 - loss: 218.4871 - val_ll_loss: 218.6836 - val_loss: 215.9575 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 538ms/step - ll_loss: 238.3814 - loss: 217.9872 - val_ll_loss: 218.2513 - val_loss: 215.5451 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 571ms/step - ll_loss: 237.8699 - loss: 217.5399 - val_ll_loss: 217.8642 - val_loss: 215.1757 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 554ms/step - ll_loss: 237.4113 - loss: 217.1389 - val_ll_loss: 217.5169 - val_loss: 214.8443 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 541ms/step - ll_loss: 236.9997 - loss: 216.7791 - val_ll_loss: 217.2050 - val_loss: 214.5467 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 534ms/step - ll_loss: 236.6298 - loss: 216.4558 - val_ll_loss: 216.9246 - val_loss: 214.2791 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 570ms/step - ll_loss: 236.2971 - loss: 216.1651 - val_ll_loss: 216.6722 - val_loss: 214.0383 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 539ms/step - ll_loss: 235.9975 - loss: 215.9033 - val_ll_loss: 216.4449 - val_loss: 213.8214 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 505ms/step - ll_loss: 235.7275 - loss: 215.6674 - val_ll_loss: 216.2400 - val_loss: 213.6258 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 557ms/step - ll_loss: 235.4841 - loss: 215.4547 - val_ll_loss: 216.0551 - val_loss: 213.4495 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 555ms/step - ll_loss: 235.2645 - loss: 215.2629 - val_ll_loss: 215.8883 - val_loss: 213.2903 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 542ms/step - ll_loss: 235.0663 - loss: 215.0897 - val_ll_loss: 215.7377 - val_loss: 213.1465 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 539ms/step - ll_loss: 234.8873 - loss: 214.9333 - val_ll_loss: 215.6016 - val_loss: 213.0166 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 556ms/step - ll_loss: 234.7256 - loss: 214.7921 - val_ll_loss: 215.4786 - val_loss: 212.8993 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 542ms/step - ll_loss: 234.5794 - loss: 214.6644 - val_ll_loss: 215.3675 - val_loss: 212.7932 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 540ms/step - ll_loss: 234.4473 - loss: 214.5490 - val_ll_loss: 215.2669 - val_loss: 212.6972 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 531ms/step - ll_loss: 234.3279 - loss: 214.4446 - val_ll_loss: 215.1761 - val_loss: 212.6105 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 569ms/step - ll_loss: 234.2198 - loss: 214.3503 - val_ll_loss: 215.0938 - val_loss: 212.5320 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 537ms/step - ll_loss: 234.1221 - loss: 214.2648 - val_ll_loss: 215.0194 - val_loss: 212.4609 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 535ms/step - ll_loss: 234.0336 - loss: 214.1875 - val_ll_loss: 214.9520 - val_loss: 212.3966 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 533ms/step - ll_loss: 233.9536 - loss: 214.1176 - val_ll_loss: 214.8910 - val_loss: 212.3385 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 580ms/step - ll_loss: 233.8812 - loss: 214.0544 - val_ll_loss: 214.8358 - val_loss: 212.2858 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 532ms/step - ll_loss: 233.8156 - loss: 213.9971 - val_ll_loss: 214.7859 - val_loss: 212.2381 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 569ms/step - ll_loss: 233.7563 - loss: 213.9452 - val_ll_loss: 214.7406 - val_loss: 212.1949 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 551ms/step - ll_loss: 233.7025 - loss: 213.8983 - val_ll_loss: 214.6997 - val_loss: 212.1559 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 557ms/step - ll_loss: 233.6539 - loss: 213.8558 - val_ll_loss: 214.6626 - val_loss: 212.1205 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 532ms/step - ll_loss: 233.6099 - loss: 213.8173 - val_ll_loss: 214.6290 - val_loss: 212.0884 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 501ms/step - ll_loss: 233.5700 - loss: 213.7825 - val_ll_loss: 214.5986 - val_loss: 212.0594 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 577ms/step - ll_loss: 233.5339 - loss: 213.7509 - val_ll_loss: 214.5710 - val_loss: 212.0331 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 550ms/step - ll_loss: 233.5012 - loss: 213.7223 - val_ll_loss: 214.5461 - val_loss: 212.0093 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 539ms/step - ll_loss: 233.4717 - loss: 213.6965 - val_ll_loss: 214.5235 - val_loss: 211.9877 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 545ms/step - ll_loss: 233.4448 - loss: 213.6730 - val_ll_loss: 214.5030 - val_loss: 211.9682 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 581ms/step - ll_loss: 233.4205 - loss: 213.6518 - val_ll_loss: 214.4845 - val_loss: 211.9505 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 542ms/step - ll_loss: 233.3986 - loss: 213.6326 - val_ll_loss: 214.4677 - val_loss: 211.9345 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 547ms/step - ll_loss: 233.3787 - loss: 213.6152 - val_ll_loss: 214.4525 - val_loss: 211.9200 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 548ms/step - ll_loss: 233.3607 - loss: 213.5995 - val_ll_loss: 214.4388 - val_loss: 211.9069 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 563ms/step - ll_loss: 233.3444 - loss: 213.5852 - val_ll_loss: 214.4263 - val_loss: 211.8950 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 544ms/step - ll_loss: 233.3296 - loss: 213.5723 - val_ll_loss: 214.4150 - val_loss: 211.8842 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 532ms/step - ll_loss: 233.3162 - loss: 213.5606 - val_ll_loss: 214.4048 - val_loss: 211.8745 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 562ms/step - ll_loss: 233.3041 - loss: 213.5500 - val_ll_loss: 214.3956 - val_loss: 211.8657 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 553ms/step - ll_loss: 233.2932 - loss: 213.5405 - val_ll_loss: 214.3872 - val_loss: 211.8577 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 536ms/step - ll_loss: 233.2832 - loss: 213.5318 - val_ll_loss: 214.3796 - val_loss: 211.8504 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 544ms/step - ll_loss: 233.2742 - loss: 213.5239 - val_ll_loss: 214.3727 - val_loss: 211.8439 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 572ms/step - ll_loss: 233.2661 - loss: 213.5168 - val_ll_loss: 214.3665 - val_loss: 211.8379 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 532ms/step - ll_loss: 233.2588 - loss: 213.5103 - val_ll_loss: 214.3609 - val_loss: 211.8326 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 534ms/step - ll_loss: 233.2521 - loss: 213.5045 - val_ll_loss: 214.3558 - val_loss: 211.8277 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/2 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K07/fold_holdout-sub-02
    \n--- holdout=sub-03 | val=sub-08 | K=7 ---
    train segs=53 short=0 | val segs=10 short=0 | test segs=8 short=0



    Loading files:   0%|          | 0/53 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 1/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 559ms/step - ll_loss: 232.8809 - loss: 235.6277 - val_ll_loss: 276.8467 - val_loss: 276.1630 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 554ms/step - ll_loss: 230.1774 - loss: 232.9128 - val_ll_loss: 273.4616 - val_loss: 272.8400 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 552ms/step - ll_loss: 227.4525 - loss: 230.1391 - val_ll_loss: 270.5189 - val_loss: 270.0224 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 546ms/step - ll_loss: 225.1266 - loss: 227.7732 - val_ll_loss: 268.0503 - val_loss: 267.6137 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 537ms/step - ll_loss: 223.1707 - loss: 225.7839 - val_ll_loss: 265.9444 - val_loss: 265.5565 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 586ms/step - ll_loss: 221.4927 - loss: 224.0770 - val_ll_loss: 264.1249 - val_loss: 263.7780 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 546ms/step - ll_loss: 220.0371 - loss: 222.5962 - val_ll_loss: 262.5401 - val_loss: 262.2282 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m43s[0m 546ms/step - ll_loss: 218.7653 - loss: 221.3022 - val_ll_loss: 261.1516 - val_loss: 260.8699 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 562ms/step - ll_loss: 217.6481 - loss: 220.1653 - val_ll_loss: 259.9292 - val_loss: 259.6739 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 572ms/step - ll_loss: 216.6625 - loss: 219.1621 - val_ll_loss: 258.8489 - val_loss: 258.6168 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 546ms/step - ll_loss: 215.7898 - loss: 218.2739 - val_ll_loss: 257.8912 - val_loss: 257.6794 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 549ms/step - ll_loss: 215.0148 - loss: 217.4850 - val_ll_loss: 257.0396 - val_loss: 256.8459 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 591ms/step - ll_loss: 214.3248 - loss: 216.7826 - val_ll_loss: 256.2806 - val_loss: 256.1030 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 569ms/step - ll_loss: 213.7092 - loss: 216.1558 - val_ll_loss: 255.6028 - val_loss: 255.4395 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 512ms/step - ll_loss: 213.1588 - loss: 215.5954 - val_ll_loss: 254.9964 - val_loss: 254.8458 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 551ms/step - ll_loss: 212.6660 - loss: 215.0936 - val_ll_loss: 254.4530 - val_loss: 254.3138 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 582ms/step - ll_loss: 212.2242 - loss: 214.6436 - val_ll_loss: 253.9654 - val_loss: 253.8364 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 560ms/step - ll_loss: 211.8274 - loss: 214.2395 - val_ll_loss: 253.5273 - val_loss: 253.4074 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 564ms/step - ll_loss: 211.4707 - loss: 213.8763 - val_ll_loss: 253.1332 - val_loss: 253.0215 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 553ms/step - ll_loss: 211.1498 - loss: 213.5494 - val_ll_loss: 252.7784 - val_loss: 252.6741 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 572ms/step - ll_loss: 210.8608 - loss: 213.2551 - val_ll_loss: 252.4587 - val_loss: 252.3611 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 558ms/step - ll_loss: 210.6004 - loss: 212.9897 - val_ll_loss: 252.1705 - val_loss: 252.0788 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 506ms/step - ll_loss: 210.3654 - loss: 212.7504 - val_ll_loss: 251.9104 - val_loss: 251.8241 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 549ms/step - ll_loss: 210.1535 - loss: 212.5345 - val_ll_loss: 251.6756 - val_loss: 251.5942 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 577ms/step - ll_loss: 209.9621 - loss: 212.3395 - val_ll_loss: 251.4636 - val_loss: 251.3865 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 579ms/step - ll_loss: 209.7892 - loss: 212.1634 - val_ll_loss: 251.2719 - val_loss: 251.1988 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 565ms/step - ll_loss: 209.6330 - loss: 212.0042 - val_ll_loss: 251.0987 - val_loss: 251.0291 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 558ms/step - ll_loss: 209.4918 - loss: 211.8603 - val_ll_loss: 250.9420 - val_loss: 250.8757 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 572ms/step - ll_loss: 209.3640 - loss: 211.7302 - val_ll_loss: 250.8003 - val_loss: 250.7369 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 532ms/step - ll_loss: 209.2485 - loss: 211.6125 - val_ll_loss: 250.6721 - val_loss: 250.6113 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 556ms/step - ll_loss: 209.1440 - loss: 211.5060 - val_ll_loss: 250.5561 - val_loss: 250.4977 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 549ms/step - ll_loss: 209.0495 - loss: 211.4097 - val_ll_loss: 250.4510 - val_loss: 250.3948 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 574ms/step - ll_loss: 208.9639 - loss: 211.3224 - val_ll_loss: 250.3559 - val_loss: 250.3017 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 571ms/step - ll_loss: 208.8864 - loss: 211.2435 - val_ll_loss: 250.2698 - val_loss: 250.2173 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 555ms/step - ll_loss: 208.8163 - loss: 211.1720 - val_ll_loss: 250.1919 - val_loss: 250.1410 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 548ms/step - ll_loss: 208.7528 - loss: 211.1073 - val_ll_loss: 250.1212 - val_loss: 250.0718 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 523ms/step - ll_loss: 208.6953 - loss: 211.0487 - val_ll_loss: 250.0573 - val_loss: 250.0092 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 540ms/step - ll_loss: 208.6432 - loss: 210.9957 - val_ll_loss: 249.9994 - val_loss: 249.9524 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 547ms/step - ll_loss: 208.5960 - loss: 210.9476 - val_ll_loss: 249.9470 - val_loss: 249.9011 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 545ms/step - ll_loss: 208.5533 - loss: 210.9041 - val_ll_loss: 249.8994 - val_loss: 249.8545 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 567ms/step - ll_loss: 208.5146 - loss: 210.8647 - val_ll_loss: 249.8564 - val_loss: 249.8124 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 579ms/step - ll_loss: 208.4796 - loss: 210.8290 - val_ll_loss: 249.8175 - val_loss: 249.7742 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 555ms/step - ll_loss: 208.4479 - loss: 210.7966 - val_ll_loss: 249.7821 - val_loss: 249.7395 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 548ms/step - ll_loss: 208.4192 - loss: 210.7673 - val_ll_loss: 249.7501 - val_loss: 249.7083 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 517ms/step - ll_loss: 208.3931 - loss: 210.7408 - val_ll_loss: 249.7211 - val_loss: 249.6798 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 570ms/step - ll_loss: 208.3696 - loss: 210.7168 - val_ll_loss: 249.6949 - val_loss: 249.6541 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 557ms/step - ll_loss: 208.3482 - loss: 210.6950 - val_ll_loss: 249.6711 - val_loss: 249.6308 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 558ms/step - ll_loss: 208.3288 - loss: 210.6753 - val_ll_loss: 249.6496 - val_loss: 249.6097 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 562ms/step - ll_loss: 208.3113 - loss: 210.6574 - val_ll_loss: 249.6300 - val_loss: 249.5906 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 588ms/step - ll_loss: 208.2955 - loss: 210.6413 - val_ll_loss: 249.6124 - val_loss: 249.5733 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 555ms/step - ll_loss: 208.2811 - loss: 210.6266 - val_ll_loss: 249.5964 - val_loss: 249.5576 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 522ms/step - ll_loss: 208.2681 - loss: 210.6134 - val_ll_loss: 249.5819 - val_loss: 249.5434 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 549ms/step - ll_loss: 208.2563 - loss: 210.6014 - val_ll_loss: 249.5687 - val_loss: 249.5305 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 580ms/step - ll_loss: 208.2456 - loss: 210.5905 - val_ll_loss: 249.5568 - val_loss: 249.5188 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 550ms/step - ll_loss: 208.2360 - loss: 210.5806 - val_ll_loss: 249.5461 - val_loss: 249.5083 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 543ms/step - ll_loss: 208.2272 - loss: 210.5717 - val_ll_loss: 249.5363 - val_loss: 249.4988 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 548ms/step - ll_loss: 208.2193 - loss: 210.5636 - val_ll_loss: 249.5275 - val_loss: 249.4901 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 569ms/step - ll_loss: 208.2121 - loss: 210.5563 - val_ll_loss: 249.5195 - val_loss: 249.4822 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 516ms/step - ll_loss: 208.2056 - loss: 210.5497 - val_ll_loss: 249.5122 - val_loss: 249.4751 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 564ms/step - ll_loss: 208.1997 - loss: 210.5437 - val_ll_loss: 249.5056 - val_loss: 249.4687 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/8 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K07/fold_holdout-sub-03
    \n--- holdout=sub-08 | val=sub-17 | K=7 ---
    train segs=51 short=0 | val segs=10 short=0 | test segs=10 short=0



    Loading files:   0%|          | 0/51 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]


    Epoch 1/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 527ms/step - ll_loss: 256.2395 - loss: 230.3987 - val_ll_loss: 275.4660 - val_loss: 265.5999 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m45s[0m 571ms/step - ll_loss: 253.8147 - loss: 228.2823 - val_ll_loss: 273.9141 - val_loss: 264.0876 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 511ms/step - ll_loss: 251.5022 - loss: 226.2949 - val_ll_loss: 271.6483 - val_loss: 261.8659 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 547ms/step - ll_loss: 249.3287 - loss: 224.4145 - val_ll_loss: 269.3291 - val_loss: 259.6818 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 569ms/step - ll_loss: 247.4325 - loss: 222.7762 - val_ll_loss: 267.3195 - val_loss: 257.7889 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 583ms/step - ll_loss: 245.7787 - loss: 221.3532 - val_ll_loss: 265.5670 - val_loss: 256.1382 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 568ms/step - ll_loss: 244.3260 - loss: 220.1006 - val_ll_loss: 264.0305 - val_loss: 254.6907 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 563ms/step - ll_loss: 243.0458 - loss: 218.9969 - val_ll_loss: 262.6772 - val_loss: 253.4158 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 513ms/step - ll_loss: 241.9138 - loss: 218.0217 - val_ll_loss: 261.4808 - val_loss: 252.2886 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 592ms/step - ll_loss: 240.9097 - loss: 217.1574 - val_ll_loss: 260.4196 - val_loss: 251.2890 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 601ms/step - ll_loss: 240.0168 - loss: 216.3893 - val_ll_loss: 259.4760 - val_loss: 250.4000 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 546ms/step - ll_loss: 239.2210 - loss: 215.7049 - val_ll_loss: 258.6350 - val_loss: 249.6076 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 560ms/step - ll_loss: 238.5103 - loss: 215.0941 - val_ll_loss: 257.8839 - val_loss: 248.9000 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 601ms/step - ll_loss: 237.8745 - loss: 214.5479 - val_ll_loss: 257.2119 - val_loss: 248.2668 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 572ms/step - ll_loss: 237.3048 - loss: 214.0586 - val_ll_loss: 256.6096 - val_loss: 247.6995 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 526ms/step - ll_loss: 236.7938 - loss: 213.6198 - val_ll_loss: 256.0692 - val_loss: 247.1903 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 553ms/step - ll_loss: 236.3347 - loss: 213.2257 - val_ll_loss: 255.5837 - val_loss: 246.7329 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 577ms/step - ll_loss: 235.9220 - loss: 212.8715 - val_ll_loss: 255.1470 - val_loss: 246.3216 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 573ms/step - ll_loss: 235.5506 - loss: 212.5528 - val_ll_loss: 254.7540 - val_loss: 245.9514 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 545ms/step - ll_loss: 235.2160 - loss: 212.2657 - val_ll_loss: 254.3999 - val_loss: 245.6179 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 561ms/step - ll_loss: 234.9145 - loss: 212.0070 - val_ll_loss: 254.0807 - val_loss: 245.3171 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 548ms/step - ll_loss: 234.6425 - loss: 211.7737 - val_ll_loss: 253.7927 - val_loss: 245.0458 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 582ms/step - ll_loss: 234.3971 - loss: 211.5632 - val_ll_loss: 253.5327 - val_loss: 244.8009 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 557ms/step - ll_loss: 234.1755 - loss: 211.3732 - val_ll_loss: 253.2980 - val_loss: 244.5798 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 560ms/step - ll_loss: 233.9753 - loss: 211.2015 - val_ll_loss: 253.0858 - val_loss: 244.3800 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 567ms/step - ll_loss: 233.7945 - loss: 211.0464 - val_ll_loss: 252.8941 - val_loss: 244.1994 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 586ms/step - ll_loss: 233.6310 - loss: 210.9062 - val_ll_loss: 252.7208 - val_loss: 244.0361 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 522ms/step - ll_loss: 233.4832 - loss: 210.7794 - val_ll_loss: 252.5640 - val_loss: 243.8884 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 523ms/step - ll_loss: 233.3495 - loss: 210.6648 - val_ll_loss: 252.4222 - val_loss: 243.7548 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 563ms/step - ll_loss: 233.2285 - loss: 210.5611 - val_ll_loss: 252.2939 - val_loss: 243.6340 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 595ms/step - ll_loss: 233.1191 - loss: 210.4672 - val_ll_loss: 252.1777 - val_loss: 243.5246 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 552ms/step - ll_loss: 233.0201 - loss: 210.3824 - val_ll_loss: 252.0727 - val_loss: 243.4256 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 558ms/step - ll_loss: 232.9305 - loss: 210.3055 - val_ll_loss: 251.9774 - val_loss: 243.3360 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 563ms/step - ll_loss: 232.8493 - loss: 210.2360 - val_ll_loss: 251.8914 - val_loss: 243.2549 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 553ms/step - ll_loss: 232.7759 - loss: 210.1730 - val_ll_loss: 251.8133 - val_loss: 243.1813 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 558ms/step - ll_loss: 232.7094 - loss: 210.1160 - val_ll_loss: 251.7426 - val_loss: 243.1148 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 550ms/step - ll_loss: 232.6492 - loss: 210.0643 - val_ll_loss: 251.6787 - val_loss: 243.0545 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 556ms/step - ll_loss: 232.5947 - loss: 210.0176 - val_ll_loss: 251.6208 - val_loss: 243.0000 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 588ms/step - ll_loss: 232.5454 - loss: 209.9752 - val_ll_loss: 251.5683 - val_loss: 242.9505 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 568ms/step - ll_loss: 232.5007 - loss: 209.9369 - val_ll_loss: 251.5207 - val_loss: 242.9058 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 522ms/step - ll_loss: 232.4602 - loss: 209.9022 - val_ll_loss: 251.4777 - val_loss: 242.8652 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 554ms/step - ll_loss: 232.4235 - loss: 209.8708 - val_ll_loss: 251.4387 - val_loss: 242.8285 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 568ms/step - ll_loss: 232.3903 - loss: 209.8423 - val_ll_loss: 251.4034 - val_loss: 242.7952 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 586ms/step - ll_loss: 232.3603 - loss: 209.8165 - val_ll_loss: 251.3714 - val_loss: 242.7651 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 557ms/step - ll_loss: 232.3330 - loss: 209.7931 - val_ll_loss: 251.3424 - val_loss: 242.7378 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 538ms/step - ll_loss: 232.3084 - loss: 209.7720 - val_ll_loss: 251.3161 - val_loss: 242.7131 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 535ms/step - ll_loss: 232.2860 - loss: 209.7528 - val_ll_loss: 251.2924 - val_loss: 242.6906 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 594ms/step - ll_loss: 232.2658 - loss: 209.7355 - val_ll_loss: 251.2708 - val_loss: 242.6703 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 560ms/step - ll_loss: 232.2475 - loss: 209.7198 - val_ll_loss: 251.2513 - val_loss: 242.6520 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 569ms/step - ll_loss: 232.2309 - loss: 209.7055 - val_ll_loss: 251.2336 - val_loss: 242.6353 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 563ms/step - ll_loss: 232.2159 - loss: 209.6926 - val_ll_loss: 251.2177 - val_loss: 242.6203 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 598ms/step - ll_loss: 232.2023 - loss: 209.6810 - val_ll_loss: 251.2032 - val_loss: 242.6066 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 539ms/step - ll_loss: 232.1899 - loss: 209.6704 - val_ll_loss: 251.1900 - val_loss: 242.5943 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 509ms/step - ll_loss: 232.1788 - loss: 209.6608 - val_ll_loss: 251.1781 - val_loss: 242.5831 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 557ms/step - ll_loss: 232.1687 - loss: 209.6522 - val_ll_loss: 251.1674 - val_loss: 242.5729 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 603ms/step - ll_loss: 232.1595 - loss: 209.6443 - val_ll_loss: 251.1576 - val_loss: 242.5637 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 565ms/step - ll_loss: 232.1513 - loss: 209.6372 - val_ll_loss: 251.1488 - val_loss: 242.5554 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 562ms/step - ll_loss: 232.1438 - loss: 209.6308 - val_ll_loss: 251.1408 - val_loss: 242.5479 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 519ms/step - ll_loss: 232.1370 - loss: 209.6249 - val_ll_loss: 251.1335 - val_loss: 242.5411 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 546ms/step - ll_loss: 232.1308 - loss: 209.6197 - val_ll_loss: 251.1270 - val_loss: 242.5350 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/10 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K07/fold_holdout-sub-08
    \n--- holdout=sub-09 | val=sub-08 | K=7 ---
    train segs=59 short=0 | val segs=10 short=0 | test segs=2 short=0



    Loading files:   0%|          | 0/59 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/2 [00:00<?, ?it/s]


    Epoch 1/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 520ms/step - ll_loss: 265.2190 - loss: 229.0117 - val_ll_loss: 250.3243 - val_loss: 246.0946 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 536ms/step - ll_loss: 262.1362 - loss: 226.1955 - val_ll_loss: 247.0882 - val_loss: 243.0690 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 549ms/step - ll_loss: 258.9190 - loss: 223.2951 - val_ll_loss: 244.1515 - val_loss: 240.2399 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 588ms/step - ll_loss: 255.8446 - loss: 220.8125 - val_ll_loss: 241.7708 - val_loss: 237.9506 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 548ms/step - ll_loss: 253.3520 - loss: 218.7940 - val_ll_loss: 239.8231 - val_loss: 236.0829 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 559ms/step - ll_loss: 251.2970 - loss: 217.1157 - val_ll_loss: 238.1700 - val_loss: 234.5001 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 551ms/step - ll_loss: 249.5477 - loss: 215.6822 - val_ll_loss: 236.7458 - val_loss: 233.1375 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 568ms/step - ll_loss: 248.0363 - loss: 214.4419 - val_ll_loss: 235.5067 - val_loss: 231.9529 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 538ms/step - ll_loss: 246.7184 - loss: 213.3595 - val_ll_loss: 234.4218 - val_loss: 230.9159 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m44s[0m 525ms/step - ll_loss: 245.5618 - loss: 212.4094 - val_ll_loss: 233.4666 - val_loss: 230.0032 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 589ms/step - ll_loss: 244.5416 - loss: 211.5712 - val_ll_loss: 232.6223 - val_loss: 229.1967 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 533ms/step - ll_loss: 243.6384 - loss: 210.8291 - val_ll_loss: 231.8735 - val_loss: 228.4814 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 561ms/step - ll_loss: 242.8360 - loss: 210.1700 - val_ll_loss: 231.2073 - val_loss: 227.8453 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 567ms/step - ll_loss: 242.1215 - loss: 209.5830 - val_ll_loss: 230.6134 - val_loss: 227.2781 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 567ms/step - ll_loss: 241.4836 - loss: 209.0592 - val_ll_loss: 230.0828 - val_loss: 226.7715 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m43s[0m 541ms/step - ll_loss: 240.9132 - loss: 208.5907 - val_ll_loss: 229.6078 - val_loss: 226.3180 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 553ms/step - ll_loss: 240.4023 - loss: 208.1712 - val_ll_loss: 229.1821 - val_loss: 225.9116 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 582ms/step - ll_loss: 239.9439 - loss: 207.7948 - val_ll_loss: 228.7999 - val_loss: 225.5467 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 553ms/step - ll_loss: 239.5323 - loss: 207.4568 - val_ll_loss: 228.4564 - val_loss: 225.2189 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 550ms/step - ll_loss: 239.1622 - loss: 207.1529 - val_ll_loss: 228.1473 - val_loss: 224.9239 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 575ms/step - ll_loss: 238.8291 - loss: 206.8794 - val_ll_loss: 227.8691 - val_loss: 224.6583 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m39s[0m 536ms/step - ll_loss: 238.5290 - loss: 206.6331 - val_ll_loss: 227.6183 - val_loss: 224.4190 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 541ms/step - ll_loss: 238.2585 - loss: 206.4110 - val_ll_loss: 227.3921 - val_loss: 224.2032 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 566ms/step - ll_loss: 238.0145 - loss: 206.2108 - val_ll_loss: 227.1881 - val_loss: 224.0085 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 564ms/step - ll_loss: 237.7943 - loss: 206.0300 - val_ll_loss: 227.0038 - val_loss: 223.8327 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 553ms/step - ll_loss: 237.5955 - loss: 205.8668 - val_ll_loss: 226.8374 - val_loss: 223.6739 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 543ms/step - ll_loss: 237.4159 - loss: 205.7193 - val_ll_loss: 226.6870 - val_loss: 223.5304 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 572ms/step - ll_loss: 237.2537 - loss: 205.5861 - val_ll_loss: 226.5511 - val_loss: 223.4007 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 556ms/step - ll_loss: 237.1069 - loss: 205.4656 - val_ll_loss: 226.4281 - val_loss: 223.2834 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 539ms/step - ll_loss: 236.9743 - loss: 205.3567 - val_ll_loss: 226.3169 - val_loss: 223.1773 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 547ms/step - ll_loss: 236.8543 - loss: 205.2582 - val_ll_loss: 226.2163 - val_loss: 223.0813 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m43s[0m 574ms/step - ll_loss: 236.7458 - loss: 205.1691 - val_ll_loss: 226.1253 - val_loss: 222.9945 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 550ms/step - ll_loss: 236.6476 - loss: 205.0884 - val_ll_loss: 226.0429 - val_loss: 222.9159 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 547ms/step - ll_loss: 236.5587 - loss: 205.0154 - val_ll_loss: 225.9683 - val_loss: 222.8448 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 573ms/step - ll_loss: 236.4783 - loss: 204.9494 - val_ll_loss: 225.9008 - val_loss: 222.7804 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 556ms/step - ll_loss: 236.4055 - loss: 204.8896 - val_ll_loss: 225.8397 - val_loss: 222.7220 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 556ms/step - ll_loss: 236.3396 - loss: 204.8354 - val_ll_loss: 225.7843 - val_loss: 222.6693 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 562ms/step - ll_loss: 236.2799 - loss: 204.7864 - val_ll_loss: 225.7342 - val_loss: 222.6215 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 590ms/step - ll_loss: 236.2259 - loss: 204.7420 - val_ll_loss: 225.6888 - val_loss: 222.5782 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 561ms/step - ll_loss: 236.1770 - loss: 204.7018 - val_ll_loss: 225.6477 - val_loss: 222.5390 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 551ms/step - ll_loss: 236.1327 - loss: 204.6654 - val_ll_loss: 225.6105 - val_loss: 222.5035 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 531ms/step - ll_loss: 236.0926 - loss: 204.6325 - val_ll_loss: 225.5768 - val_loss: 222.4714 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 550ms/step - ll_loss: 236.0562 - loss: 204.6026 - val_ll_loss: 225.5463 - val_loss: 222.4422 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 512ms/step - ll_loss: 236.0234 - loss: 204.5756 - val_ll_loss: 225.5186 - val_loss: 222.4158 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 542ms/step - ll_loss: 235.9936 - loss: 204.5511 - val_ll_loss: 225.4935 - val_loss: 222.3919 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 575ms/step - ll_loss: 235.9667 - loss: 204.5289 - val_ll_loss: 225.4709 - val_loss: 222.3703 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 571ms/step - ll_loss: 235.9422 - loss: 204.5089 - val_ll_loss: 225.4503 - val_loss: 222.3507 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 549ms/step - ll_loss: 235.9201 - loss: 204.4906 - val_ll_loss: 225.4317 - val_loss: 222.3330 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 556ms/step - ll_loss: 235.9001 - loss: 204.4742 - val_ll_loss: 225.4148 - val_loss: 222.3169 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 588ms/step - ll_loss: 235.8819 - loss: 204.4593 - val_ll_loss: 225.3996 - val_loss: 222.3024 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 543ms/step - ll_loss: 235.8655 - loss: 204.4458 - val_ll_loss: 225.3858 - val_loss: 222.2892 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 534ms/step - ll_loss: 235.8506 - loss: 204.4335 - val_ll_loss: 225.3732 - val_loss: 222.2772 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 554ms/step - ll_loss: 235.8372 - loss: 204.4225 - val_ll_loss: 225.3619 - val_loss: 222.2665 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 560ms/step - ll_loss: 235.8250 - loss: 204.4125 - val_ll_loss: 225.3517 - val_loss: 222.2567 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 559ms/step - ll_loss: 235.8139 - loss: 204.4034 - val_ll_loss: 225.3424 - val_loss: 222.2478 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 544ms/step - ll_loss: 235.8040 - loss: 204.3952 - val_ll_loss: 225.3340 - val_loss: 222.2398 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 581ms/step - ll_loss: 235.7949 - loss: 204.3877 - val_ll_loss: 225.3263 - val_loss: 222.2325 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 556ms/step - ll_loss: 235.7867 - loss: 204.3810 - val_ll_loss: 225.3194 - val_loss: 222.2260 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 547ms/step - ll_loss: 235.7793 - loss: 204.3748 - val_ll_loss: 225.3132 - val_loss: 222.2200 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 551ms/step - ll_loss: 235.7726 - loss: 204.3693 - val_ll_loss: 225.3075 - val_loss: 222.2146 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/2 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K07/fold_holdout-sub-09
    \n--- holdout=sub-13 | val=sub-08 | K=7 ---
    train segs=52 short=0 | val segs=10 short=0 | test segs=9 short=0



    Loading files:   0%|          | 0/52 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/9 [00:00<?, ?it/s]


    Epoch 1/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 601ms/step - ll_loss: 254.4176 - loss: 231.4651 - val_ll_loss: 232.1756 - val_loss: 228.4788 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 560ms/step - ll_loss: 252.0582 - loss: 229.3053 - val_ll_loss: 229.5883 - val_loss: 226.0543 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 564ms/step - ll_loss: 249.2388 - loss: 226.6844 - val_ll_loss: 227.2585 - val_loss: 223.7729 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 528ms/step - ll_loss: 246.4978 - loss: 224.2734 - val_ll_loss: 225.3426 - val_loss: 221.9174 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 581ms/step - ll_loss: 244.2813 - loss: 222.3439 - val_ll_loss: 223.7784 - val_loss: 220.4096 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 557ms/step - ll_loss: 242.4601 - loss: 220.7542 - val_ll_loss: 222.4549 - val_loss: 219.1368 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 571ms/step - ll_loss: 240.9142 - loss: 219.4032 - val_ll_loss: 221.3157 - val_loss: 218.0427 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 568ms/step - ll_loss: 239.5803 - loss: 218.2369 - val_ll_loss: 220.3248 - val_loss: 217.0919 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 593ms/step - ll_loss: 238.4179 - loss: 217.2202 - val_ll_loss: 219.4568 - val_loss: 216.2596 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 561ms/step - ll_loss: 237.3980 - loss: 216.3282 - val_ll_loss: 218.6925 - val_loss: 215.5269 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 516ms/step - ll_loss: 236.4984 - loss: 215.5414 - val_ll_loss: 218.0165 - val_loss: 214.8793 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 551ms/step - ll_loss: 235.7019 - loss: 214.8447 - val_ll_loss: 217.4167 - val_loss: 214.3047 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 589ms/step - ll_loss: 234.9943 - loss: 214.2259 - val_ll_loss: 216.8830 - val_loss: 213.7936 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 576ms/step - ll_loss: 234.3641 - loss: 213.6748 - val_ll_loss: 216.4070 - val_loss: 213.3379 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 561ms/step - ll_loss: 233.8014 - loss: 213.1828 - val_ll_loss: 215.9816 - val_loss: 212.9306 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 556ms/step - ll_loss: 233.2982 - loss: 212.7429 - val_ll_loss: 215.6007 - val_loss: 212.5661 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m43s[0m 589ms/step - ll_loss: 232.8474 - loss: 212.3488 - val_ll_loss: 215.2594 - val_loss: 212.2394 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 544ms/step - ll_loss: 232.4431 - loss: 211.9953 - val_ll_loss: 214.9528 - val_loss: 211.9460 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 552ms/step - ll_loss: 232.0799 - loss: 211.6778 - val_ll_loss: 214.6773 - val_loss: 211.6825 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 521ms/step - ll_loss: 231.7534 - loss: 211.3924 - val_ll_loss: 214.4295 - val_loss: 211.4454 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 585ms/step - ll_loss: 231.4596 - loss: 211.1355 - val_ll_loss: 214.2064 - val_loss: 211.2319 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 558ms/step - ll_loss: 231.1949 - loss: 210.9042 - val_ll_loss: 214.0053 - val_loss: 211.0396 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 549ms/step - ll_loss: 230.9565 - loss: 210.6957 - val_ll_loss: 213.8241 - val_loss: 210.8662 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 559ms/step - ll_loss: 230.7414 - loss: 210.5077 - val_ll_loss: 213.6605 - val_loss: 210.7098 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 596ms/step - ll_loss: 230.5474 - loss: 210.3381 - val_ll_loss: 213.5129 - val_loss: 210.5686 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 554ms/step - ll_loss: 230.3722 - loss: 210.1850 - val_ll_loss: 213.3795 - val_loss: 210.4411 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 541ms/step - ll_loss: 230.2140 - loss: 210.0467 - val_ll_loss: 213.2591 - val_loss: 210.3259 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 554ms/step - ll_loss: 230.0711 - loss: 209.9217 - val_ll_loss: 213.1502 - val_loss: 210.2218 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 639ms/step - ll_loss: 229.9419 - loss: 209.8088 - val_ll_loss: 213.0518 - val_loss: 210.1277 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 568ms/step - ll_loss: 229.8251 - loss: 209.7067 - val_ll_loss: 212.9628 - val_loss: 210.0426 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 552ms/step - ll_loss: 229.7196 - loss: 209.6144 - val_ll_loss: 212.8824 - val_loss: 209.9657 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 565ms/step - ll_loss: 229.6241 - loss: 209.5309 - val_ll_loss: 212.8096 - val_loss: 209.8961 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m44s[0m 592ms/step - ll_loss: 229.5377 - loss: 209.4554 - val_ll_loss: 212.7437 - val_loss: 209.8331 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 559ms/step - ll_loss: 229.4596 - loss: 209.3871 - val_ll_loss: 212.6841 - val_loss: 209.7762 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 573ms/step - ll_loss: 229.3889 - loss: 209.3252 - val_ll_loss: 212.6302 - val_loss: 209.7246 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m43s[0m 585ms/step - ll_loss: 229.3249 - loss: 209.2693 - val_ll_loss: 212.5813 - val_loss: 209.6779 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 573ms/step - ll_loss: 229.2669 - loss: 209.2186 - val_ll_loss: 212.5371 - val_loss: 209.6357 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 559ms/step - ll_loss: 229.2146 - loss: 209.1728 - val_ll_loss: 212.4971 - val_loss: 209.5974 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 575ms/step - ll_loss: 229.1671 - loss: 209.1313 - val_ll_loss: 212.4609 - val_loss: 209.5628 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m43s[0m 605ms/step - ll_loss: 229.1242 - loss: 209.0937 - val_ll_loss: 212.4281 - val_loss: 209.5314 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 569ms/step - ll_loss: 229.0853 - loss: 209.0597 - val_ll_loss: 212.3984 - val_loss: 209.5031 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 585ms/step - ll_loss: 229.0501 - loss: 209.0289 - val_ll_loss: 212.3715 - val_loss: 209.4774 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 560ms/step - ll_loss: 229.0183 - loss: 209.0010 - val_ll_loss: 212.3471 - val_loss: 209.4541 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 555ms/step - ll_loss: 228.9894 - loss: 208.9758 - val_ll_loss: 212.3251 - val_loss: 209.4330 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 553ms/step - ll_loss: 228.9633 - loss: 208.9529 - val_ll_loss: 212.3051 - val_loss: 209.4139 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m45s[0m 556ms/step - ll_loss: 228.9396 - loss: 208.9322 - val_ll_loss: 212.2870 - val_loss: 209.3966 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 542ms/step - ll_loss: 228.9182 - loss: 208.9135 - val_ll_loss: 212.2707 - val_loss: 209.3810 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 608ms/step - ll_loss: 228.8988 - loss: 208.8965 - val_ll_loss: 212.2559 - val_loss: 209.3669 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 559ms/step - ll_loss: 228.8813 - loss: 208.8812 - val_ll_loss: 212.2424 - val_loss: 209.3540 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 568ms/step - ll_loss: 228.8654 - loss: 208.8673 - val_ll_loss: 212.2303 - val_loss: 209.3424 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 570ms/step - ll_loss: 228.8510 - loss: 208.8547 - val_ll_loss: 212.2192 - val_loss: 209.3319 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 574ms/step - ll_loss: 228.8380 - loss: 208.8432 - val_ll_loss: 212.2093 - val_loss: 209.3224 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 548ms/step - ll_loss: 228.8261 - loss: 208.8329 - val_ll_loss: 212.2003 - val_loss: 209.3137 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 521ms/step - ll_loss: 228.8155 - loss: 208.8235 - val_ll_loss: 212.1921 - val_loss: 209.3059 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 580ms/step - ll_loss: 228.8058 - loss: 208.8151 - val_ll_loss: 212.1847 - val_loss: 209.2988 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 593ms/step - ll_loss: 228.7970 - loss: 208.8074 - val_ll_loss: 212.1780 - val_loss: 209.2924 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 554ms/step - ll_loss: 228.7891 - loss: 208.8005 - val_ll_loss: 212.1719 - val_loss: 209.2866 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 568ms/step - ll_loss: 228.7819 - loss: 208.7942 - val_ll_loss: 212.1664 - val_loss: 209.2814 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 559ms/step - ll_loss: 228.7754 - loss: 208.7885 - val_ll_loss: 212.1615 - val_loss: 209.2767 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 585ms/step - ll_loss: 228.7695 - loss: 208.7834 - val_ll_loss: 212.1569 - val_loss: 209.2723 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/9 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K07/fold_holdout-sub-13
    \n--- holdout=sub-14 | val=sub-08 | K=7 ---
    train segs=57 short=0 | val segs=10 short=0 | test segs=4 short=0



    Loading files:   0%|          | 0/57 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/4 [00:00<?, ?it/s]


    Epoch 1/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 561ms/step - ll_loss: 256.6787 - loss: 231.1038 - val_ll_loss: 239.5816 - val_loss: 236.1522 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 568ms/step - ll_loss: 253.5755 - loss: 228.2478 - val_ll_loss: 236.6910 - val_loss: 233.3680 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 592ms/step - ll_loss: 250.5217 - loss: 225.6112 - val_ll_loss: 234.3868 - val_loss: 231.1615 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 570ms/step - ll_loss: 248.0307 - loss: 223.4597 - val_ll_loss: 232.4576 - val_loss: 229.3146 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 567ms/step - ll_loss: 245.9214 - loss: 221.6379 - val_ll_loss: 230.8062 - val_loss: 227.7337 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 572ms/step - ll_loss: 244.1037 - loss: 220.0687 - val_ll_loss: 229.3764 - val_loss: 226.3649 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 594ms/step - ll_loss: 242.5221 - loss: 218.7039 - val_ll_loss: 228.1290 - val_loss: 225.1707 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 532ms/step - ll_loss: 241.1369 - loss: 217.5092 - val_ll_loss: 227.0347 - val_loss: 224.1230 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 527ms/step - ll_loss: 239.9175 - loss: 216.4580 - val_ll_loss: 226.0702 - val_loss: 223.1995 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 565ms/step - ll_loss: 238.8396 - loss: 215.5293 - val_ll_loss: 225.2170 - val_loss: 222.3827 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 583ms/step - ll_loss: 237.8840 - loss: 214.7062 - val_ll_loss: 224.4601 - val_loss: 221.6579 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 568ms/step - ll_loss: 237.0343 - loss: 213.9746 - val_ll_loss: 223.7867 - val_loss: 221.0130 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 564ms/step - ll_loss: 236.2770 - loss: 213.3228 - val_ll_loss: 223.1862 - val_loss: 220.4380 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 579ms/step - ll_loss: 235.6006 - loss: 212.7408 - val_ll_loss: 222.6497 - val_loss: 219.9242 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 565ms/step - ll_loss: 234.9954 - loss: 212.2202 - val_ll_loss: 222.1695 - val_loss: 219.4643 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 564ms/step - ll_loss: 234.4532 - loss: 211.7538 - val_ll_loss: 221.7391 - val_loss: 219.0521 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 556ms/step - ll_loss: 233.9666 - loss: 211.3354 - val_ll_loss: 221.3527 - val_loss: 218.6821 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 593ms/step - ll_loss: 233.5295 - loss: 210.9596 - val_ll_loss: 221.0056 - val_loss: 218.3495 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 583ms/step - ll_loss: 233.1364 - loss: 210.6217 - val_ll_loss: 220.6933 - val_loss: 218.0504 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 557ms/step - ll_loss: 232.7826 - loss: 210.3176 - val_ll_loss: 220.4120 - val_loss: 217.7811 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 558ms/step - ll_loss: 232.4639 - loss: 210.0436 - val_ll_loss: 220.1586 - val_loss: 217.5384 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 585ms/step - ll_loss: 232.1765 - loss: 209.7966 - val_ll_loss: 219.9301 - val_loss: 217.3195 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 553ms/step - ll_loss: 231.9173 - loss: 209.5739 - val_ll_loss: 219.7239 - val_loss: 217.1220 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 564ms/step - ll_loss: 231.6833 - loss: 209.3729 - val_ll_loss: 219.5378 - val_loss: 216.9437 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 566ms/step - ll_loss: 231.4720 - loss: 209.1913 - val_ll_loss: 219.3696 - val_loss: 216.7827 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 570ms/step - ll_loss: 231.2812 - loss: 209.0273 - val_ll_loss: 219.2176 - val_loss: 216.6371 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 516ms/step - ll_loss: 231.1087 - loss: 208.8791 - val_ll_loss: 219.0803 - val_loss: 216.5056 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 558ms/step - ll_loss: 230.9528 - loss: 208.7451 - val_ll_loss: 218.9561 - val_loss: 216.3866 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 568ms/step - ll_loss: 230.8117 - loss: 208.6239 - val_ll_loss: 218.8437 - val_loss: 216.2790 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 569ms/step - ll_loss: 230.6842 - loss: 208.5143 - val_ll_loss: 218.7421 - val_loss: 216.1817 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 559ms/step - ll_loss: 230.5687 - loss: 208.4151 - val_ll_loss: 218.6501 - val_loss: 216.0936 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 576ms/step - ll_loss: 230.4643 - loss: 208.3254 - val_ll_loss: 218.5668 - val_loss: 216.0138 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 589ms/step - ll_loss: 230.3698 - loss: 208.2442 - val_ll_loss: 218.4915 - val_loss: 215.9417 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 563ms/step - ll_loss: 230.2843 - loss: 208.1707 - val_ll_loss: 218.4232 - val_loss: 215.8763 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 566ms/step - ll_loss: 230.2068 - loss: 208.1042 - val_ll_loss: 218.3615 - val_loss: 215.8172 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 550ms/step - ll_loss: 230.1367 - loss: 208.0439 - val_ll_loss: 218.3055 - val_loss: 215.7636 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 599ms/step - ll_loss: 230.0733 - loss: 207.9893 - val_ll_loss: 218.2549 - val_loss: 215.7151 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 562ms/step - ll_loss: 230.0158 - loss: 207.9399 - val_ll_loss: 218.2090 - val_loss: 215.6712 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 559ms/step - ll_loss: 229.9637 - loss: 207.8952 - val_ll_loss: 218.1674 - val_loss: 215.6314 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 569ms/step - ll_loss: 229.9166 - loss: 207.8547 - val_ll_loss: 218.1298 - val_loss: 215.5953 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 577ms/step - ll_loss: 229.8739 - loss: 207.8180 - val_ll_loss: 218.0957 - val_loss: 215.5627 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 557ms/step - ll_loss: 229.8352 - loss: 207.7848 - val_ll_loss: 218.0648 - val_loss: 215.5332 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 561ms/step - ll_loss: 229.8002 - loss: 207.7547 - val_ll_loss: 218.0369 - val_loss: 215.5064 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 583ms/step - ll_loss: 229.7685 - loss: 207.7274 - val_ll_loss: 218.0116 - val_loss: 215.4821 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 548ms/step - ll_loss: 229.7398 - loss: 207.7028 - val_ll_loss: 217.9886 - val_loss: 215.4602 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 524ms/step - ll_loss: 229.7138 - loss: 207.6804 - val_ll_loss: 217.9679 - val_loss: 215.4403 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 566ms/step - ll_loss: 229.6903 - loss: 207.6601 - val_ll_loss: 217.9491 - val_loss: 215.4222 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 599ms/step - ll_loss: 229.6690 - loss: 207.6418 - val_ll_loss: 217.9320 - val_loss: 215.4059 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 568ms/step - ll_loss: 229.6497 - loss: 207.6252 - val_ll_loss: 217.9166 - val_loss: 215.3912 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 567ms/step - ll_loss: 229.6322 - loss: 207.6102 - val_ll_loss: 217.9026 - val_loss: 215.3778 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 560ms/step - ll_loss: 229.6163 - loss: 207.5965 - val_ll_loss: 217.8899 - val_loss: 215.3656 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 581ms/step - ll_loss: 229.6020 - loss: 207.5842 - val_ll_loss: 217.8784 - val_loss: 215.3546 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 557ms/step - ll_loss: 229.5890 - loss: 207.5730 - val_ll_loss: 217.8680 - val_loss: 215.3447 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 559ms/step - ll_loss: 229.5772 - loss: 207.5629 - val_ll_loss: 217.8587 - val_loss: 215.3357 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 581ms/step - ll_loss: 229.5666 - loss: 207.5538 - val_ll_loss: 217.8501 - val_loss: 215.3275 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 578ms/step - ll_loss: 229.5569 - loss: 207.5454 - val_ll_loss: 217.8424 - val_loss: 215.3201 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 553ms/step - ll_loss: 229.5482 - loss: 207.5379 - val_ll_loss: 217.8354 - val_loss: 215.3135 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 559ms/step - ll_loss: 229.5403 - loss: 207.5311 - val_ll_loss: 217.8291 - val_loss: 215.3074 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 580ms/step - ll_loss: 229.5331 - loss: 207.5250 - val_ll_loss: 217.8234 - val_loss: 215.3019 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 560ms/step - ll_loss: 229.5266 - loss: 207.5194 - val_ll_loss: 217.8182 - val_loss: 215.2970 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/4 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K07/fold_holdout-sub-14
    \n--- holdout=sub-16 | val=sub-08 | K=7 ---
    train segs=56 short=0 | val segs=10 short=0 | test segs=5 short=0



    Loading files:   0%|          | 0/56 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/5 [00:00<?, ?it/s]


    Epoch 1/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 562ms/step - ll_loss: 257.5751 - loss: 231.5805 - val_ll_loss: 236.2230 - val_loss: 232.6670 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 562ms/step - ll_loss: 255.1043 - loss: 229.4477 - val_ll_loss: 234.0054 - val_loss: 230.5334 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 590ms/step - ll_loss: 252.5868 - loss: 227.2893 - val_ll_loss: 231.8187 - val_loss: 228.4217 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 561ms/step - ll_loss: 249.9882 - loss: 224.9317 - val_ll_loss: 229.7951 - val_loss: 226.4743 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 526ms/step - ll_loss: 247.2013 - loss: 222.4262 - val_ll_loss: 227.8344 - val_loss: 224.6576 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 584ms/step - ll_loss: 244.6399 - loss: 220.1871 - val_ll_loss: 225.9054 - val_loss: 222.7834 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 574ms/step - ll_loss: 242.3659 - loss: 218.2773 - val_ll_loss: 224.3282 - val_loss: 221.2584 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 604ms/step - ll_loss: 240.5768 - loss: 216.7621 - val_ll_loss: 223.0349 - val_loss: 220.0129 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 556ms/step - ll_loss: 239.1055 - loss: 215.5124 - val_ll_loss: 221.9630 - val_loss: 219.0133 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 596ms/step - ll_loss: 237.8655 - loss: 214.4577 - val_ll_loss: 221.0334 - val_loss: 218.1212 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 569ms/step - ll_loss: 236.8028 - loss: 213.5530 - val_ll_loss: 220.2285 - val_loss: 217.3482 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 561ms/step - ll_loss: 235.8816 - loss: 212.7684 - val_ll_loss: 219.5260 - val_loss: 216.6739 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 577ms/step - ll_loss: 235.0767 - loss: 212.0825 - val_ll_loss: 218.9092 - val_loss: 216.0823 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 603ms/step - ll_loss: 234.3690 - loss: 211.4794 - val_ll_loss: 218.3649 - val_loss: 215.5603 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 568ms/step - ll_loss: 233.7440 - loss: 210.9467 - val_ll_loss: 217.8828 - val_loss: 215.0982 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 565ms/step - ll_loss: 233.1901 - loss: 210.4744 - val_ll_loss: 217.4545 - val_loss: 214.6877 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 576ms/step - ll_loss: 232.6975 - loss: 210.0545 - val_ll_loss: 217.0728 - val_loss: 214.3219 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 601ms/step - ll_loss: 232.2583 - loss: 209.6800 - val_ll_loss: 216.7320 - val_loss: 213.9955 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 577ms/step - ll_loss: 231.8661 - loss: 209.3456 - val_ll_loss: 216.4272 - val_loss: 213.7035 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 571ms/step - ll_loss: 231.5151 - loss: 209.0463 - val_ll_loss: 216.1541 - val_loss: 213.4420 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 587ms/step - ll_loss: 231.2006 - loss: 208.7780 - val_ll_loss: 215.9090 - val_loss: 213.2074 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 571ms/step - ll_loss: 230.9183 - loss: 208.5373 - val_ll_loss: 215.6889 - val_loss: 212.9966 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 517ms/step - ll_loss: 230.6647 - loss: 208.3210 - val_ll_loss: 215.4910 - val_loss: 212.8072 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 568ms/step - ll_loss: 230.4366 - loss: 208.1265 - val_ll_loss: 215.3129 - val_loss: 212.6367 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 591ms/step - ll_loss: 230.2313 - loss: 207.9514 - val_ll_loss: 215.1525 - val_loss: 212.4832 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 579ms/step - ll_loss: 230.0465 - loss: 207.7937 - val_ll_loss: 215.0079 - val_loss: 212.3448 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 575ms/step - ll_loss: 229.8799 - loss: 207.6515 - val_ll_loss: 214.8775 - val_loss: 212.2201 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 562ms/step - ll_loss: 229.7296 - loss: 207.5233 - val_ll_loss: 214.7599 - val_loss: 212.1075 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 592ms/step - ll_loss: 229.5940 - loss: 207.4077 - val_ll_loss: 214.6537 - val_loss: 212.0059 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 564ms/step - ll_loss: 229.4716 - loss: 207.3032 - val_ll_loss: 214.5578 - val_loss: 211.9141 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 564ms/step - ll_loss: 229.3611 - loss: 207.2089 - val_ll_loss: 214.4712 - val_loss: 211.8312 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 575ms/step - ll_loss: 229.2613 - loss: 207.1237 - val_ll_loss: 214.3929 - val_loss: 211.7564 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 596ms/step - ll_loss: 229.1711 - loss: 207.0467 - val_ll_loss: 214.3221 - val_loss: 211.6887 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 570ms/step - ll_loss: 229.0896 - loss: 206.9772 - val_ll_loss: 214.2582 - val_loss: 211.6275 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 561ms/step - ll_loss: 229.0159 - loss: 206.9143 - val_ll_loss: 214.2003 - val_loss: 211.5722 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 569ms/step - ll_loss: 228.9493 - loss: 206.8574 - val_ll_loss: 214.1480 - val_loss: 211.5221 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 580ms/step - ll_loss: 228.8891 - loss: 206.8060 - val_ll_loss: 214.1007 - val_loss: 211.4769 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 518ms/step - ll_loss: 228.8345 - loss: 206.7594 - val_ll_loss: 214.0579 - val_loss: 211.4359 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m47s[0m 591ms/step - ll_loss: 228.7853 - loss: 206.7173 - val_ll_loss: 214.0191 - val_loss: 211.3989 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 604ms/step - ll_loss: 228.7407 - loss: 206.6793 - val_ll_loss: 213.9841 - val_loss: 211.3653 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 563ms/step - ll_loss: 228.7003 - loss: 206.6448 - val_ll_loss: 213.9524 - val_loss: 211.3350 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 569ms/step - ll_loss: 228.6639 - loss: 206.6136 - val_ll_loss: 213.9236 - val_loss: 211.3076 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m44s[0m 589ms/step - ll_loss: 228.6308 - loss: 206.5854 - val_ll_loss: 213.8977 - val_loss: 211.2827 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 564ms/step - ll_loss: 228.6009 - loss: 206.5599 - val_ll_loss: 213.8742 - val_loss: 211.2603 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 574ms/step - ll_loss: 228.5738 - loss: 206.5368 - val_ll_loss: 213.8529 - val_loss: 211.2399 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 568ms/step - ll_loss: 228.5494 - loss: 206.5159 - val_ll_loss: 213.8336 - val_loss: 211.2215 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 589ms/step - ll_loss: 228.5272 - loss: 206.4969 - val_ll_loss: 213.8162 - val_loss: 211.2048 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 567ms/step - ll_loss: 228.5072 - loss: 206.4798 - val_ll_loss: 213.8004 - val_loss: 211.1897 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 575ms/step - ll_loss: 228.4890 - loss: 206.4643 - val_ll_loss: 213.7861 - val_loss: 211.1760 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 583ms/step - ll_loss: 228.4725 - loss: 206.4502 - val_ll_loss: 213.7731 - val_loss: 211.1637 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 582ms/step - ll_loss: 228.4577 - loss: 206.4375 - val_ll_loss: 213.7614 - val_loss: 211.1525 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 572ms/step - ll_loss: 228.4442 - loss: 206.4260 - val_ll_loss: 213.7508 - val_loss: 211.1424 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 577ms/step - ll_loss: 228.4320 - loss: 206.4156 - val_ll_loss: 213.7413 - val_loss: 211.1332 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 594ms/step - ll_loss: 228.4210 - loss: 206.4062 - val_ll_loss: 213.7325 - val_loss: 211.1249 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 555ms/step - ll_loss: 228.4110 - loss: 206.3976 - val_ll_loss: 213.7247 - val_loss: 211.1174 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 522ms/step - ll_loss: 228.4020 - loss: 206.3899 - val_ll_loss: 213.7176 - val_loss: 211.1105 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 551ms/step - ll_loss: 228.3938 - loss: 206.3829 - val_ll_loss: 213.7111 - val_loss: 211.1044 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 601ms/step - ll_loss: 228.3864 - loss: 206.3766 - val_ll_loss: 213.7053 - val_loss: 211.0988 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 567ms/step - ll_loss: 228.3797 - loss: 206.3708 - val_ll_loss: 213.7000 - val_loss: 211.0938 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 567ms/step - ll_loss: 228.3736 - loss: 206.3657 - val_ll_loss: 213.6952 - val_loss: 211.0892 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/5 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K07/fold_holdout-sub-16
    \n--- holdout=sub-17 | val=sub-08 | K=7 ---
    train segs=51 short=0 | val segs=10 short=0 | test segs=10 short=0



    Loading files:   0%|          | 0/51 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]


    Epoch 1/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 598ms/step - ll_loss: 257.8932 - loss: 231.5934 - val_ll_loss: 248.2869 - val_loss: 243.8865 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 587ms/step - ll_loss: 255.1588 - loss: 229.2180 - val_ll_loss: 245.8463 - val_loss: 241.5471 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 564ms/step - ll_loss: 252.5466 - loss: 226.9688 - val_ll_loss: 243.7754 - val_loss: 239.5663 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 560ms/step - ll_loss: 250.2825 - loss: 225.0171 - val_ll_loss: 242.0194 - val_loss: 237.8839 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 582ms/step - ll_loss: 248.3390 - loss: 223.3437 - val_ll_loss: 240.5051 - val_loss: 236.4317 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 543ms/step - ll_loss: 246.6503 - loss: 221.8910 - val_ll_loss: 239.1871 - val_loss: 235.1670 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 577ms/step - ll_loss: 245.1723 - loss: 220.6206 - val_ll_loss: 238.0328 - val_loss: 234.0588 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 573ms/step - ll_loss: 243.8721 - loss: 219.5038 - val_ll_loss: 237.0168 - val_loss: 233.0831 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 590ms/step - ll_loss: 242.7235 - loss: 218.5177 - val_ll_loss: 236.1192 - val_loss: 232.2208 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 596ms/step - ll_loss: 241.7056 - loss: 217.6443 - val_ll_loss: 235.3236 - val_loss: 231.4563 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 570ms/step - ll_loss: 240.8009 - loss: 216.8685 - val_ll_loss: 234.6165 - val_loss: 230.7768 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 574ms/step - ll_loss: 239.9950 - loss: 216.1775 - val_ll_loss: 233.9865 - val_loss: 230.1712 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 524ms/step - ll_loss: 239.2755 - loss: 215.5609 - val_ll_loss: 233.4240 - val_loss: 229.6305 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 593ms/step - ll_loss: 238.6321 - loss: 215.0097 - val_ll_loss: 232.9210 - val_loss: 229.1469 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 583ms/step - ll_loss: 238.0559 - loss: 214.5161 - val_ll_loss: 232.4704 - val_loss: 228.7135 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 564ms/step - ll_loss: 237.5390 - loss: 214.0735 - val_ll_loss: 232.0662 - val_loss: 228.3249 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 582ms/step - ll_loss: 237.0748 - loss: 213.6761 - val_ll_loss: 231.7032 - val_loss: 227.9757 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 613ms/step - ll_loss: 236.6576 - loss: 213.3189 - val_ll_loss: 231.3768 - val_loss: 227.6618 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 562ms/step - ll_loss: 236.2822 - loss: 212.9976 - val_ll_loss: 231.0831 - val_loss: 227.3793 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 524ms/step - ll_loss: 235.9440 - loss: 212.7082 - val_ll_loss: 230.8185 - val_loss: 227.1248 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 572ms/step - ll_loss: 235.6393 - loss: 212.4474 - val_ll_loss: 230.5799 - val_loss: 226.8954 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 609ms/step - ll_loss: 235.3645 - loss: 212.2123 - val_ll_loss: 230.3648 - val_loss: 226.6884 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 586ms/step - ll_loss: 235.1165 - loss: 212.0002 - val_ll_loss: 230.1707 - val_loss: 226.5017 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 581ms/step - ll_loss: 234.8927 - loss: 211.8086 - val_ll_loss: 229.9954 - val_loss: 226.3331 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 593ms/step - ll_loss: 234.6906 - loss: 211.6357 - val_ll_loss: 229.8370 - val_loss: 226.1807 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 592ms/step - ll_loss: 234.5079 - loss: 211.4794 - val_ll_loss: 229.6939 - val_loss: 226.0430 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 545ms/step - ll_loss: 234.3428 - loss: 211.3382 - val_ll_loss: 229.5645 - val_loss: 225.9186 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 561ms/step - ll_loss: 234.1936 - loss: 211.2105 - val_ll_loss: 229.4475 - val_loss: 225.8060 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 580ms/step - ll_loss: 234.0585 - loss: 211.0950 - val_ll_loss: 229.3416 - val_loss: 225.7042 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 589ms/step - ll_loss: 233.9364 - loss: 210.9906 - val_ll_loss: 229.2459 - val_loss: 225.6121 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 583ms/step - ll_loss: 233.8260 - loss: 210.8960 - val_ll_loss: 229.1593 - val_loss: 225.5288 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 565ms/step - ll_loss: 233.7260 - loss: 210.8105 - val_ll_loss: 229.0808 - val_loss: 225.4533 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 576ms/step - ll_loss: 233.6355 - loss: 210.7331 - val_ll_loss: 229.0098 - val_loss: 225.3850 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m42s[0m 601ms/step - ll_loss: 233.5536 - loss: 210.6631 - val_ll_loss: 228.9456 - val_loss: 225.3232 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 580ms/step - ll_loss: 233.4795 - loss: 210.5996 - val_ll_loss: 228.8874 - val_loss: 225.2672 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 577ms/step - ll_loss: 233.4124 - loss: 210.5422 - val_ll_loss: 228.8347 - val_loss: 225.2166 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 540ms/step - ll_loss: 233.3517 - loss: 210.4902 - val_ll_loss: 228.7870 - val_loss: 225.1707 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 603ms/step - ll_loss: 233.2966 - loss: 210.4431 - val_ll_loss: 228.7438 - val_loss: 225.1291 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 579ms/step - ll_loss: 233.2468 - loss: 210.4005 - val_ll_loss: 228.7046 - val_loss: 225.0915 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 565ms/step - ll_loss: 233.2017 - loss: 210.3619 - val_ll_loss: 228.6692 - val_loss: 225.0574 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 574ms/step - ll_loss: 233.1608 - loss: 210.3270 - val_ll_loss: 228.6371 - val_loss: 225.0265 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 610ms/step - ll_loss: 233.1239 - loss: 210.2953 - val_ll_loss: 228.6080 - val_loss: 224.9986 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 587ms/step - ll_loss: 233.0904 - loss: 210.2666 - val_ll_loss: 228.5817 - val_loss: 224.9732 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 527ms/step - ll_loss: 233.0600 - loss: 210.2406 - val_ll_loss: 228.5579 - val_loss: 224.9503 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 562ms/step - ll_loss: 233.0325 - loss: 210.2171 - val_ll_loss: 228.5363 - val_loss: 224.9295 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 597ms/step - ll_loss: 233.0076 - loss: 210.1958 - val_ll_loss: 228.5167 - val_loss: 224.9107 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 583ms/step - ll_loss: 232.9851 - loss: 210.1766 - val_ll_loss: 228.4990 - val_loss: 224.8937 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 554ms/step - ll_loss: 232.9647 - loss: 210.1591 - val_ll_loss: 228.4829 - val_loss: 224.8782 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 564ms/step - ll_loss: 232.9462 - loss: 210.1433 - val_ll_loss: 228.4684 - val_loss: 224.8642 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 596ms/step - ll_loss: 232.9295 - loss: 210.1289 - val_ll_loss: 228.4552 - val_loss: 224.8516 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 552ms/step - ll_loss: 232.9143 - loss: 210.1160 - val_ll_loss: 228.4433 - val_loss: 224.8401 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 566ms/step - ll_loss: 232.9006 - loss: 210.1042 - val_ll_loss: 228.4325 - val_loss: 224.8297 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 562ms/step - ll_loss: 232.8882 - loss: 210.0936 - val_ll_loss: 228.4227 - val_loss: 224.8203 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 581ms/step - ll_loss: 232.8769 - loss: 210.0839 - val_ll_loss: 228.4139 - val_loss: 224.8118 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 580ms/step - ll_loss: 232.8667 - loss: 210.0752 - val_ll_loss: 228.4058 - val_loss: 224.8041 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 551ms/step - ll_loss: 232.8575 - loss: 210.0673 - val_ll_loss: 228.3986 - val_loss: 224.7971 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 523ms/step - ll_loss: 232.8492 - loss: 210.0602 - val_ll_loss: 228.3920 - val_loss: 224.7908 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 572ms/step - ll_loss: 232.8416 - loss: 210.0537 - val_ll_loss: 228.3860 - val_loss: 224.7850 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 601ms/step - ll_loss: 232.8347 - loss: 210.0478 - val_ll_loss: 228.3807 - val_loss: 224.7798 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 564ms/step - ll_loss: 232.8286 - loss: 210.0425 - val_ll_loss: 228.3758 - val_loss: 224.7752 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/10 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K07/fold_holdout-sub-17
    \n--- holdout=sub-18 | val=sub-08 | K=7 ---
    train segs=55 short=0 | val segs=10 short=0 | test segs=6 short=0



    Loading files:   0%|          | 0/55 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/6 [00:00<?, ?it/s]


    Epoch 1/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 569ms/step - ll_loss: 262.1854 - loss: 233.6892 - val_ll_loss: 247.6716 - val_loss: 244.1866 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 574ms/step - ll_loss: 259.6943 - loss: 231.4445 - val_ll_loss: 245.2916 - val_loss: 241.9224 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 577ms/step - ll_loss: 257.0087 - loss: 229.0558 - val_ll_loss: 243.1418 - val_loss: 239.8682 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 641ms/step - ll_loss: 254.4496 - loss: 226.6752 - val_ll_loss: 241.2060 - val_loss: 238.0518 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m48s[0m 869ms/step - ll_loss: 252.1361 - loss: 224.5097 - val_ll_loss: 239.3825 - val_loss: 236.3656 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m48s[0m 882ms/step - ll_loss: 250.0145 - loss: 222.5331 - val_ll_loss: 237.6133 - val_loss: 234.7077 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m49s[0m 888ms/step - ll_loss: 247.7862 - loss: 220.6875 - val_ll_loss: 235.9623 - val_loss: 233.1092 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m48s[0m 878ms/step - ll_loss: 245.9214 - loss: 219.1671 - val_ll_loss: 234.6170 - val_loss: 231.8066 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m49s[0m 896ms/step - ll_loss: 244.4215 - loss: 217.9254 - val_ll_loss: 233.4875 - val_loss: 230.7117 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m47s[0m 865ms/step - ll_loss: 243.1694 - loss: 216.8828 - val_ll_loss: 232.5216 - val_loss: 229.7774 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m45s[0m 814ms/step - ll_loss: 242.0950 - loss: 215.9859 - val_ll_loss: 231.6882 - val_loss: 228.9758 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m49s[0m 902ms/step - ll_loss: 241.1638 - loss: 215.2065 - val_ll_loss: 230.9700 - val_loss: 228.3031 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m47s[0m 858ms/step - ll_loss: 240.3505 - loss: 214.5247 - val_ll_loss: 230.3321 - val_loss: 227.6914 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m46s[0m 833ms/step - ll_loss: 239.6359 - loss: 213.9248 - val_ll_loss: 229.7680 - val_loss: 227.1483 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m48s[0m 877ms/step - ll_loss: 239.0048 - loss: 213.3944 - val_ll_loss: 229.2682 - val_loss: 226.6674 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m43s[0m 789ms/step - ll_loss: 238.4453 - loss: 212.9240 - val_ll_loss: 228.8240 - val_loss: 226.2403 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m87s[0m 845ms/step - ll_loss: 237.9477 - loss: 212.5052 - val_ll_loss: 228.4281 - val_loss: 225.8599 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m47s[0m 826ms/step - ll_loss: 237.5041 - loss: 212.1318 - val_ll_loss: 228.0746 - val_loss: 225.5203 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m48s[0m 868ms/step - ll_loss: 237.1078 - loss: 211.7979 - val_ll_loss: 227.7583 - val_loss: 225.2166 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m47s[0m 852ms/step - ll_loss: 236.7531 - loss: 211.4991 - val_ll_loss: 227.4748 - val_loss: 224.9444 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m48s[0m 871ms/step - ll_loss: 236.4351 - loss: 211.2310 - val_ll_loss: 227.2204 - val_loss: 224.7003 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m46s[0m 845ms/step - ll_loss: 236.1498 - loss: 210.9904 - val_ll_loss: 226.9919 - val_loss: 224.4811 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m48s[0m 884ms/step - ll_loss: 235.8933 - loss: 210.7741 - val_ll_loss: 226.7863 - val_loss: 224.2839 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m49s[0m 901ms/step - ll_loss: 235.6628 - loss: 210.5795 - val_ll_loss: 226.6013 - val_loss: 224.1065 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m47s[0m 856ms/step - ll_loss: 235.4552 - loss: 210.4044 - val_ll_loss: 226.4347 - val_loss: 223.9467 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m48s[0m 876ms/step - ll_loss: 235.2682 - loss: 210.2465 - val_ll_loss: 226.2844 - val_loss: 223.8027 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m49s[0m 898ms/step - ll_loss: 235.0997 - loss: 210.1042 - val_ll_loss: 226.1490 - val_loss: 223.6729 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m46s[0m 830ms/step - ll_loss: 234.9477 - loss: 209.9759 - val_ll_loss: 226.0267 - val_loss: 223.5557 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m49s[0m 887ms/step - ll_loss: 234.8106 - loss: 209.8600 - val_ll_loss: 225.9163 - val_loss: 223.4499 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m47s[0m 855ms/step - ll_loss: 234.6867 - loss: 209.7554 - val_ll_loss: 225.8166 - val_loss: 223.3544 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m47s[0m 856ms/step - ll_loss: 234.5750 - loss: 209.6609 - val_ll_loss: 225.7265 - val_loss: 223.2682 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m47s[0m 847ms/step - ll_loss: 234.4740 - loss: 209.5756 - val_ll_loss: 225.6452 - val_loss: 223.1902 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m47s[0m 857ms/step - ll_loss: 234.3828 - loss: 209.4985 - val_ll_loss: 225.5716 - val_loss: 223.1198 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m49s[0m 888ms/step - ll_loss: 234.3003 - loss: 209.4288 - val_ll_loss: 225.5051 - val_loss: 223.0561 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m46s[0m 845ms/step - ll_loss: 234.2258 - loss: 209.3658 - val_ll_loss: 225.4449 - val_loss: 222.9985 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m47s[0m 855ms/step - ll_loss: 234.1584 - loss: 209.3087 - val_ll_loss: 225.3905 - val_loss: 222.9464 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m47s[0m 856ms/step - ll_loss: 234.0974 - loss: 209.2572 - val_ll_loss: 225.3413 - val_loss: 222.8993 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m47s[0m 858ms/step - ll_loss: 234.0423 - loss: 209.2106 - val_ll_loss: 225.2968 - val_loss: 222.8566 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m48s[0m 868ms/step - ll_loss: 233.9924 - loss: 209.1684 - val_ll_loss: 225.2565 - val_loss: 222.8181 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m48s[0m 882ms/step - ll_loss: 233.9473 - loss: 209.1302 - val_ll_loss: 225.2200 - val_loss: 222.7832 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m47s[0m 861ms/step - ll_loss: 233.9065 - loss: 209.0956 - val_ll_loss: 225.1870 - val_loss: 222.7516 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m48s[0m 874ms/step - ll_loss: 233.8695 - loss: 209.0644 - val_ll_loss: 225.1571 - val_loss: 222.7230 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m45s[0m 829ms/step - ll_loss: 233.8361 - loss: 209.0361 - val_ll_loss: 225.1301 - val_loss: 222.6972 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m46s[0m 837ms/step - ll_loss: 233.8059 - loss: 209.0105 - val_ll_loss: 225.1056 - val_loss: 222.6738 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m46s[0m 844ms/step - ll_loss: 233.7785 - loss: 208.9873 - val_ll_loss: 225.0835 - val_loss: 222.6526 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m48s[0m 882ms/step - ll_loss: 233.7538 - loss: 208.9664 - val_ll_loss: 225.0634 - val_loss: 222.6334 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m49s[0m 885ms/step - ll_loss: 233.7313 - loss: 208.9474 - val_ll_loss: 225.0453 - val_loss: 222.6160 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m46s[0m 831ms/step - ll_loss: 233.7110 - loss: 208.9302 - val_ll_loss: 225.0289 - val_loss: 222.6003 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m50s[0m 912ms/step - ll_loss: 233.6927 - loss: 208.9146 - val_ll_loss: 225.0140 - val_loss: 222.5861 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 920ms/step - ll_loss: 233.6761 - loss: 208.9006 - val_ll_loss: 225.0006 - val_loss: 222.5732 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m48s[0m 870ms/step - ll_loss: 233.6610 - loss: 208.8878 - val_ll_loss: 224.9884 - val_loss: 222.5616 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m50s[0m 911ms/step - ll_loss: 233.6474 - loss: 208.8763 - val_ll_loss: 224.9773 - val_loss: 222.5510 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m48s[0m 874ms/step - ll_loss: 233.6351 - loss: 208.8659 - val_ll_loss: 224.9673 - val_loss: 222.5414 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m87s[0m 860ms/step - ll_loss: 233.6239 - loss: 208.8564 - val_ll_loss: 224.9583 - val_loss: 222.5328 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m46s[0m 832ms/step - ll_loss: 233.6138 - loss: 208.8479 - val_ll_loss: 224.9501 - val_loss: 222.5249 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 929ms/step - ll_loss: 233.6047 - loss: 208.8401 - val_ll_loss: 224.9427 - val_loss: 222.5178 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m45s[0m 827ms/step - ll_loss: 233.5964 - loss: 208.8331 - val_ll_loss: 224.9360 - val_loss: 222.5115 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m50s[0m 912ms/step - ll_loss: 233.5889 - loss: 208.8267 - val_ll_loss: 224.9299 - val_loss: 222.5056 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m47s[0m 849ms/step - ll_loss: 233.5822 - loss: 208.8210 - val_ll_loss: 224.9245 - val_loss: 222.5004 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m45s[0m 818ms/step - ll_loss: 233.5760 - loss: 208.8158 - val_ll_loss: 224.9195 - val_loss: 222.4957 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/6 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K07/fold_holdout-sub-18
    \n--- holdout=sub-20 | val=sub-08 | K=7 ---
    train segs=58 short=0 | val segs=10 short=0 | test segs=3 short=0



    Loading files:   0%|          | 0/58 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/3 [00:00<?, ?it/s]


    Epoch 1/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 867ms/step - ll_loss: 261.0052 - loss: 233.2813 - val_ll_loss: 245.1001 - val_loss: 242.3374 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m48s[0m 823ms/step - ll_loss: 258.2744 - loss: 230.8986 - val_ll_loss: 242.3767 - val_loss: 239.7371 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 905ms/step - ll_loss: 255.4364 - loss: 228.4015 - val_ll_loss: 239.7401 - val_loss: 237.2288 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m48s[0m 827ms/step - ll_loss: 252.6653 - loss: 226.0048 - val_ll_loss: 237.6003 - val_loss: 235.1861 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 875ms/step - ll_loss: 250.3842 - loss: 224.0337 - val_ll_loss: 235.8135 - val_loss: 233.4759 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m50s[0m 864ms/step - ll_loss: 248.4518 - loss: 222.3654 - val_ll_loss: 234.2879 - val_loss: 232.0134 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m48s[0m 831ms/step - ll_loss: 246.7872 - loss: 220.9308 - val_ll_loss: 232.9685 - val_loss: 230.7472 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 884ms/step - ll_loss: 245.3386 - loss: 219.6840 - val_ll_loss: 231.8178 - val_loss: 229.6422 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m49s[0m 854ms/step - ll_loss: 244.0690 - loss: 218.5926 - val_ll_loss: 230.8083 - val_loss: 228.6720 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m49s[0m 852ms/step - ll_loss: 242.9506 - loss: 217.6320 - val_ll_loss: 229.9182 - val_loss: 227.8163 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 881ms/step - ll_loss: 241.9614 - loss: 216.7831 - val_ll_loss: 229.1304 - val_loss: 227.0586 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m47s[0m 807ms/step - ll_loss: 241.0835 - loss: 216.0303 - val_ll_loss: 228.4311 - val_loss: 226.3858 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 902ms/step - ll_loss: 240.3022 - loss: 215.3607 - val_ll_loss: 227.8086 - val_loss: 225.7868 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m85s[0m 888ms/step - ll_loss: 239.6054 - loss: 214.7638 - val_ll_loss: 227.2532 - val_loss: 225.2522 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 931ms/step - ll_loss: 238.9826 - loss: 214.2306 - val_ll_loss: 226.7566 - val_loss: 224.7741 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m50s[0m 857ms/step - ll_loss: 238.4250 - loss: 213.7534 - val_ll_loss: 226.3119 - val_loss: 224.3459 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 927ms/step - ll_loss: 237.9250 - loss: 213.3257 - val_ll_loss: 225.9132 - val_loss: 223.9619 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m49s[0m 852ms/step - ll_loss: 237.4762 - loss: 212.9418 - val_ll_loss: 225.5550 - val_loss: 223.6169 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 913ms/step - ll_loss: 237.0728 - loss: 212.5968 - val_ll_loss: 225.2331 - val_loss: 223.3068 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m48s[0m 819ms/step - ll_loss: 236.7098 - loss: 212.2865 - val_ll_loss: 224.9433 - val_loss: 223.0277 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 886ms/step - ll_loss: 236.3830 - loss: 212.0072 - val_ll_loss: 224.6824 - val_loss: 222.7763 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m50s[0m 858ms/step - ll_loss: 236.0885 - loss: 211.7555 - val_ll_loss: 224.4472 - val_loss: 222.5497 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m48s[0m 825ms/step - ll_loss: 235.8229 - loss: 211.5285 - val_ll_loss: 224.2350 - val_loss: 222.3453 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 890ms/step - ll_loss: 235.5833 - loss: 211.3238 - val_ll_loss: 224.0435 - val_loss: 222.1608 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m48s[0m 824ms/step - ll_loss: 235.3669 - loss: 211.1390 - val_ll_loss: 223.8707 - val_loss: 221.9942 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 914ms/step - ll_loss: 235.1715 - loss: 210.9720 - val_ll_loss: 223.7144 - val_loss: 221.8437 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m49s[0m 840ms/step - ll_loss: 234.9950 - loss: 210.8212 - val_ll_loss: 223.5733 - val_loss: 221.7077 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 883ms/step - ll_loss: 234.8354 - loss: 210.6849 - val_ll_loss: 223.4457 - val_loss: 221.5847 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m50s[0m 859ms/step - ll_loss: 234.6912 - loss: 210.5617 - val_ll_loss: 223.3302 - val_loss: 221.4734 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m50s[0m 870ms/step - ll_loss: 234.5607 - loss: 210.4502 - val_ll_loss: 223.2258 - val_loss: 221.3729 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 884ms/step - ll_loss: 234.4427 - loss: 210.3494 - val_ll_loss: 223.1314 - val_loss: 221.2818 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m49s[0m 846ms/step - ll_loss: 234.3359 - loss: 210.2582 - val_ll_loss: 223.0459 - val_loss: 221.1994 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m48s[0m 838ms/step - ll_loss: 234.2393 - loss: 210.1757 - val_ll_loss: 222.9685 - val_loss: 221.1249 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 916ms/step - ll_loss: 234.1519 - loss: 210.1010 - val_ll_loss: 222.8985 - val_loss: 221.0574 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m48s[0m 838ms/step - ll_loss: 234.0727 - loss: 210.0334 - val_ll_loss: 222.8351 - val_loss: 220.9963 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 907ms/step - ll_loss: 234.0011 - loss: 209.9722 - val_ll_loss: 222.7777 - val_loss: 220.9410 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m43s[0m 744ms/step - ll_loss: 233.9362 - loss: 209.9167 - val_ll_loss: 222.7258 - val_loss: 220.8909 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 578ms/step - ll_loss: 233.8775 - loss: 209.8666 - val_ll_loss: 222.6787 - val_loss: 220.8456 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 579ms/step - ll_loss: 233.8243 - loss: 209.8212 - val_ll_loss: 222.6361 - val_loss: 220.8045 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 609ms/step - ll_loss: 233.7761 - loss: 209.7801 - val_ll_loss: 222.5975 - val_loss: 220.7673 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 542ms/step - ll_loss: 233.7325 - loss: 209.7428 - val_ll_loss: 222.5625 - val_loss: 220.7336 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 511ms/step - ll_loss: 233.6931 - loss: 209.7090 - val_ll_loss: 222.5308 - val_loss: 220.7031 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 510ms/step - ll_loss: 233.6573 - loss: 209.6785 - val_ll_loss: 222.5022 - val_loss: 220.6754 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 493ms/step - ll_loss: 233.6249 - loss: 209.6508 - val_ll_loss: 222.4762 - val_loss: 220.6504 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 566ms/step - ll_loss: 233.5956 - loss: 209.6258 - val_ll_loss: 222.4527 - val_loss: 220.6277 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 600ms/step - ll_loss: 233.5690 - loss: 209.6031 - val_ll_loss: 222.4314 - val_loss: 220.6072 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 551ms/step - ll_loss: 233.5450 - loss: 209.5825 - val_ll_loss: 222.4121 - val_loss: 220.5886 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 550ms/step - ll_loss: 233.5233 - loss: 209.5639 - val_ll_loss: 222.3946 - val_loss: 220.5718 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 527ms/step - ll_loss: 233.5035 - loss: 209.5471 - val_ll_loss: 222.3788 - val_loss: 220.5565 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 536ms/step - ll_loss: 233.4857 - loss: 209.5318 - val_ll_loss: 222.3645 - val_loss: 220.5427 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 521ms/step - ll_loss: 233.4695 - loss: 209.5180 - val_ll_loss: 222.3515 - val_loss: 220.5302 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 573ms/step - ll_loss: 233.4549 - loss: 209.5055 - val_ll_loss: 222.3397 - val_loss: 220.5189 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 562ms/step - ll_loss: 233.4416 - loss: 209.4941 - val_ll_loss: 222.3291 - val_loss: 220.5086 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 585ms/step - ll_loss: 233.4296 - loss: 209.4839 - val_ll_loss: 222.3194 - val_loss: 220.4993 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 593ms/step - ll_loss: 233.4187 - loss: 209.4746 - val_ll_loss: 222.3107 - val_loss: 220.4909 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 571ms/step - ll_loss: 233.4089 - loss: 209.4662 - val_ll_loss: 222.3028 - val_loss: 220.4833 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 562ms/step - ll_loss: 233.4000 - loss: 209.4585 - val_ll_loss: 222.2956 - val_loss: 220.4764 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 542ms/step - ll_loss: 233.3919 - loss: 209.4517 - val_ll_loss: 222.2891 - val_loss: 220.4702 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 603ms/step - ll_loss: 233.3846 - loss: 209.4454 - val_ll_loss: 222.2832 - val_loss: 220.4645 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 561ms/step - ll_loss: 233.3780 - loss: 209.4397 - val_ll_loss: 222.2780 - val_loss: 220.4594 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/3 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K07/fold_holdout-sub-20
    \n--- holdout=sub-21 | val=sub-08 | K=7 ---
    train segs=55 short=0 | val segs=10 short=0 | test segs=6 short=0



    Loading files:   0%|          | 0/55 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/6 [00:00<?, ?it/s]


    Epoch 1/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 593ms/step - ll_loss: 257.8069 - loss: 230.2342 - val_ll_loss: 242.9790 - val_loss: 239.5494 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 532ms/step - ll_loss: 255.0618 - loss: 227.8782 - val_ll_loss: 240.4472 - val_loss: 237.1186 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 570ms/step - ll_loss: 252.1645 - loss: 225.2445 - val_ll_loss: 237.7947 - val_loss: 234.5512 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 529ms/step - ll_loss: 249.1049 - loss: 222.6158 - val_ll_loss: 235.4423 - val_loss: 232.2753 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 558ms/step - ll_loss: 246.5045 - loss: 220.4236 - val_ll_loss: 233.6135 - val_loss: 230.5153 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 593ms/step - ll_loss: 244.4557 - loss: 218.6821 - val_ll_loss: 232.0989 - val_loss: 229.0745 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 580ms/step - ll_loss: 242.7363 - loss: 217.2187 - val_ll_loss: 230.8116 - val_loss: 227.8419 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 577ms/step - ll_loss: 241.2771 - loss: 215.9748 - val_ll_loss: 229.7037 - val_loss: 226.7796 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 587ms/step - ll_loss: 240.0204 - loss: 214.9024 - val_ll_loss: 228.7411 - val_loss: 225.8573 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 613ms/step - ll_loss: 238.9283 - loss: 213.9702 - val_ll_loss: 227.8962 - val_loss: 225.0481 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 562ms/step - ll_loss: 237.9685 - loss: 213.1510 - val_ll_loss: 227.1504 - val_loss: 224.3341 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m44s[0m 591ms/step - ll_loss: 237.1196 - loss: 212.4263 - val_ll_loss: 226.4917 - val_loss: 223.7037 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 612ms/step - ll_loss: 236.3686 - loss: 211.7851 - val_ll_loss: 225.9077 - val_loss: 223.1450 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 597ms/step - ll_loss: 235.7019 - loss: 211.2160 - val_ll_loss: 225.3885 - val_loss: 222.6483 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 555ms/step - ll_loss: 235.1085 - loss: 210.7093 - val_ll_loss: 224.9256 - val_loss: 222.2055 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 556ms/step - ll_loss: 234.5789 - loss: 210.2572 - val_ll_loss: 224.5121 - val_loss: 221.8101 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 603ms/step - ll_loss: 234.1056 - loss: 209.8530 - val_ll_loss: 224.1421 - val_loss: 221.4563 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 578ms/step - ll_loss: 233.6816 - loss: 209.4911 - val_ll_loss: 223.8103 - val_loss: 221.1391 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 525ms/step - ll_loss: 233.3013 - loss: 209.1664 - val_ll_loss: 223.5126 - val_loss: 220.8545 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 579ms/step - ll_loss: 232.9599 - loss: 208.8749 - val_ll_loss: 223.2451 - val_loss: 220.5988 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 616ms/step - ll_loss: 232.6530 - loss: 208.6129 - val_ll_loss: 223.0045 - val_loss: 220.3688 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 543ms/step - ll_loss: 232.3768 - loss: 208.3770 - val_ll_loss: 222.7878 - val_loss: 220.1617 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 551ms/step - ll_loss: 232.1280 - loss: 208.1647 - val_ll_loss: 222.5926 - val_loss: 219.9752 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 592ms/step - ll_loss: 231.9039 - loss: 207.9733 - val_ll_loss: 222.4166 - val_loss: 219.8070 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 603ms/step - ll_loss: 231.7019 - loss: 207.8008 - val_ll_loss: 222.2578 - val_loss: 219.6552 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 574ms/step - ll_loss: 231.5195 - loss: 207.6451 - val_ll_loss: 222.1145 - val_loss: 219.5183 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 552ms/step - ll_loss: 231.3549 - loss: 207.5045 - val_ll_loss: 221.9851 - val_loss: 219.3946 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 546ms/step - ll_loss: 231.2063 - loss: 207.3775 - val_ll_loss: 221.8681 - val_loss: 219.2829 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 590ms/step - ll_loss: 231.0720 - loss: 207.2629 - val_ll_loss: 221.7625 - val_loss: 219.1820 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 546ms/step - ll_loss: 230.9506 - loss: 207.1592 - val_ll_loss: 221.6669 - val_loss: 219.0907 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 561ms/step - ll_loss: 230.8409 - loss: 207.0655 - val_ll_loss: 221.5806 - val_loss: 219.0082 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 505ms/step - ll_loss: 230.7418 - loss: 206.9808 - val_ll_loss: 221.5024 - val_loss: 218.9336 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 608ms/step - ll_loss: 230.6521 - loss: 206.9042 - val_ll_loss: 221.4318 - val_loss: 218.8661 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 559ms/step - ll_loss: 230.5710 - loss: 206.8349 - val_ll_loss: 221.3678 - val_loss: 218.8049 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m46s[0m 584ms/step - ll_loss: 230.4976 - loss: 206.7722 - val_ll_loss: 221.3099 - val_loss: 218.7497 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 596ms/step - ll_loss: 230.4312 - loss: 206.7154 - val_ll_loss: 221.2575 - val_loss: 218.6997 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 480ms/step - ll_loss: 230.3711 - loss: 206.6641 - val_ll_loss: 221.2102 - val_loss: 218.6544 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 554ms/step - ll_loss: 230.3167 - loss: 206.6176 - val_ll_loss: 221.1672 - val_loss: 218.6134 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 554ms/step - ll_loss: 230.2675 - loss: 206.5755 - val_ll_loss: 221.1284 - val_loss: 218.5764 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 622ms/step - ll_loss: 230.2229 - loss: 206.5374 - val_ll_loss: 221.0932 - val_loss: 218.5428 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 589ms/step - ll_loss: 230.1826 - loss: 206.5030 - val_ll_loss: 221.0614 - val_loss: 218.5124 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 603ms/step - ll_loss: 230.1461 - loss: 206.4718 - val_ll_loss: 221.0325 - val_loss: 218.4848 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 604ms/step - ll_loss: 230.1130 - loss: 206.4435 - val_ll_loss: 221.0064 - val_loss: 218.4599 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 630ms/step - ll_loss: 230.0831 - loss: 206.4179 - val_ll_loss: 220.9828 - val_loss: 218.4373 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 553ms/step - ll_loss: 230.0560 - loss: 206.3948 - val_ll_loss: 220.9614 - val_loss: 218.4169 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 582ms/step - ll_loss: 230.0315 - loss: 206.3738 - val_ll_loss: 220.9420 - val_loss: 218.3984 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 604ms/step - ll_loss: 230.0093 - loss: 206.3548 - val_ll_loss: 220.9245 - val_loss: 218.3816 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 532ms/step - ll_loss: 229.9892 - loss: 206.3376 - val_ll_loss: 220.9086 - val_loss: 218.3665 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 568ms/step - ll_loss: 229.9710 - loss: 206.3220 - val_ll_loss: 220.8942 - val_loss: 218.3527 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 554ms/step - ll_loss: 229.9546 - loss: 206.3080 - val_ll_loss: 220.8812 - val_loss: 218.3403 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 599ms/step - ll_loss: 229.9396 - loss: 206.2952 - val_ll_loss: 220.8694 - val_loss: 218.3291 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 577ms/step - ll_loss: 229.9261 - loss: 206.2837 - val_ll_loss: 220.8587 - val_loss: 218.3188 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 576ms/step - ll_loss: 229.9139 - loss: 206.2732 - val_ll_loss: 220.8490 - val_loss: 218.3096 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 568ms/step - ll_loss: 229.9028 - loss: 206.2637 - val_ll_loss: 220.8403 - val_loss: 218.3013 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 603ms/step - ll_loss: 229.8928 - loss: 206.2552 - val_ll_loss: 220.8324 - val_loss: 218.2937 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 573ms/step - ll_loss: 229.8838 - loss: 206.2474 - val_ll_loss: 220.8252 - val_loss: 218.2868 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 569ms/step - ll_loss: 229.8755 - loss: 206.2404 - val_ll_loss: 220.8187 - val_loss: 218.2807 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 561ms/step - ll_loss: 229.8681 - loss: 206.2340 - val_ll_loss: 220.8128 - val_loss: 218.2751 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 582ms/step - ll_loss: 229.8614 - loss: 206.2283 - val_ll_loss: 220.8075 - val_loss: 218.2700 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 551ms/step - ll_loss: 229.8553 - loss: 206.2231 - val_ll_loss: 220.8026 - val_loss: 218.2653 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/6 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K07/fold_holdout-sub-21
    \n==================== K=8 ====================
    \n--- holdout=sub-01 | val=sub-08 | K=8 ---
    train segs=55 short=0 | val segs=10 short=0 | test segs=6 short=0



    Loading files:   0%|          | 0/55 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/6 [00:00<?, ?it/s]


    Epoch 1/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 643ms/step - ll_loss: 271.1994 - loss: 224.1921 - val_ll_loss: 251.3709 - val_loss: 246.8999 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 618ms/step - ll_loss: 268.8456 - loss: 222.1916 - val_ll_loss: 249.3144 - val_loss: 244.9625 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 592ms/step - ll_loss: 266.5368 - loss: 220.3676 - val_ll_loss: 247.2690 - val_loss: 243.0273 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 575ms/step - ll_loss: 264.2545 - loss: 218.5280 - val_ll_loss: 245.2255 - val_loss: 241.0982 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 605ms/step - ll_loss: 261.9156 - loss: 216.5157 - val_ll_loss: 243.2551 - val_loss: 239.2440 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 608ms/step - ll_loss: 259.0811 - loss: 214.2353 - val_ll_loss: 241.5981 - val_loss: 237.7226 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 560ms/step - ll_loss: 256.8088 - loss: 212.4555 - val_ll_loss: 240.1626 - val_loss: 236.4203 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 564ms/step - ll_loss: 254.9076 - loss: 210.9805 - val_ll_loss: 238.8068 - val_loss: 235.1089 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m44s[0m 623ms/step - ll_loss: 253.2735 - loss: 209.7202 - val_ll_loss: 237.5945 - val_loss: 233.9196 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 582ms/step - ll_loss: 251.8108 - loss: 208.6222 - val_ll_loss: 236.5244 - val_loss: 232.8913 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 578ms/step - ll_loss: 250.5650 - loss: 207.6896 - val_ll_loss: 235.6046 - val_loss: 232.0077 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 575ms/step - ll_loss: 249.5085 - loss: 206.8944 - val_ll_loss: 234.8046 - val_loss: 231.2341 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 590ms/step - ll_loss: 248.6007 - loss: 206.2097 - val_ll_loss: 234.1063 - val_loss: 230.5595 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 539ms/step - ll_loss: 247.8118 - loss: 205.6145 - val_ll_loss: 233.4930 - val_loss: 229.9676 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 576ms/step - ll_loss: 247.1224 - loss: 205.0946 - val_ll_loss: 232.9521 - val_loss: 229.4462 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 594ms/step - ll_loss: 246.5149 - loss: 204.6360 - val_ll_loss: 232.4736 - val_loss: 228.9852 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 615ms/step - ll_loss: 245.9768 - loss: 204.2293 - val_ll_loss: 232.0489 - val_loss: 228.5763 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 609ms/step - ll_loss: 245.4989 - loss: 203.8678 - val_ll_loss: 231.6709 - val_loss: 228.2127 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 629ms/step - ll_loss: 245.0736 - loss: 203.5458 - val_ll_loss: 231.3338 - val_loss: 227.8885 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 587ms/step - ll_loss: 244.6941 - loss: 203.2584 - val_ll_loss: 231.0325 - val_loss: 227.5988 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 588ms/step - ll_loss: 244.3551 - loss: 203.0015 - val_ll_loss: 230.7628 - val_loss: 227.3396 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 560ms/step - ll_loss: 244.0515 - loss: 202.7714 - val_ll_loss: 230.5209 - val_loss: 227.1072 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 560ms/step - ll_loss: 243.7794 - loss: 202.5650 - val_ll_loss: 230.3038 - val_loss: 226.8987 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 591ms/step - ll_loss: 243.5351 - loss: 202.3797 - val_ll_loss: 230.1086 - val_loss: 226.7113 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 570ms/step - ll_loss: 243.3157 - loss: 202.2132 - val_ll_loss: 229.9332 - val_loss: 226.5429 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 562ms/step - ll_loss: 243.1183 - loss: 202.0633 - val_ll_loss: 229.7751 - val_loss: 226.3912 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 559ms/step - ll_loss: 242.9406 - loss: 201.9284 - val_ll_loss: 229.6328 - val_loss: 226.2546 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 601ms/step - ll_loss: 242.7805 - loss: 201.8069 - val_ll_loss: 229.5045 - val_loss: 226.1315 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 642ms/step - ll_loss: 242.6363 - loss: 201.6973 - val_ll_loss: 229.3887 - val_loss: 226.0205 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 649ms/step - ll_loss: 242.5063 - loss: 201.5985 - val_ll_loss: 229.2843 - val_loss: 225.9204 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 661ms/step - ll_loss: 242.3889 - loss: 201.5093 - val_ll_loss: 229.1900 - val_loss: 225.8300 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m46s[0m 833ms/step - ll_loss: 242.2830 - loss: 201.4288 - val_ll_loss: 229.1049 - val_loss: 225.7484 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m94s[0m 970ms/step - ll_loss: 242.1874 - loss: 201.3560 - val_ll_loss: 229.0280 - val_loss: 225.6747 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 934ms/step - ll_loss: 242.1010 - loss: 201.2903 - val_ll_loss: 228.9585 - val_loss: 225.6081 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 966ms/step - ll_loss: 242.0229 - loss: 201.2309 - val_ll_loss: 228.8957 - val_loss: 225.5479 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m50s[0m 913ms/step - ll_loss: 241.9523 - loss: 201.1772 - val_ll_loss: 228.8389 - val_loss: 225.4935 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 1s/step - ll_loss: 241.8884 - loss: 201.1285 - val_ll_loss: 228.7876 - val_loss: 225.4443 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m49s[0m 900ms/step - ll_loss: 241.8305 - loss: 201.0844 - val_ll_loss: 228.7412 - val_loss: 225.3998 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 1s/step - ll_loss: 241.7781 - loss: 201.0444 - val_ll_loss: 228.6992 - val_loss: 225.3596 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 920ms/step - ll_loss: 241.7306 - loss: 201.0081 - val_ll_loss: 228.6613 - val_loss: 225.3232 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 996ms/step - ll_loss: 241.6875 - loss: 200.9751 - val_ll_loss: 228.6269 - val_loss: 225.2903 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 938ms/step - ll_loss: 241.6485 - loss: 200.9453 - val_ll_loss: 228.5959 - val_loss: 225.2606 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 957ms/step - ll_loss: 241.6133 - loss: 200.9184 - val_ll_loss: 228.5677 - val_loss: 225.2337 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 1s/step - ll_loss: 241.5815 - loss: 200.8940 - val_ll_loss: 228.5423 - val_loss: 225.2093 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 927ms/step - ll_loss: 241.5528 - loss: 200.8721 - val_ll_loss: 228.5193 - val_loss: 225.1873 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 981ms/step - ll_loss: 241.5268 - loss: 200.8522 - val_ll_loss: 228.4984 - val_loss: 225.1673 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m48s[0m 875ms/step - ll_loss: 241.5034 - loss: 200.8344 - val_ll_loss: 228.4796 - val_loss: 225.1493 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 981ms/step - ll_loss: 241.4821 - loss: 200.8181 - val_ll_loss: 228.4625 - val_loss: 225.1329 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m48s[0m 874ms/step - ll_loss: 241.4630 - loss: 200.8035 - val_ll_loss: 228.4471 - val_loss: 225.1181 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 975ms/step - ll_loss: 241.4456 - loss: 200.7903 - val_ll_loss: 228.4331 - val_loss: 225.1048 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m50s[0m 914ms/step - ll_loss: 241.4299 - loss: 200.7783 - val_ll_loss: 228.4204 - val_loss: 225.0926 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 943ms/step - ll_loss: 241.4157 - loss: 200.7674 - val_ll_loss: 228.4090 - val_loss: 225.0817 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 967ms/step - ll_loss: 241.4028 - loss: 200.7575 - val_ll_loss: 228.3986 - val_loss: 225.0718 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 947ms/step - ll_loss: 241.3912 - loss: 200.7487 - val_ll_loss: 228.3893 - val_loss: 225.0628 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m85s[0m 923ms/step - ll_loss: 241.3806 - loss: 200.7406 - val_ll_loss: 228.3808 - val_loss: 225.0547 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 938ms/step - ll_loss: 241.3711 - loss: 200.7333 - val_ll_loss: 228.3731 - val_loss: 225.0473 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 978ms/step - ll_loss: 241.3625 - loss: 200.7267 - val_ll_loss: 228.3661 - val_loss: 225.0406 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 935ms/step - ll_loss: 241.3546 - loss: 200.7207 - val_ll_loss: 228.3598 - val_loss: 225.0346 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 966ms/step - ll_loss: 241.3476 - loss: 200.7153 - val_ll_loss: 228.3541 - val_loss: 225.0292 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m85s[0m 951ms/step - ll_loss: 241.3411 - loss: 200.7104 - val_ll_loss: 228.3490 - val_loss: 225.0242 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/6 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K08/fold_holdout-sub-01
    \n--- holdout=sub-02 | val=sub-08 | K=8 ---
    train segs=59 short=0 | val segs=10 short=0 | test segs=2 short=0



    Loading files:   0%|          | 0/59 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/2 [00:00<?, ?it/s]


    Epoch 1/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m58s[0m 979ms/step - ll_loss: 260.1964 - loss: 237.3493 - val_ll_loss: 234.7331 - val_loss: 231.5318 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 901ms/step - ll_loss: 257.4030 - loss: 234.6903 - val_ll_loss: 232.1742 - val_loss: 229.0839 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m58s[0m 988ms/step - ll_loss: 254.7907 - loss: 232.1955 - val_ll_loss: 229.9608 - val_loss: 226.9587 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 945ms/step - ll_loss: 252.2371 - loss: 229.9743 - val_ll_loss: 228.1003 - val_loss: 225.1739 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 936ms/step - ll_loss: 250.0676 - loss: 228.0844 - val_ll_loss: 226.5039 - val_loss: 223.6432 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m87s[0m 943ms/step - ll_loss: 248.2008 - loss: 226.4546 - val_ll_loss: 225.1192 - val_loss: 222.3159 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m81s[0m 865ms/step - ll_loss: 246.5771 - loss: 225.0361 - val_ll_loss: 223.9099 - val_loss: 221.1568 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 953ms/step - ll_loss: 245.1553 - loss: 223.7936 - val_ll_loss: 222.8479 - val_loss: 220.1391 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 905ms/step - ll_loss: 243.9039 - loss: 222.7000 - val_ll_loss: 221.9115 - val_loss: 219.2417 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m87s[0m 940ms/step - ll_loss: 242.7981 - loss: 221.7336 - val_ll_loss: 221.0830 - val_loss: 218.4477 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 868ms/step - ll_loss: 241.8177 - loss: 220.8770 - val_ll_loss: 220.3475 - val_loss: 217.7429 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 930ms/step - ll_loss: 240.9460 - loss: 220.1154 - val_ll_loss: 219.6930 - val_loss: 217.1157 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 924ms/step - ll_loss: 240.1692 - loss: 219.4368 - val_ll_loss: 219.1092 - val_loss: 216.5563 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 962ms/step - ll_loss: 239.4754 - loss: 218.8309 - val_ll_loss: 218.5876 - val_loss: 216.0565 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 868ms/step - ll_loss: 238.8548 - loss: 218.2889 - val_ll_loss: 218.1206 - val_loss: 215.6090 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 940ms/step - ll_loss: 238.2986 - loss: 217.8034 - val_ll_loss: 217.7020 - val_loss: 215.2079 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 909ms/step - ll_loss: 237.7997 - loss: 217.3677 - val_ll_loss: 217.3262 - val_loss: 214.8478 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 944ms/step - ll_loss: 237.3514 - loss: 216.9764 - val_ll_loss: 216.9884 - val_loss: 214.5242 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 974ms/step - ll_loss: 236.9483 - loss: 216.6245 - val_ll_loss: 216.6845 - val_loss: 214.2330 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 895ms/step - ll_loss: 236.5854 - loss: 216.3079 - val_ll_loss: 216.4109 - val_loss: 213.9708 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m58s[0m 987ms/step - ll_loss: 236.2586 - loss: 216.0226 - val_ll_loss: 216.1643 - val_loss: 213.7346 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 886ms/step - ll_loss: 235.9639 - loss: 215.7654 - val_ll_loss: 215.9419 - val_loss: 213.5215 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 976ms/step - ll_loss: 235.6981 - loss: 215.5333 - val_ll_loss: 215.7412 - val_loss: 213.3292 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 930ms/step - ll_loss: 235.4581 - loss: 215.3239 - val_ll_loss: 215.5600 - val_loss: 213.1556 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 947ms/step - ll_loss: 235.2415 - loss: 215.1349 - val_ll_loss: 215.3963 - val_loss: 212.9987 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 943ms/step - ll_loss: 235.0457 - loss: 214.9640 - val_ll_loss: 215.2484 - val_loss: 212.8570 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 973ms/step - ll_loss: 234.8688 - loss: 214.8096 - val_ll_loss: 215.1146 - val_loss: 212.7289 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 892ms/step - ll_loss: 234.7089 - loss: 214.6700 - val_ll_loss: 214.9937 - val_loss: 212.6130 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 922ms/step - ll_loss: 234.5643 - loss: 214.5437 - val_ll_loss: 214.8843 - val_loss: 212.5082 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 869ms/step - ll_loss: 234.4334 - loss: 214.4296 - val_ll_loss: 214.7853 - val_loss: 212.4134 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 971ms/step - ll_loss: 234.3151 - loss: 214.3262 - val_ll_loss: 214.6957 - val_loss: 212.3276 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 869ms/step - ll_loss: 234.2080 - loss: 214.2327 - val_ll_loss: 214.6146 - val_loss: 212.2499 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m48s[0m 808ms/step - ll_loss: 234.1110 - loss: 214.1481 - val_ll_loss: 214.5412 - val_loss: 212.1796 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 604ms/step - ll_loss: 234.0233 - loss: 214.0715 - val_ll_loss: 214.4748 - val_loss: 212.1159 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 620ms/step - ll_loss: 233.9439 - loss: 214.0021 - val_ll_loss: 214.4146 - val_loss: 212.0583 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 591ms/step - ll_loss: 233.8719 - loss: 213.9393 - val_ll_loss: 214.3601 - val_loss: 212.0061 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 586ms/step - ll_loss: 233.8068 - loss: 213.8825 - val_ll_loss: 214.3108 - val_loss: 211.9588 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 604ms/step - ll_loss: 233.7479 - loss: 213.8310 - val_ll_loss: 214.2661 - val_loss: 211.9160 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 590ms/step - ll_loss: 233.6945 - loss: 213.7844 - val_ll_loss: 214.2256 - val_loss: 211.8773 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 606ms/step - ll_loss: 233.6461 - loss: 213.7421 - val_ll_loss: 214.1889 - val_loss: 211.8421 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 609ms/step - ll_loss: 233.6023 - loss: 213.7039 - val_ll_loss: 214.1557 - val_loss: 211.8103 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 607ms/step - ll_loss: 233.5627 - loss: 213.6692 - val_ll_loss: 214.1256 - val_loss: 211.7815 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 558ms/step - ll_loss: 233.5268 - loss: 213.6378 - val_ll_loss: 214.0984 - val_loss: 211.7554 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 556ms/step - ll_loss: 233.4942 - loss: 213.6095 - val_ll_loss: 214.0737 - val_loss: 211.7318 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 576ms/step - ll_loss: 233.4648 - loss: 213.5837 - val_ll_loss: 214.0513 - val_loss: 211.7104 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 576ms/step - ll_loss: 233.4381 - loss: 213.5604 - val_ll_loss: 214.0311 - val_loss: 211.6910 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 554ms/step - ll_loss: 233.4140 - loss: 213.5393 - val_ll_loss: 214.0128 - val_loss: 211.6735 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 552ms/step - ll_loss: 233.3921 - loss: 213.5202 - val_ll_loss: 213.9962 - val_loss: 211.6575 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 592ms/step - ll_loss: 233.3723 - loss: 213.5029 - val_ll_loss: 213.9811 - val_loss: 211.6431 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 547ms/step - ll_loss: 233.3543 - loss: 213.4872 - val_ll_loss: 213.9675 - val_loss: 211.6301 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 557ms/step - ll_loss: 233.3381 - loss: 213.4730 - val_ll_loss: 213.9552 - val_loss: 211.6183 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 571ms/step - ll_loss: 233.3233 - loss: 213.4601 - val_ll_loss: 213.9440 - val_loss: 211.6076 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 574ms/step - ll_loss: 233.3100 - loss: 213.4485 - val_ll_loss: 213.9339 - val_loss: 211.5979 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 549ms/step - ll_loss: 233.2979 - loss: 213.4379 - val_ll_loss: 213.9247 - val_loss: 211.5891 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 560ms/step - ll_loss: 233.2870 - loss: 213.4284 - val_ll_loss: 213.9164 - val_loss: 211.5811 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 585ms/step - ll_loss: 233.2771 - loss: 213.4197 - val_ll_loss: 213.9089 - val_loss: 211.5739 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 516ms/step - ll_loss: 233.2682 - loss: 213.4119 - val_ll_loss: 213.9021 - val_loss: 211.5674 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 552ms/step - ll_loss: 233.2600 - loss: 213.4048 - val_ll_loss: 213.8959 - val_loss: 211.5615 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 555ms/step - ll_loss: 233.2527 - loss: 213.3984 - val_ll_loss: 213.8903 - val_loss: 211.5562 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 596ms/step - ll_loss: 233.2460 - loss: 213.3926 - val_ll_loss: 213.8853 - val_loss: 211.5514 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/2 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K08/fold_holdout-sub-02
    \n--- holdout=sub-03 | val=sub-08 | K=8 ---
    train segs=53 short=0 | val segs=10 short=0 | test segs=8 short=0



    Loading files:   0%|          | 0/53 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 1/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 518ms/step - ll_loss: 233.6342 - loss: 236.3739 - val_ll_loss: 278.0517 - val_loss: 277.4518 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 568ms/step - ll_loss: 230.9303 - loss: 233.6390 - val_ll_loss: 274.6825 - val_loss: 274.1480 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 562ms/step - ll_loss: 228.2881 - loss: 230.9566 - val_ll_loss: 271.8156 - val_loss: 271.3396 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 595ms/step - ll_loss: 226.0351 - loss: 228.6725 - val_ll_loss: 269.3844 - val_loss: 268.9809 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 555ms/step - ll_loss: 224.1064 - loss: 226.7114 - val_ll_loss: 267.2814 - val_loss: 266.9192 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 562ms/step - ll_loss: 222.4398 - loss: 225.0166 - val_ll_loss: 265.4557 - val_loss: 265.1295 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 562ms/step - ll_loss: 220.9880 - loss: 223.5400 - val_ll_loss: 263.8605 - val_loss: 263.5661 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 562ms/step - ll_loss: 219.7158 - loss: 222.2459 - val_ll_loss: 262.4597 - val_loss: 262.1934 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 520ms/step - ll_loss: 218.5959 - loss: 221.1065 - val_ll_loss: 261.2245 - val_loss: 260.9830 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 568ms/step - ll_loss: 217.6062 - loss: 220.0995 - val_ll_loss: 260.1318 - val_loss: 259.9122 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 563ms/step - ll_loss: 216.7289 - loss: 219.2068 - val_ll_loss: 259.1620 - val_loss: 258.9619 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 601ms/step - ll_loss: 215.9490 - loss: 218.4131 - val_ll_loss: 258.2990 - val_loss: 258.1165 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 558ms/step - ll_loss: 215.2541 - loss: 217.7059 - val_ll_loss: 257.5296 - val_loss: 257.3625 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 579ms/step - ll_loss: 214.6337 - loss: 217.0744 - val_ll_loss: 256.8421 - val_loss: 256.6890 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 575ms/step - ll_loss: 214.0788 - loss: 216.5095 - val_ll_loss: 256.2267 - val_loss: 256.0861 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 646ms/step - ll_loss: 213.5817 - loss: 216.0034 - val_ll_loss: 255.6751 - val_loss: 255.5457 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 527ms/step - ll_loss: 213.1357 - loss: 215.5494 - val_ll_loss: 255.1800 - val_loss: 255.0606 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 571ms/step - ll_loss: 212.7352 - loss: 215.1416 - val_ll_loss: 254.7350 - val_loss: 254.6247 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 587ms/step - ll_loss: 212.3750 - loss: 214.7748 - val_ll_loss: 254.3347 - val_loss: 254.2325 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 583ms/step - ll_loss: 212.0508 - loss: 214.4447 - val_ll_loss: 253.9742 - val_loss: 253.8793 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 568ms/step - ll_loss: 211.7588 - loss: 214.1474 - val_ll_loss: 253.6493 - val_loss: 253.5611 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 563ms/step - ll_loss: 211.4955 - loss: 213.8793 - val_ll_loss: 253.3564 - val_loss: 253.2742 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 588ms/step - ll_loss: 211.2581 - loss: 213.6375 - val_ll_loss: 253.0921 - val_loss: 253.0152 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 584ms/step - ll_loss: 211.0438 - loss: 213.4192 - val_ll_loss: 252.8533 - val_loss: 252.7813 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 574ms/step - ll_loss: 210.8503 - loss: 213.2221 - val_ll_loss: 252.6377 - val_loss: 252.5701 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 529ms/step - ll_loss: 210.6754 - loss: 213.0440 - val_ll_loss: 252.4428 - val_loss: 252.3792 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 584ms/step - ll_loss: 210.5174 - loss: 212.8831 - val_ll_loss: 252.2666 - val_loss: 252.2066 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 593ms/step - ll_loss: 210.3745 - loss: 212.7376 - val_ll_loss: 252.1073 - val_loss: 252.0504 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 573ms/step - ll_loss: 210.2453 - loss: 212.6060 - val_ll_loss: 251.9632 - val_loss: 251.9093 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 567ms/step - ll_loss: 210.1285 - loss: 212.4869 - val_ll_loss: 251.8328 - val_loss: 251.7815 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 582ms/step - ll_loss: 210.0227 - loss: 212.3792 - val_ll_loss: 251.7147 - val_loss: 251.6659 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 588ms/step - ll_loss: 209.9270 - loss: 212.2817 - val_ll_loss: 251.6079 - val_loss: 251.5612 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 573ms/step - ll_loss: 209.8404 - loss: 212.1935 - val_ll_loss: 251.5112 - val_loss: 251.4664 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 531ms/step - ll_loss: 209.7620 - loss: 212.1136 - val_ll_loss: 251.4235 - val_loss: 251.3806 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 588ms/step - ll_loss: 209.6910 - loss: 212.0413 - val_ll_loss: 251.3442 - val_loss: 251.3029 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 578ms/step - ll_loss: 209.6267 - loss: 211.9758 - val_ll_loss: 251.2724 - val_loss: 251.2325 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 572ms/step - ll_loss: 209.5685 - loss: 211.9165 - val_ll_loss: 251.2073 - val_loss: 251.1687 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 610ms/step - ll_loss: 209.5158 - loss: 211.8628 - val_ll_loss: 251.1484 - val_loss: 251.1110 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 603ms/step - ll_loss: 209.4680 - loss: 211.8141 - val_ll_loss: 251.0950 - val_loss: 251.0587 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 622ms/step - ll_loss: 209.4248 - loss: 211.7701 - val_ll_loss: 251.0467 - val_loss: 251.0114 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m45s[0m 655ms/step - ll_loss: 209.3856 - loss: 211.7302 - val_ll_loss: 251.0029 - val_loss: 250.9684 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 655ms/step - ll_loss: 209.3502 - loss: 211.6941 - val_ll_loss: 250.9632 - val_loss: 250.9295 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 650ms/step - ll_loss: 209.3180 - loss: 211.6613 - val_ll_loss: 250.9272 - val_loss: 250.8943 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 617ms/step - ll_loss: 209.2889 - loss: 211.6317 - val_ll_loss: 250.8947 - val_loss: 250.8624 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 628ms/step - ll_loss: 209.2626 - loss: 211.6048 - val_ll_loss: 250.8652 - val_loss: 250.8335 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 653ms/step - ll_loss: 209.2387 - loss: 211.5805 - val_ll_loss: 250.8384 - val_loss: 250.8073 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 631ms/step - ll_loss: 209.2171 - loss: 211.5584 - val_ll_loss: 250.8142 - val_loss: 250.7836 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 630ms/step - ll_loss: 209.1975 - loss: 211.5385 - val_ll_loss: 250.7923 - val_loss: 250.7621 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 644ms/step - ll_loss: 209.1797 - loss: 211.5204 - val_ll_loss: 250.7724 - val_loss: 250.7427 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 639ms/step - ll_loss: 209.1637 - loss: 211.5040 - val_ll_loss: 250.7544 - val_loss: 250.7251 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 623ms/step - ll_loss: 209.1491 - loss: 211.4892 - val_ll_loss: 250.7381 - val_loss: 250.7091 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 630ms/step - ll_loss: 209.1359 - loss: 211.4758 - val_ll_loss: 250.7234 - val_loss: 250.6946 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 654ms/step - ll_loss: 209.1240 - loss: 211.4636 - val_ll_loss: 250.7100 - val_loss: 250.6815 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 633ms/step - ll_loss: 209.1131 - loss: 211.4526 - val_ll_loss: 250.6979 - val_loss: 250.6696 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 632ms/step - ll_loss: 209.1034 - loss: 211.4426 - val_ll_loss: 250.6869 - val_loss: 250.6589 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 654ms/step - ll_loss: 209.0945 - loss: 211.4336 - val_ll_loss: 250.6770 - val_loss: 250.6492 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 653ms/step - ll_loss: 209.0865 - loss: 211.4254 - val_ll_loss: 250.6680 - val_loss: 250.6404 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 631ms/step - ll_loss: 209.0792 - loss: 211.4180 - val_ll_loss: 250.6598 - val_loss: 250.6324 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 624ms/step - ll_loss: 209.0726 - loss: 211.4113 - val_ll_loss: 250.6525 - val_loss: 250.6252 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m53/53[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 777ms/step - ll_loss: 209.0667 - loss: 211.4052 - val_ll_loss: 250.6458 - val_loss: 250.6186 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/8 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K08/fold_holdout-sub-03
    \n--- holdout=sub-08 | val=sub-17 | K=8 ---
    train segs=51 short=0 | val segs=10 short=0 | test segs=10 short=0



    Loading files:   0%|          | 0/51 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]


    Epoch 1/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 1s/step - ll_loss: 256.6559 - loss: 230.6839 - val_ll_loss: 279.3431 - val_loss: 269.1479 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m50s[0m 981ms/step - ll_loss: 254.0059 - loss: 228.4169 - val_ll_loss: 277.5080 - val_loss: 267.2623 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 1s/step - ll_loss: 251.5063 - loss: 226.2769 - val_ll_loss: 274.5406 - val_loss: 264.4481 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m50s[0m 990ms/step - ll_loss: 249.0189 - loss: 224.1211 - val_ll_loss: 271.6650 - val_loss: 261.7321 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 997ms/step - ll_loss: 246.7035 - loss: 222.1125 - val_ll_loss: 269.3157 - val_loss: 259.5184 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 997ms/step - ll_loss: 244.8038 - loss: 220.4682 - val_ll_loss: 267.3364 - val_loss: 257.6542 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m50s[0m 973ms/step - ll_loss: 243.1996 - loss: 219.0827 - val_ll_loss: 265.6396 - val_loss: 256.0566 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m49s[0m 971ms/step - ll_loss: 241.8204 - loss: 217.8935 - val_ll_loss: 264.1682 - val_loss: 254.6715 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m50s[0m 987ms/step - ll_loss: 240.6217 - loss: 216.8611 - val_ll_loss: 262.8819 - val_loss: 253.4608 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m47s[0m 922ms/step - ll_loss: 239.5716 - loss: 215.9576 - val_ll_loss: 261.7509 - val_loss: 252.3964 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m50s[0m 988ms/step - ll_loss: 238.6478 - loss: 215.1633 - val_ll_loss: 260.7498 - val_loss: 251.4544 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m49s[0m 914ms/step - ll_loss: 237.8303 - loss: 214.4612 - val_ll_loss: 259.8575 - val_loss: 250.6147 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m47s[0m 927ms/step - ll_loss: 237.1000 - loss: 213.8346 - val_ll_loss: 259.0646 - val_loss: 249.8686 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m50s[0m 976ms/step - ll_loss: 236.4500 - loss: 213.2772 - val_ll_loss: 258.3586 - val_loss: 249.2043 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m47s[0m 915ms/step - ll_loss: 235.8704 - loss: 212.7802 - val_ll_loss: 257.7283 - val_loss: 248.6113 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 1s/step - ll_loss: 235.3524 - loss: 212.3362 - val_ll_loss: 257.1646 - val_loss: 248.0810 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m47s[0m 925ms/step - ll_loss: 234.8886 - loss: 211.9388 - val_ll_loss: 256.6597 - val_loss: 247.6060 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 1s/step - ll_loss: 234.4728 - loss: 211.5826 - val_ll_loss: 256.2068 - val_loss: 247.1799 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 964ms/step - ll_loss: 234.0996 - loss: 211.2630 - val_ll_loss: 255.8000 - val_loss: 246.7973 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m49s[0m 971ms/step - ll_loss: 233.7642 - loss: 210.9758 - val_ll_loss: 255.4342 - val_loss: 246.4532 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 1s/step - ll_loss: 233.4625 - loss: 210.7174 - val_ll_loss: 255.1050 - val_loss: 246.1435 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m47s[0m 915ms/step - ll_loss: 233.1908 - loss: 210.4848 - val_ll_loss: 254.8085 - val_loss: 245.8646 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 1s/step - ll_loss: 232.9460 - loss: 210.2753 - val_ll_loss: 254.5411 - val_loss: 245.6131 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m48s[0m 951ms/step - ll_loss: 232.7253 - loss: 210.0864 - val_ll_loss: 254.3000 - val_loss: 245.3863 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 1s/step - ll_loss: 232.5262 - loss: 209.9160 - val_ll_loss: 254.0824 - val_loss: 245.1817 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m50s[0m 972ms/step - ll_loss: 232.3465 - loss: 209.7622 - val_ll_loss: 253.8860 - val_loss: 244.9969 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m47s[0m 924ms/step - ll_loss: 232.1843 - loss: 209.6234 - val_ll_loss: 253.7085 - val_loss: 244.8301 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 986ms/step - ll_loss: 232.0378 - loss: 209.4980 - val_ll_loss: 253.5482 - val_loss: 244.6792 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m46s[0m 912ms/step - ll_loss: 231.9053 - loss: 209.3846 - val_ll_loss: 253.4032 - val_loss: 244.5429 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 1s/step - ll_loss: 231.7857 - loss: 209.2822 - val_ll_loss: 253.2721 - val_loss: 244.4197 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m47s[0m 932ms/step - ll_loss: 231.6775 - loss: 209.1897 - val_ll_loss: 253.1536 - val_loss: 244.3082 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m49s[0m 965ms/step - ll_loss: 231.5796 - loss: 209.1059 - val_ll_loss: 253.0464 - val_loss: 244.2074 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 994ms/step - ll_loss: 231.4911 - loss: 209.0302 - val_ll_loss: 252.9494 - val_loss: 244.1161 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m47s[0m 922ms/step - ll_loss: 231.4111 - loss: 208.9617 - val_ll_loss: 252.8616 - val_loss: 244.0336 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 996ms/step - ll_loss: 231.3387 - loss: 208.8997 - val_ll_loss: 252.7822 - val_loss: 243.9589 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m47s[0m 931ms/step - ll_loss: 231.2731 - loss: 208.8436 - val_ll_loss: 252.7103 - val_loss: 243.8913 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m48s[0m 938ms/step - ll_loss: 231.2138 - loss: 208.7928 - val_ll_loss: 252.6452 - val_loss: 243.8301 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 997ms/step - ll_loss: 231.1601 - loss: 208.7469 - val_ll_loss: 252.5863 - val_loss: 243.7747 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m47s[0m 917ms/step - ll_loss: 231.1115 - loss: 208.7053 - val_ll_loss: 252.5330 - val_loss: 243.7245 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 1s/step - ll_loss: 231.0675 - loss: 208.6677 - val_ll_loss: 252.4847 - val_loss: 243.6791 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m47s[0m 924ms/step - ll_loss: 231.0277 - loss: 208.6336 - val_ll_loss: 252.4409 - val_loss: 243.6380 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m50s[0m 979ms/step - ll_loss: 230.9917 - loss: 208.6027 - val_ll_loss: 252.4013 - val_loss: 243.6008 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m49s[0m 960ms/step - ll_loss: 230.9590 - loss: 208.5748 - val_ll_loss: 252.3655 - val_loss: 243.5671 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m46s[0m 902ms/step - ll_loss: 230.9295 - loss: 208.5495 - val_ll_loss: 252.3330 - val_loss: 243.5365 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 1s/step - ll_loss: 230.9027 - loss: 208.5266 - val_ll_loss: 252.3036 - val_loss: 243.5089 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m47s[0m 924ms/step - ll_loss: 230.8785 - loss: 208.5059 - val_ll_loss: 252.2770 - val_loss: 243.4839 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 1s/step - ll_loss: 230.8566 - loss: 208.4871 - val_ll_loss: 252.2529 - val_loss: 243.4612 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m49s[0m 969ms/step - ll_loss: 230.8367 - loss: 208.4701 - val_ll_loss: 252.2310 - val_loss: 243.4406 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m48s[0m 954ms/step - ll_loss: 230.8188 - loss: 208.4548 - val_ll_loss: 252.2113 - val_loss: 243.4221 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m49s[0m 973ms/step - ll_loss: 230.8025 - loss: 208.4408 - val_ll_loss: 252.1934 - val_loss: 243.4053 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m49s[0m 956ms/step - ll_loss: 230.7878 - loss: 208.4282 - val_ll_loss: 252.1772 - val_loss: 243.3900 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m48s[0m 953ms/step - ll_loss: 230.7744 - loss: 208.4168 - val_ll_loss: 252.1625 - val_loss: 243.3762 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m48s[0m 940ms/step - ll_loss: 230.7624 - loss: 208.4064 - val_ll_loss: 252.1492 - val_loss: 243.3637 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m90s[0m 968ms/step - ll_loss: 230.7514 - loss: 208.3971 - val_ll_loss: 252.1372 - val_loss: 243.3524 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m48s[0m 936ms/step - ll_loss: 230.7415 - loss: 208.3886 - val_ll_loss: 252.1263 - val_loss: 243.3422 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 1s/step - ll_loss: 230.7326 - loss: 208.3810 - val_ll_loss: 252.1164 - val_loss: 243.3329 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m48s[0m 940ms/step - ll_loss: 230.7245 - loss: 208.3740 - val_ll_loss: 252.1075 - val_loss: 243.3245 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m48s[0m 948ms/step - ll_loss: 230.7171 - loss: 208.3677 - val_ll_loss: 252.0994 - val_loss: 243.3169 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m50s[0m 988ms/step - ll_loss: 230.7105 - loss: 208.3620 - val_ll_loss: 252.0921 - val_loss: 243.3100 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m48s[0m 948ms/step - ll_loss: 230.7045 - loss: 208.3569 - val_ll_loss: 252.0854 - val_loss: 243.3038 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/10 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K08/fold_holdout-sub-08
    \n--- holdout=sub-09 | val=sub-08 | K=8 ---
    train segs=59 short=0 | val segs=10 short=0 | test segs=2 short=0



    Loading files:   0%|          | 0/59 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/2 [00:00<?, ?it/s]


    Epoch 1/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m59s[0m 995ms/step - ll_loss: 265.5175 - loss: 228.3524 - val_ll_loss: 248.0870 - val_loss: 244.1215 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 903ms/step - ll_loss: 262.5905 - loss: 225.9445 - val_ll_loss: 245.3400 - val_loss: 241.5049 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 977ms/step - ll_loss: 259.7305 - loss: 223.5535 - val_ll_loss: 242.9702 - val_loss: 239.2443 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 909ms/step - ll_loss: 257.2017 - loss: 221.4375 - val_ll_loss: 240.9373 - val_loss: 237.3036 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 963ms/step - ll_loss: 255.0192 - loss: 219.6223 - val_ll_loss: 239.1833 - val_loss: 235.6283 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m58s[0m 993ms/step - ll_loss: 253.1328 - loss: 218.0571 - val_ll_loss: 237.6581 - val_loss: 234.1709 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 950ms/step - ll_loss: 251.4882 - loss: 216.6949 - val_ll_loss: 236.3236 - val_loss: 232.8956 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 950ms/step - ll_loss: 250.0456 - loss: 215.5020 - val_ll_loss: 235.1503 - val_loss: 231.7740 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 947ms/step - ll_loss: 248.7742 - loss: 214.4520 - val_ll_loss: 234.1147 - val_loss: 230.7839 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 931ms/step - ll_loss: 247.6496 - loss: 213.5244 - val_ll_loss: 233.1975 - val_loss: 229.9068 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 951ms/step - ll_loss: 246.6517 - loss: 212.7022 - val_ll_loss: 232.3830 - val_loss: 229.1279 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 910ms/step - ll_loss: 245.7640 - loss: 211.9714 - val_ll_loss: 231.6579 - val_loss: 228.4344 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 969ms/step - ll_loss: 244.9726 - loss: 211.3203 - val_ll_loss: 231.0108 - val_loss: 227.8155 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 904ms/step - ll_loss: 244.2654 - loss: 210.7390 - val_ll_loss: 230.4324 - val_loss: 227.2622 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m58s[0m 986ms/step - ll_loss: 243.6325 - loss: 210.2190 - val_ll_loss: 229.9145 - val_loss: 226.7668 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 921ms/step - ll_loss: 243.0652 - loss: 209.7532 - val_ll_loss: 229.4500 - val_loss: 226.3224 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 937ms/step - ll_loss: 242.5561 - loss: 209.3353 - val_ll_loss: 229.0330 - val_loss: 225.9234 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m58s[0m 987ms/step - ll_loss: 242.0987 - loss: 208.9599 - val_ll_loss: 228.6581 - val_loss: 225.5648 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m49s[0m 829ms/step - ll_loss: 241.6872 - loss: 208.6224 - val_ll_loss: 228.3208 - val_loss: 225.2421 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 888ms/step - ll_loss: 241.3167 - loss: 208.3186 - val_ll_loss: 228.0169 - val_loss: 224.9513 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m43s[0m 732ms/step - ll_loss: 240.9829 - loss: 208.0450 - val_ll_loss: 227.7430 - val_loss: 224.6893 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m43s[0m 728ms/step - ll_loss: 240.6819 - loss: 207.7983 - val_ll_loss: 227.4959 - val_loss: 224.4529 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m47s[0m 798ms/step - ll_loss: 240.4104 - loss: 207.5758 - val_ll_loss: 227.2730 - val_loss: 224.2396 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m42s[0m 720ms/step - ll_loss: 240.1652 - loss: 207.3749 - val_ll_loss: 227.0717 - val_loss: 224.0470 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m43s[0m 736ms/step - ll_loss: 239.9438 - loss: 207.1935 - val_ll_loss: 226.8898 - val_loss: 223.8729 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m47s[0m 801ms/step - ll_loss: 239.7437 - loss: 207.0296 - val_ll_loss: 226.7254 - val_loss: 223.7156 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m43s[0m 732ms/step - ll_loss: 239.5630 - loss: 206.8815 - val_ll_loss: 226.5767 - val_loss: 223.5734 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m44s[0m 744ms/step - ll_loss: 239.3994 - loss: 206.7476 - val_ll_loss: 226.4423 - val_loss: 223.4448 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m90s[0m 807ms/step - ll_loss: 239.2516 - loss: 206.6265 - val_ll_loss: 226.3207 - val_loss: 223.3284 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 688ms/step - ll_loss: 239.1178 - loss: 206.5169 - val_ll_loss: 226.2106 - val_loss: 223.2231 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 583ms/step - ll_loss: 238.9968 - loss: 206.4178 - val_ll_loss: 226.1110 - val_loss: 223.1278 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 577ms/step - ll_loss: 238.8872 - loss: 206.3281 - val_ll_loss: 226.0209 - val_loss: 223.0415 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 563ms/step - ll_loss: 238.7881 - loss: 206.2469 - val_ll_loss: 225.9393 - val_loss: 222.9635 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 605ms/step - ll_loss: 238.6984 - loss: 206.1734 - val_ll_loss: 225.8654 - val_loss: 222.8927 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 560ms/step - ll_loss: 238.6171 - loss: 206.1068 - val_ll_loss: 225.7984 - val_loss: 222.8287 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 566ms/step - ll_loss: 238.5436 - loss: 206.0466 - val_ll_loss: 225.7378 - val_loss: 222.7707 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 571ms/step - ll_loss: 238.4770 - loss: 205.9920 - val_ll_loss: 225.6829 - val_loss: 222.7182 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 592ms/step - ll_loss: 238.4166 - loss: 205.9426 - val_ll_loss: 225.6332 - val_loss: 222.6706 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 606ms/step - ll_loss: 238.3620 - loss: 205.8978 - val_ll_loss: 225.5882 - val_loss: 222.6276 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 689ms/step - ll_loss: 238.3126 - loss: 205.8573 - val_ll_loss: 225.5474 - val_loss: 222.5885 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 598ms/step - ll_loss: 238.2677 - loss: 205.8206 - val_ll_loss: 225.5104 - val_loss: 222.5532 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 571ms/step - ll_loss: 238.2271 - loss: 205.7874 - val_ll_loss: 225.4769 - val_loss: 222.5212 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 579ms/step - ll_loss: 238.1904 - loss: 205.7572 - val_ll_loss: 225.4466 - val_loss: 222.4922 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 593ms/step - ll_loss: 238.1571 - loss: 205.7300 - val_ll_loss: 225.4192 - val_loss: 222.4659 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 594ms/step - ll_loss: 238.1270 - loss: 205.7053 - val_ll_loss: 225.3943 - val_loss: 222.4421 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 597ms/step - ll_loss: 238.0997 - loss: 205.6829 - val_ll_loss: 225.3717 - val_loss: 222.4205 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 576ms/step - ll_loss: 238.0750 - loss: 205.6626 - val_ll_loss: 225.3513 - val_loss: 222.4010 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m43s[0m 573ms/step - ll_loss: 238.0526 - loss: 205.6443 - val_ll_loss: 225.3328 - val_loss: 222.3833 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 574ms/step - ll_loss: 238.0323 - loss: 205.6276 - val_ll_loss: 225.3161 - val_loss: 222.3673 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 562ms/step - ll_loss: 238.0139 - loss: 205.6126 - val_ll_loss: 225.3009 - val_loss: 222.3528 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 609ms/step - ll_loss: 237.9973 - loss: 205.5989 - val_ll_loss: 225.2872 - val_loss: 222.3397 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 599ms/step - ll_loss: 237.9822 - loss: 205.5866 - val_ll_loss: 225.2747 - val_loss: 222.3278 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 616ms/step - ll_loss: 237.9686 - loss: 205.5754 - val_ll_loss: 225.2635 - val_loss: 222.3170 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m43s[0m 724ms/step - ll_loss: 237.9562 - loss: 205.5653 - val_ll_loss: 225.2533 - val_loss: 222.3072 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 652ms/step - ll_loss: 237.9450 - loss: 205.5561 - val_ll_loss: 225.2440 - val_loss: 222.2984 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 595ms/step - ll_loss: 237.9349 - loss: 205.5478 - val_ll_loss: 225.2356 - val_loss: 222.2904 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 590ms/step - ll_loss: 237.9257 - loss: 205.5403 - val_ll_loss: 225.2281 - val_loss: 222.2832 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 586ms/step - ll_loss: 237.9174 - loss: 205.5335 - val_ll_loss: 225.2212 - val_loss: 222.2765 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 580ms/step - ll_loss: 237.9099 - loss: 205.5273 - val_ll_loss: 225.2150 - val_loss: 222.2706 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 591ms/step - ll_loss: 237.9031 - loss: 205.5217 - val_ll_loss: 225.2094 - val_loss: 222.2652 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/2 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K08/fold_holdout-sub-09
    \n--- holdout=sub-13 | val=sub-08 | K=8 ---
    train segs=52 short=0 | val segs=10 short=0 | test segs=9 short=0



    Loading files:   0%|          | 0/52 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/9 [00:00<?, ?it/s]


    Epoch 1/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 641ms/step - ll_loss: 254.9032 - loss: 231.5714 - val_ll_loss: 231.7910 - val_loss: 228.1263 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 588ms/step - ll_loss: 252.3255 - loss: 229.2637 - val_ll_loss: 229.5219 - val_loss: 225.9084 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 601ms/step - ll_loss: 249.8194 - loss: 227.1143 - val_ll_loss: 227.3838 - val_loss: 223.8691 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 674ms/step - ll_loss: 247.4339 - loss: 225.0113 - val_ll_loss: 225.5268 - val_loss: 222.1002 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 668ms/step - ll_loss: 245.3378 - loss: 223.1444 - val_ll_loss: 223.9765 - val_loss: 220.6220 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 646ms/step - ll_loss: 243.5503 - loss: 221.5580 - val_ll_loss: 222.6531 - val_loss: 219.3589 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 651ms/step - ll_loss: 242.0089 - loss: 220.1922 - val_ll_loss: 221.5076 - val_loss: 218.2650 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 688ms/step - ll_loss: 240.6655 - loss: 219.0080 - val_ll_loss: 220.5112 - val_loss: 217.3133 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 658ms/step - ll_loss: 239.4957 - loss: 217.9782 - val_ll_loss: 219.6387 - val_loss: 216.4796 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 682ms/step - ll_loss: 238.4695 - loss: 217.0758 - val_ll_loss: 218.8706 - val_loss: 215.7457 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 670ms/step - ll_loss: 237.5646 - loss: 216.2807 - val_ll_loss: 218.1915 - val_loss: 215.0966 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 672ms/step - ll_loss: 236.7631 - loss: 215.5770 - val_ll_loss: 217.5890 - val_loss: 214.5207 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 641ms/step - ll_loss: 236.0510 - loss: 214.9523 - val_ll_loss: 217.0531 - val_loss: 214.0083 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 647ms/step - ll_loss: 235.4167 - loss: 214.3960 - val_ll_loss: 216.5751 - val_loss: 213.5513 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 691ms/step - ll_loss: 234.8504 - loss: 213.8996 - val_ll_loss: 216.1480 - val_loss: 213.1430 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 629ms/step - ll_loss: 234.3438 - loss: 213.4557 - val_ll_loss: 215.7657 - val_loss: 212.7774 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 624ms/step - ll_loss: 233.8900 - loss: 213.0582 - val_ll_loss: 215.4230 - val_loss: 212.4498 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 623ms/step - ll_loss: 233.4828 - loss: 212.7017 - val_ll_loss: 215.1154 - val_loss: 212.1556 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 648ms/step - ll_loss: 233.1171 - loss: 212.3815 - val_ll_loss: 214.8390 - val_loss: 211.8912 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 551ms/step - ll_loss: 232.7882 - loss: 212.0937 - val_ll_loss: 214.5904 - val_loss: 211.6534 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 638ms/step - ll_loss: 232.4923 - loss: 211.8347 - val_ll_loss: 214.3665 - val_loss: 211.4393 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 642ms/step - ll_loss: 232.2257 - loss: 211.6016 - val_ll_loss: 214.1648 - val_loss: 211.2464 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 647ms/step - ll_loss: 231.9855 - loss: 211.3914 - val_ll_loss: 213.9829 - val_loss: 211.0725 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 652ms/step - ll_loss: 231.7688 - loss: 211.2019 - val_ll_loss: 213.8188 - val_loss: 210.9155 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 656ms/step - ll_loss: 231.5733 - loss: 211.0309 - val_ll_loss: 213.6707 - val_loss: 210.7739 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 648ms/step - ll_loss: 231.3968 - loss: 210.8765 - val_ll_loss: 213.5370 - val_loss: 210.6460 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 618ms/step - ll_loss: 231.2374 - loss: 210.7371 - val_ll_loss: 213.4162 - val_loss: 210.5305 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 656ms/step - ll_loss: 231.0933 - loss: 210.6111 - val_ll_loss: 213.3071 - val_loss: 210.4260 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 667ms/step - ll_loss: 230.9632 - loss: 210.4973 - val_ll_loss: 213.2083 - val_loss: 210.3317 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 654ms/step - ll_loss: 230.8455 - loss: 210.3944 - val_ll_loss: 213.1191 - val_loss: 210.2463 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 628ms/step - ll_loss: 230.7391 - loss: 210.3014 - val_ll_loss: 213.0384 - val_loss: 210.1691 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 609ms/step - ll_loss: 230.6428 - loss: 210.2173 - val_ll_loss: 212.9654 - val_loss: 210.0993 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 651ms/step - ll_loss: 230.5558 - loss: 210.1411 - val_ll_loss: 212.8994 - val_loss: 210.0361 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 625ms/step - ll_loss: 230.4770 - loss: 210.0723 - val_ll_loss: 212.8396 - val_loss: 209.9789 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 595ms/step - ll_loss: 230.4058 - loss: 210.0099 - val_ll_loss: 212.7855 - val_loss: 209.9272 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 588ms/step - ll_loss: 230.3413 - loss: 209.9536 - val_ll_loss: 212.7365 - val_loss: 209.8804 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 620ms/step - ll_loss: 230.2829 - loss: 209.9025 - val_ll_loss: 212.6922 - val_loss: 209.8380 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 573ms/step - ll_loss: 230.2301 - loss: 209.8563 - val_ll_loss: 212.6521 - val_loss: 209.7996 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 538ms/step - ll_loss: 230.1822 - loss: 209.8145 - val_ll_loss: 212.6158 - val_loss: 209.7649 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 590ms/step - ll_loss: 230.1389 - loss: 209.7766 - val_ll_loss: 212.5828 - val_loss: 209.7334 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 653ms/step - ll_loss: 230.0998 - loss: 209.7424 - val_ll_loss: 212.5531 - val_loss: 209.7049 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 625ms/step - ll_loss: 230.0643 - loss: 209.7113 - val_ll_loss: 212.5261 - val_loss: 209.6791 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 621ms/step - ll_loss: 230.0322 - loss: 209.6832 - val_ll_loss: 212.5017 - val_loss: 209.6558 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 632ms/step - ll_loss: 230.0031 - loss: 209.6578 - val_ll_loss: 212.4796 - val_loss: 209.6346 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 643ms/step - ll_loss: 229.9767 - loss: 209.6348 - val_ll_loss: 212.4595 - val_loss: 209.6155 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 612ms/step - ll_loss: 229.9529 - loss: 209.6139 - val_ll_loss: 212.4414 - val_loss: 209.5982 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 614ms/step - ll_loss: 229.9313 - loss: 209.5950 - val_ll_loss: 212.4250 - val_loss: 209.5825 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 670ms/step - ll_loss: 229.9117 - loss: 209.5779 - val_ll_loss: 212.4101 - val_loss: 209.5682 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 625ms/step - ll_loss: 229.8941 - loss: 209.5625 - val_ll_loss: 212.3967 - val_loss: 209.5553 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 632ms/step - ll_loss: 229.8780 - loss: 209.5484 - val_ll_loss: 212.3845 - val_loss: 209.5437 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 654ms/step - ll_loss: 229.8635 - loss: 209.5357 - val_ll_loss: 212.3735 - val_loss: 209.5332 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 673ms/step - ll_loss: 229.8504 - loss: 209.5243 - val_ll_loss: 212.3635 - val_loss: 209.5236 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 612ms/step - ll_loss: 229.8385 - loss: 209.5139 - val_ll_loss: 212.3544 - val_loss: 209.5149 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 635ms/step - ll_loss: 229.8277 - loss: 209.5044 - val_ll_loss: 212.3462 - val_loss: 209.5071 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 627ms/step - ll_loss: 229.8180 - loss: 209.4959 - val_ll_loss: 212.3388 - val_loss: 209.5000 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 614ms/step - ll_loss: 229.8091 - loss: 209.4882 - val_ll_loss: 212.3321 - val_loss: 209.4936 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 568ms/step - ll_loss: 229.8011 - loss: 209.4812 - val_ll_loss: 212.3260 - val_loss: 209.4878 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 622ms/step - ll_loss: 229.7939 - loss: 209.4749 - val_ll_loss: 212.3205 - val_loss: 209.4825 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 672ms/step - ll_loss: 229.7873 - loss: 209.4691 - val_ll_loss: 212.3155 - val_loss: 209.4778 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m52/52[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 560ms/step - ll_loss: 229.7814 - loss: 209.4639 - val_ll_loss: 212.3110 - val_loss: 209.4734 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/9 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K08/fold_holdout-sub-13
    \n--- holdout=sub-14 | val=sub-08 | K=8 ---
    train segs=57 short=0 | val segs=10 short=0 | test segs=4 short=0



    Loading files:   0%|          | 0/57 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/4 [00:00<?, ?it/s]


    Epoch 1/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 581ms/step - ll_loss: 254.5410 - loss: 229.2840 - val_ll_loss: 237.9058 - val_loss: 234.5270 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 578ms/step - ll_loss: 251.7708 - loss: 226.8432 - val_ll_loss: 235.7090 - val_loss: 232.4293 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 590ms/step - ll_loss: 249.3033 - loss: 224.6718 - val_ll_loss: 233.6913 - val_loss: 230.5120 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 618ms/step - ll_loss: 247.0003 - loss: 222.6560 - val_ll_loss: 231.8259 - val_loss: 228.7320 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 533ms/step - ll_loss: 244.9506 - loss: 220.8744 - val_ll_loss: 230.2166 - val_loss: 227.1944 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 616ms/step - ll_loss: 243.1746 - loss: 219.3338 - val_ll_loss: 228.8184 - val_loss: 225.8576 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 510ms/step - ll_loss: 241.6244 - loss: 217.9911 - val_ll_loss: 227.5955 - val_loss: 224.6880 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 572ms/step - ll_loss: 240.2636 - loss: 216.8138 - val_ll_loss: 226.5207 - val_loss: 223.6597 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 571ms/step - ll_loss: 239.0638 - loss: 215.7767 - val_ll_loss: 225.5721 - val_loss: 222.7519 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 603ms/step - ll_loss: 238.0020 - loss: 214.8597 - val_ll_loss: 224.7321 - val_loss: 221.9479 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 565ms/step - ll_loss: 237.0595 - loss: 214.0462 - val_ll_loss: 223.9860 - val_loss: 221.2337 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 580ms/step - ll_loss: 236.2207 - loss: 213.3227 - val_ll_loss: 223.3218 - val_loss: 220.5978 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 604ms/step - ll_loss: 235.4726 - loss: 212.6778 - val_ll_loss: 222.7292 - val_loss: 220.0303 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 569ms/step - ll_loss: 234.8042 - loss: 212.1017 - val_ll_loss: 222.1993 - val_loss: 219.5230 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 578ms/step - ll_loss: 234.2058 - loss: 211.5863 - val_ll_loss: 221.7249 - val_loss: 219.0686 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 580ms/step - ll_loss: 233.6694 - loss: 211.1244 - val_ll_loss: 221.2995 - val_loss: 218.6612 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 621ms/step - ll_loss: 233.1879 - loss: 210.7099 - val_ll_loss: 220.9175 - val_loss: 218.2953 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m46s[0m 803ms/step - ll_loss: 232.7552 - loss: 210.3375 - val_ll_loss: 220.5741 - val_loss: 217.9664 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 960ms/step - ll_loss: 232.3660 - loss: 210.0025 - val_ll_loss: 220.2651 - val_loss: 217.6704 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m50s[0m 879ms/step - ll_loss: 232.0155 - loss: 209.7010 - val_ll_loss: 219.9869 - val_loss: 217.4038 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 1s/step - ll_loss: 231.6997 - loss: 209.4294 - val_ll_loss: 219.7360 - val_loss: 217.1636 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 948ms/step - ll_loss: 231.4150 - loss: 209.1845 - val_ll_loss: 219.5098 - val_loss: 216.9468 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 943ms/step - ll_loss: 231.1581 - loss: 208.9635 - val_ll_loss: 219.3056 - val_loss: 216.7513 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m87s[0m 952ms/step - ll_loss: 230.9262 - loss: 208.7641 - val_ll_loss: 219.1213 - val_loss: 216.5746 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 927ms/step - ll_loss: 230.7167 - loss: 208.5840 - val_ll_loss: 218.9547 - val_loss: 216.4151 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 959ms/step - ll_loss: 230.5275 - loss: 208.4213 - val_ll_loss: 218.8042 - val_loss: 216.2709 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 997ms/step - ll_loss: 230.3564 - loss: 208.2742 - val_ll_loss: 218.6681 - val_loss: 216.1405 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 970ms/step - ll_loss: 230.2017 - loss: 208.1412 - val_ll_loss: 218.5451 - val_loss: 216.0226 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m58s[0m 1s/step - ll_loss: 230.0619 - loss: 208.0210 - val_ll_loss: 218.4337 - val_loss: 215.9159 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 933ms/step - ll_loss: 229.9353 - loss: 207.9122 - val_ll_loss: 218.3330 - val_loss: 215.8194 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m60s[0m 1s/step - ll_loss: 229.8208 - loss: 207.8138 - val_ll_loss: 218.2418 - val_loss: 215.7321 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 969ms/step - ll_loss: 229.7172 - loss: 207.7247 - val_ll_loss: 218.1593 - val_loss: 215.6531 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 939ms/step - ll_loss: 229.6235 - loss: 207.6441 - val_ll_loss: 218.0846 - val_loss: 215.5815 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 957ms/step - ll_loss: 229.5386 - loss: 207.5711 - val_ll_loss: 218.0170 - val_loss: 215.5167 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 960ms/step - ll_loss: 229.4617 - loss: 207.5050 - val_ll_loss: 217.9558 - val_loss: 215.4580 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 965ms/step - ll_loss: 229.3921 - loss: 207.4452 - val_ll_loss: 217.9003 - val_loss: 215.4049 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 916ms/step - ll_loss: 229.3291 - loss: 207.3910 - val_ll_loss: 217.8501 - val_loss: 215.3568 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 1s/step - ll_loss: 229.2721 - loss: 207.3419 - val_ll_loss: 217.8046 - val_loss: 215.3132 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 958ms/step - ll_loss: 229.2204 - loss: 207.2976 - val_ll_loss: 217.7634 - val_loss: 215.2737 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 993ms/step - ll_loss: 229.1736 - loss: 207.2573 - val_ll_loss: 217.7261 - val_loss: 215.2380 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 988ms/step - ll_loss: 229.1313 - loss: 207.2209 - val_ll_loss: 217.6923 - val_loss: 215.2056 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m59s[0m 1s/step - ll_loss: 229.0928 - loss: 207.1879 - val_ll_loss: 217.6616 - val_loss: 215.1763 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 902ms/step - ll_loss: 229.0581 - loss: 207.1580 - val_ll_loss: 217.6339 - val_loss: 215.1497 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m58s[0m 1s/step - ll_loss: 229.0266 - loss: 207.1309 - val_ll_loss: 217.6088 - val_loss: 215.1256 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 928ms/step - ll_loss: 228.9981 - loss: 207.1064 - val_ll_loss: 217.5861 - val_loss: 215.1039 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m91s[0m 1s/step - ll_loss: 228.9723 - loss: 207.0842 - val_ll_loss: 217.5654 - val_loss: 215.0841 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 966ms/step - ll_loss: 228.9489 - loss: 207.0641 - val_ll_loss: 217.5468 - val_loss: 215.0662 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 1000ms/step - ll_loss: 228.9277 - loss: 207.0459 - val_ll_loss: 217.5299 - val_loss: 215.0500 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 977ms/step - ll_loss: 228.9086 - loss: 207.0294 - val_ll_loss: 217.5145 - val_loss: 215.0353 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 956ms/step - ll_loss: 228.8912 - loss: 207.0144 - val_ll_loss: 217.5007 - val_loss: 215.0221 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 961ms/step - ll_loss: 228.8755 - loss: 207.0009 - val_ll_loss: 217.4881 - val_loss: 215.0100 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 987ms/step - ll_loss: 228.8612 - loss: 206.9887 - val_ll_loss: 217.4768 - val_loss: 214.9991 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 923ms/step - ll_loss: 228.8483 - loss: 206.9775 - val_ll_loss: 217.4664 - val_loss: 214.9893 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 980ms/step - ll_loss: 228.8367 - loss: 206.9675 - val_ll_loss: 217.4571 - val_loss: 214.9803 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 937ms/step - ll_loss: 228.8261 - loss: 206.9584 - val_ll_loss: 217.4486 - val_loss: 214.9722 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 965ms/step - ll_loss: 228.8165 - loss: 206.9502 - val_ll_loss: 217.4410 - val_loss: 214.9649 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 946ms/step - ll_loss: 228.8078 - loss: 206.9427 - val_ll_loss: 217.4341 - val_loss: 214.9583 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 895ms/step - ll_loss: 228.7999 - loss: 206.9359 - val_ll_loss: 217.4278 - val_loss: 214.9522 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 1s/step - ll_loss: 228.7928 - loss: 206.9298 - val_ll_loss: 217.4221 - val_loss: 214.9468 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m57/57[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 891ms/step - ll_loss: 228.7864 - loss: 206.9243 - val_ll_loss: 217.4169 - val_loss: 214.9418 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/4 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K08/fold_holdout-sub-14
    \n--- holdout=sub-16 | val=sub-08 | K=8 ---
    train segs=56 short=0 | val segs=10 short=0 | test segs=5 short=0



    Loading files:   0%|          | 0/56 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/5 [00:00<?, ?it/s]


    Epoch 1/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 978ms/step - ll_loss: 256.4309 - loss: 230.7827 - val_ll_loss: 234.7033 - val_loss: 231.0833 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m50s[0m 889ms/step - ll_loss: 253.7547 - loss: 228.3209 - val_ll_loss: 232.3600 - val_loss: 228.8503 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m92s[0m 937ms/step - ll_loss: 251.2893 - loss: 226.0448 - val_ll_loss: 230.2313 - val_loss: 226.8119 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 977ms/step - ll_loss: 248.8647 - loss: 223.9793 - val_ll_loss: 228.4355 - val_loss: 225.0939 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 965ms/step - ll_loss: 246.8033 - loss: 222.2235 - val_ll_loss: 226.8962 - val_loss: 223.6227 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 944ms/step - ll_loss: 245.0347 - loss: 220.7123 - val_ll_loss: 225.5607 - val_loss: 222.3469 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 994ms/step - ll_loss: 243.4984 - loss: 219.3978 - val_ll_loss: 224.3933 - val_loss: 221.2321 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 981ms/step - ll_loss: 242.1547 - loss: 218.2475 - val_ll_loss: 223.3651 - val_loss: 220.2502 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m58s[0m 1s/step - ll_loss: 240.9715 - loss: 217.2344 - val_ll_loss: 222.4545 - val_loss: 219.3806 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 953ms/step - ll_loss: 239.9213 - loss: 216.3352 - val_ll_loss: 221.6490 - val_loss: 218.6114 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 1s/step - ll_loss: 238.9905 - loss: 215.5381 - val_ll_loss: 220.9343 - val_loss: 217.9289 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 966ms/step - ll_loss: 238.1632 - loss: 214.8296 - val_ll_loss: 220.2983 - val_loss: 217.3216 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 1s/step - ll_loss: 237.4259 - loss: 214.1983 - val_ll_loss: 219.7310 - val_loss: 216.7800 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 963ms/step - ll_loss: 236.7676 - loss: 213.6347 - val_ll_loss: 219.2241 - val_loss: 216.2959 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 932ms/step - ll_loss: 236.1786 - loss: 213.1304 - val_ll_loss: 218.7703 - val_loss: 215.8626 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 1s/step - ll_loss: 235.6510 - loss: 212.6786 - val_ll_loss: 218.3636 - val_loss: 215.4742 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 921ms/step - ll_loss: 235.1775 - loss: 212.2733 - val_ll_loss: 217.9983 - val_loss: 215.1255 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 965ms/step - ll_loss: 234.7523 - loss: 211.9092 - val_ll_loss: 217.6701 - val_loss: 214.8122 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m50s[0m 901ms/step - ll_loss: 234.3699 - loss: 211.5819 - val_ll_loss: 217.3748 - val_loss: 214.5302 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 984ms/step - ll_loss: 234.0256 - loss: 211.2872 - val_ll_loss: 217.1089 - val_loss: 214.2764 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m50s[0m 897ms/step - ll_loss: 233.7156 - loss: 211.0218 - val_ll_loss: 216.8693 - val_loss: 214.0476 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 983ms/step - ll_loss: 233.4361 - loss: 210.7825 - val_ll_loss: 216.6532 - val_loss: 213.8413 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 926ms/step - ll_loss: 233.1840 - loss: 210.5667 - val_ll_loss: 216.4583 - val_loss: 213.6551 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 983ms/step - ll_loss: 232.9565 - loss: 210.3719 - val_ll_loss: 216.2823 - val_loss: 213.4871 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 1s/step - ll_loss: 232.7510 - loss: 210.1960 - val_ll_loss: 216.1232 - val_loss: 213.3353 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m49s[0m 876ms/step - ll_loss: 232.5654 - loss: 210.0371 - val_ll_loss: 215.9796 - val_loss: 213.1981 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m58s[0m 1s/step - ll_loss: 232.3977 - loss: 209.8935 - val_ll_loss: 215.8497 - val_loss: 213.0741 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 934ms/step - ll_loss: 232.2461 - loss: 209.7637 - val_ll_loss: 215.7322 - val_loss: 212.9620 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 1s/step - ll_loss: 232.1090 - loss: 209.6463 - val_ll_loss: 215.6260 - val_loss: 212.8606 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 915ms/step - ll_loss: 231.9851 - loss: 209.5401 - val_ll_loss: 215.5299 - val_loss: 212.7689 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 1s/step - ll_loss: 231.8729 - loss: 209.4441 - val_ll_loss: 215.4429 - val_loss: 212.6858 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 1s/step - ll_loss: 231.7714 - loss: 209.3571 - val_ll_loss: 215.3642 - val_loss: 212.6107 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 967ms/step - ll_loss: 231.6795 - loss: 209.2785 - val_ll_loss: 215.2930 - val_loss: 212.5427 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 1s/step - ll_loss: 231.5964 - loss: 209.2073 - val_ll_loss: 215.2285 - val_loss: 212.4811 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 951ms/step - ll_loss: 231.5211 - loss: 209.1428 - val_ll_loss: 215.1701 - val_loss: 212.4254 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 1s/step - ll_loss: 231.4530 - loss: 209.0845 - val_ll_loss: 215.1172 - val_loss: 212.3749 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 996ms/step - ll_loss: 231.3913 - loss: 209.0316 - val_ll_loss: 215.0693 - val_loss: 212.3293 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m45s[0m 807ms/step - ll_loss: 231.3355 - loss: 208.9838 - val_ll_loss: 215.0259 - val_loss: 212.2878 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 665ms/step - ll_loss: 231.2849 - loss: 208.9404 - val_ll_loss: 214.9867 - val_loss: 212.2504 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 650ms/step - ll_loss: 231.2392 - loss: 208.9012 - val_ll_loss: 214.9511 - val_loss: 212.2164 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 639ms/step - ll_loss: 231.1977 - loss: 208.8657 - val_ll_loss: 214.9189 - val_loss: 212.1857 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 623ms/step - ll_loss: 231.1602 - loss: 208.8335 - val_ll_loss: 214.8897 - val_loss: 212.1578 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 630ms/step - ll_loss: 231.1261 - loss: 208.8044 - val_ll_loss: 214.8633 - val_loss: 212.1326 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 616ms/step - ll_loss: 231.0954 - loss: 208.7780 - val_ll_loss: 214.8394 - val_loss: 212.1098 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 610ms/step - ll_loss: 231.0674 - loss: 208.7541 - val_ll_loss: 214.8177 - val_loss: 212.0891 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 619ms/step - ll_loss: 231.0422 - loss: 208.7324 - val_ll_loss: 214.7980 - val_loss: 212.0704 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 614ms/step - ll_loss: 231.0193 - loss: 208.7128 - val_ll_loss: 214.7802 - val_loss: 212.0534 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 630ms/step - ll_loss: 230.9986 - loss: 208.6951 - val_ll_loss: 214.7641 - val_loss: 212.0380 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 721ms/step - ll_loss: 230.9799 - loss: 208.6790 - val_ll_loss: 214.7496 - val_loss: 212.0241 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 644ms/step - ll_loss: 230.9629 - loss: 208.6644 - val_ll_loss: 214.7364 - val_loss: 212.0115 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 589ms/step - ll_loss: 230.9475 - loss: 208.6513 - val_ll_loss: 214.7244 - val_loss: 212.0001 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 564ms/step - ll_loss: 230.9336 - loss: 208.6393 - val_ll_loss: 214.7135 - val_loss: 211.9897 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m39s[0m 689ms/step - ll_loss: 230.9210 - loss: 208.6285 - val_ll_loss: 214.7037 - val_loss: 211.9804 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 629ms/step - ll_loss: 230.9095 - loss: 208.6187 - val_ll_loss: 214.6949 - val_loss: 211.9719 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 610ms/step - ll_loss: 230.8992 - loss: 208.6098 - val_ll_loss: 214.6868 - val_loss: 211.9642 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 641ms/step - ll_loss: 230.8898 - loss: 208.6018 - val_ll_loss: 214.6795 - val_loss: 211.9573 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 579ms/step - ll_loss: 230.8814 - loss: 208.5945 - val_ll_loss: 214.6729 - val_loss: 211.9510 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m46s[0m 630ms/step - ll_loss: 230.8737 - loss: 208.5880 - val_ll_loss: 214.6669 - val_loss: 211.9453 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 606ms/step - ll_loss: 230.8667 - loss: 208.5820 - val_ll_loss: 214.6615 - val_loss: 211.9401 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m56/56[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 581ms/step - ll_loss: 230.8605 - loss: 208.5766 - val_ll_loss: 214.6566 - val_loss: 211.9354 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/5 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K08/fold_holdout-sub-16
    \n--- holdout=sub-17 | val=sub-08 | K=8 ---
    train segs=51 short=0 | val segs=10 short=0 | test segs=10 short=0



    Loading files:   0%|          | 0/51 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]


    Epoch 1/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 581ms/step - ll_loss: 257.4224 - loss: 231.4752 - val_ll_loss: 247.8105 - val_loss: 243.3390 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 600ms/step - ll_loss: 254.9879 - loss: 229.3433 - val_ll_loss: 245.7376 - val_loss: 241.3575 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 667ms/step - ll_loss: 252.5825 - loss: 227.2150 - val_ll_loss: 243.8618 - val_loss: 239.5797 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 655ms/step - ll_loss: 250.2511 - loss: 225.0955 - val_ll_loss: 241.6572 - val_loss: 237.4654 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 619ms/step - ll_loss: 247.8269 - loss: 222.9840 - val_ll_loss: 239.6554 - val_loss: 235.5205 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 601ms/step - ll_loss: 245.5856 - loss: 221.0612 - val_ll_loss: 238.0541 - val_loss: 233.9706 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 652ms/step - ll_loss: 243.7522 - loss: 219.4861 - val_ll_loss: 236.7250 - val_loss: 232.6868 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 604ms/step - ll_loss: 242.2250 - loss: 218.1744 - val_ll_loss: 235.5950 - val_loss: 231.5966 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 622ms/step - ll_loss: 240.9237 - loss: 217.0572 - val_ll_loss: 234.6203 - val_loss: 230.6571 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 660ms/step - ll_loss: 239.7995 - loss: 216.0923 - val_ll_loss: 233.7716 - val_loss: 229.8394 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 628ms/step - ll_loss: 238.8191 - loss: 215.2513 - val_ll_loss: 233.0273 - val_loss: 229.1227 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 600ms/step - ll_loss: 237.9587 - loss: 214.5133 - val_ll_loss: 232.3714 - val_loss: 228.4913 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 629ms/step - ll_loss: 237.1994 - loss: 213.8624 - val_ll_loss: 231.7908 - val_loss: 227.9326 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 658ms/step - ll_loss: 236.5271 - loss: 213.2860 - val_ll_loss: 231.2752 - val_loss: 227.4365 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 596ms/step - ll_loss: 235.9295 - loss: 212.7740 - val_ll_loss: 230.8162 - val_loss: 226.9949 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 526ms/step - ll_loss: 235.3971 - loss: 212.3179 - val_ll_loss: 230.4065 - val_loss: 226.6008 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 570ms/step - ll_loss: 234.9218 - loss: 211.9107 - val_ll_loss: 230.0401 - val_loss: 226.2485 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 581ms/step - ll_loss: 234.4966 - loss: 211.5465 - val_ll_loss: 229.7119 - val_loss: 225.9329 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 588ms/step - ll_loss: 234.1156 - loss: 211.2202 - val_ll_loss: 229.4175 - val_loss: 225.6498 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 567ms/step - ll_loss: 233.7738 - loss: 210.9276 - val_ll_loss: 229.1531 - val_loss: 225.3957 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 551ms/step - ll_loss: 233.4668 - loss: 210.6647 - val_ll_loss: 228.9154 - val_loss: 225.1672 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 523ms/step - ll_loss: 233.1907 - loss: 210.4283 - val_ll_loss: 228.7015 - val_loss: 224.9616 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 604ms/step - ll_loss: 232.9423 - loss: 210.2156 - val_ll_loss: 228.5089 - val_loss: 224.7765 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 553ms/step - ll_loss: 232.7185 - loss: 210.0241 - val_ll_loss: 228.3352 - val_loss: 224.6096 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 594ms/step - ll_loss: 232.5169 - loss: 209.8515 - val_ll_loss: 228.1786 - val_loss: 224.4591 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 575ms/step - ll_loss: 232.3350 - loss: 209.6958 - val_ll_loss: 228.0374 - val_loss: 224.3234 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 626ms/step - ll_loss: 232.1710 - loss: 209.5554 - val_ll_loss: 227.9098 - val_loss: 224.2008 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 568ms/step - ll_loss: 232.0229 - loss: 209.4286 - val_ll_loss: 227.7946 - val_loss: 224.0902 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 549ms/step - ll_loss: 231.8892 - loss: 209.3141 - val_ll_loss: 227.6906 - val_loss: 223.9902 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 646ms/step - ll_loss: 231.7684 - loss: 209.2108 - val_ll_loss: 227.5965 - val_loss: 223.8998 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 650ms/step - ll_loss: 231.6593 - loss: 209.1173 - val_ll_loss: 227.5114 - val_loss: 223.8182 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 657ms/step - ll_loss: 231.5606 - loss: 209.0329 - val_ll_loss: 227.4346 - val_loss: 223.7444 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 631ms/step - ll_loss: 231.4714 - loss: 208.9565 - val_ll_loss: 227.3650 - val_loss: 223.6776 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 646ms/step - ll_loss: 231.3907 - loss: 208.8875 - val_ll_loss: 227.3022 - val_loss: 223.6172 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 574ms/step - ll_loss: 231.3178 - loss: 208.8250 - val_ll_loss: 227.2453 - val_loss: 223.5625 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 616ms/step - ll_loss: 231.2519 - loss: 208.7686 - val_ll_loss: 227.1938 - val_loss: 223.5130 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 571ms/step - ll_loss: 231.1922 - loss: 208.7175 - val_ll_loss: 227.1471 - val_loss: 223.4683 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 607ms/step - ll_loss: 231.1382 - loss: 208.6712 - val_ll_loss: 227.1050 - val_loss: 223.4278 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 618ms/step - ll_loss: 231.0893 - loss: 208.6294 - val_ll_loss: 227.0668 - val_loss: 223.3912 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 524ms/step - ll_loss: 231.0451 - loss: 208.5915 - val_ll_loss: 227.0323 - val_loss: 223.3580 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 574ms/step - ll_loss: 231.0051 - loss: 208.5573 - val_ll_loss: 227.0010 - val_loss: 223.3280 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 638ms/step - ll_loss: 230.9689 - loss: 208.5263 - val_ll_loss: 226.9727 - val_loss: 223.3008 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 619ms/step - ll_loss: 230.9361 - loss: 208.4982 - val_ll_loss: 226.9471 - val_loss: 223.2762 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 614ms/step - ll_loss: 230.9064 - loss: 208.4728 - val_ll_loss: 226.9239 - val_loss: 223.2539 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 607ms/step - ll_loss: 230.8796 - loss: 208.4498 - val_ll_loss: 226.9029 - val_loss: 223.2337 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 700ms/step - ll_loss: 230.8553 - loss: 208.4290 - val_ll_loss: 226.8838 - val_loss: 223.2155 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 607ms/step - ll_loss: 230.8332 - loss: 208.4101 - val_ll_loss: 226.8666 - val_loss: 223.1990 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 583ms/step - ll_loss: 230.8133 - loss: 208.3931 - val_ll_loss: 226.8510 - val_loss: 223.1840 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 524ms/step - ll_loss: 230.7953 - loss: 208.3776 - val_ll_loss: 226.8369 - val_loss: 223.1705 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 608ms/step - ll_loss: 230.7790 - loss: 208.3636 - val_ll_loss: 226.8241 - val_loss: 223.1582 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 586ms/step - ll_loss: 230.7642 - loss: 208.3509 - val_ll_loss: 226.8126 - val_loss: 223.1471 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 588ms/step - ll_loss: 230.7508 - loss: 208.3395 - val_ll_loss: 226.8020 - val_loss: 223.1370 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 569ms/step - ll_loss: 230.7387 - loss: 208.3291 - val_ll_loss: 226.7926 - val_loss: 223.1279 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 626ms/step - ll_loss: 230.7277 - loss: 208.3197 - val_ll_loss: 226.7840 - val_loss: 223.1196 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 554ms/step - ll_loss: 230.7178 - loss: 208.3112 - val_ll_loss: 226.7762 - val_loss: 223.1122 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 515ms/step - ll_loss: 230.7088 - loss: 208.3035 - val_ll_loss: 226.7692 - val_loss: 223.1054 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 571ms/step - ll_loss: 230.7007 - loss: 208.2965 - val_ll_loss: 226.7628 - val_loss: 223.0993 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 593ms/step - ll_loss: 230.6933 - loss: 208.2902 - val_ll_loss: 226.7570 - val_loss: 223.0938 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 568ms/step - ll_loss: 230.6867 - loss: 208.2845 - val_ll_loss: 226.7518 - val_loss: 223.0888 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m51/51[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 569ms/step - ll_loss: 230.6806 - loss: 208.2794 - val_ll_loss: 226.7470 - val_loss: 223.0842 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/10 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K08/fold_holdout-sub-17
    \n--- holdout=sub-18 | val=sub-08 | K=8 ---
    train segs=55 short=0 | val segs=10 short=0 | test segs=6 short=0



    Loading files:   0%|          | 0/55 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/6 [00:00<?, ?it/s]


    Epoch 1/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 549ms/step - ll_loss: 263.0052 - loss: 234.2428 - val_ll_loss: 248.5775 - val_loss: 245.2992 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 605ms/step - ll_loss: 260.2431 - loss: 231.7359 - val_ll_loss: 245.8253 - val_loss: 242.6315 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 545ms/step - ll_loss: 257.4138 - loss: 229.2503 - val_ll_loss: 243.2811 - val_loss: 240.1590 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 561ms/step - ll_loss: 254.6476 - loss: 226.9461 - val_ll_loss: 240.8326 - val_loss: 237.8355 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 523ms/step - ll_loss: 252.0640 - loss: 224.7548 - val_ll_loss: 238.8154 - val_loss: 235.8958 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m47s[0m 589ms/step - ll_loss: 249.8712 - loss: 222.8795 - val_ll_loss: 237.1721 - val_loss: 234.3172 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 568ms/step - ll_loss: 248.0670 - loss: 221.3348 - val_ll_loss: 235.7898 - val_loss: 232.9904 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 550ms/step - ll_loss: 246.5422 - loss: 220.0309 - val_ll_loss: 234.6058 - val_loss: 231.8544 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 633ms/step - ll_loss: 245.2317 - loss: 218.9113 - val_ll_loss: 233.5796 - val_loss: 230.8703 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 513ms/step - ll_loss: 244.0929 - loss: 217.9393 - val_ll_loss: 232.6831 - val_loss: 230.0106 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 564ms/step - ll_loss: 243.0960 - loss: 217.0888 - val_ll_loss: 231.8953 - val_loss: 229.2552 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 553ms/step - ll_loss: 242.2182 - loss: 216.3405 - val_ll_loss: 231.1995 - val_loss: 228.5883 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 592ms/step - ll_loss: 241.4419 - loss: 215.6790 - val_ll_loss: 230.5829 - val_loss: 227.9973 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 597ms/step - ll_loss: 240.7530 - loss: 215.0923 - val_ll_loss: 230.0347 - val_loss: 227.4719 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 594ms/step - ll_loss: 240.1399 - loss: 214.5703 - val_ll_loss: 229.5462 - val_loss: 227.0037 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m47s[0m 864ms/step - ll_loss: 239.5930 - loss: 214.1049 - val_ll_loss: 229.1098 - val_loss: 226.5856 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 990ms/step - ll_loss: 239.1041 - loss: 213.6890 - val_ll_loss: 228.7194 - val_loss: 226.2114 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 991ms/step - ll_loss: 238.6664 - loss: 213.3166 - val_ll_loss: 228.3694 - val_loss: 225.8761 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m50s[0m 914ms/step - ll_loss: 238.2739 - loss: 212.9828 - val_ll_loss: 228.0554 - val_loss: 225.5752 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 995ms/step - ll_loss: 237.9216 - loss: 212.6832 - val_ll_loss: 227.7732 - val_loss: 225.3049 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m49s[0m 892ms/step - ll_loss: 237.6048 - loss: 212.4139 - val_ll_loss: 227.5194 - val_loss: 225.0618 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 976ms/step - ll_loss: 237.3198 - loss: 212.1717 - val_ll_loss: 227.2909 - val_loss: 224.8429 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m50s[0m 910ms/step - ll_loss: 237.0632 - loss: 211.9536 - val_ll_loss: 227.0851 - val_loss: 224.6457 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 948ms/step - ll_loss: 236.8320 - loss: 211.7571 - val_ll_loss: 226.8996 - val_loss: 224.4679 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 959ms/step - ll_loss: 236.6236 - loss: 211.5799 - val_ll_loss: 226.7322 - val_loss: 224.3076 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 983ms/step - ll_loss: 236.4355 - loss: 211.4202 - val_ll_loss: 226.5811 - val_loss: 224.1629 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 1s/step - ll_loss: 236.2658 - loss: 211.2759 - val_ll_loss: 226.4447 - val_loss: 224.0322 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 984ms/step - ll_loss: 236.1125 - loss: 211.1457 - val_ll_loss: 226.3215 - val_loss: 223.9142 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 997ms/step - ll_loss: 235.9741 - loss: 211.0281 - val_ll_loss: 226.2101 - val_loss: 223.8076 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 947ms/step - ll_loss: 235.8491 - loss: 210.9218 - val_ll_loss: 226.1095 - val_loss: 223.7112 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 1s/step - ll_loss: 235.7360 - loss: 210.8258 - val_ll_loss: 226.0185 - val_loss: 223.6240 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 960ms/step - ll_loss: 235.6338 - loss: 210.7389 - val_ll_loss: 225.9362 - val_loss: 223.5452 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 1s/step - ll_loss: 235.5414 - loss: 210.6604 - val_ll_loss: 225.8617 - val_loss: 223.4739 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 953ms/step - ll_loss: 235.4578 - loss: 210.5893 - val_ll_loss: 225.7944 - val_loss: 223.4094 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 959ms/step - ll_loss: 235.3822 - loss: 210.5251 - val_ll_loss: 225.7335 - val_loss: 223.3511 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 946ms/step - ll_loss: 235.3138 - loss: 210.4669 - val_ll_loss: 225.6783 - val_loss: 223.2982 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 988ms/step - ll_loss: 235.2519 - loss: 210.4143 - val_ll_loss: 225.6284 - val_loss: 223.2504 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 932ms/step - ll_loss: 235.1958 - loss: 210.3667 - val_ll_loss: 225.5832 - val_loss: 223.2072 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 974ms/step - ll_loss: 235.1452 - loss: 210.3236 - val_ll_loss: 225.5423 - val_loss: 223.1680 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 937ms/step - ll_loss: 235.0993 - loss: 210.2846 - val_ll_loss: 225.5053 - val_loss: 223.1326 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m89s[0m 979ms/step - ll_loss: 235.0577 - loss: 210.2493 - val_ll_loss: 225.4718 - val_loss: 223.1005 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 1s/step - ll_loss: 235.0201 - loss: 210.2174 - val_ll_loss: 225.4415 - val_loss: 223.0714 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 929ms/step - ll_loss: 234.9861 - loss: 210.1885 - val_ll_loss: 225.4140 - val_loss: 223.0451 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 1s/step - ll_loss: 234.9553 - loss: 210.1623 - val_ll_loss: 225.3891 - val_loss: 223.0213 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m50s[0m 912ms/step - ll_loss: 234.9274 - loss: 210.1386 - val_ll_loss: 225.3666 - val_loss: 222.9998 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 1s/step - ll_loss: 234.9022 - loss: 210.1171 - val_ll_loss: 225.3462 - val_loss: 222.9802 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 951ms/step - ll_loss: 234.8793 - loss: 210.0977 - val_ll_loss: 225.3277 - val_loss: 222.9625 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 1s/step - ll_loss: 234.8586 - loss: 210.0801 - val_ll_loss: 225.3110 - val_loss: 222.9465 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 934ms/step - ll_loss: 234.8399 - loss: 210.0641 - val_ll_loss: 225.2959 - val_loss: 222.9320 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 889ms/step - ll_loss: 234.8229 - loss: 210.0497 - val_ll_loss: 225.2821 - val_loss: 222.9189 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 972ms/step - ll_loss: 234.8076 - loss: 210.0367 - val_ll_loss: 225.2698 - val_loss: 222.9070 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 900ms/step - ll_loss: 234.7937 - loss: 210.0248 - val_ll_loss: 225.2585 - val_loss: 222.8963 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 949ms/step - ll_loss: 234.7811 - loss: 210.0142 - val_ll_loss: 225.2484 - val_loss: 222.8865 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 943ms/step - ll_loss: 234.7697 - loss: 210.0045 - val_ll_loss: 225.2391 - val_loss: 222.8777 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 1s/step - ll_loss: 234.7594 - loss: 209.9957 - val_ll_loss: 225.2308 - val_loss: 222.8698 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 936ms/step - ll_loss: 234.7501 - loss: 209.9877 - val_ll_loss: 225.2233 - val_loss: 222.8625 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 985ms/step - ll_loss: 234.7416 - loss: 209.9806 - val_ll_loss: 225.2164 - val_loss: 222.8560 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 963ms/step - ll_loss: 234.7340 - loss: 209.9741 - val_ll_loss: 225.2102 - val_loss: 222.8501 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 979ms/step - ll_loss: 234.7270 - loss: 209.9681 - val_ll_loss: 225.2046 - val_loss: 222.8447 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 971ms/step - ll_loss: 234.7207 - loss: 209.9628 - val_ll_loss: 225.1996 - val_loss: 222.8398 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/6 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K08/fold_holdout-sub-18
    \n--- holdout=sub-20 | val=sub-08 | K=8 ---
    train segs=58 short=0 | val segs=10 short=0 | test segs=3 short=0



    Loading files:   0%|          | 0/58 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/3 [00:00<?, ?it/s]


    Epoch 1/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 901ms/step - ll_loss: 260.0280 - loss: 232.4330 - val_ll_loss: 243.6982 - val_loss: 241.2904 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m58s[0m 1s/step - ll_loss: 257.0445 - loss: 229.7092 - val_ll_loss: 240.8085 - val_loss: 238.4493 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 890ms/step - ll_loss: 254.0139 - loss: 226.9712 - val_ll_loss: 238.1356 - val_loss: 235.8609 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m63s[0m 1s/step - ll_loss: 251.1305 - loss: 224.5018 - val_ll_loss: 236.0143 - val_loss: 233.8164 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 956ms/step - ll_loss: 248.8132 - loss: 222.5194 - val_ll_loss: 234.2541 - val_loss: 232.1191 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 955ms/step - ll_loss: 246.8718 - loss: 220.8583 - val_ll_loss: 232.7561 - val_loss: 230.6743 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 935ms/step - ll_loss: 245.2095 - loss: 219.4366 - val_ll_loss: 231.4636 - val_loss: 229.4274 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 936ms/step - ll_loss: 243.7688 - loss: 218.2049 - val_ll_loss: 230.3383 - val_loss: 228.3416 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m58s[0m 1s/step - ll_loss: 242.5099 - loss: 217.1293 - val_ll_loss: 229.3523 - val_loss: 227.3899 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m59s[0m 1s/step - ll_loss: 241.4033 - loss: 216.1843 - val_ll_loss: 228.4837 - val_loss: 226.5515 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 921ms/step - ll_loss: 240.4261 - loss: 215.3503 - val_ll_loss: 227.7156 - val_loss: 225.8100 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 969ms/step - ll_loss: 239.5601 - loss: 214.6115 - val_ll_loss: 227.0340 - val_loss: 225.1519 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 988ms/step - ll_loss: 238.7903 - loss: 213.9549 - val_ll_loss: 226.4276 - val_loss: 224.5664 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 909ms/step - ll_loss: 238.1043 - loss: 213.3701 - val_ll_loss: 225.8868 - val_loss: 224.0441 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m58s[0m 1s/step - ll_loss: 237.4916 - loss: 212.8479 - val_ll_loss: 225.4035 - val_loss: 223.5774 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 915ms/step - ll_loss: 236.9435 - loss: 212.3809 - val_ll_loss: 224.9707 - val_loss: 223.1595 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m60s[0m 1s/step - ll_loss: 236.4523 - loss: 211.9625 - val_ll_loss: 224.5828 - val_loss: 222.7847 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 980ms/step - ll_loss: 236.0115 - loss: 211.5871 - val_ll_loss: 224.2345 - val_loss: 222.4483 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m58s[0m 1s/step - ll_loss: 235.6155 - loss: 211.2499 - val_ll_loss: 223.9214 - val_loss: 222.1459 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 987ms/step - ll_loss: 235.2593 - loss: 210.9467 - val_ll_loss: 223.6396 - val_loss: 221.8738 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m60s[0m 1s/step - ll_loss: 234.9388 - loss: 210.6738 - val_ll_loss: 223.3859 - val_loss: 221.6288 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 968ms/step - ll_loss: 234.6500 - loss: 210.4279 - val_ll_loss: 223.1573 - val_loss: 221.4079 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m60s[0m 1s/step - ll_loss: 234.3896 - loss: 210.2063 - val_ll_loss: 222.9511 - val_loss: 221.2087 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 970ms/step - ll_loss: 234.1547 - loss: 210.0065 - val_ll_loss: 222.7650 - val_loss: 221.0290 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 983ms/step - ll_loss: 233.9428 - loss: 209.8260 - val_ll_loss: 222.5970 - val_loss: 220.8667 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 994ms/step - ll_loss: 233.7513 - loss: 209.6632 - val_ll_loss: 222.4452 - val_loss: 220.7200 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m59s[0m 1s/step - ll_loss: 233.5784 - loss: 209.5160 - val_ll_loss: 222.3080 - val_loss: 220.5875 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 961ms/step - ll_loss: 233.4221 - loss: 209.3830 - val_ll_loss: 222.1841 - val_loss: 220.4678 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m59s[0m 1s/step - ll_loss: 233.2809 - loss: 209.2628 - val_ll_loss: 222.0719 - val_loss: 220.3595 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 991ms/step - ll_loss: 233.1532 - loss: 209.1541 - val_ll_loss: 221.9705 - val_loss: 220.2615 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 959ms/step - ll_loss: 233.0377 - loss: 209.0558 - val_ll_loss: 221.8788 - val_loss: 220.1729 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 653ms/step - ll_loss: 232.9332 - loss: 208.9669 - val_ll_loss: 221.7957 - val_loss: 220.0927 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m42s[0m 684ms/step - ll_loss: 232.8386 - loss: 208.8864 - val_ll_loss: 221.7206 - val_loss: 220.0201 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 643ms/step - ll_loss: 232.7531 - loss: 208.8136 - val_ll_loss: 221.6526 - val_loss: 219.9544 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 645ms/step - ll_loss: 232.6756 - loss: 208.7477 - val_ll_loss: 221.5910 - val_loss: 219.8950 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m39s[0m 673ms/step - ll_loss: 232.6056 - loss: 208.6880 - val_ll_loss: 221.5353 - val_loss: 219.8411 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 632ms/step - ll_loss: 232.5421 - loss: 208.6340 - val_ll_loss: 221.4848 - val_loss: 219.7923 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 624ms/step - ll_loss: 232.4846 - loss: 208.5851 - val_ll_loss: 221.4391 - val_loss: 219.7482 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 687ms/step - ll_loss: 232.4326 - loss: 208.5408 - val_ll_loss: 221.3977 - val_loss: 219.7083 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 644ms/step - ll_loss: 232.3856 - loss: 208.5008 - val_ll_loss: 221.3602 - val_loss: 219.6721 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 641ms/step - ll_loss: 232.3429 - loss: 208.4644 - val_ll_loss: 221.3263 - val_loss: 219.6393 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 628ms/step - ll_loss: 232.3043 - loss: 208.4316 - val_ll_loss: 221.2956 - val_loss: 219.6096 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 625ms/step - ll_loss: 232.2693 - loss: 208.4018 - val_ll_loss: 221.2677 - val_loss: 219.5827 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 662ms/step - ll_loss: 232.2377 - loss: 208.3749 - val_ll_loss: 221.2425 - val_loss: 219.5583 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 642ms/step - ll_loss: 232.2090 - loss: 208.3505 - val_ll_loss: 221.2197 - val_loss: 219.5363 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 665ms/step - ll_loss: 232.1831 - loss: 208.3284 - val_ll_loss: 221.1990 - val_loss: 219.5164 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 648ms/step - ll_loss: 232.1596 - loss: 208.3084 - val_ll_loss: 221.1803 - val_loss: 219.4983 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 635ms/step - ll_loss: 232.1383 - loss: 208.2902 - val_ll_loss: 221.1633 - val_loss: 219.4818 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m39s[0m 680ms/step - ll_loss: 232.1190 - loss: 208.2738 - val_ll_loss: 221.1479 - val_loss: 219.4670 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 622ms/step - ll_loss: 232.1016 - loss: 208.2589 - val_ll_loss: 221.1340 - val_loss: 219.4536 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 619ms/step - ll_loss: 232.0858 - loss: 208.2455 - val_ll_loss: 221.1214 - val_loss: 219.4414 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 689ms/step - ll_loss: 232.0715 - loss: 208.2333 - val_ll_loss: 221.1100 - val_loss: 219.4304 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 640ms/step - ll_loss: 232.0585 - loss: 208.2223 - val_ll_loss: 221.0997 - val_loss: 219.4204 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 628ms/step - ll_loss: 232.0468 - loss: 208.2123 - val_ll_loss: 221.0903 - val_loss: 219.4114 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 665ms/step - ll_loss: 232.0361 - loss: 208.2032 - val_ll_loss: 221.0818 - val_loss: 219.4032 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m39s[0m 666ms/step - ll_loss: 232.0265 - loss: 208.1950 - val_ll_loss: 221.0741 - val_loss: 219.3958 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 628ms/step - ll_loss: 232.0178 - loss: 208.1876 - val_ll_loss: 221.0672 - val_loss: 219.3891 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 664ms/step - ll_loss: 232.0099 - loss: 208.1809 - val_ll_loss: 221.0609 - val_loss: 219.3830 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 643ms/step - ll_loss: 232.0028 - loss: 208.1748 - val_ll_loss: 221.0552 - val_loss: 219.3775 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 629ms/step - ll_loss: 231.9963 - loss: 208.1693 - val_ll_loss: 221.0500 - val_loss: 219.3725 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/3 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K08/fold_holdout-sub-20
    \n--- holdout=sub-21 | val=sub-08 | K=8 ---
    train segs=55 short=0 | val segs=10 short=0 | test segs=6 short=0



    Loading files:   0%|          | 0/55 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/10 [00:00<?, ?it/s]



    Loading files:   0%|          | 0/6 [00:00<?, ?it/s]


    Epoch 1/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 648ms/step - ll_loss: 258.9948 - loss: 231.7529 - val_ll_loss: 245.2664 - val_loss: 241.7424 - learning_rate: 0.0010 - rho: 0.2853
    Epoch 2/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 623ms/step - ll_loss: 256.3765 - loss: 229.3166 - val_ll_loss: 242.8471 - val_loss: 239.5301 - learning_rate: 9.0484e-04 - rho: 0.2403
    Epoch 3/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 614ms/step - ll_loss: 253.5101 - loss: 226.5781 - val_ll_loss: 240.3573 - val_loss: 237.2527 - learning_rate: 8.1873e-04 - rho: 0.2094
    Epoch 4/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 624ms/step - ll_loss: 250.8201 - loss: 224.0042 - val_ll_loss: 237.9720 - val_loss: 234.9237 - learning_rate: 7.4082e-04 - rho: 0.1866
    Epoch 5/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 684ms/step - ll_loss: 248.2142 - loss: 221.7000 - val_ll_loss: 235.8303 - val_loss: 232.8311 - learning_rate: 6.7032e-04 - rho: 0.1691
    Epoch 6/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 624ms/step - ll_loss: 245.8920 - loss: 219.7770 - val_ll_loss: 234.1547 - val_loss: 231.2058 - learning_rate: 6.0653e-04 - rho: 0.1551
    Epoch 7/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 627ms/step - ll_loss: 244.0364 - loss: 218.2270 - val_ll_loss: 232.7619 - val_loss: 229.8601 - learning_rate: 5.4881e-04 - rho: 0.1436
    Epoch 8/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 663ms/step - ll_loss: 242.4856 - loss: 216.9250 - val_ll_loss: 231.5903 - val_loss: 228.7560 - learning_rate: 4.9659e-04 - rho: 0.1340
    Epoch 9/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 662ms/step - ll_loss: 241.1621 - loss: 215.8069 - val_ll_loss: 230.5706 - val_loss: 227.7760 - learning_rate: 4.4933e-04 - rho: 0.1258
    Epoch 10/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 644ms/step - ll_loss: 240.0188 - loss: 214.8392 - val_ll_loss: 229.6822 - val_loss: 226.9209 - learning_rate: 4.0657e-04 - rho: 0.1187
    Epoch 11/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 641ms/step - ll_loss: 239.0218 - loss: 213.9944 - val_ll_loss: 228.9035 - val_loss: 226.1721 - learning_rate: 3.6788e-04 - rho: 0.1125
    Epoch 12/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 674ms/step - ll_loss: 238.1465 - loss: 213.2520 - val_ll_loss: 228.2175 - val_loss: 225.5126 - learning_rate: 3.3287e-04 - rho: 0.1071
    Epoch 13/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 624ms/step - ll_loss: 237.3742 - loss: 212.5965 - val_ll_loss: 227.6105 - val_loss: 224.9295 - learning_rate: 3.0119e-04 - rho: 0.1022
    Epoch 14/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 641ms/step - ll_loss: 236.6900 - loss: 212.0156 - val_ll_loss: 227.0716 - val_loss: 224.4120 - learning_rate: 2.7253e-04 - rho: 0.0979
    Epoch 15/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 661ms/step - ll_loss: 236.0819 - loss: 211.4991 - val_ll_loss: 226.5918 - val_loss: 223.9514 - learning_rate: 2.4660e-04 - rho: 0.0939
    Epoch 16/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 634ms/step - ll_loss: 235.5399 - loss: 211.0387 - val_ll_loss: 226.1637 - val_loss: 223.5405 - learning_rate: 2.2313e-04 - rho: 0.0904
    Epoch 17/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 619ms/step - ll_loss: 235.0560 - loss: 210.6275 - val_ll_loss: 225.7810 - val_loss: 223.1732 - learning_rate: 2.0190e-04 - rho: 0.0871
    Epoch 18/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 626ms/step - ll_loss: 234.6230 - loss: 210.2595 - val_ll_loss: 225.4382 - val_loss: 222.8443 - learning_rate: 1.8268e-04 - rho: 0.0841
    Epoch 19/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 604ms/step - ll_loss: 234.2350 - loss: 209.9296 - val_ll_loss: 225.1307 - val_loss: 222.5494 - learning_rate: 1.6530e-04 - rho: 0.0814
    Epoch 20/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 577ms/step - ll_loss: 233.8868 - loss: 209.6335 - val_ll_loss: 224.8546 - val_loss: 222.2846 - learning_rate: 1.4957e-04 - rho: 0.0789
    Epoch 21/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 601ms/step - ll_loss: 233.5740 - loss: 209.3676 - val_ll_loss: 224.6064 - val_loss: 222.0465 - learning_rate: 1.3534e-04 - rho: 0.0765
    Epoch 22/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 605ms/step - ll_loss: 233.2927 - loss: 209.1283 - val_ll_loss: 224.3830 - val_loss: 221.8324 - learning_rate: 1.2246e-04 - rho: 0.0743
    Epoch 23/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 588ms/step - ll_loss: 233.0395 - loss: 208.9129 - val_ll_loss: 224.1818 - val_loss: 221.6395 - learning_rate: 1.1080e-04 - rho: 0.0723
    Epoch 24/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 515ms/step - ll_loss: 232.8114 - loss: 208.7189 - val_ll_loss: 224.0005 - val_loss: 221.4657 - learning_rate: 1.0026e-04 - rho: 0.0704
    Epoch 25/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 557ms/step - ll_loss: 232.6058 - loss: 208.5439 - val_ll_loss: 223.8370 - val_loss: 221.3089 - learning_rate: 9.0718e-05 - rho: 0.0686
    Epoch 26/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 659ms/step - ll_loss: 232.4204 - loss: 208.3862 - val_ll_loss: 223.6894 - val_loss: 221.1675 - learning_rate: 8.2085e-05 - rho: 0.0669
    Epoch 27/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 605ms/step - ll_loss: 232.2531 - loss: 208.2437 - val_ll_loss: 223.5562 - val_loss: 221.0399 - learning_rate: 7.4274e-05 - rho: 0.0653
    Epoch 28/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 559ms/step - ll_loss: 232.1020 - loss: 208.1152 - val_ll_loss: 223.4359 - val_loss: 220.9246 - learning_rate: 6.7206e-05 - rho: 0.0638
    Epoch 29/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 553ms/step - ll_loss: 231.9655 - loss: 207.9990 - val_ll_loss: 223.3272 - val_loss: 220.8205 - learning_rate: 6.0810e-05 - rho: 0.0624
    Epoch 30/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 583ms/step - ll_loss: 231.8423 - loss: 207.8941 - val_ll_loss: 223.2290 - val_loss: 220.7263 - learning_rate: 5.5023e-05 - rho: 0.0610
    Epoch 31/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 565ms/step - ll_loss: 231.7309 - loss: 207.7992 - val_ll_loss: 223.1402 - val_loss: 220.6412 - learning_rate: 4.9787e-05 - rho: 0.0597
    Epoch 32/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 576ms/step - ll_loss: 231.6302 - loss: 207.7135 - val_ll_loss: 223.0599 - val_loss: 220.5643 - learning_rate: 4.5049e-05 - rho: 0.0585
    Epoch 33/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 575ms/step - ll_loss: 231.5391 - loss: 207.6359 - val_ll_loss: 222.9872 - val_loss: 220.4948 - learning_rate: 4.0762e-05 - rho: 0.0574
    Epoch 34/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 577ms/step - ll_loss: 231.4567 - loss: 207.5658 - val_ll_loss: 222.9215 - val_loss: 220.4318 - learning_rate: 3.6883e-05 - rho: 0.0563
    Epoch 35/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 523ms/step - ll_loss: 231.3823 - loss: 207.5023 - val_ll_loss: 222.8621 - val_loss: 220.3749 - learning_rate: 3.3373e-05 - rho: 0.0552
    Epoch 36/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 553ms/step - ll_loss: 231.3149 - loss: 207.4449 - val_ll_loss: 222.8083 - val_loss: 220.3233 - learning_rate: 3.0197e-05 - rho: 0.0542
    Epoch 37/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 591ms/step - ll_loss: 231.2539 - loss: 207.3930 - val_ll_loss: 222.7596 - val_loss: 220.2767 - learning_rate: 2.7324e-05 - rho: 0.0532
    Epoch 38/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 612ms/step - ll_loss: 231.1987 - loss: 207.3460 - val_ll_loss: 222.7155 - val_loss: 220.2345 - learning_rate: 2.4724e-05 - rho: 0.0523
    Epoch 39/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 552ms/step - ll_loss: 231.1488 - loss: 207.3034 - val_ll_loss: 222.6756 - val_loss: 220.1963 - learning_rate: 2.2371e-05 - rho: 0.0514
    Epoch 40/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 577ms/step - ll_loss: 231.1036 - loss: 207.2649 - val_ll_loss: 222.6395 - val_loss: 220.1617 - learning_rate: 2.0242e-05 - rho: 0.0506
    Epoch 41/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 599ms/step - ll_loss: 231.0627 - loss: 207.2300 - val_ll_loss: 222.6068 - val_loss: 220.1304 - learning_rate: 1.8316e-05 - rho: 0.0498
    Epoch 42/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 547ms/step - ll_loss: 231.0257 - loss: 207.1985 - val_ll_loss: 222.5772 - val_loss: 220.1021 - learning_rate: 1.6573e-05 - rho: 0.0490
    Epoch 43/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 551ms/step - ll_loss: 230.9922 - loss: 207.1699 - val_ll_loss: 222.5504 - val_loss: 220.0765 - learning_rate: 1.4996e-05 - rho: 0.0482
    Epoch 44/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 490ms/step - ll_loss: 230.9619 - loss: 207.1441 - val_ll_loss: 222.5262 - val_loss: 220.0532 - learning_rate: 1.3569e-05 - rho: 0.0475
    Epoch 45/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 589ms/step - ll_loss: 230.9344 - loss: 207.1206 - val_ll_loss: 222.5042 - val_loss: 220.0322 - learning_rate: 1.2277e-05 - rho: 0.0468
    Epoch 46/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 546ms/step - ll_loss: 230.9095 - loss: 207.0994 - val_ll_loss: 222.4843 - val_loss: 220.0132 - learning_rate: 1.1109e-05 - rho: 0.0461
    Epoch 47/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 554ms/step - ll_loss: 230.8870 - loss: 207.0802 - val_ll_loss: 222.4663 - val_loss: 219.9959 - learning_rate: 1.0052e-05 - rho: 0.0455
    Epoch 48/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 561ms/step - ll_loss: 230.8667 - loss: 207.0629 - val_ll_loss: 222.4500 - val_loss: 219.9803 - learning_rate: 9.0953e-06 - rho: 0.0449
    Epoch 49/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 626ms/step - ll_loss: 230.8483 - loss: 207.0471 - val_ll_loss: 222.4353 - val_loss: 219.9662 - learning_rate: 8.2297e-06 - rho: 0.0442
    Epoch 50/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 553ms/step - ll_loss: 230.8315 - loss: 207.0329 - val_ll_loss: 222.4219 - val_loss: 219.9534 - learning_rate: 7.4466e-06 - rho: 0.0437
    Epoch 51/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 572ms/step - ll_loss: 230.8165 - loss: 207.0200 - val_ll_loss: 222.4098 - val_loss: 219.9418 - learning_rate: 6.7379e-06 - rho: 0.0431
    Epoch 52/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 589ms/step - ll_loss: 230.8027 - loss: 207.0083 - val_ll_loss: 222.3988 - val_loss: 219.9314 - learning_rate: 6.0967e-06 - rho: 0.0425
    Epoch 53/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 620ms/step - ll_loss: 230.7904 - loss: 206.9978 - val_ll_loss: 222.3889 - val_loss: 219.9219 - learning_rate: 5.5166e-06 - rho: 0.0420
    Epoch 54/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 613ms/step - ll_loss: 230.7791 - loss: 206.9882 - val_ll_loss: 222.3799 - val_loss: 219.9132 - learning_rate: 4.9916e-06 - rho: 0.0415
    Epoch 55/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 525ms/step - ll_loss: 230.7690 - loss: 206.9795 - val_ll_loss: 222.3718 - val_loss: 219.9055 - learning_rate: 4.5166e-06 - rho: 0.0410
    Epoch 56/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 539ms/step - ll_loss: 230.7598 - loss: 206.9717 - val_ll_loss: 222.3644 - val_loss: 219.8984 - learning_rate: 4.0868e-06 - rho: 0.0405
    Epoch 57/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 613ms/step - ll_loss: 230.7514 - loss: 206.9646 - val_ll_loss: 222.3578 - val_loss: 219.8920 - learning_rate: 3.6979e-06 - rho: 0.0400
    Epoch 58/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 636ms/step - ll_loss: 230.7439 - loss: 206.9581 - val_ll_loss: 222.3517 - val_loss: 219.8862 - learning_rate: 3.3460e-06 - rho: 0.0395
    Epoch 59/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 672ms/step - ll_loss: 230.7371 - loss: 206.9523 - val_ll_loss: 222.3462 - val_loss: 219.8810 - learning_rate: 3.0276e-06 - rho: 0.0391
    Epoch 60/60
    [1m55/55[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m39s[0m 720ms/step - ll_loss: 230.7309 - loss: 206.9470 - val_ll_loss: 222.3413 - val_loss: 219.8763 - learning_rate: 2.7394e-06 - rho: 0.0386



    Getting alpha:   0%|          | 0/6 [00:00<?, ?it/s]


      saved to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/K08/fold_holdout-sub-21
    \nSaved: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/fusion_hmm_LOSO_param_stability/intermediate_nolags_minlen15/cv_fold_summary.tsv



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>heldout_subject</th>
      <th>val_subject</th>
      <th>K</th>
      <th>n_train_segments</th>
      <th>n_val_segments</th>
      <th>n_test_segments</th>
      <th>n_train_short</th>
      <th>n_val_short</th>
      <th>n_test_short</th>
      <th>outdir</th>
      <th>...</th>
      <th>dwellA_TR_s01</th>
      <th>dwellA_TR_s02</th>
      <th>dwellA_TR_s03</th>
      <th>dwellA_TR_s04</th>
      <th>dwellA_TR_s05</th>
      <th>dwellA_TR_s06</th>
      <th>FO_s07</th>
      <th>dwellA_TR_s07</th>
      <th>FO_s08</th>
      <th>dwellA_TR_s08</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sub-01</td>
      <td>sub-08</td>
      <td>6</td>
      <td>55</td>
      <td>10</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fu...</td>
      <td>...</td>
      <td>1.0</td>
      <td>4194304.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sub-02</td>
      <td>sub-08</td>
      <td>6</td>
      <td>59</td>
      <td>10</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fu...</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>8388608.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sub-03</td>
      <td>sub-08</td>
      <td>6</td>
      <td>53</td>
      <td>10</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fu...</td>
      <td>...</td>
      <td>1.0</td>
      <td>8388608.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sub-08</td>
      <td>sub-17</td>
      <td>6</td>
      <td>51</td>
      <td>10</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fu...</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>8388608.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sub-09</td>
      <td>sub-08</td>
      <td>6</td>
      <td>59</td>
      <td>10</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fu...</td>
      <td>...</td>
      <td>8388608.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>



```python
# =========================
# Cell 9 — Quick sanity check: how many folds were saved per K?
# =========================
import glob

for K in K_LIST:
    pattern = str(OUT_ROOT / f"K{K:02d}" / "fold_holdout-*" / "covs_pca.npy")
    hits = glob.glob(pattern)
    print(f"K={K}: saved folds =", len(hits))

```

    K=6: saved folds = 12
    K=7: saved folds = 12
    K=8: saved folds = 12

