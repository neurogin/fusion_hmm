# R01 Pipeline B (STRICT-INTERMEDIATE) — Build X_segments + segments_manifest.tsv (from Pipeline A)

This notebook builds **HMM observation segments** (`X_segments`) from Pipeline A outputs.

**What it does**
- Loads per-run `eeg_power_tr_lags.npy`, `bold_tr.npy`, and `keep_center_minlen{10,15}_<LTAG>.npy`
- Enforces **contiguous segments** with **no NaN holes** (finite-row check)
- Writes one `.npy` per segment and a `segments_manifest.tsv`
- Produces global QC plots (segment lengths, usable minutes, etc.)

You should only edit **Cell 0** unless you are changing methods.


```python
# Cell 0 — User inputs (edit only this cell)

from pathlib import Path
import numpy as np
import pandas as pd

# ---------- USER INPUTS ----------
TR_SEC = 2.1

# Which fusion design to build X_segments for?
#   "lags"   = EEG power features with lags (e.g., -1/0/+1 TR), concatenated with BOLD
#   "nolags" = EEG power features at the same TR only (no temporal context), concatenated with BOLD
FEATURE_MODE = "nolags"   # <-- set to "lags" or "nolags"

# Minimum contiguous segment length (in TRs) after lag-centering / masking
MINLEN_PRIMARY = 10
MINLEN_OPTIONAL = 15   # set to None to skip building optional set

# Lags used in Pipeline A outputs (tags are used to pick the correct keep_center mask)
LAGS_TR_LAGS   = (-1, 0, 1)
LTAG_LAGS      = "lags-1_0_1"
LAGS_TR_NOLAGS = (0,)
LTAG_NOLAGS    = "lags0"

if FEATURE_MODE.lower() == "lags":
    LAGS_TR = LAGS_TR_LAGS
    LTAG    = LTAG_LAGS
    EEG_FILE = "eeg_power_tr_lags.npy"   # (TR × 600)
elif FEATURE_MODE.lower() == "nolags":
    LAGS_TR = LAGS_TR_NOLAGS
    LTAG    = LTAG_NOLAGS
    EEG_FILE = "eeg_power_tr.npy"        # (TR × 200)
else:
    raise ValueError("FEATURE_MODE must be 'lags' or 'nolags'")

# Where FINAL_v3_gnorm_allTR lives (WSL path)
FINAL_ROOT = Path("/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/align_trmask_lags/FINAL_v3_gnorm_allTR/intermediate")
PER_RUN = FINAL_ROOT / "per_run"

# Output folder for HMM segments (variant-specific)
OUT = FINAL_ROOT / f"hmm_segments_minlen{MINLEN_PRIMARY}_{FEATURE_MODE.lower()}"
OUT_SEG = OUT / "segments"     # one .npy per segment
OUT_QC = OUT / "qc"
OUT_PLOTS = OUT_QC / "plots"

for d in [OUT, OUT_SEG, OUT_QC, OUT_PLOTS]:
    d.mkdir(parents=True, exist_ok=True)

print("FEATURE_MODE:", FEATURE_MODE)
print("PER_RUN:", PER_RUN)
print("OUT:", OUT)
print("EEG_FILE:", EEG_FILE, "| LTAG:", LTAG, "| LAGS_TR:", LAGS_TR)

```


```python
# Cell 1 — Helpers (segments, safe loads, validation)

import json

def contiguous_true_segments(mask_bool):
    """Return list of (start, end) half-open intervals where mask_bool is True."""
    mask = np.asarray(mask_bool).astype(bool)
    if mask.ndim != 1:
        raise ValueError("mask must be 1D")
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    # split at breaks
    breaks = np.where(np.diff(idx) != 1)[0] + 1
    blocks = np.split(idx, breaks)
    segs = [(b[0], b[-1] + 1) for b in blocks]  # half-open
    return segs

def filter_segments_by_minlen(segs, minlen):
    return [(s,e) for (s,e) in segs if (e - s) >= int(minlen)]

def safe_load_npy(path: Path):
    if not path.exists():
        raise FileNotFoundError(str(path))
    arr = np.load(path, allow_pickle=False)
    return arr

def ensure_2d_time_by_features(X, name):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2D (T x F). got {X.shape}")
    return X

def finite_rows_mask(*arrays_2d):
    """Rows where all arrays have finite values."""
    m = None
    for A in arrays_2d:
        A = np.asarray(A)
        r = np.isfinite(A).all(axis=1)
        m = r if m is None else (m & r)
    return m

print("Helpers ready.")
```


```python
# Cell 2 — Discover runs and required files

from pathlib import Path
import numpy as np

# Discover run folders
runs = sorted([p.name for p in PER_RUN.iterdir() if p.is_dir()])
print("Discovered run folders:", len(runs))

# Required files from Pipeline A
def required_files_for_mode():
    # keep_center_minlen mask must match the design:
    #  - lags:   keep_center_minlen10_lags-1_0_1.npy
    #  - nolags: keep_center_minlen10_lags0.npy
    keep_mask = f"keep_center_minlen{MINLEN_PRIMARY}_{LTAG}.npy"
    return {
        "bold": "bold_pc1.npy",
        "eeg": EEG_FILE,
        "keep": keep_mask,
    }

REQ = required_files_for_mode()
print("REQ:", REQ)

missing = []
for r in runs:
    rdir = PER_RUN / r
    for key, fname in REQ.items():
        if not (rdir / fname).exists():
            missing.append((r, key, fname))

print("Missing entries:", len(missing))
if len(missing) > 0:
    print("First 20 missing:")
    for x in missing[:20]:
        print("  ", x)

# Keep only runs that have all required files
ok_runs = []
for r in runs:
    rdir = PER_RUN / r
    if all((rdir / f).exists() for f in REQ.values()):
        ok_runs.append(r)

print("OK runs for this mode:", len(ok_runs))
ok_runs[:10]

```


```python
# Cell 3 — Build and write segments (minlen=10)

manifest_rows = []
per_run_rows = []

for run in ok_runs:
    rp = PER_RUN / run

    # load
    bold = ensure_2d_time_by_features(safe_load_npy(rp / "bold_pc1.npy"), "bold_pc1")
    eeg  = ensure_2d_time_by_features(safe_load_npy(rp / EEG_FILE), EEG_FILE.replace(".npy",""))
    edges = np.asarray(safe_load_npy(rp / "tr_edges_sec.npy")).astype(float).ravel()
    keep10 = np.asarray(safe_load_npy(rp / f"keep_center_minlen10_{LTAG}.npy")).astype(bool).ravel()

    # harmonize lengths (defensive)
    T = min(bold.shape[0], eeg.shape[0], keep10.size, edges.size - 1)
    bold = bold[:T]
    eeg  = eeg[:T]
    keep10 = keep10[:T]
    edges = edges[:T+1]

    # enforce finite rows (HMM cannot train on NaNs reliably)
    finite = finite_rows_mask(bold, eeg)
    valid = keep10 & finite

    segs = contiguous_true_segments(valid)
    # if finite-row filtering created small holes, enforce minlen again:
    segs = filter_segments_by_minlen(segs, MINLEN_PRIMARY)

    # write segments
    n_written = 0
    total_tr = 0
    max_len = 0

    for si, (s,e) in enumerate(segs):
        X = np.concatenate([bold[s:e], eeg[s:e]], axis=1).astype(np.float32)
        seg_id = f"{run}__seg{si:04d}"
        seg_path = OUT_SEG / f"{seg_id}.npy"
        np.save(seg_path, X)

        dur_sec = (e - s) * TR_SEC
        manifest_rows.append({
            "run": run,
            "feature_mode": FEATURE_MODE.lower(),
            "lags_tr": ",".join([str(int(x)) for x in LAGS_TR]),
            "seg_id": seg_id,
            "start_TR": s,
            "end_TR": e,                # half-open
            "len_TR": e - s,
            "start_sec": float(edges[s]),
            "end_sec": float(edges[e]),
            "dur_sec": float(dur_sec),
            "n_features": X.shape[1],
            "seg_path": str(seg_path),
        })

        n_written += 1
        total_tr += (e - s)
        max_len = max(max_len, (e - s))

    per_run_rows.append({
        "run": run,
            "feature_mode": FEATURE_MODE.lower(),
            "lags_tr": ",".join([str(int(x)) for x in LAGS_TR]),
        "T_total_TR": int(T),
        "n_segments_minlen10": int(n_written),
        "kept_TR_minlen10": int(total_tr),
        "usable_min_minlen10": float(total_tr * TR_SEC / 60.0),
        "maxSeg_TR_minlen10": int(max_len),
        "finite_drop_TR": int((keep10 & ~finite).sum()),
    })

    print(f"[OK] {run}: wrote {n_written} segments, kept {total_tr}/{T} TR (minlen10)")

manifest = pd.DataFrame(manifest_rows)
manifest_path = OUT / "segments_manifest.tsv"
manifest.to_csv(manifest_path, sep="\t", index=False)

per_run_qc = pd.DataFrame(per_run_rows).sort_values(["usable_min_minlen10","maxSeg_TR_minlen10"], ascending=[False, False])
per_run_qc_path = OUT_QC / "per_run_segments_minlen10.csv"
per_run_qc.to_csv(per_run_qc_path, index=False)

print("\nSaved:")
print(" manifest:", manifest_path)
print(" per-run qc:", per_run_qc_path)
print(" total segments:", len(manifest))
print(" feature dim check (unique):", sorted(manifest["n_features"].unique()))
```


```python
# Cell 4 — Optional: build minlen=15 version (same outputs, separate folder)

if MINLEN_OPTIONAL is None:
    print("MINLEN_OPTIONAL=None -> skipping minlen15 build.")
else:
    OUT15 = FINAL_ROOT / f"hmm_segments_minlen{MINLEN_OPTIONAL}_{FEATURE_MODE.lower()}"
    OUT15_SEG = OUT15 / "segments"
    OUT15_QC = OUT15 / "qc"
    OUT15_PLOTS = OUT15_QC / "plots"
    for d in [OUT15, OUT15_SEG, OUT15_QC, OUT15_PLOTS]:
        d.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    per_run_rows = []

    for run in ok_runs:
        rp = PER_RUN / run

        bold = ensure_2d_time_by_features(safe_load_npy(rp / "bold_pc1.npy"), "bold_pc1")
        eeg  = ensure_2d_time_by_features(safe_load_npy(rp / EEG_FILE), EEG_FILE.replace(".npy",""))
        edges = np.asarray(safe_load_npy(rp / "tr_edges_sec.npy")).astype(float).ravel()

        keep15_path = rp / f"keep_center_minlen15_{LTAG}.npy"
        if not keep15_path.exists():
            print(f"[SKIP] {run}: missing {keep15_path.name}")
            continue

        keep15 = np.asarray(safe_load_npy(keep15_path)).astype(bool).ravel()

        T = min(bold.shape[0], eeg.shape[0], keep15.size, edges.size - 1)
        bold = bold[:T]
        eeg  = eeg[:T]
        keep15 = keep15[:T]
        edges = edges[:T+1]

        finite = finite_rows_mask(bold, eeg)
        valid = keep15 & finite

        segs = contiguous_true_segments(valid)
        segs = filter_segments_by_minlen(segs, MINLEN_OPTIONAL)

        n_written = 0
        total_tr = 0
        max_len = 0

        for si, (s,e) in enumerate(segs):
            X = np.concatenate([bold[s:e], eeg[s:e]], axis=1).astype(np.float32)
            seg_id = f"{run}__seg{si:04d}"
            seg_path = OUT15_SEG / f"{seg_id}.npy"
            np.save(seg_path, X)

            dur_sec = (e - s) * TR_SEC
            manifest_rows.append({
                "run": run,
                "feature_mode": FEATURE_MODE.lower(),
                "lags_tr": ",".join([str(int(x)) for x in LAGS_TR]),
                "seg_id": seg_id,
                "start_TR": s,
                "end_TR": e,
                "len_TR": e - s,
                "start_sec": float(edges[s]),
                "end_sec": float(edges[e]),
                "dur_sec": float(dur_sec),
                "n_features": X.shape[1],
                "seg_path": str(seg_path),
            })

            n_written += 1
            total_tr += (e - s)
            max_len = max(max_len, (e - s))

        per_run_rows.append({
            "run": run,
                "feature_mode": FEATURE_MODE.lower(),
                "lags_tr": ",".join([str(int(x)) for x in LAGS_TR]),
            "T_total_TR": int(T),
            "n_segments_minlen15": int(n_written),
            "kept_TR_minlen15": int(total_tr),
            "usable_min_minlen15": float(total_tr * TR_SEC / 60.0),
            "maxSeg_TR_minlen15": int(max_len),
        })

        print(f"[OK] {run}: wrote {n_written} segments, kept {total_tr}/{T} TR (minlen15)")

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = OUT15 / "segments_manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)

    per_run_qc = pd.DataFrame(per_run_rows).sort_values(["usable_min_minlen15","maxSeg_TR_minlen15"], ascending=[False, False])
    per_run_qc_path = OUT15_QC / "per_run_segments_minlen15.csv"
    per_run_qc.to_csv(per_run_qc_path, index=False)

    print("\nSaved minlen15:")
    print(" manifest:", manifest_path)
    print(" per-run qc:", per_run_qc_path)
```


```python
# Cell 5 — Global QC plots (“usable minutes” and “segment count”)

import matplotlib.pyplot as plt

qc = pd.read_csv(OUT_QC / "per_run_segments_minlen10.csv")

# Usable minutes bar plot
plt.figure()
plt.bar(qc["run"], qc["usable_min_minlen10"])
plt.xticks(rotation=60, ha="right")
plt.ylabel("usable minutes (minlen10)")
plt.title("Usable minutes per run after minlen10")
plt.tight_layout()
p1 = OUT_PLOTS / "usable_minutes_minlen10.png"
plt.savefig(p1, dpi=180)
plt.close()

# Segment count bar plot
plt.figure()
plt.bar(qc["run"], qc["n_segments_minlen10"])
plt.xticks(rotation=60, ha="right")
plt.ylabel("segment count (minlen10)")
plt.title("Segment count per run after minlen10")
plt.tight_layout()
p2 = OUT_PLOTS / "segment_count_minlen10.png"
plt.savefig(p2, dpi=180)
plt.close()

# Segment length distribution (all segments)
manifest = pd.read_csv(OUT / "segments_manifest.tsv", sep="\t")
plt.figure()
plt.hist(manifest["len_TR"], bins=30)
plt.xlabel("segment length (TR)")
plt.ylabel("count")
plt.title("Distribution of segment lengths (minlen10)")
plt.tight_layout()
p3 = OUT_PLOTS / "segment_length_distribution_minlen10.png"
plt.savefig(p3, dpi=180)
plt.close()

print("Saved plots:")
print(" ", p1)
print(" ", p2)
print(" ", p3)
```


```python
# Cell 6 — (Preview) Create an osl-dynamics Data object from segments

# Optional sanity check: create a Data object (requires osl-dynamics installed in this env)
# If you want to postpone API wiring until the next notebook, you can skip this cell.

from osl_dynamics.data import Data

manifest = pd.read_csv(OUT / "segments_manifest.tsv", sep="\t")
seg_paths = manifest["seg_path"].tolist()

# Load into memory (fine for moderate total size).
X_list = [np.load(p).astype(np.float32) for p in seg_paths]

data = Data(X_list)

# Standardize features (recommended before HMM)
# (Use the method your osl-dynamics version supports; this is typical usage.)
data.standardize()

print(data)
print("n_sessions:", data.n_sessions)
print("n_channels/features:", data.n_channels)
```
