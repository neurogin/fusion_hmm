```python
# ============================================================
# R01_BOLD_ParcelPC1_Export_v6.py  (merged v2+v5)
# ============================================================

from __future__ import annotations
import re, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from nilearn.image import resample_to_img
from scipy.io import savemat
from templateflow.api import get as tf_get


# -----------------------
# CONFIG
# -----------------------
FMRI_ROOT = Path("/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids")
OUT_ROOT  = Path("/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bold_parcel/parcel_pc1_v6")

ATLAS_SPACE = "MNI152NLin2009cAsym"
ATLAS_NAME  = "Schaefer2018"
ATLAS_DESC  = "200Parcels7Networks"
ATLAS_RESOLUTION = 2   # res-02 (2mm). Set to 1 if you prefer res-01.

N_PARCELS = 200

# ---- Choose which BOLD you want, explicitly ----
# If you want strict comparability with your older exports, use only desc-preproc.
BOLD_PREFERENCE = [
    f"*_task-rest_*space-{ATLAS_SPACE}_desc-preproc_bold.nii.gz",
    # Uncomment if you *intentionally* want AROMA/smoothAROMA:
    # f"*_task-rest_*space-{ATLAS_SPACE}_desc-smoothAROMAnonaggr_bold.nii.gz",
    # f"*_task-rest_*space-{ATLAS_SPACE}_desc-AROMAnonaggr_bold.nii.gz",
]

# Confounds
USE_MOTION24   = True
USE_WM_CSF     = True          # matches v2 behavior
USE_COSINES    = True          # fmriprep cosine drift terms if present
N_ACOMPCOR     = 10            # a_comp_cor_XX (prefers Retained==True if JSON present)
ADD_NONSTEADY  = True

# Keep-TRs “scrubbing via regression”
ADD_FD_SPIKES    = True
ADD_DVARS_SPIKES = False
FD_THRESHOLD     = 0.5         # mm
DVARS_ZTHRESH    = 2.5         # MAD-z on std_dvars if dvars_outlier* absent
BLOCK_MIN_LEN    = 3           # contiguous spikes length >=3 -> block regressor, else one-hot

# PCA extraction
MIN_VOXELS_FOR_PCA  = 10
MIN_VOXELS_FOR_ANY  = 1
ZSCOR_BEFORE_PCA    = True

# Outputs
SAVE_NPY = True
SAVE_MAT = True
SAVE_ATLAS_ON_GRID = True
SAVE_QC_FIGS = True

OUT_ROOT.mkdir(parents=True, exist_ok=True)
for subd in ["npy", "mat", "atlas_on_grid", "qc", "atlas_source"]:
    (OUT_ROOT / subd).mkdir(parents=True, exist_ok=True)


# -----------------------
# UTILITIES
# -----------------------
def tf_get_single(*args, **kwargs) -> Path:
    out = tf_get(*args, **kwargs)
    if isinstance(out, (list, tuple)):
        out = out[0]
    return Path(out)

def discover_func_dirs(root: Path) -> list[Path]:
    return sorted(root.glob("sub-*_ses-*/fmriprep/sub-*/ses-*/func"))

def infer_runTag(func_dir: Path) -> str:
    sub = func_dir.parts[-3]
    ses = func_dir.parts[-2]
    return f"{sub}_{ses}_task-rest"

def pick_bold_file(func_dir: Path) -> Path | None:
    for pat in BOLD_PREFERENCE:
        cands = sorted(func_dir.glob(pat))
        if cands:
            return cands[0]
    return None

def derive_stem_space_from_bold(bold_path: Path) -> str:
    m = re.match(r"^(.*)_desc-[^_]+_bold\.nii\.gz$", bold_path.name)
    if not m:
        raise ValueError(f"Unrecognized BOLD filename: {bold_path.name}")
    return m.group(1)

def load_confounds(conf_tsv: Path, conf_json: Path | None):
    conf = pd.read_csv(conf_tsv, sep="\t")
    meta = {}
    if conf_json and conf_json.exists():
        meta = json.loads(conf_json.read_text())
    return conf, meta

def build_label_map(tsv_path: Path) -> dict[int, str]:
    if not tsv_path.exists():
        return {}
    df = pd.read_csv(tsv_path, sep="\t")
    if df.shape[1] < 2:
        return {}
    k, v = df.columns[0], df.columns[1]
    return dict(zip(df[k].astype(int).tolist(), df[v].astype(str).tolist()))

def _collect_motion24(conf: pd.DataFrame) -> list[str]:
    base = ["trans_x","trans_y","trans_z","rot_x","rot_y","rot_z"]
    cols = []
    for b in base:
        cols += [b, f"{b}_derivative1", f"{b}_power2", f"{b}_derivative1_power2"]
    return [c for c in cols if c in conf.columns]

def _collect_wm_csf(conf: pd.DataFrame) -> list[str]:
    base = ["white_matter","csf"]
    cols = []
    for b in base:
        cols += [b, f"{b}_derivative1", f"{b}_power2", f"{b}_derivative1_power2"]
    return [c for c in cols if c in conf.columns]

def _collect_cosines(conf: pd.DataFrame) -> list[str]:
    return sorted([c for c in conf.columns if c.startswith("cosine")])

def _collect_nonsteady(conf: pd.DataFrame) -> list[str]:
    return sorted([c for c in conf.columns if c.startswith("non_steady_state_outlier")])

def _collect_acompcor_retained(conf: pd.DataFrame, meta: dict, n_max: int) -> list[str]:
    cols = sorted([c for c in conf.columns if c.startswith("a_comp_cor_")])
    if not cols:
        return []
    retained = []
    for c in cols:
        if c in meta and isinstance(meta[c], dict) and meta[c].get("Retained", None) is True:
            retained.append(c)
    use = sorted(retained) if retained else cols
    return use[:min(n_max, len(use))]

def detect_fd(conf: pd.DataFrame) -> np.ndarray:
    fd = conf["framewise_displacement"].to_numpy() if "framewise_displacement" in conf.columns else np.zeros(len(conf))
    return np.nan_to_num(fd, nan=0.0).astype(np.float32)

def detect_dvars_spikes(conf: pd.DataFrame, zthr: float) -> tuple[np.ndarray, str]:
    out_cols = [c for c in conf.columns if c.startswith("dvars_outlier")]
    if out_cols:
        mask = conf[out_cols].fillna(0).to_numpy().sum(axis=1) > 0
        return mask.astype(bool), "dvars_outlier_cols"

    if "std_dvars" in conf.columns:
        v = conf["std_dvars"].to_numpy()
    elif "dvars" in conf.columns:
        v = conf["dvars"].to_numpy()
    else:
        return np.zeros(len(conf), dtype=bool), "none"

    v = np.nan_to_num(v, nan=np.nanmedian(v))
    med = np.median(v)
    mad = np.median(np.abs(v - med)) + 1e-8
    vz  = (v - med) / mad
    return (vz > zthr).astype(bool), "mad_z"

def _spike_onehot(mask: np.ndarray, prefix: str) -> tuple[np.ndarray, list[str]]:
    idx = np.where(mask)[0]
    X = np.zeros((mask.shape[0], len(idx)), dtype=np.float32)
    for j, t in enumerate(idx):
        X[t, j] = 1.0
    names = [f"{prefix}_{t:04d}" for t in idx]
    return X, names

def _spike_blocks(mask: np.ndarray, prefix: str) -> tuple[np.ndarray, list[str]]:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return np.zeros((mask.shape[0], 0), dtype=np.float32), []
    blocks = []
    start = prev = idx[0]
    for t in idx[1:]:
        if t == prev + 1:
            prev = t
        else:
            blocks.append((start, prev))
            start = prev = t
    blocks.append((start, prev))

    X = np.zeros((mask.shape[0], len(blocks)), dtype=np.float32)
    names = []
    for j, (a, b) in enumerate(blocks):
        X[a:b+1, j] = 1.0
        names.append(f"{prefix}_block_{a:04d}_{b:04d}")
    return X, names

def spikes_to_design(mask: np.ndarray, prefix: str, block_min_len: int):
    """Hybrid: segments >= block_min_len become blocks; short segments become one-hots."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return np.zeros((mask.shape[0], 0), dtype=np.float32), []

    # Build contiguous segments
    segs = []
    a = b = idx[0]
    for t in idx[1:]:
        if t == b + 1:
            b = t
        else:
            segs.append((a, b))
            a = b = t
    segs.append((a, b))

    # Convert
    cols = []
    names = []
    for (a, b) in segs:
        L = b - a + 1
        if L >= block_min_len:
            col = np.zeros((mask.shape[0],), dtype=np.float32)
            col[a:b+1] = 1.0
            cols.append(col)
            names.append(f"{prefix}_block_{a:04d}_{b:04d}")
        else:
            for t in range(a, b+1):
                col = np.zeros((mask.shape[0],), dtype=np.float32)
                col[t] = 1.0
                cols.append(col)
                names.append(f"{prefix}_{t:04d}")

    X = np.stack(cols, axis=1) if cols else np.zeros((mask.shape[0], 0), dtype=np.float32)
    return X, names

def build_design_matrix(conf: pd.DataFrame, meta: dict):
    """
    Design matrix:
      - z-scored continuous confounds (motion24, WM/CSF, cosines, aCompCor, nonsteady)
      - spike regressors:
          * FD spikes: FD > FD_THRESHOLD
          * PLUS catastrophic expansion: if FD > FD_CAT, also flag t..t+EXPAND_POST (and optionally t-EXPAND_PRE)
          * motion_outlier* (fMRIPrep): include ONLY those outliers occurring at FD <= FD_THRESHOLD
            (so they catch "FD-misses-it" intensity transients like sub-03 TR19)
      - optional DVARS spikes (kept as diagnostic even if not included)
      - intercept
    Returns: X, names, qc, fd
    """

    # ---------- local helpers ----------
    def _expand_mask(mask: np.ndarray, pre: int = 0, post: int = 2) -> np.ndarray:
        """Expand True indices in `mask` by [t-pre, ..., t+post]."""
        Tloc = mask.shape[0]
        out = mask.copy()
        idx = np.where(mask)[0]
        for t in idx:
            a = max(0, t - pre)
            b = min(Tloc, t + post + 1)
            out[a:b] = True
        return out

    def _collect_motion_outliers(conf_: pd.DataFrame) -> list[str]:
        """fMRIPrep one-hot outlier columns."""
        return sorted([c for c in conf_.columns if c.startswith("motion_outlier")])

    # ---------- config for catastrophic expansion ----------
    FD_CAT = 2.0          # mm; "catastrophic" FD threshold for expansion
    EXPAND_PRE = 0        # set to 1 if you want t-1 included
    EXPAND_POST = 2       # include t+1 and t+2

    T = len(conf)
    cont_cols: list[str] = []

    # ---------- continuous confounds ----------
    if USE_MOTION24:
        cont_cols += _collect_motion24(conf)
    if USE_WM_CSF:
        cont_cols += _collect_wm_csf(conf)
    if USE_COSINES:
        cont_cols += _collect_cosines(conf)

    cont_cols += _collect_acompcor_retained(conf, meta, N_ACOMPCOR)

    if ADD_NONSTEADY:
        cont_cols += _collect_nonsteady(conf)

    # de-duplicate while preserving order
    seen = set()
    cont_cols = [c for c in cont_cols if (c not in seen and not seen.add(c))]

    X_cont = conf[cont_cols].copy() if cont_cols else pd.DataFrame(index=np.arange(T))
    X_cont = X_cont.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    X_parts: list[np.ndarray] = []
    names: list[str] = []

    # z-score continuous regressors (conditioning)
    if X_cont.shape[1] > 0:
        Xc = X_cont.to_numpy(dtype=np.float32)
        mu = Xc.mean(axis=0, keepdims=True)
        sd = Xc.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Xc = (Xc - mu) / sd
        X_parts.append(Xc)
        names += list(X_cont.columns)

    # ---------- FD and spikes ----------
    fd = detect_fd(conf)

    fd_base = (fd > FD_THRESHOLD)

    # catastrophic expansion (adds t..t+2 around FD>FD_CAT)
    fd_cat = (fd > FD_CAT)
    fd_spikes = fd_base | _expand_mask(fd_cat, pre=EXPAND_PRE, post=EXPAND_POST)

    # DVARS spikes computed for QC diagnostics (even if not added)
    dvars_spikes, dvars_mode = detect_dvars_spikes(conf, DVARS_ZTHRESH)

    Xfd = np.zeros((T, 0), dtype=np.float32)
    Xdv = np.zeros((T, 0), dtype=np.float32)
    Xmo = np.zeros((T, 0), dtype=np.float32)

    if ADD_FD_SPIKES:
        Xfd, nfd = spikes_to_design(fd_spikes, "fd", BLOCK_MIN_LEN)
        X_parts.append(Xfd)
        names += nfd

    if ADD_DVARS_SPIKES:
        Xdv, ndv = spikes_to_design(dvars_spikes, "dvars", BLOCK_MIN_LEN)
        X_parts.append(Xdv)
        names += ndv

    # ---------- fMRIPrep motion_outlier* spikes (filtered to low-FD only) ----------
    motion_out_cols = _collect_motion_outliers(conf)

    # union mask (diagnostic)
    motion_out_any = np.zeros(T, dtype=bool)
    if len(motion_out_cols) > 0:
        motmat = conf[motion_out_cols].fillna(0.0).to_numpy(dtype=np.float32)
        motion_out_any = (motmat.sum(axis=1) > 0)

        # keep only columns whose spike occurs at FD <= FD_THRESHOLD
        kept_cols = []
        for c in motion_out_cols:
            idx = np.where(conf[c].fillna(0.0).to_numpy() > 0)[0]
            if len(idx) == 0:
                continue
            # typical one-hot: len(idx)==1; if multiple, require ALL to be low-FD
            if np.all(fd[idx] <= FD_THRESHOLD):
                kept_cols.append(c)

        if len(kept_cols) > 0:
            Xmo = conf[kept_cols].fillna(0.0).to_numpy(dtype=np.float32)

            # drop any all-zero columns (safety)
            keep = (Xmo.sum(axis=0) > 0)
            Xmo = Xmo[:, keep]
            kept_cols = [c for c, k in zip(kept_cols, keep) if k]

            if Xmo.shape[1] > 0:
                X_parts.append(Xmo)
                names += kept_cols

    # ---------- intercept ----------
    X_parts.append(np.ones((T, 1), dtype=np.float32))
    names += ["intercept"]

    X = np.concatenate(X_parts, axis=1) if X_parts else np.ones((T, 1), dtype=np.float32)

    # diagnostics for motion_outlier
    motion_out_kept_any = float((Xmo.sum(axis=1) > 0).mean()) if Xmo.shape[1] > 0 else 0.0

    qc = dict(
        qc_fd_mean=float(fd.mean()),
        qc_fd_p95=float(np.percentile(fd, 95)),
        qc_fd_max=float(fd.max()),
        pct_fd_spikes=float(fd_spikes.mean()),
        pct_dvars_spikes=float(dvars_spikes.mean()),  # diagnostic only if DVARS spikes not added
        n_fd_units=int(Xfd.shape[1]) if ADD_FD_SPIKES else 0,
        n_dvars_units=int(Xdv.shape[1]) if ADD_DVARS_SPIKES else 0,
        dvars_mode=dvars_mode,
        fd_cat_threshold=float(FD_CAT),
        fd_expand_pre=int(EXPAND_PRE),
        fd_expand_post=int(EXPAND_POST),
        pct_fd_cat=float(fd_cat.mean()),
        n_motion_out_cols_total=int(len(motion_out_cols)),
        n_motion_out_units=int(Xmo.shape[1]),
        pct_motion_out_any_total=float(motion_out_any.mean()),
        pct_motion_out_any_kept=float(motion_out_kept_any),
        n_conf_cont=int(X_cont.shape[1]),
        n_total_regressors=int(X.shape[1]),
    )

    return X, names, qc, fd

def regress_out(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    # Y is (T x V)
    beta, *_ = np.linalg.lstsq(X.astype(np.float64), Y.astype(np.float64), rcond=None)
    return (Y - X @ beta).astype(np.float32)

def parcel_pc1(Y_resid: np.ndarray, atlas_lbl_3d: np.ndarray, mask_3d: np.ndarray):
    assert atlas_lbl_3d.shape == mask_3d.shape, f"atlas/mask mismatch {atlas_lbl_3d.shape} vs {mask_3d.shape}"

    mask_flat  = mask_3d.reshape(-1)
    atlas_flat = atlas_lbl_3d.reshape(-1)[mask_flat]  # (V,)

    T = Y_resid.shape[0]
    pc = np.zeros((T, N_PARCELS), dtype=np.float32)
    vox_counts = np.zeros(N_PARCELS, dtype=int)

    n_nan = 0
    n_mean = 0

    for p in range(1, N_PARCELS + 1):
        idx = np.where(atlas_flat == p)[0]
        vox_counts[p-1] = len(idx)

        if len(idx) < MIN_VOXELS_FOR_ANY:
            pc[:, p-1] = np.nan
            n_nan += 1
            continue

        Xp = Y_resid[:, idx]
        Xp = np.nan_to_num(Xp, nan=0.0)

        # center each voxel
        Xp = Xp - Xp.mean(axis=0, keepdims=True)

        if ZSCOR_BEFORE_PCA:
            sd = Xp.std(axis=0, keepdims=True)
            sd = np.where(sd < 1e-8, 1.0, sd)
            Xp = Xp / sd

        if len(idx) < MIN_VOXELS_FOR_PCA:
            ts = Xp.mean(axis=1)
            n_mean += 1
        else:
            U, S, _ = np.linalg.svd(Xp, full_matrices=False)
            ts = U[:, 0] * S[0]
            # sign to match mean
            m = Xp.mean(axis=1)
            if np.corrcoef(ts, m)[0, 1] < 0:
                ts = -ts

        pc[:, p-1] = ts.astype(np.float32)

    qc = dict(
        qc_pct_all_nan_parcels=float(n_nan / N_PARCELS),
        qc_n_mean_fallback=int(n_mean),
        min_voxels_per_parcel=int(vox_counts.min()),
        p10_voxels_per_parcel=float(np.percentile(vox_counts, 10)),
        median_voxels_per_parcel=float(np.median(vox_counts)),
    )
    return pc, vox_counts, qc


# -----------------------
# LOAD ATLAS + LABELS (freeze into OUT_ROOT)
# -----------------------
atlas_nii = tf_get_single(
    ATLAS_SPACE,
    atlas=ATLAS_NAME,
    desc=ATLAS_DESC,
    suffix="dseg",
    extension=".nii.gz",
    resolution=ATLAS_RESOLUTION,
)
atlas_tsv = tf_get_single(
    ATLAS_SPACE,
    atlas=ATLAS_NAME,
    desc=ATLAS_DESC,
    suffix="dseg",
    extension=".tsv",
)

# Freeze copies
atlas_nii_frozen = OUT_ROOT / "atlas_source" / atlas_nii.name
atlas_tsv_frozen = OUT_ROOT / "atlas_source" / atlas_tsv.name
if not atlas_nii_frozen.exists():
    atlas_nii_frozen.write_bytes(atlas_nii.read_bytes())
if not atlas_tsv_frozen.exists():
    atlas_tsv_frozen.write_bytes(atlas_tsv.read_bytes())

atlas_img = nib.load(str(atlas_nii_frozen))
label_map = build_label_map(atlas_tsv_frozen)

parcel_labels = np.arange(1, N_PARCELS + 1, dtype=np.int32)
parcel_names  = np.array([label_map.get(int(l), f"ROI_{int(l)}") for l in parcel_labels], dtype=object)

print("Atlas frozen:", atlas_nii_frozen)
print("TSV frozen:", atlas_tsv_frozen)


# -----------------------
# BUILD RUN INDEX (STRICT PATHS)
# -----------------------
rows = []
for func_dir in discover_func_dirs(FMRI_ROOT):
    runTag = infer_runTag(func_dir)
    bold = pick_bold_file(func_dir)
    if bold is None:
        continue

    stem_space = derive_stem_space_from_bold(bold)
    brain_mask = func_dir / f"{stem_space}_desc-brain_mask.nii.gz"
    boldref    = func_dir / f"{stem_space}_boldref.nii.gz"
    conf_tsv   = func_dir / f"{runTag}_desc-confounds_timeseries.tsv"
    conf_json  = func_dir / f"{runTag}_desc-confounds_timeseries.json"

    rows.append(dict(
        runTag=runTag,
        func_dir=str(func_dir),
        bold_file=str(bold),
        brain_mask=str(brain_mask),
        boldref=str(boldref) if boldref.exists() else "",
        confounds_tsv=str(conf_tsv),
        confounds_json=str(conf_json) if conf_json.exists() else "",
        stem_space=stem_space,
    ))

index = pd.DataFrame(rows).sort_values("runTag").reset_index(drop=True)
print("Runs to export:", len(index))

# Fail fast
for col in ["bold_file", "brain_mask", "confounds_tsv"]:
    bad = index.loc[index[col].apply(lambda p: not Path(p).exists()), ["runTag", col, "stem_space", "func_dir"]]
    if len(bad):
        raise RuntimeError(f"Missing required {col}:\n{bad.to_string(index=False)}")


# -----------------------
# EXPORT LOOP
# -----------------------
qc_rows = []

for k, r in index.iterrows():
    runTag = r["runTag"]
    print(f"[{k+1}/{len(index)}] {runTag}")

    bold_img_run = nib.load(r["bold_file"])
    data = bold_img_run.get_fdata().astype(np.float32)  # (X,Y,Z,T)

    mask_img = nib.load(r["brain_mask"])
    # resample mask to bold grid if needed
    if mask_img.shape != data.shape[:3] or (not np.allclose(mask_img.affine, bold_img_run.affine, atol=1e-4)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask_img = resample_to_img(mask_img, bold_img_run.slicer[:, :, :, 0],
                                       interpolation="nearest", force_resample=True, copy_header=True)
    mask = mask_img.get_fdata().astype(np.float32) > 0.5

    conf, meta = load_confounds(Path(r["confounds_tsv"]), Path(r["confounds_json"]) if r["confounds_json"] else None)
    X, xnames, qc_motion, fd = build_design_matrix(conf, meta)

    T = data.shape[3]
    if X.shape[0] != T:
        raise RuntimeError(f"TR mismatch for {runTag}: bold T={T}, confounds T={X.shape[0]}")

    # masked voxel series (T x V)
    Y = data[mask].T
    Y_resid = regress_out(X, Y)

    # atlas on grid
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atlas_res = resample_to_img(atlas_img, bold_img_run.slicer[:, :, :, 0],
                                    interpolation="nearest", force_resample=True, copy_header=True)
    atlas_lbl = atlas_res.get_fdata().astype(np.int32)

    if SAVE_ATLAS_ON_GRID:
        atlas_on_grid_path = OUT_ROOT / "atlas_on_grid" / f"{runTag}_atlas_on_grid.nii.gz"
        nib.save(atlas_res, str(atlas_on_grid_path))
    else:
        atlas_on_grid_path = ""

    pc, vox_counts, qc_parc = parcel_pc1(Y_resid, atlas_lbl, mask)

    # Save outputs
    out_pc1   = OUT_ROOT / "npy" / f"{runTag}_parcel_pc1.npy"
    out_lab   = OUT_ROOT / "npy" / f"{runTag}_parcel_labels.npy"
    out_names = OUT_ROOT / "npy" / f"{runTag}_parcel_names.npy"
    out_nvox  = OUT_ROOT / "npy" / f"{runTag}_parcel_nvox.npy"

    if SAVE_NPY:
        np.save(out_pc1, pc.astype(np.float32))
        np.save(out_lab, parcel_labels)
        np.save(out_names, parcel_names)
        np.save(out_nvox, vox_counts.astype(np.int32))

    out_mat = ""
    if SAVE_MAT:
        out_mat = OUT_ROOT / "mat" / f"{runTag}_parcel_PCs_final.mat"
        savemat(out_mat, {
            "runTag": runTag,
            "parcel_pc1": pc.astype(np.float32),
            "labels": parcel_labels,
            "parcel_names": parcel_names,
            "parcel_nvox": vox_counts.astype(np.int32),
        }, do_compression=True)

    # QC figure: first 5 parcel traces + FD
    qc_fig = ""
    if SAVE_QC_FIGS:
        qc_fig = OUT_ROOT / "qc" / f"{runTag}_qc_pc1_fd.png"
        plt.figure(figsize=(10, 5))
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(pc[:, :5])
        ax1.set_title(f"{runTag} — first 5 parcel PC1 traces (after nuisance regression)")
        ax1.set_ylabel("a.u.")
        ax1.set_xlabel("TR")

        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        ax2.plot(fd)
        ax2.set_title("Framewise Displacement (FD)")
        ax2.set_ylabel("mm")
        ax2.set_xlabel("TR")

        plt.tight_layout()
        plt.savefig(qc_fig, dpi=150)
        plt.close()

    qc_rows.append({
        "runTag": runTag,
        "n_volumes": int(pc.shape[0]),
        "n_parcels": int(pc.shape[1]),
        **qc_motion,
        **qc_parc,
        "bold_file": r["bold_file"],
        "brain_mask": r["brain_mask"],
        "confounds_tsv": r["confounds_tsv"],
        "confounds_json": r["confounds_json"],
        "atlas_source_nii": str(atlas_nii_frozen),
        "atlas_source_tsv": str(atlas_tsv_frozen),
        "atlas_on_grid": str(atlas_on_grid_path) if atlas_on_grid_path else "",
        "out_npy_pc1": str(out_pc1),
        "out_npy_labels": str(out_lab),
        "out_npy_names": str(out_names),
        "out_npy_nvox": str(out_nvox),
        "out_mat": str(out_mat) if out_mat else "",
        "qc_fig": str(qc_fig) if qc_fig else "",
    })

qc_df = pd.DataFrame(qc_rows).sort_values("runTag").reset_index(drop=True)
qc_df.to_csv(OUT_ROOT / "dataset_index.csv", index=False)
print("Wrote:", OUT_ROOT / "dataset_index.csv")
qc_df

```

    Atlas frozen: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bold_parcel/parcel_pc1_v6/atlas_source/tpl-MNI152NLin2009cAsym_res-02_atlas-Schaefer2018_desc-200Parcels7Networks_dseg.nii.gz
    TSV frozen: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bold_parcel/parcel_pc1_v6/atlas_source/tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-200Parcels7Networks_dseg.tsv
    Runs to export: 15
    [1/15] sub-01_ses-01_task-rest
    [2/15] sub-02_ses-01_task-rest
    [3/15] sub-03_ses-01_task-rest
    [4/15] sub-08_ses-01_task-rest
    [5/15] sub-08_ses-02_task-rest
    [6/15] sub-09_ses-01_task-rest
    [7/15] sub-13_ses-01_task-rest
    [8/15] sub-13_ses-02_task-rest
    [9/15] sub-14_ses-01_task-rest
    [10/15] sub-16_ses-01_task-rest
    [11/15] sub-17_ses-01_task-rest
    [12/15] sub-17_ses-02_task-rest
    [13/15] sub-18_ses-01_task-rest
    [14/15] sub-20_ses-01_task-rest
    [15/15] sub-21_ses-01_task-rest
    Wrote: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bold_parcel/parcel_pc1_v6/dataset_index.csv





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
      <th>runTag</th>
      <th>n_volumes</th>
      <th>n_parcels</th>
      <th>qc_fd_mean</th>
      <th>qc_fd_p95</th>
      <th>qc_fd_max</th>
      <th>pct_fd_spikes</th>
      <th>pct_dvars_spikes</th>
      <th>n_fd_units</th>
      <th>n_dvars_units</th>
      <th>...</th>
      <th>confounds_json</th>
      <th>atlas_source_nii</th>
      <th>atlas_source_tsv</th>
      <th>atlas_on_grid</th>
      <th>out_npy_pc1</th>
      <th>out_npy_labels</th>
      <th>out_npy_names</th>
      <th>out_npy_nvox</th>
      <th>out_mat</th>
      <th>qc_fig</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sub-01_ses-01_task-rest</td>
      <td>288</td>
      <td>200</td>
      <td>0.148735</td>
      <td>0.276897</td>
      <td>0.542274</td>
      <td>0.003472</td>
      <td>0.100694</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sub-02_ses-01_task-rest</td>
      <td>288</td>
      <td>200</td>
      <td>0.126517</td>
      <td>0.225350</td>
      <td>0.454937</td>
      <td>0.000000</td>
      <td>0.062500</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sub-03_ses-01_task-rest</td>
      <td>288</td>
      <td>200</td>
      <td>0.366118</td>
      <td>0.888737</td>
      <td>4.246836</td>
      <td>0.222222</td>
      <td>0.218750</td>
      <td>41</td>
      <td>0</td>
      <td>...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sub-08_ses-01_task-rest</td>
      <td>288</td>
      <td>200</td>
      <td>0.167428</td>
      <td>0.710896</td>
      <td>1.332223</td>
      <td>0.097222</td>
      <td>0.232639</td>
      <td>22</td>
      <td>0</td>
      <td>...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sub-08_ses-02_task-rest</td>
      <td>288</td>
      <td>200</td>
      <td>0.236912</td>
      <td>0.747938</td>
      <td>1.915868</td>
      <td>0.121528</td>
      <td>0.284722</td>
      <td>19</td>
      <td>0</td>
      <td>...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>sub-09_ses-01_task-rest</td>
      <td>288</td>
      <td>200</td>
      <td>0.300285</td>
      <td>0.551479</td>
      <td>11.917186</td>
      <td>0.079861</td>
      <td>0.197917</td>
      <td>10</td>
      <td>0</td>
      <td>...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>sub-13_ses-01_task-rest</td>
      <td>288</td>
      <td>200</td>
      <td>0.118586</td>
      <td>0.241643</td>
      <td>2.275126</td>
      <td>0.010417</td>
      <td>0.211806</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>sub-13_ses-02_task-rest</td>
      <td>288</td>
      <td>200</td>
      <td>0.129775</td>
      <td>0.391650</td>
      <td>2.473213</td>
      <td>0.038194</td>
      <td>0.201389</td>
      <td>5</td>
      <td>0</td>
      <td>...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>sub-14_ses-01_task-rest</td>
      <td>288</td>
      <td>200</td>
      <td>0.106983</td>
      <td>0.269055</td>
      <td>0.567419</td>
      <td>0.003472</td>
      <td>0.173611</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>sub-16_ses-01_task-rest</td>
      <td>288</td>
      <td>200</td>
      <td>0.158516</td>
      <td>0.318929</td>
      <td>0.460479</td>
      <td>0.000000</td>
      <td>0.111111</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>sub-17_ses-01_task-rest</td>
      <td>288</td>
      <td>200</td>
      <td>0.200932</td>
      <td>0.685560</td>
      <td>2.748104</td>
      <td>0.069444</td>
      <td>0.274306</td>
      <td>9</td>
      <td>0</td>
      <td>...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>sub-17_ses-02_task-rest</td>
      <td>288</td>
      <td>200</td>
      <td>0.110854</td>
      <td>0.213021</td>
      <td>0.703244</td>
      <td>0.003472</td>
      <td>0.215278</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>sub-18_ses-01_task-rest</td>
      <td>288</td>
      <td>200</td>
      <td>0.159986</td>
      <td>0.384813</td>
      <td>0.987393</td>
      <td>0.020833</td>
      <td>0.170139</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>sub-20_ses-01_task-rest</td>
      <td>288</td>
      <td>200</td>
      <td>0.079698</td>
      <td>0.238022</td>
      <td>0.777578</td>
      <td>0.010417</td>
      <td>0.232639</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>sub-21_ses-01_task-rest</td>
      <td>288</td>
      <td>200</td>
      <td>0.226257</td>
      <td>0.593136</td>
      <td>1.581828</td>
      <td>0.065972</td>
      <td>0.166667</td>
      <td>17</td>
      <td>0</td>
      <td>...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...</td>
    </tr>
  </tbody>
</table>
<p>15 rows × 39 columns</p>
</div>




```python
import pandas as pd
from pathlib import Path

OUT_ROOT = Path("/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bold_parcel/parcel_pc1_v6")
df = pd.read_csv(OUT_ROOT / "dataset_index.csv")

# thresholds you can tune
WARN_PCT_SPIKES = 0.10     # >10% TRs flagged as FD spikes
FLAG_PCT_SPIKES = 0.20     # >20%
WARN_FD_MAX = 2.0          # mm
FLAG_FD_MAX = 5.0          # mm

def label_row(r):
    worst = 0
    if r["pct_fd_spikes"] >= FLAG_PCT_SPIKES: worst = max(worst, 2)
    elif r["pct_fd_spikes"] >= WARN_PCT_SPIKES: worst = max(worst, 1)

    if r["qc_fd_max"] >= FLAG_FD_MAX: worst = max(worst, 2)
    elif r["qc_fd_max"] >= WARN_FD_MAX: worst = max(worst, 1)

    return ["OK","WARN","FLAG"][worst]

df["motion_flag"] = df.apply(label_row, axis=1)

show = df[["runTag","qc_fd_mean","qc_fd_p95","qc_fd_max","pct_fd_spikes","pct_dvars_spikes","motion_flag"]]
show = show.sort_values(["motion_flag","pct_fd_spikes","qc_fd_max"], ascending=[True, False, False])

print(show.to_string(index=False))
show.to_csv(OUT_ROOT / "motion_qc_flags.csv", index=False)
print("\nWrote:", OUT_ROOT / "motion_qc_flags.csv")

```

                     runTag  qc_fd_mean  qc_fd_p95  qc_fd_max  pct_fd_spikes  pct_dvars_spikes motion_flag
    sub-03_ses-01_task-rest    0.366118   0.888737   4.246836       0.222222          0.218750        FLAG
    sub-09_ses-01_task-rest    0.300285   0.551479  11.917186       0.079861          0.197917        FLAG
    sub-08_ses-01_task-rest    0.167428   0.710896   1.332223       0.097222          0.232639          OK
    sub-21_ses-01_task-rest    0.226257   0.593136   1.581828       0.065972          0.166667          OK
    sub-18_ses-01_task-rest    0.159986   0.384813   0.987393       0.020833          0.170139          OK
    sub-20_ses-01_task-rest    0.079698   0.238022   0.777578       0.010417          0.232639          OK
    sub-17_ses-02_task-rest    0.110854   0.213021   0.703244       0.003472          0.215278          OK
    sub-14_ses-01_task-rest    0.106983   0.269055   0.567419       0.003472          0.173611          OK
    sub-01_ses-01_task-rest    0.148735   0.276897   0.542274       0.003472          0.100694          OK
    sub-16_ses-01_task-rest    0.158516   0.318929   0.460479       0.000000          0.111111          OK
    sub-02_ses-01_task-rest    0.126517   0.225350   0.454937       0.000000          0.062500          OK
    sub-08_ses-02_task-rest    0.236912   0.747938   1.915868       0.121528          0.284722        WARN
    sub-17_ses-01_task-rest    0.200932   0.685560   2.748104       0.069444          0.274306        WARN
    sub-13_ses-02_task-rest    0.129775   0.391650   2.473213       0.038194          0.201389        WARN
    sub-13_ses-01_task-rest    0.118586   0.241643   2.275126       0.010417          0.211806        WARN
    
    Wrote: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bold_parcel/parcel_pc1_v6/motion_qc_flags.csv



```python
import numpy as np
import pandas as pd

df = pd.read_csv("/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bold_parcel/parcel_pc1_v6/dataset_index.csv")  # path to your v6 output CSV

rows = []
for _, r in df.iterrows():
    pc = np.load(r["out_npy_pc1"])  # (T,200)
    conf = pd.read_csv(r["confounds_tsv"], sep="\t")
    fd = conf["framewise_displacement"].fillna(0).to_numpy()

    # z-score
    pc_z = (pc - pc.mean(axis=0, keepdims=True)) / (pc.std(axis=0, keepdims=True) + 1e-8)
    fd_z = (fd - fd.mean()) / (fd.std() + 1e-8)

    # corr(fd, pc[:,j]) for each parcel j
    corr = (fd_z[:, None] * pc_z).mean(axis=0)

    rows.append({
        "runTag": r["runTag"],
        "fd_mean": r["qc_fd_mean"],
        "fd_max": r["qc_fd_max"],
        "pct_fd_spikes": r["pct_fd_spikes"],
        "median_abs_corr_fd_pc": float(np.median(np.abs(corr))),
        "p95_abs_corr_fd_pc": float(np.percentile(np.abs(corr), 95)),
        "max_abs_corr_fd_pc": float(np.max(np.abs(corr))),
    })

qc = pd.DataFrame(rows).sort_values(["max_abs_corr_fd_pc","p95_abs_corr_fd_pc"], ascending=False)
from pathlib import Path
out = Path(df.loc[0, "out_npy_pc1"]).parents[1]  # .../parcel_pc1_v6/npy -> parcel_pc1_v6
qc.to_csv(out / "qc_motion_to_pc.csv", index=False)

print(qc.to_string(index=False))
print("\nWrote qc_motion_to_pc.csv")

```

                     runTag  fd_mean    fd_max  pct_fd_spikes  median_abs_corr_fd_pc  p95_abs_corr_fd_pc  max_abs_corr_fd_pc
    sub-16_ses-01_task-rest 0.158516  0.460479       0.000000               0.020583            0.048055            0.063241
    sub-02_ses-01_task-rest 0.126517  0.454937       0.000000               0.013115            0.036776            0.062903
    sub-21_ses-01_task-rest 0.226257  1.581828       0.065972               0.016774            0.040830            0.062788
    sub-14_ses-01_task-rest 0.106983  0.567419       0.003472               0.019359            0.047722            0.056139
    sub-08_ses-02_task-rest 0.236912  1.915868       0.121528               0.013814            0.042127            0.050439
    sub-01_ses-01_task-rest 0.148735  0.542274       0.003472               0.014924            0.036000            0.049499
    sub-18_ses-01_task-rest 0.159986  0.987393       0.020833               0.014122            0.033048            0.041373
    sub-17_ses-02_task-rest 0.110854  0.703244       0.003472               0.008149            0.022824            0.038000
    sub-13_ses-01_task-rest 0.118586  2.275126       0.010417               0.008940            0.028435            0.036381
    sub-03_ses-01_task-rest 0.366118  4.246836       0.222222               0.010080            0.022578            0.035397
    sub-13_ses-02_task-rest 0.129775  2.473213       0.038194               0.009364            0.023539            0.033287
    sub-17_ses-01_task-rest 0.200932  2.748104       0.069444               0.008676            0.019736            0.025856
    sub-20_ses-01_task-rest 0.079698  0.777578       0.010417               0.005231            0.015234            0.019281
    sub-08_ses-01_task-rest 0.167428  1.332223       0.097222               0.004720            0.013427            0.017190
    sub-09_ses-01_task-rest 0.300285 11.917186       0.079861               0.003350            0.009580            0.016910
    
    Wrote qc_motion_to_pc.csv



```python
# ============================================================
# QC: "artifact structure" — fraction of parcels with |z|>3 per TR
# Overlays with FD to catch mass-parcel blowups not captured by FD↔PC corr.
#
# Requires: dataset_index.csv produced by exporter (with out_npy_pc1, confounds_tsv).
# Outputs:
#   - qc_parcel_blowups.csv  (run-level summary)
#   - qc_parcel_blowups_<runTag>.png for each run (time series overlay)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Point this to your exporter output folder that contains dataset_index.csv
OUT_ROOT = Path(r"/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bold_parcel/parcel_pc1_v6")
idx_path = OUT_ROOT / "dataset_index.csv"

Z_THR = 3.0                 # threshold for |z|
PCT_THR_MARK = 0.30         # mark TRs where >=30% parcels exceed |z|>3
SAVE_PLOTS = True

df = pd.read_csv(idx_path)

qc_rows = []
for _, r in df.iterrows():
    runTag = r["runTag"]

    pc = np.load(r["out_npy_pc1"]).astype(np.float32)      # (T, 200)
    T, P = pc.shape

    # z-score each parcel time series across time
    mu = pc.mean(axis=0, keepdims=True)
    sd = pc.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    pc_z = (pc - mu) / sd

    # fraction of parcels exceeding |z|>3 at each TR
    frac = (np.abs(pc_z) > Z_THR).mean(axis=1)             # (T,)

    # load FD for the same run
    conf = pd.read_csv(r["confounds_tsv"], sep="\t")
    fd = conf["framewise_displacement"].fillna(0).to_numpy().astype(np.float32)
    if len(fd) != T:
        raise RuntimeError(f"TR mismatch for {runTag}: pc T={T} vs FD T={len(fd)}")

    # summary metrics
    qc_rows.append({
        "runTag": runTag,
        "T": int(T),
        "P": int(P),
        "max_frac_absz_gt3": float(frac.max()),
        "p95_frac_absz_gt3": float(np.percentile(frac, 95)),
        "mean_frac_absz_gt3": float(frac.mean()),
        "pct_TR_frac_gt_0p30": float((frac >= PCT_THR_MARK).mean()),
        "fd_mean": float(fd.mean()),
        "fd_p95": float(np.percentile(fd, 95)),
        "fd_max": float(fd.max()),
        "pct_fd_gt_0p5": float((fd > 0.5).mean()),
    })

    if SAVE_PLOTS:
        fig_path = OUT_ROOT / "qc" / f"qc_parcel_blowups_{runTag}.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)

        x = np.arange(T)

        plt.figure(figsize=(12, 5))

        ax1 = plt.gca()
        ax1.plot(x, frac)
        ax1.set_ylim(0, 1)
        ax1.set_ylabel(f"Fraction parcels with |z|>{Z_THR}")
        ax1.set_xlabel("TR")
        ax1.set_title(f"{runTag} — parcel blowup fraction (|z|>{Z_THR}) with FD overlay")

        # mark "mass blowup" TRs
        bad = np.where(frac >= PCT_THR_MARK)[0]
        if len(bad) > 0:
            ax1.scatter(bad, frac[bad], marker="x")

        # FD overlay on right axis
        ax2 = ax1.twinx()
        ax2.plot(x, fd)
        ax2.set_ylabel("FD (mm)")

        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()

qc = pd.DataFrame(qc_rows).sort_values(
    ["max_frac_absz_gt3", "p95_frac_absz_gt3", "pct_TR_frac_gt_0p30"],
    ascending=False
).reset_index(drop=True)

# Save + print top offenders
out_csv = OUT_ROOT / "qc" / "qc_parcel_blowups.csv"
qc.to_csv(out_csv, index=False)

print("Wrote:", out_csv)
print(qc.head(10).to_string(index=False))

```

    Wrote: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bold_parcel/parcel_pc1_v6/qc/qc_parcel_blowups.csv
                     runTag   T   P  max_frac_absz_gt3  p95_frac_absz_gt3  mean_frac_absz_gt3  pct_TR_frac_gt_0p30  fd_mean   fd_p95    fd_max  pct_fd_gt_0p5
    sub-09_ses-01_task-rest 288 200              0.200            0.02000            0.004705                  0.0 0.300285 0.551479 11.917186       0.069444
    sub-08_ses-02_task-rest 288 200              0.150            0.04325            0.009115                  0.0 0.236912 0.747938  1.915868       0.121528
    sub-03_ses-01_task-rest 288 200              0.140            0.05825            0.011042                  0.0 0.366118 0.888737  4.246836       0.211806
    sub-17_ses-02_task-rest 288 200              0.130            0.02500            0.005781                  0.0 0.110854 0.213021  0.703244       0.003472
    sub-21_ses-01_task-rest 288 200              0.130            0.02500            0.005642                  0.0 0.226257 0.593136  1.581828       0.065972
    sub-13_ses-01_task-rest 288 200              0.120            0.02000            0.004705                  0.0 0.118586 0.241643  2.275126       0.003472
    sub-17_ses-01_task-rest 288 200              0.110            0.02000            0.004983                  0.0 0.200932 0.685560  2.748104       0.065972
    sub-13_ses-02_task-rest 288 200              0.100            0.02500            0.004497                  0.0 0.129775 0.391650  2.473213       0.034722
    sub-18_ses-01_task-rest 288 200              0.095            0.02000            0.004323                  0.0 0.159986 0.384813  0.987393       0.020833
    sub-20_ses-01_task-rest 288 200              0.080            0.02325            0.004948                  0.0 0.079698 0.238022  0.777578       0.010417



```python
import numpy as np
import pandas as pd

# point to your v6 output dataset_index.csv
df = pd.read_csv(r"/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bold_parcel/parcel_pc1_v6/dataset_index.csv")

RUN = "sub-03_ses-01_task-rest"
Z_THR = 3.0
FRAC_THR = 0.30
FD_THRESHOLD = 0.5
FD_CAT = 2.0

r = df[df["runTag"] == RUN].iloc[0]

pc = np.load(r["out_npy_pc1"]).astype(np.float32)  # (T,200)
conf = pd.read_csv(r["confounds_tsv"], sep="\t")
fd = conf["framewise_displacement"].fillna(0).to_numpy().astype(np.float32)

# z-score each parcel across time
mu = pc.mean(axis=0, keepdims=True)
sd = pc.std(axis=0, keepdims=True)
sd = np.where(sd < 1e-8, 1.0, sd)
pc_z = (pc - mu) / sd

frac = (np.abs(pc_z) > Z_THR).mean(axis=1)

# TRs with mass blowup
bad = np.where(frac >= FRAC_THR)[0]

print("Run:", RUN)
print("Worst TR:", int(frac.argmax()), "max_frac:", float(frac.max()), "FD:", float(fd[int(frac.argmax())]))
print("TRs with frac>=0.30:", bad.tolist())

# Print a small table for the bad TR(s)
if len(bad) > 0:
    out = []
    for t in bad:
        out.append({
            "TR": int(t),
            "frac_absz_gt3": float(frac[t]),
            "FD": float(fd[t]),
            "FD_gt_0p5": bool(fd[t] > FD_THRESHOLD),
            "FD_gt_2p0": bool(fd[t] > FD_CAT),
        })
    print(pd.DataFrame(out).to_string(index=False))

# Also show top-10 TRs by fraction (helps if FRAC_THR misses near-events)
top = np.argsort(-frac)[:10]
out2 = pd.DataFrame({
    "TR": top.astype(int),
    "frac_absz_gt3": frac[top],
    "FD": fd[top],
    "FD_gt_0p5": fd[top] > FD_THRESHOLD,
    "FD_gt_2p0": fd[top] > FD_CAT,
})
print("\nTop-10 TRs by frac_absz_gt3:")
print(out2.to_string(index=False))

```

    Run: sub-03_ses-01_task-rest
    Worst TR: 47 max_frac: 0.14 FD: 0.25061675906181335
    TRs with frac>=0.30: []
    
    Top-10 TRs by frac_absz_gt3:
     TR  frac_absz_gt3       FD  FD_gt_0p5  FD_gt_2p0
     47          0.140 0.250617      False      False
    201          0.140 0.651455       True      False
     37          0.125 0.221472      False      False
    267          0.120 0.306818      False      False
    109          0.100 0.155712      False      False
     79          0.095 0.338390      False      False
    203          0.090 1.141041       True      False
    139          0.090 0.489973      False      False
     15          0.090 0.144206      False      False
     87          0.075 0.252880      False      False



```python
import pandas as pd

conf = pd.read_csv(r"/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/sub-03_ses-01/fmriprep/sub-03/ses-01/func/sub-03_ses-01_task-rest_desc-confounds_timeseries.tsv", sep="\t")
out_cols = [c for c in conf.columns if c.startswith("motion_outlier")]
print("n motion_outlier cols:", len(out_cols))
print("TR 19 outliers:", {c: int(conf.loc[19, c]) for c in out_cols if conf.loc[19, c] == 1})

```

    n motion_outlier cols: 112
    TR 19 outliers: {'motion_outlier02': 1}



```python

```
