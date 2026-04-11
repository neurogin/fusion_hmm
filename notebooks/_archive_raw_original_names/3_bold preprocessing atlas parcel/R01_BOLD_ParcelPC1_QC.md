# R01 rerun — BOLD parcel PC1 QC (reproducibility + sign convention) — v2

This QC recomputes parcel PC1 for a random subset of parcels per run and correlates them with the saved outputs.
**Critical patch:** it uses the exact FD spike TR list recorded in `dataset_index.csv` (`fd_spike_trs`) so the recomputation matches the exporter.



```python

# ========================
# Cell 1 — Configuration
# ========================
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

OUT_ROOT = Path(r"/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bold_parcel/parcel_pc1_v6")
INDEX_CSV = OUT_ROOT / "dataset_index.csv"
QC_DIR = OUT_ROOT / "qc_sign_v6"
QC_DIR.mkdir(parents=True, exist_ok=True)

PARCELS_PER_RUN = 25
CORR_THR = 0.99
RANDOM_SEED = 1
np.random.seed(RANDOM_SEED)

# Mirror exporter key settings (must match your exporter!)
FD_THRESHOLD = 0.5
FD_CAT = 2.0
EXPAND_PRE = 0
EXPAND_POST = 2
BLOCK_MIN_LEN = 3

USE_MOTION24 = True
USE_WM_CSF = True
USE_COSINES = True
ADD_NONSTEADY = True
N_ACOMPCOR = 10

ADD_FD_SPIKES = True
ADD_DVARS_SPIKES = False  # match your final nuisance model

MIN_VOXELS_FOR_PCA = 10
ZSCOR_BEFORE_PCA = True

print("INDEX_CSV:", INDEX_CSV)
print("QC_DIR:", QC_DIR)


```

    INDEX_CSV: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bold_parcel/parcel_pc1_v6/dataset_index.csv
    QC_DIR: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bold_parcel/parcel_pc1_v6/qc_sign_v6



```python

# ========================
# Cell 2 — Helpers (mirror exporter)
# ========================
def safe_read_json(p):
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def safe_corr(a, b):
    a = np.asarray(a).astype(np.float64)
    b = np.asarray(b).astype(np.float64)
    da = a - a.mean()
    db = b - b.mean()
    va = np.sqrt(np.sum(da**2))
    vb = np.sqrt(np.sum(db**2))
    if va == 0 or vb == 0:
        return np.nan
    return float(np.sum(da*db) / (va*vb))

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
    if "framewise_displacement" not in conf.columns:
        return np.zeros(len(conf), dtype=np.float32)
    return conf["framewise_displacement"].fillna(0.0).to_numpy(dtype=np.float32)

def detect_dvars_spikes(conf: pd.DataFrame, zthr: float = 2.5):
    # only for diagnostics in this QC (DVARS spikes are off)
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

def _expand_mask(mask: np.ndarray, pre: int = 0, post: int = 2) -> np.ndarray:
    T = len(mask)
    out = mask.copy()
    idx = np.where(mask)[0]
    for t in idx:
        a = max(0, t - pre)
        b = min(T, t + post + 1)
        out[a:b] = True
    return out

def spikes_to_design(mask: np.ndarray, prefix: str, block_min_len: int):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return np.zeros((mask.shape[0], 0), dtype=np.float32), []

    # contiguous segments
    segs = []
    a = b = idx[0]
    for t in idx[1:]:
        if t == b + 1:
            b = t
        else:
            segs.append((a, b))
            a = b = t
    segs.append((a, b))

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

def _collect_motion_outliers(conf: pd.DataFrame) -> list[str]:
    return sorted([c for c in conf.columns if c.startswith("motion_outlier")])

def build_design_matrix_v6(conf: pd.DataFrame, meta: dict):
    T = len(conf)

    # continuous confounds
    cont_cols = []
    if USE_MOTION24:
        cont_cols += _collect_motion24(conf)
    if USE_WM_CSF:
        cont_cols += _collect_wm_csf(conf)
    if USE_COSINES:
        cont_cols += _collect_cosines(conf)

    cont_cols += _collect_acompcor_retained(conf, meta, N_ACOMPCOR)
    if ADD_NONSTEADY:
        cont_cols += _collect_nonsteady(conf)

    # de-duplicate preserve order
    seen=set()
    cont_cols = [c for c in cont_cols if (c not in seen and not seen.add(c))]

    X_cont = conf[cont_cols].copy() if cont_cols else pd.DataFrame(index=np.arange(T))
    X_cont = X_cont.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    X_parts = []
    names = []

    # z-score continuous
    if X_cont.shape[1] > 0:
        Xc = X_cont.to_numpy(dtype=np.float32)
        mu = Xc.mean(axis=0, keepdims=True)
        sd = Xc.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Xc = (Xc - mu) / sd
        X_parts.append(Xc)
        names += list(X_cont.columns)

    fd = detect_fd(conf)
    fd_base = (fd > FD_THRESHOLD)
    fd_cat = (fd > FD_CAT)
    fd_spikes = fd_base | _expand_mask(fd_cat, pre=EXPAND_PRE, post=EXPAND_POST)

    # diagnostic only
    dvars_spikes, dvars_mode = detect_dvars_spikes(conf, zthr=2.5)

    Xfd = np.zeros((T,0), dtype=np.float32)
    Xmo = np.zeros((T,0), dtype=np.float32)

    if ADD_FD_SPIKES:
        Xfd, nfd = spikes_to_design(fd_spikes, "fd", BLOCK_MIN_LEN)
        X_parts.append(Xfd); names += nfd

    # motion_outlier columns filtered to FD<=FD_THRESHOLD (exactly like exporter)
    motion_out_cols = _collect_motion_outliers(conf)
    motion_out_any = np.zeros(T, dtype=bool)
    if len(motion_out_cols) > 0:
        motmat = conf[motion_out_cols].fillna(0.0).to_numpy(dtype=np.float32)
        motion_out_any = (motmat.sum(axis=1) > 0)

        kept_cols = []
        for c in motion_out_cols:
            idx = np.where(conf[c].fillna(0.0).to_numpy() > 0)[0]
            if len(idx) == 0:
                continue
            if np.all(fd[idx] <= FD_THRESHOLD):
                kept_cols.append(c)

        if len(kept_cols) > 0:
            Xmo = conf[kept_cols].fillna(0.0).to_numpy(dtype=np.float32)
            keep = (Xmo.sum(axis=0) > 0)
            Xmo = Xmo[:, keep]
            kept_cols = [c for c, k in zip(kept_cols, keep) if k]
            if Xmo.shape[1] > 0:
                X_parts.append(Xmo); names += kept_cols

    # intercept
    X_parts.append(np.ones((T,1), dtype=np.float32)); names += ["intercept"]
    X = np.concatenate(X_parts, axis=1)

    qc = dict(
        n_conf_cont=int(X_cont.shape[1]),
        n_fd_units=int(Xfd.shape[1]),
        n_motion_out_units=int(Xmo.shape[1]),
        n_motion_out_cols_total=int(len(motion_out_cols)),
        pct_fd_spikes=float(fd_spikes.mean()),
        pct_motion_out_any_total=float(motion_out_any.mean()),
        pct_motion_out_any_kept=float((Xmo.sum(axis=1)>0).mean()) if Xmo.shape[1] else 0.0,
        dvars_mode=dvars_mode,
        pct_dvars_spikes=float(dvars_spikes.mean()),
        n_total_regressors=int(X.shape[1]),
    )
    return X, names, qc, fd

def regress_out(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X.astype(np.float64), Y.astype(np.float64), rcond=None)
    return (Y - X @ beta).astype(np.float32)

def parcel_pc1_from_voxels(Yp: np.ndarray):
    # Yp: (T, V)
    X = Yp.astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)

    if ZSCOR_BEFORE_PCA:
        sd = X.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)
        X = X / sd

    if X.shape[1] < MIN_VOXELS_FOR_PCA:
        ts = X.mean(axis=1).astype(np.float32)
        return ts

    U, S, _ = np.linalg.svd(X, full_matrices=False)
    pc1 = (U[:,0] * S[0]).astype(np.float32)

    # sign convention: align with mean across voxels
    mean_ts = X.mean(axis=1)
    r = safe_corr(pc1, mean_ts)
    if np.isfinite(r) and r < 0:
        pc1 = -pc1
    return pc1

```


```python

# ========================
# Cell 3 — Run QC
# ========================
index = pd.read_csv(INDEX_CSV)

details = []
summary_rows = []

for _, r in index.iterrows():
    runTag = r["runTag"]

    pc1_path = r["out_npy_pc1"]
    lab_path = r["out_npy_labels"]
    atlas_on_grid = r["atlas_on_grid"]

    if not (Path(pc1_path).exists() and Path(lab_path).exists() and Path(atlas_on_grid).exists()):
        print("Missing outputs for", runTag)
        continue

    pc1_saved = np.load(pc1_path)                 # (T,200)
    labels = np.load(lab_path).astype(int)        # (200,)

    rng = np.random.default_rng(abs(hash(runTag)) % (2**32))
    pick = rng.choice(len(labels), size=min(PARCELS_PER_RUN, len(labels)), replace=False)

    bold_img = nib.load(r["bold_file"])
    mask_img = nib.load(r["brain_mask"])
    bold = bold_img.get_fdata().astype(np.float32)        # (X,Y,Z,T)
    mask = mask_img.get_fdata().astype(np.float32) > 0
    T = bold.shape[3]

    # masked voxels as (T,V)
    vox_idx = np.where(mask.reshape(-1))[0]
    Y = bold.reshape(-1, T)[vox_idx].T

    atlas_img = nib.load(atlas_on_grid)
    atlas_data = atlas_img.get_fdata().astype(np.int32)
    atlas_data[~mask] = 0
    flat_atlas = atlas_data.reshape(-1)[vox_idx]

    conf = pd.read_csv(r["confounds_tsv"], sep="\t")
    meta = safe_read_json(r["confounds_json"]) if isinstance(r["confounds_json"], str) and len(r["confounds_json"]) else {}
    X, xnames, qc_x, fd = build_design_matrix_v6(conf, meta)

    if X.shape[0] != T:
        raise RuntimeError(f"TR mismatch for {runTag}: bold T={T}, conf T={X.shape[0]}")

    Y_res = regress_out(Y, X)

    corr_list = []
    for pidx in pick:
        lab = labels[pidx]
        cols = np.where(flat_atlas == lab)[0]
        if len(cols) < MIN_VOXELS_FOR_PCA:
            continue

        pc1_re = parcel_pc1_from_voxels(Y_res[:, cols])
        pc1_sv = pc1_saved[:, pidx]
        c = safe_corr(pc1_sv, pc1_re)

        corr_list.append(c)
        details.append({
            "runTag": runTag,
            "parcel_index": int(pidx),
            "label": int(lab),
            "n_vox": int(len(cols)),
            "corr": c
        })

    corr_arr = np.array(corr_list, dtype=float)
    n_tested = len(corr_arr)
    n_pass = int(np.sum(corr_arr >= CORR_THR))

    summary_rows.append({
        "runTag": runTag,
        "n_tested": int(n_tested),
        "n_pass": int(n_pass),
        "pass_rate": float(n_pass / n_tested) if n_tested else np.nan,
        "corr_median": float(np.nanmedian(corr_arr)) if n_tested else np.nan,
        "corr_min": float(np.nanmin(corr_arr)) if n_tested else np.nan,
        "corr_p10": float(np.nanquantile(corr_arr, 0.10)) if n_tested else np.nan,
        # also record X sizes for a quick match check
        "n_total_regressors_qc": int(qc_x["n_total_regressors"]),
        "n_fd_units_qc": int(qc_x["n_fd_units"]),
        "n_motion_out_units_qc": int(qc_x["n_motion_out_units"]),
    })

details_df = pd.DataFrame(details)
summary_df = pd.DataFrame(summary_rows).sort_values("runTag").reset_index(drop=True)

details_df.to_csv(QC_DIR / "qc_sign_details.csv", index=False)
summary_df.to_csv(QC_DIR / "qc_sign_summary.csv", index=False)

print("Wrote:", QC_DIR / "qc_sign_details.csv")
print("Wrote:", QC_DIR / "qc_sign_summary.csv")
display(summary_df)

```

    Wrote: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bold_parcel/parcel_pc1_v6/qc_sign_v6/qc_sign_details.csv
    Wrote: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bold_parcel/parcel_pc1_v6/qc_sign_v6/qc_sign_summary.csv



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
      <th>n_tested</th>
      <th>n_pass</th>
      <th>pass_rate</th>
      <th>corr_median</th>
      <th>corr_min</th>
      <th>corr_p10</th>
      <th>n_total_regressors_qc</th>
      <th>n_fd_units_qc</th>
      <th>n_motion_out_units_qc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sub-01_ses-01_task-rest</td>
      <td>25</td>
      <td>25</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>64</td>
      <td>1</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sub-02_ses-01_task-rest</td>
      <td>25</td>
      <td>25</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>59</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sub-03_ses-01_task-rest</td>
      <td>25</td>
      <td>25</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>145</td>
      <td>41</td>
      <td>51</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sub-08_ses-01_task-rest</td>
      <td>25</td>
      <td>25</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>95</td>
      <td>22</td>
      <td>22</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sub-08_ses-02_task-rest</td>
      <td>25</td>
      <td>25</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>126</td>
      <td>19</td>
      <td>56</td>
    </tr>
    <tr>
      <th>5</th>
      <td>sub-09_ses-01_task-rest</td>
      <td>25</td>
      <td>25</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>71</td>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>6</th>
      <td>sub-13_ses-01_task-rest</td>
      <td>25</td>
      <td>25</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>59</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>sub-13_ses-02_task-rest</td>
      <td>25</td>
      <td>25</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>78</td>
      <td>5</td>
      <td>21</td>
    </tr>
    <tr>
      <th>8</th>
      <td>sub-14_ses-01_task-rest</td>
      <td>25</td>
      <td>25</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>55</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>sub-16_ses-01_task-rest</td>
      <td>25</td>
      <td>25</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>55</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>sub-17_ses-01_task-rest</td>
      <td>25</td>
      <td>25</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>81</td>
      <td>9</td>
      <td>20</td>
    </tr>
    <tr>
      <th>11</th>
      <td>sub-17_ses-02_task-rest</td>
      <td>25</td>
      <td>25</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>90</td>
      <td>1</td>
      <td>37</td>
    </tr>
    <tr>
      <th>12</th>
      <td>sub-18_ses-01_task-rest</td>
      <td>25</td>
      <td>25</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>64</td>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>13</th>
      <td>sub-20_ses-01_task-rest</td>
      <td>25</td>
      <td>25</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>89</td>
      <td>3</td>
      <td>33</td>
    </tr>
    <tr>
      <th>14</th>
      <td>sub-21_ses-01_task-rest</td>
      <td>25</td>
      <td>25</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>89</td>
      <td>17</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>



```python

# ========================
# Cell 4 — QC plots (fixes 'Too many bins' on degenerate ranges)
# ========================
details_df = pd.read_csv(QC_DIR / "qc_sign_details.csv")
summary_df = pd.read_csv(QC_DIR / "qc_sign_summary.csv")

vals = details_df["corr"].dropna().values.astype(float)
vals = vals[np.isfinite(vals)]

plt.figure(figsize=(7,4))
if vals.size == 0:
    plt.text(0.5, 0.5, "No finite corr values", ha="center", va="center")
else:
    rmin, rmax = float(vals.min()), float(vals.max())
    if (rmax - rmin) < 1e-9:
        rmin -= 1e-3; rmax += 1e-3
    bins = min(50, max(10, int(np.sqrt(vals.size))))
    plt.hist(vals, bins=bins, range=(rmin, rmax))
plt.title("BOLD parcel PC1 QC — correlation (saved vs recomputed)")
plt.xlabel("corr"); plt.ylabel("count")
plt.tight_layout()
plt.savefig(QC_DIR / "fig_corr_hist.png", dpi=150)
plt.close()

plt.figure(figsize=(10,4))
plt.plot(summary_df["pass_rate"].values, marker="o")
plt.xticks(range(len(summary_df)), summary_df["runTag"].values, rotation=60, ha="right")
plt.ylim(-0.05, 1.05)
plt.title(f"QC pass rate per run (corr >= {CORR_THR})")
plt.ylabel("pass_rate")
plt.tight_layout()
plt.savefig(QC_DIR / "fig_passrate_per_run.png", dpi=150)
plt.close()

print("Saved figures to:", QC_DIR)

```

    Saved figures to: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bold_parcel/parcel_pc1_v6/qc_sign_v6

