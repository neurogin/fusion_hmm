```python
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# User settings
# -----------------------
ROOT = Path(r"/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/eeg_source/parcel_pc1")  # WSL style
# ROOT = Path(r"C:\EEGFMRI\hmm\R01_rerun\02_derivatives\eeg_source\parcel_pc1")   # Windows style

OUT = ROOT / "figs_v3"
OUT.mkdir(parents=True, exist_ok=True)

def _safe_read_csv(p: Path):
    if not p.exists():
        print(f"[WARN] Missing: {p}")
        return None
    return pd.read_csv(p)

def _pick_col(df, preferred, contains=None):
    if preferred in df.columns:
        return preferred
    if contains:
        for c in df.columns:
            if all(s.lower() in c.lower() for s in contains):
                return c
    return None

# -----------------------
# 1) Gain normalization summary (pc1_std_med per run)
# -----------------------
gain_csv = ROOT / "qc_v3" / "qc_run_timeseries_gain_summary.csv"
df = _safe_read_csv(gain_csv)
if df is not None:
    run_col = _pick_col(df, "runTag") or df.columns[0]
    y_col = _pick_col(df, "pc1_std_med", contains=["pc1", "std"]) or df.columns[1]
    df = df.sort_values(run_col)

    plt.figure()
    plt.plot(df[run_col], df[y_col], marker="o")
    plt.xticks(rotation=90)
    plt.ylabel(y_col)
    plt.title("EEG parcel PC1 scale after gnorm (median std)")
    plt.tight_layout()
    plt.savefig(OUT / "fig_gain_pc1_std_med_per_run_v3.png", dpi=200)
    plt.close()
    print("[OK] Wrote fig_gain_pc1_std_med_per_run_v3.png")

# -----------------------
# 2) PVE1 pooled histogram (from binned counts)
# -----------------------
hist_csv = ROOT / "batch_pve1_histogram_v3.csv"
df = _safe_read_csv(hist_csv)
if df is not None and len(df.columns) >= 2:
    # Expect something like: bin_center, count  (or bin_left/bin_right, count)
    x_col = df.columns[0]
    y_col = df.columns[1]

    plt.figure()
    plt.bar(df[x_col], df[y_col], width=0.01)  # default width; OK if bins are fine-grained
    plt.xlabel("PVE1 (PC1 / trace(Cp))")
    plt.ylabel("Count")
    plt.title("PVE1 pooled across runs/parcels")
    plt.tight_layout()
    plt.savefig(OUT / "fig_pve1_histogram_pooled_v3.png", dpi=200)
    plt.close()
    print("[OK] Wrote fig_pve1_histogram_pooled_v3.png")

# -----------------------
# 3) Run-wise PVE1 quantiles (q10/q50/q90)
# -----------------------
quant_csv = ROOT / "batch_pve1_run_quantiles_v3.csv"
df = _safe_read_csv(quant_csv)
if df is not None:
    run_col = _pick_col(df, "runTag") or df.columns[0]
    q10 = _pick_col(df, "q10", contains=["q10"]) or _pick_col(df, "p10", contains=["p10"])
    q50 = _pick_col(df, "q50", contains=["q50"]) or _pick_col(df, "median", contains=["median"])
    q90 = _pick_col(df, "q90", contains=["q90"]) or _pick_col(df, "p90", contains=["p90"])
    if q10 and q50 and q90:
        df = df.sort_values(run_col)
        plt.figure()
        plt.plot(df[run_col], df[q50], marker="o", label="q50")
        plt.plot(df[run_col], df[q10], marker="o", label="q10")
        plt.plot(df[run_col], df[q90], marker="o", label="q90")
        plt.xticks(rotation=90)
        plt.ylabel("PVE1")
        plt.title("Run-wise PVE1 quantiles")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT / "fig_pve1_run_quantiles_v3.png", dpi=200)
        plt.close()
        print("[OK] Wrote fig_pve1_run_quantiles_v3.png")
    else:
        print("[WARN] Could not find q10/q50/q90 columns in run quantiles CSV.")

# -----------------------
# 4) Low-PVE parcels frequency (top 20, named)
# -----------------------
lowfreq_csv = ROOT / "batch_pve1_lowparcels_frequency_named_v3.csv"
df = _safe_read_csv(lowfreq_csv)
if df is not None and len(df.columns) >= 2:
    # Expect: parcelName, frequency (or count)
    name_col = df.columns[0]
    freq_col = df.columns[1]
    df = df.sort_values(freq_col, ascending=False).head(20)

    plt.figure()
    plt.bar(df[name_col].astype(str), df[freq_col])
    plt.xticks(rotation=90)
    plt.ylabel(freq_col)
    plt.title("Top-20 parcels appearing in bottom-PVE set (frequency)")
    plt.tight_layout()
    plt.savefig(OUT / "fig_lowpve_frequency_top20_v3.png", dpi=200)
    plt.close()
    print("[OK] Wrote fig_lowpve_frequency_top20_v3.png")

# -----------------------
# 5) Sign QC pass rate per run (optional)
# -----------------------
sign_csv = ROOT / "qc_v3_sign" / "qc_sign_v3_summary.csv"
df = _safe_read_csv(sign_csv)
if df is not None:
    run_col = _pick_col(df, "runTag") or df.columns[0]
    pass_col = _pick_col(df, "PassRate", contains=["pass"]) or _pick_col(df, "pass_rate", contains=["pass"])
    if pass_col:
        df = df.sort_values(run_col)
        plt.figure()
        plt.plot(df[run_col], df[pass_col], marker="o")
        plt.xticks(rotation=90)
        plt.ylim(-0.05, 1.05)
        plt.ylabel("Pass rate")
        plt.title("Sign convention QC pass rate per run")
        plt.tight_layout()
        plt.savefig(OUT / "fig_sign_qc_passrate_v3.png", dpi=200)
        plt.close()
        print("[OK] Wrote fig_sign_qc_passrate_v3.png")
    else:
        print("[WARN] Could not find pass-rate column in sign QC summary CSV.")

print(f"\nDone. Figures in: {OUT}")

```

    [OK] Wrote fig_gain_pc1_std_med_per_run_v3.png
    [OK] Wrote fig_pve1_histogram_pooled_v3.png
    [OK] Wrote fig_pve1_run_quantiles_v3.png
    [OK] Wrote fig_lowpve_frequency_top20_v3.png
    [OK] Wrote fig_sign_qc_passrate_v3.png
    
    Done. Figures in: /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/eeg_source/parcel_pc1/figs_v3



```python
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import h5py

OUTDIR = Path("/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/eeg_source/parcel_pc1")
MIN_DIPOLES = 40
OUT = OUTDIR / "figs_tables"
OUT.mkdir(parents=True, exist_ok=True)

mat_files = sorted(OUTDIR.glob("*_parcelPC.mat"))
print(f"Found {len(mat_files)} MAT files.")
print("Example:", mat_files[0].name)

# ---------- helpers ----------
def _read_dataset(h5, key):
    if key not in h5:
        return None
    obj = h5[key]
    if isinstance(obj, h5py.Dataset):
        return obj[()]
    return None

def _as1d(x):
    return np.asarray(x).squeeze()

def _read_logical(x):
    a = np.asarray(x).squeeze()
    if a.dtype == bool:
        return a
    return a.astype(np.int8) != 0

def _read_matlab_char(arr):
    a = np.asarray(arr).squeeze()
    if a.dtype == np.uint16:
        return "".join(chr(int(c)) for c in a if int(c) != 0).strip()
    if a.dtype.kind == "S":
        try:
            return b"".join(a.tolist()).decode("utf-8", errors="ignore").strip()
        except Exception:
            return str(a)
    return str(a)

def _try_read_cellstr_refs(h5, ds):
    """
    Attempt to read a dataset of HDF5 references -> list[str]
    """
    data = ds[()]
    data = np.asarray(data).squeeze()
    out = []
    for ref in np.ravel(data):
        try:
            targ = h5[ref]
            out.append(_read_matlab_char(targ[()]))
        except Exception:
            out.append("")
    return out

def _read_parcel_names(h5, n_expected, debug=False, file_label=""):
    """
    Robust reader for parcel_names in MATLAB v7.3.
    Returns list[str] length n_expected OR None.
    """
    if "parcel_names" not in h5:
        return None

    ds = h5["parcel_names"]
    if not isinstance(ds, h5py.Dataset):
        return None

    # Debug: show dtype/shape once for the first file
    if debug:
        print(f"[DEBUG] {file_label} parcel_names: shape={ds.shape}, dtype={ds.dtype}")

    # Case 1: dataset of references (classic cell array)
    # h5py reports ref dtype as object/ref-like; safest is to try reading refs if possible
    try:
        if "ref" in str(ds.dtype).lower() or ds.dtype == h5py.ref_dtype:
            names = _try_read_cellstr_refs(h5, ds)
            return names
    except Exception:
        pass

    # Case 2: MATLAB string array sometimes stored as uint16 char matrix or similar
    raw = ds[()]
    raw = np.asarray(raw)

    # If it looks like a char matrix, convert to a single string and then split is not reliable
    # So: treat as one string (fallback) and let caller broadcast blanks
    try:
        s = _read_matlab_char(raw)
        # If somehow it contains line breaks and equals n_expected lines, use that
        parts = [p for p in s.splitlines() if p.strip() != ""]
        if len(parts) == n_expected:
            return parts
        # Otherwise return single-item list
        return [s]
    except Exception:
        return None

def q(x, p):
    return float(np.nanquantile(np.asarray(x, dtype=float), p))

# ---------- load rows ----------
rows = []

for k, f in enumerate(mat_files):
    runTag = f.name.replace("_parcelPC.mat", "")
    with h5py.File(f, "r") as h5:
        parcel_ids = _as1d(_read_dataset(h5, "parcel_ids")).astype(int)
        n_dipoles  = _as1d(_read_dataset(h5, "n_dipoles")).astype(float)
        pve1       = _as1d(_read_dataset(h5, "PVE1")).astype(float)
        vmask      = _read_logical(_read_dataset(h5, "valid_parcel_mask"))

        nP = len(parcel_ids)

        # Only print debug for first file so you can see how parcel_names is stored
        debug = (k == 0)
        parcel_names = _read_parcel_names(h5, nP, debug=debug, file_label=f.name)

        # Normalize names to length nP
        if parcel_names is None:
            parcel_names = [""] * nP
        else:
            if len(parcel_names) != nP:
                # Most likely len=1 due to MATLAB string encoding; do not crash.
                # Use blanks (or broadcast the single name if you prefer).
                if debug:
                    print(f"[DEBUG] {f.name}: parcel_names length={len(parcel_names)}; forcing blank names.")
                parcel_names = [""] * nP

        # Sanity: numeric lengths must match
        if not (len(n_dipoles) == len(pve1) == len(vmask) == nP):
            raise ValueError(
                f"Numeric length mismatch in {f.name}: "
                f"ids={nP} dip={len(n_dipoles)} pve1={len(pve1)} mask={len(vmask)}"
            )

        for i in range(nP):
            rows.append({
                "runTag": runTag,
                "parcel_id": int(parcel_ids[i]),
                "parcel_name": str(parcel_names[i]),
                "n_dipoles": float(n_dipoles[i]) if np.isfinite(n_dipoles[i]) else np.nan,
                "PVE1": float(pve1[i]) if np.isfinite(pve1[i]) else np.nan,
                "valid": bool(vmask[i]),
            })

df = pd.DataFrame(rows)
dfv = df[df["valid"]].copy()

print(f"Total points (run×parcel): {len(df)}")
print(f"Valid points: {len(dfv)}")

# =========================
# TABLE: per-run dipole support summary
# =========================
run_summary = (
    df.groupby("runTag")
      .apply(lambda g: pd.Series({
          "n_total_parcels": g.shape[0],
          "n_valid_parcels": int(g["valid"].sum()),
          "median_n_dipoles_valid": float(np.nanmedian(g.loc[g["valid"], "n_dipoles"])),
          "q10_n_dipoles_valid": q(g.loc[g["valid"], "n_dipoles"], 0.10),
          "q90_n_dipoles_valid": q(g.loc[g["valid"], "n_dipoles"], 0.90),
          "min_n_dipoles_valid": float(np.nanmin(g.loc[g["valid"], "n_dipoles"])),
          "max_n_dipoles_valid": float(np.nanmax(g.loc[g["valid"], "n_dipoles"])),
          "frac_parcels_below_MinDipoles": float((g["n_dipoles"] < MIN_DIPOLES).mean()),
      }))
      .reset_index()
      .sort_values("runTag")
)

t_run = OUT / "table_run_dipole_support_summary.csv"
run_summary.to_csv(t_run, index=False)
print("Wrote:", t_run)

# =========================
# FIG: dipole support distribution by run (median ± [q10,q90])
# =========================
x = np.arange(len(run_summary))
med = run_summary["median_n_dipoles_valid"].to_numpy()
q10v = run_summary["q10_n_dipoles_valid"].to_numpy()
q90v = run_summary["q90_n_dipoles_valid"].to_numpy()
yerr = np.vstack([med - q10v, q90v - med])

plt.figure(figsize=(12, 4))
plt.errorbar(x, med, yerr=yerr, fmt="o", capsize=3)
plt.axhline(MIN_DIPOLES, linestyle="--", linewidth=1)
plt.xticks(x, run_summary["runTag"], rotation=60, ha="right", fontsize=8)
plt.ylabel("n_dipoles per parcel (valid parcels): median ± [q10,q90]")
plt.title(f"Per-run dipole support distribution (MinDipoles={MIN_DIPOLES})")
plt.grid(True, axis="y", linestyle=":", linewidth=0.5)
plt.tight_layout()
f_dip = OUT / "fig_dipole_support_by_run.png"
plt.savefig(f_dip, dpi=200)
plt.close()
print("Wrote:", f_dip)

# =========================
# FIG: support vs PVE1 scatter (valid parcels only)
# =========================
def spearman_corr(a, b):
    a = np.asarray(a); b = np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    ra = pd.Series(a[m]).rank(method="average").to_numpy()
    rb = pd.Series(b[m]).rank(method="average").to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])

rho = spearman_corr(dfv["n_dipoles"].to_numpy(), dfv["PVE1"].to_numpy())

plt.figure(figsize=(6, 5))
plt.scatter(dfv["n_dipoles"], dfv["PVE1"], s=12)
plt.xlabel("n_dipoles (parcel support; valid parcels)")
plt.ylabel("PVE1 (PC1 explained variance fraction)")
plt.title(f"Support vs PVE1 (valid parcels). Spearman ρ={rho:.3f}")
plt.grid(True, linestyle=":", linewidth=0.5)
plt.tight_layout()
f_sc = OUT / "fig_support_vs_pve1_scatter.png"
plt.savefig(f_sc, dpi=200)
plt.close()
print("Wrote:", f_sc)

# =========================
# TABLE: per-parcel pooled summary across runs (valid-only)
# =========================
per_parcel = (
    dfv.groupby(["parcel_id", "parcel_name"])
       .apply(lambda g: pd.Series({
           "n_runs_contributed": int(g["runTag"].nunique()),
           "median_n_dipoles": float(np.nanmedian(g["n_dipoles"])),
           "q10_n_dipoles": q(g["n_dipoles"], 0.10),
           "min_n_dipoles": float(np.nanmin(g["n_dipoles"])),
           "median_PVE1": float(np.nanmedian(g["PVE1"])),
           "q10_PVE1": q(g["PVE1"], 0.10),
           "min_PVE1": float(np.nanmin(g["PVE1"])),
       }))
       .reset_index()
       .sort_values("parcel_id")
)

t_par = OUT / "table_parcel_support_pve1_summary.csv"
per_parcel.to_csv(t_par, index=False)
print("Wrote:", t_par)

print("\nDONE.")

```


```python
from pathlib import Path
import pandas as pd
import numpy as np

OUTDIR = Path("/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/eeg_source/parcel_pc1")
FIGDIR = OUTDIR / "figs_tables"

# TemplateFlow TSV you already confirmed exists:
TSV = Path("/mnt/c/EEGFMRI_PIPELINE/templateflow/tpl-MNI152NLin2009cAsym/"
           "tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-200Parcels7Networks_dseg.tsv")

if not TSV.exists():
    raise FileNotFoundError(f"Missing TSV: {TSV}")

T = pd.read_csv(TSV, sep="\t")

# Robustly pick the ID and name columns
# TemplateFlow dseg.tsv typically has an integer index column plus a label/name column.
id_col = None
for c in T.columns:
    if c.lower() in ("index", "id", "label", "parcel_id"):
        id_col = c
        break
if id_col is None:
    # fallback: assume first column is the ID
    id_col = T.columns[0]

name_col = None
for c in T.columns:
    if c.lower() in ("name", "label", "region", "roi"):
        name_col = c
        break
if name_col is None:
    # fallback: use the second column
    name_col = T.columns[1] if len(T.columns) > 1 else T.columns[0]

name_map = (
    T[[id_col, name_col]]
    .dropna()
    .assign(**{id_col: T[id_col].astype(int)})
    .rename(columns={id_col: "parcel_id", name_col: "parcel_name"})
)

# Patch table_parcel_support_pve1_summary.csv
par = pd.read_csv(FIGDIR / "table_parcel_support_pve1_summary.csv")
par = par.drop(columns=[c for c in par.columns if c == "parcel_name"], errors="ignore")
par = par.merge(name_map, on="parcel_id", how="left")
par["parcel_name"] = par["parcel_name"].fillna("")
par.to_csv(FIGDIR / "table_parcel_support_pve1_summary.csv", index=False)

print("Patched:", FIGDIR / "table_parcel_support_pve1_summary.csv")
print("Example names:", par[["parcel_id", "parcel_name"]].head(5).to_string(index=False))

```


```python

```
