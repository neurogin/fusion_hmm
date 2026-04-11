# R01 — BOLD atlas registration to fMRIPrep BOLD grid (MNI152NLin2009cAsym)

**Purpose**
Create per-run label atlases on the *exact voxel grid* used by each run's fMRIPrep BOLD outputs, so parcel extraction is voxel-index consistent and reproducible.

**Inputs**
- Canonical atlas (Schaefer2018 200 parcels / 7 networks) in MNI152NLin2009cAsym
- fMRIPrep BOLD reference images (`*_space-MNI152NLin2009cAsym_*boldref.nii.gz`)
- Optional: fMRIPrep brain masks (`*_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz`)
- Optional: BIDS raw JSON sidecars for metadata logging (TR, TE, etc.)

**Outputs**
For each runTag:
- `<runTag>_space-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-200Parcels7Networks_dseg_onbold.nii.gz`
- Optional overlay PNGs for QC
And a global QC CSV:
- `qc_atlas_on_boldgrid_summary.csv`

**Reproducibility notes**
- Label resampling uses nearest-neighbor interpolation (required for discrete labels).
- This notebook is safe to rerun; set OVERWRITE to control behavior.



```python
# Cell 1 — Config (edit paths once)

from pathlib import Path

# -------- EDIT THESE --------
FMRI_ROOT = Path(r"/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids")   # where fmriprep outputs live under sub-*/ses-*/func
OUT_ROOT  = Path(r"/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bold_parcel/atlas_on_boldgrid")

ATLAS_NII = Path(r"/mnt/c/EEGFMRI_PIPELINE/templateflow/tpl-MNI152NLin2009cAsym"
                 r"/tpl-MNI152NLin2009cAsym_res-01_atlas-Schaefer2018_desc-200Parcels7Networks_dseg.nii.gz")
ATLAS_TSV = Path(str(ATLAS_NII).replace("_dseg.nii.gz", "_dseg.tsv"))

SPACE_TAG = "space-MNI152NLin2009cAsym"
OVERWRITE = True

# Optional: if your raw BIDS JSONs are elsewhere, set this; if None, notebook will try to find them under FMRI_ROOT
RAW_BIDS_ROOT = None  # e.g., Path(r"C:\EEGFMRI\hmm\R01_rerun\01_raw\fmri_bids")

```


```python
# Cell 2 — Imports + environment checks

import re
import json
import numpy as np
import pandas as pd
import nibabel as nib

from nilearn.image import resample_to_img
from nilearn.plotting import plot_roi
import matplotlib.pyplot as plt

def assert_exists(p: Path, label: str):
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")

assert_exists(ATLAS_NII, "ATLAS_NII")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

print("OK: atlas and output directories are accessible.")

```

    OK: atlas and output directories are accessible.



```python
# Cell 3 — Find BOLD references (robust search)

def find_boldrefs(root: Path) -> list[Path]:
    patterns = [
        f"**/func/*_{SPACE_TAG}_boldref.nii.gz",
        f"**/func/*_{SPACE_TAG}_desc-coreg_boldref.nii.gz",
        f"**/func/*_{SPACE_TAG}_desc-preproc_boldref.nii.gz",
    ]
    found = []
    for pat in patterns:
        found.extend(root.glob(pat))
    return sorted(set(found))

boldrefs = find_boldrefs(FMRI_ROOT)
if not boldrefs:
    raise RuntimeError(f"No boldref files found under {FMRI_ROOT} for {SPACE_TAG}.")

print(f"Found {len(boldrefs)} boldref files.")
boldrefs[:3]

```

    Found 15 boldref files.





    [PosixPath('/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/sub-01_ses-01/fmriprep/sub-01/ses-01/func/sub-01_ses-01_task-rest_space-MNI152NLin2009cAsym_boldref.nii.gz'),
     PosixPath('/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/sub-02_ses-01/fmriprep/sub-02/ses-01/func/sub-02_ses-01_task-rest_space-MNI152NLin2009cAsym_boldref.nii.gz'),
     PosixPath('/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/sub-03_ses-01/fmriprep/sub-03/ses-01/func/sub-03_ses-01_task-rest_space-MNI152NLin2009cAsym_boldref.nii.gz')]




```python
# Cell 4 — Helpers: runTag parsing, matching mask, optional raw JSON metadata

def infer_runTag_from_boldref(fname: str) -> str:
    """
    Keep BIDS entities up to (but not including) space/desc.
    Example:
      sub-01_ses-01_task-rest_run-01_space-..._boldref.nii.gz
      -> sub-01_ses-01_task-rest_run-01
    """
    m = re.match(r"^(sub-[^_]+_ses-[^_]+(?:_task-[^_]+)?(?:_run-[^_]+)?)_.*$", fname)
    if m:
        return m.group(1)
    # fallback: strip from _space- onwards if present
    return fname.split("_" + SPACE_TAG)[0].replace(".nii.gz","")

def find_mask_for_boldref(boldref: Path) -> Path | None:
    folder = boldref.parent
    stem = boldref.name
    cand = folder / stem.replace("_boldref.nii.gz", "_desc-brain_mask.nii.gz")
    if cand.exists():
        return cand
    # fallback
    prefix = stem.split("_" + SPACE_TAG)[0]
    masks = sorted(folder.glob(f"{prefix}_{SPACE_TAG}_desc-brain_mask.nii.gz"))
    return masks[0] if masks else None

def find_raw_json_for_run(runTag: str) -> Path | None:
    """
    Try to locate the raw BIDS JSON sidecar:
      sub-01_ses-01_task-rest_bold.json  (run may or may not be included)
    We search under RAW_BIDS_ROOT (if set) else FMRI_ROOT.
    """
    root = RAW_BIDS_ROOT if RAW_BIDS_ROOT is not None else FMRI_ROOT
    sub = re.search(r"(sub-[^_]+)", runTag)
    ses = re.search(r"(ses-[^_]+)", runTag)
    task = re.search(r"(task-[^_]+)", runTag)
    run = re.search(r"(run-[^_]+)", runTag)

    if not sub or not ses:
        return None

    # Build plausible JSON patterns
    base = f"{sub.group(1)}_{ses.group(1)}"
    pats = []
    if task:
        pats.append(f"**/func/{base}_{task.group(1)}_bold.json")
        if run:
            pats.append(f"**/func/{base}_{task.group(1)}_{run.group(1)}_bold.json")
    else:
        pats.append(f"**/func/{base}_bold.json")

    for pat in pats:
        hits = list(Path(root).glob(pat))
        if hits:
            return sorted(hits)[0]
    return None

def load_json_fields(jpath: Path | None) -> dict:
    if jpath is None or not jpath.exists():
        return {}
    with open(jpath, "r", encoding="utf-8") as f:
        d = json.load(f)
    # keep only a small set for provenance
    keep = ["TaskName","RepetitionTime","EchoTime","PhaseEncodingDirection","TotalReadoutTime","SliceTiming"]
    return {k: d.get(k, None) for k in keep}

```


```python
# Cell 5 — Resample atlas to each boldref grid + write outputs + QC summary

atlas_img = nib.load(str(ATLAS_NII))
atlas_labels = np.unique(atlas_img.get_fdata()).astype(int)
atlas_labels = atlas_labels[atlas_labels != 0]

rows = []

for boldref in boldrefs:
    runTag = infer_runTag_from_boldref(boldref.name)
    out_dir = OUT_ROOT / runTag
    out_dir.mkdir(parents=True, exist_ok=True)

    out_atlas = out_dir / f"{runTag}_{SPACE_TAG}_atlas-Schaefer2018_desc-200Parcels7Networks_dseg_onbold.nii.gz"
    out_png   = out_dir / f"{runTag}_{SPACE_TAG}_atlas_overlay.png"

    if out_atlas.exists() and not OVERWRITE:
        print(f"[SKIP] {runTag} (exists)")
        continue

    mask = find_mask_for_boldref(boldref)
    boldref_img = nib.load(str(boldref))

    # Resample labels -> boldref grid (nearest-neighbor)
    atlas_on_grid = resample_to_img(atlas_img, boldref_img, interpolation="nearest")
    atlas_data = atlas_on_grid.get_fdata().astype(np.int32)

    # Apply brain mask if available
    if mask is not None and mask.exists():
        mask_data = (nib.load(str(mask)).get_fdata() > 0).astype(np.uint8)
        atlas_data = atlas_data * mask_data

    # Save
    nib.save(nib.Nifti1Image(atlas_data, boldref_img.affine, boldref_img.header), str(out_atlas))

    # Copy TSV for convenience
    if ATLAS_TSV.exists():
        out_tsv = out_dir / ATLAS_TSV.name
        if not out_tsv.exists() or OVERWRITE:
            out_tsv.write_bytes(ATLAS_TSV.read_bytes())

    # QC: label survival
    present = np.unique(atlas_data)
    present = present[present != 0]
    missing = np.setdiff1d(atlas_labels, present)

    # Optional: quick overlay PNG (atlas contours on boldref)
    try:
        disp = plot_roi(nib.load(str(out_atlas)), bg_img=boldref_img, title=runTag, display_mode="ortho", cut_coords=(0,0,0))
        disp.savefig(str(out_png))
        disp.close()
    except Exception as e:
        print(f"[WARN] overlay failed for {runTag}: {e}")

    # Optional raw JSON metadata
    raw_json = find_raw_json_for_run(runTag)
    meta = load_json_fields(raw_json)

    rows.append({
        "runTag": runTag,
        "boldref": str(boldref),
        "mask": str(mask) if mask is not None else "",
        "raw_json": str(raw_json) if raw_json is not None else "",
        **meta,
        "n_labels_expected": int(len(atlas_labels)),
        "n_labels_present": int(len(present)),
        "n_labels_missing": int(len(missing)),
        "missing_labels_head": " ".join(map(str, missing.tolist()[:30])) + (" ..." if len(missing) > 30 else ""),
        "out_atlas": str(out_atlas),
        "out_overlay_png": str(out_png) if out_png.exists() else "",
    })

    print(f"[OK] {runTag}: {len(present)}/{len(atlas_labels)} labels -> {out_atlas.name}")

qc = pd.DataFrame(rows).sort_values("runTag")
qc_csv = OUT_ROOT / "qc_atlas_on_boldgrid_summary.csv"
qc.to_csv(qc_csv, index=False)

qc.head(), qc_csv

```

    /tmp/ipykernel_1556/1496662992.py:25: FutureWarning: 'force_resample' will be set to 'True' by default in Nilearn 0.13.0.
    Use 'force_resample=True' to suppress this warning.
      atlas_on_grid = resample_to_img(atlas_img, boldref_img, interpolation="nearest")
    /tmp/ipykernel_1556/1496662992.py:25: FutureWarning: From release 0.13.0 onwards, this function will, by default, copy the header of the input image to the output. Currently, the header is reset to the default Nifti1Header. To suppress this warning and use the new behavior, set `copy_header=True`.
      atlas_on_grid = resample_to_img(atlas_img, boldref_img, interpolation="nearest")


    [OK] sub-01_ses-01_task-rest: 200/200 labels -> sub-01_ses-01_task-rest_space-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-200Parcels7Networks_dseg_onbold.nii.gz


    /tmp/ipykernel_1556/1496662992.py:25: FutureWarning: 'force_resample' will be set to 'True' by default in Nilearn 0.13.0.
    Use 'force_resample=True' to suppress this warning.
      atlas_on_grid = resample_to_img(atlas_img, boldref_img, interpolation="nearest")
    /tmp/ipykernel_1556/1496662992.py:25: FutureWarning: From release 0.13.0 onwards, this function will, by default, copy the header of the input image to the output. Currently, the header is reset to the default Nifti1Header. To suppress this warning and use the new behavior, set `copy_header=True`.
      atlas_on_grid = resample_to_img(atlas_img, boldref_img, interpolation="nearest")


    [OK] sub-02_ses-01_task-rest: 200/200 labels -> sub-02_ses-01_task-rest_space-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-200Parcels7Networks_dseg_onbold.nii.gz


    /tmp/ipykernel_1556/1496662992.py:25: FutureWarning: 'force_resample' will be set to 'True' by default in Nilearn 0.13.0.
    Use 'force_resample=True' to suppress this warning.
      atlas_on_grid = resample_to_img(atlas_img, boldref_img, interpolation="nearest")
    /tmp/ipykernel_1556/1496662992.py:25: FutureWarning: From release 0.13.0 onwards, this function will, by default, copy the header of the input image to the output. Currently, the header is reset to the default Nifti1Header. To suppress this warning and use the new behavior, set `copy_header=True`.
      atlas_on_grid = resample_to_img(atlas_img, boldref_img, interpolation="nearest")


    [OK] sub-03_ses-01_task-rest: 200/200 labels -> sub-03_ses-01_task-rest_space-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-200Parcels7Networks_dseg_onbold.nii.gz


    /tmp/ipykernel_1556/1496662992.py:25: FutureWarning: 'force_resample' will be set to 'True' by default in Nilearn 0.13.0.
    Use 'force_resample=True' to suppress this warning.
      atlas_on_grid = resample_to_img(atlas_img, boldref_img, interpolation="nearest")
    /tmp/ipykernel_1556/1496662992.py:25: FutureWarning: From release 0.13.0 onwards, this function will, by default, copy the header of the input image to the output. Currently, the header is reset to the default Nifti1Header. To suppress this warning and use the new behavior, set `copy_header=True`.
      atlas_on_grid = resample_to_img(atlas_img, boldref_img, interpolation="nearest")


    [OK] sub-08_ses-01_task-rest: 200/200 labels -> sub-08_ses-01_task-rest_space-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-200Parcels7Networks_dseg_onbold.nii.gz


    /tmp/ipykernel_1556/1496662992.py:25: FutureWarning: 'force_resample' will be set to 'True' by default in Nilearn 0.13.0.
    Use 'force_resample=True' to suppress this warning.
      atlas_on_grid = resample_to_img(atlas_img, boldref_img, interpolation="nearest")
    /tmp/ipykernel_1556/1496662992.py:25: FutureWarning: From release 0.13.0 onwards, this function will, by default, copy the header of the input image to the output. Currently, the header is reset to the default Nifti1Header. To suppress this warning and use the new behavior, set `copy_header=True`.
      atlas_on_grid = resample_to_img(atlas_img, boldref_img, interpolation="nearest")


    [OK] sub-08_ses-02_task-rest: 200/200 labels -> sub-08_ses-02_task-rest_space-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-200Parcels7Networks_dseg_onbold.nii.gz


    /tmp/ipykernel_1556/1496662992.py:25: FutureWarning: 'force_resample' will be set to 'True' by default in Nilearn 0.13.0.
    Use 'force_resample=True' to suppress this warning.
      atlas_on_grid = resample_to_img(atlas_img, boldref_img, interpolation="nearest")
    /tmp/ipykernel_1556/1496662992.py:25: FutureWarning: From release 0.13.0 onwards, this function will, by default, copy the header of the input image to the output. Currently, the header is reset to the default Nifti1Header. To suppress this warning and use the new behavior, set `copy_header=True`.
      atlas_on_grid = resample_to_img(atlas_img, boldref_img, interpolation="nearest")


    [OK] sub-09_ses-01_task-rest: 200/200 labels -> sub-09_ses-01_task-rest_space-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-200Parcels7Networks_dseg_onbold.nii.gz


    /tmp/ipykernel_1556/1496662992.py:25: FutureWarning: 'force_resample' will be set to 'True' by default in Nilearn 0.13.0.
    Use 'force_resample=True' to suppress this warning.
      atlas_on_grid = resample_to_img(atlas_img, boldref_img, interpolation="nearest")
    /tmp/ipykernel_1556/1496662992.py:25: FutureWarning: From release 0.13.0 onwards, this function will, by default, copy the header of the input image to the output. Currently, the header is reset to the default Nifti1Header. To suppress this warning and use the new behavior, set `copy_header=True`.
      atlas_on_grid = resample_to_img(atlas_img, boldref_img, interpolation="nearest")


    [OK] sub-13_ses-01_task-rest: 200/200 labels -> sub-13_ses-01_task-rest_space-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-200Parcels7Networks_dseg_onbold.nii.gz


    /tmp/ipykernel_1556/1496662992.py:25: FutureWarning: 'force_resample' will be set to 'True' by default in Nilearn 0.13.0.
    Use 'force_resample=True' to suppress this warning.
      atlas_on_grid = resample_to_img(atlas_img, boldref_img, interpolation="nearest")
    /tmp/ipykernel_1556/1496662992.py:25: FutureWarning: From release 0.13.0 onwards, this function will, by default, copy the header of the input image to the output. Currently, the header is reset to the default Nifti1Header. To suppress this warning and use the new behavior, set `copy_header=True`.
      atlas_on_grid = resample_to_img(atlas_img, boldref_img, interpolation="nearest")


    [OK] sub-13_ses-02_task-rest: 200/200 labels -> sub-13_ses-02_task-rest_space-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-200Parcels7Networks_dseg_onbold.nii.gz


    /tmp/ipykernel_1556/1496662992.py:25: FutureWarning: 'force_resample' will be set to 'True' by default in Nilearn 0.13.0.
    Use 'force_resample=True' to suppress this warning.
      atlas_on_grid = resample_to_img(atlas_img, boldref_img, interpolation="nearest")
    /tmp/ipykernel_1556/1496662992.py:25: FutureWarning: From release 0.13.0 onwards, this function will, by default, copy the header of the input image to the output. Currently, the header is reset to the default Nifti1Header. To suppress this warning and use the new behavior, set `copy_header=True`.
      atlas_on_grid = resample_to_img(atlas_img, boldref_img, interpolation="nearest")


    [OK] sub-14_ses-01_task-rest: 200/200 labels -> sub-14_ses-01_task-rest_space-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-200Parcels7Networks_dseg_onbold.nii.gz


    /tmp/ipykernel_1556/1496662992.py:25: FutureWarning: 'force_resample' will be set to 'True' by default in Nilearn 0.13.0.
    Use 'force_resample=True' to suppress this warning.
      atlas_on_grid = resample_to_img(atlas_img, boldref_img, interpolation="nearest")
    /tmp/ipykernel_1556/1496662992.py:25: FutureWarning: From release 0.13.0 onwards, this function will, by default, copy the header of the input image to the output. Currently, the header is reset to the default Nifti1Header. To suppress this warning and use the new behavior, set `copy_header=True`.
      atlas_on_grid = resample_to_img(atlas_img, boldref_img, interpolation="nearest")


    [OK] sub-16_ses-01_task-rest: 200/200 labels -> sub-16_ses-01_task-rest_space-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-200Parcels7Networks_dseg_onbold.nii.gz


    /tmp/ipykernel_1556/1496662992.py:25: FutureWarning: 'force_resample' will be set to 'True' by default in Nilearn 0.13.0.
    Use 'force_resample=True' to suppress this warning.
      atlas_on_grid = resample_to_img(atlas_img, boldref_img, interpolation="nearest")
    /tmp/ipykernel_1556/1496662992.py:25: FutureWarning: From release 0.13.0 onwards, this function will, by default, copy the header of the input image to the output. Currently, the header is reset to the default Nifti1Header. To suppress this warning and use the new behavior, set `copy_header=True`.
      atlas_on_grid = resample_to_img(atlas_img, boldref_img, interpolation="nearest")


    [OK] sub-17_ses-01_task-rest: 200/200 labels -> sub-17_ses-01_task-rest_space-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-200Parcels7Networks_dseg_onbold.nii.gz


    /tmp/ipykernel_1556/1496662992.py:25: FutureWarning: 'force_resample' will be set to 'True' by default in Nilearn 0.13.0.
    Use 'force_resample=True' to suppress this warning.
      atlas_on_grid = resample_to_img(atlas_img, boldref_img, interpolation="nearest")
    /tmp/ipykernel_1556/1496662992.py:25: FutureWarning: From release 0.13.0 onwards, this function will, by default, copy the header of the input image to the output. Currently, the header is reset to the default Nifti1Header. To suppress this warning and use the new behavior, set `copy_header=True`.
      atlas_on_grid = resample_to_img(atlas_img, boldref_img, interpolation="nearest")


    [OK] sub-17_ses-02_task-rest: 200/200 labels -> sub-17_ses-02_task-rest_space-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-200Parcels7Networks_dseg_onbold.nii.gz


    /tmp/ipykernel_1556/1496662992.py:25: FutureWarning: 'force_resample' will be set to 'True' by default in Nilearn 0.13.0.
    Use 'force_resample=True' to suppress this warning.
      atlas_on_grid = resample_to_img(atlas_img, boldref_img, interpolation="nearest")
    /tmp/ipykernel_1556/1496662992.py:25: FutureWarning: From release 0.13.0 onwards, this function will, by default, copy the header of the input image to the output. Currently, the header is reset to the default Nifti1Header. To suppress this warning and use the new behavior, set `copy_header=True`.
      atlas_on_grid = resample_to_img(atlas_img, boldref_img, interpolation="nearest")


    [OK] sub-18_ses-01_task-rest: 200/200 labels -> sub-18_ses-01_task-rest_space-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-200Parcels7Networks_dseg_onbold.nii.gz


    /tmp/ipykernel_1556/1496662992.py:25: FutureWarning: 'force_resample' will be set to 'True' by default in Nilearn 0.13.0.
    Use 'force_resample=True' to suppress this warning.
      atlas_on_grid = resample_to_img(atlas_img, boldref_img, interpolation="nearest")
    /tmp/ipykernel_1556/1496662992.py:25: FutureWarning: From release 0.13.0 onwards, this function will, by default, copy the header of the input image to the output. Currently, the header is reset to the default Nifti1Header. To suppress this warning and use the new behavior, set `copy_header=True`.
      atlas_on_grid = resample_to_img(atlas_img, boldref_img, interpolation="nearest")


    [OK] sub-20_ses-01_task-rest: 200/200 labels -> sub-20_ses-01_task-rest_space-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-200Parcels7Networks_dseg_onbold.nii.gz


    /tmp/ipykernel_1556/1496662992.py:25: FutureWarning: 'force_resample' will be set to 'True' by default in Nilearn 0.13.0.
    Use 'force_resample=True' to suppress this warning.
      atlas_on_grid = resample_to_img(atlas_img, boldref_img, interpolation="nearest")
    /tmp/ipykernel_1556/1496662992.py:25: FutureWarning: From release 0.13.0 onwards, this function will, by default, copy the header of the input image to the output. Currently, the header is reset to the default Nifti1Header. To suppress this warning and use the new behavior, set `copy_header=True`.
      atlas_on_grid = resample_to_img(atlas_img, boldref_img, interpolation="nearest")


    [OK] sub-21_ses-01_task-rest: 200/200 labels -> sub-21_ses-01_task-rest_space-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-200Parcels7Networks_dseg_onbold.nii.gz





    (                    runTag                                            boldref  \
     0  sub-01_ses-01_task-rest  /mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/...   
     1  sub-02_ses-01_task-rest  /mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/...   
     2  sub-03_ses-01_task-rest  /mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/...   
     3  sub-08_ses-01_task-rest  /mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/...   
     4  sub-08_ses-02_task-rest  /mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/...   
     
                                                     mask  \
     0  /mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/...   
     1  /mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/...   
     2  /mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/...   
     3  /mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/...   
     4  /mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/...   
     
                                                 raw_json TaskName  RepetitionTime  \
     0  /mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/...     rest             2.1   
     1  /mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/...     rest             2.1   
     2  /mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/...     rest             2.1   
     3  /mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/...     rest             2.1   
     4  /mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/...     rest             2.1   
     
        EchoTime PhaseEncodingDirection  TotalReadoutTime  \
     0    0.0246                     j-           0.02961   
     1    0.0246                     j-           0.02961   
     2    0.0246                     j-           0.02961   
     3    0.0246                     j-           0.02961   
     4    0.0246                     j-           0.02961   
     
                                              SliceTiming  n_labels_expected  \
     0  [1.0525, 0, 1.11, 0.055, 1.165, 0.11, 1.22, 0....                200   
     1  [1.055, 0, 1.11, 0.0575, 1.165, 0.1125, 1.2225...                200   
     2  [1.0525, 0, 1.11, 0.055, 1.165, 0.11, 1.22, 0....                200   
     3  [1.055, 0, 1.11, 0.055, 1.165, 0.11, 1.22, 0.1...                200   
     4  [1.055, 0, 1.11, 0.055, 1.165, 0.1125, 1.22, 0...                200   
     
        n_labels_present  n_labels_missing missing_labels_head  \
     0               200                 0                       
     1               200                 0                       
     2               200                 0                       
     3               200                 0                       
     4               200                 0                       
     
                                                out_atlas  \
     0  /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...   
     1  /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...   
     2  /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...   
     3  /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...   
     4  /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...   
     
                                          out_overlay_png  
     0  /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...  
     1  /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...  
     2  /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...  
     3  /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...  
     4  /mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bo...  ,
     PosixPath('/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bold_parcel/atlas_on_boldgrid/qc_atlas_on_boldgrid_summary.csv'))




```python
# Cell 6 — Display QC summary (quick read)

pd.set_option("display.max_colwidth", 120)
qc[["runTag","TaskName","RepetitionTime","n_labels_present","n_labels_missing","missing_labels_head"]]

```




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
      <th>TaskName</th>
      <th>RepetitionTime</th>
      <th>n_labels_present</th>
      <th>n_labels_missing</th>
      <th>missing_labels_head</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sub-01_ses-01_task-rest</td>
      <td>rest</td>
      <td>2.1</td>
      <td>200</td>
      <td>0</td>
      <td></td>
    </tr>
    <tr>
      <th>1</th>
      <td>sub-02_ses-01_task-rest</td>
      <td>rest</td>
      <td>2.1</td>
      <td>200</td>
      <td>0</td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>sub-03_ses-01_task-rest</td>
      <td>rest</td>
      <td>2.1</td>
      <td>200</td>
      <td>0</td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td>sub-08_ses-01_task-rest</td>
      <td>rest</td>
      <td>2.1</td>
      <td>200</td>
      <td>0</td>
      <td></td>
    </tr>
    <tr>
      <th>4</th>
      <td>sub-08_ses-02_task-rest</td>
      <td>rest</td>
      <td>2.1</td>
      <td>200</td>
      <td>0</td>
      <td></td>
    </tr>
    <tr>
      <th>5</th>
      <td>sub-09_ses-01_task-rest</td>
      <td>rest</td>
      <td>2.1</td>
      <td>200</td>
      <td>0</td>
      <td></td>
    </tr>
    <tr>
      <th>6</th>
      <td>sub-13_ses-01_task-rest</td>
      <td>rest</td>
      <td>2.1</td>
      <td>200</td>
      <td>0</td>
      <td></td>
    </tr>
    <tr>
      <th>7</th>
      <td>sub-13_ses-02_task-rest</td>
      <td>rest</td>
      <td>2.1</td>
      <td>200</td>
      <td>0</td>
      <td></td>
    </tr>
    <tr>
      <th>8</th>
      <td>sub-14_ses-01_task-rest</td>
      <td>rest</td>
      <td>2.1</td>
      <td>200</td>
      <td>0</td>
      <td></td>
    </tr>
    <tr>
      <th>9</th>
      <td>sub-16_ses-01_task-rest</td>
      <td>rest</td>
      <td>2.1</td>
      <td>200</td>
      <td>0</td>
      <td></td>
    </tr>
    <tr>
      <th>10</th>
      <td>sub-17_ses-01_task-rest</td>
      <td>rest</td>
      <td>2.1</td>
      <td>200</td>
      <td>0</td>
      <td></td>
    </tr>
    <tr>
      <th>11</th>
      <td>sub-17_ses-02_task-rest</td>
      <td>rest</td>
      <td>2.1</td>
      <td>200</td>
      <td>0</td>
      <td></td>
    </tr>
    <tr>
      <th>12</th>
      <td>sub-18_ses-01_task-rest</td>
      <td>rest</td>
      <td>2.1</td>
      <td>200</td>
      <td>0</td>
      <td></td>
    </tr>
    <tr>
      <th>13</th>
      <td>sub-20_ses-01_task-rest</td>
      <td>rest</td>
      <td>2.1</td>
      <td>200</td>
      <td>0</td>
      <td></td>
    </tr>
    <tr>
      <th>14</th>
      <td>sub-21_ses-01_task-rest</td>
      <td>rest</td>
      <td>2.1</td>
      <td>200</td>
      <td>0</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>




```python
from pathlib import Path

FMRI_ROOT = Path(r"/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids")
SPACE_TAG = "space-MNI152NLin2009cAsym"

boldrefs = sorted(FMRI_ROOT.glob(f"**/fmriprep/**/func/*_{SPACE_TAG}_*boldref.nii.gz"))
print("boldrefs found:", len(boldrefs))
print("example:", boldrefs[0] if boldrefs else "NONE")

preproc = sorted(FMRI_ROOT.glob(f"**/fmriprep/**/func/*_{SPACE_TAG}_desc-preproc_bold.nii.gz"))
print("preproc bold found:", len(preproc))
print("example:", preproc[0] if preproc else "NONE")

masks = sorted(FMRI_ROOT.glob(f"**/fmriprep/**/func/*_{SPACE_TAG}_desc-brain_mask.nii.gz"))
print("brain masks found:", len(masks))
print("example:", masks[0] if masks else "NONE")

```

    boldrefs found: 15
    example: /mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/sub-01_ses-01/fmriprep/sub-01/ses-01/func/sub-01_ses-01_task-rest_space-MNI152NLin2009cAsym_boldref.nii.gz
    preproc bold found: 15
    example: /mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/sub-01_ses-01/fmriprep/sub-01/ses-01/func/sub-01_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
    brain masks found: 15
    example: /mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/sub-01_ses-01/fmriprep/sub-01/ses-01/func/sub-01_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz



```python
jsons = sorted(FMRI_ROOT.glob("**/*.json"))
print("json files found under FMRI_ROOT:", len(jsons))
print("example:", jsons[0] if jsons else "NONE")

```

    json files found under FMRI_ROOT: 300
    example: /mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/sub-01_ses-01/fmriprep/dataset_description.json



```python
from pathlib import Path
import re
import pandas as pd

FMRI_ROOT = Path("/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids")
SPACE_TAG = "space-MNI152NLin2009cAsym"

# 1) list all fmriprep func directories
func_dirs = sorted(FMRI_ROOT.glob("sub-*_ses-*/fmriprep/sub-*/ses-*/func"))
print("func_dirs found:", len(func_dirs))
print("example:", func_dirs[0] if func_dirs else "NONE")

def infer_subses_from_dir(d: Path) -> str:
    # robustly get sub-XX and ses-YY from path
    msub = re.search(r"(sub-[^/]+)", str(d))
    mses = re.search(r"(ses-[^/]+)", str(d))
    sub = msub.group(1) if msub else "sub-?"
    ses = mses.group(1) if mses else "ses-?"
    return f"{sub}_{ses}"

rows = []
for d in func_dirs:
    subses = infer_subses_from_dir(d)

    # files we expect in MNI space
    boldref = sorted(d.glob(f"*_{SPACE_TAG}_*boldref.nii.gz"))
    preproc = sorted(d.glob(f"*_{SPACE_TAG}_desc-preproc_bold.nii.gz"))
    mask    = sorted(d.glob(f"*_{SPACE_TAG}_desc-brain_mask.nii.gz"))

    # any boldrefs in any space (helps diagnose “wrong space only”)
    any_boldref = sorted(d.glob("*boldref*.nii.gz"))
    any_preproc = sorted(d.glob("*desc-preproc_bold.nii.gz"))
    any_mask    = sorted(d.glob("*desc-brain_mask.nii.gz"))

    rows.append({
        "subses": subses,
        "func_dir": str(d),
        "n_boldref_MNI": len(boldref),
        "n_preproc_MNI": len(preproc),
        "n_mask_MNI": len(mask),
        "n_boldref_any": len(any_boldref),
        "n_preproc_any": len(any_preproc),
        "n_mask_any": len(any_mask),
        "example_boldref_any": any_boldref[0].name if any_boldref else "",
        "example_preproc_any": any_preproc[0].name if any_preproc else "",
        "example_mask_any": any_mask[0].name if any_mask else "",
    })

df = pd.DataFrame(rows).sort_values(["subses","func_dir"]).reset_index(drop=True)
df

```

    func_dirs found: 15
    example: /mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/sub-01_ses-01/fmriprep/sub-01/ses-01/func





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
      <th>subses</th>
      <th>func_dir</th>
      <th>n_boldref_MNI</th>
      <th>n_preproc_MNI</th>
      <th>n_mask_MNI</th>
      <th>n_boldref_any</th>
      <th>n_preproc_any</th>
      <th>n_mask_any</th>
      <th>example_boldref_any</th>
      <th>example_preproc_any</th>
      <th>example_mask_any</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sub-01_ses-01_ses-01</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/sub-01_ses-01/fmriprep/sub-01/ses-01/func</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>sub-01_ses-01_task-rest_desc-coreg_boldref.nii.gz</td>
      <td>sub-01_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz</td>
      <td>sub-01_ses-01_task-rest_desc-brain_mask.nii.gz</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sub-02_ses-01_ses-01</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/sub-02_ses-01/fmriprep/sub-02/ses-01/func</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>sub-02_ses-01_task-rest_desc-coreg_boldref.nii.gz</td>
      <td>sub-02_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz</td>
      <td>sub-02_ses-01_task-rest_desc-brain_mask.nii.gz</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sub-03_ses-01_ses-01</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/sub-03_ses-01/fmriprep/sub-03/ses-01/func</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>sub-03_ses-01_task-rest_desc-coreg_boldref.nii.gz</td>
      <td>sub-03_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz</td>
      <td>sub-03_ses-01_task-rest_desc-brain_mask.nii.gz</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sub-08_ses-01_ses-01</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/sub-08_ses-01/fmriprep/sub-08/ses-01/func</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>sub-08_ses-01_task-rest_desc-coreg_boldref.nii.gz</td>
      <td>sub-08_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz</td>
      <td>sub-08_ses-01_task-rest_desc-brain_mask.nii.gz</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sub-08_ses-02_ses-02</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/sub-08_ses-02/fmriprep/sub-08/ses-02/func</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>sub-08_ses-02_task-rest_desc-coreg_boldref.nii.gz</td>
      <td>sub-08_ses-02_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz</td>
      <td>sub-08_ses-02_task-rest_desc-brain_mask.nii.gz</td>
    </tr>
    <tr>
      <th>5</th>
      <td>sub-09_ses-01_ses-01</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/sub-09_ses-01/fmriprep/sub-09/ses-01/func</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>sub-09_ses-01_task-rest_desc-coreg_boldref.nii.gz</td>
      <td>sub-09_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz</td>
      <td>sub-09_ses-01_task-rest_desc-brain_mask.nii.gz</td>
    </tr>
    <tr>
      <th>6</th>
      <td>sub-13_ses-01_ses-01</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/sub-13_ses-01/fmriprep/sub-13/ses-01/func</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>sub-13_ses-01_task-rest_desc-coreg_boldref.nii.gz</td>
      <td>sub-13_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz</td>
      <td>sub-13_ses-01_task-rest_desc-brain_mask.nii.gz</td>
    </tr>
    <tr>
      <th>7</th>
      <td>sub-13_ses-02_ses-02</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/sub-13_ses-02/fmriprep/sub-13/ses-02/func</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>sub-13_ses-02_task-rest_desc-coreg_boldref.nii.gz</td>
      <td>sub-13_ses-02_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz</td>
      <td>sub-13_ses-02_task-rest_desc-brain_mask.nii.gz</td>
    </tr>
    <tr>
      <th>8</th>
      <td>sub-14_ses-01_ses-01</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/sub-14_ses-01/fmriprep/sub-14/ses-01/func</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>sub-14_ses-01_task-rest_desc-coreg_boldref.nii.gz</td>
      <td>sub-14_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz</td>
      <td>sub-14_ses-01_task-rest_desc-brain_mask.nii.gz</td>
    </tr>
    <tr>
      <th>9</th>
      <td>sub-16_ses-01_ses-01</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/sub-16_ses-01/fmriprep/sub-16/ses-01/func</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>sub-16_ses-01_task-rest_desc-coreg_boldref.nii.gz</td>
      <td>sub-16_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz</td>
      <td>sub-16_ses-01_task-rest_desc-brain_mask.nii.gz</td>
    </tr>
    <tr>
      <th>10</th>
      <td>sub-17_ses-01_ses-01</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/sub-17_ses-01/fmriprep/sub-17/ses-01/func</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>sub-17_ses-01_task-rest_desc-coreg_boldref.nii.gz</td>
      <td>sub-17_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz</td>
      <td>sub-17_ses-01_task-rest_desc-brain_mask.nii.gz</td>
    </tr>
    <tr>
      <th>11</th>
      <td>sub-17_ses-02_ses-02</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/sub-17_ses-02/fmriprep/sub-17/ses-02/func</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>sub-17_ses-02_task-rest_desc-coreg_boldref.nii.gz</td>
      <td>sub-17_ses-02_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz</td>
      <td>sub-17_ses-02_task-rest_desc-brain_mask.nii.gz</td>
    </tr>
    <tr>
      <th>12</th>
      <td>sub-18_ses-01_ses-01</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/sub-18_ses-01/fmriprep/sub-18/ses-01/func</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>sub-18_ses-01_task-rest_desc-coreg_boldref.nii.gz</td>
      <td>sub-18_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz</td>
      <td>sub-18_ses-01_task-rest_desc-brain_mask.nii.gz</td>
    </tr>
    <tr>
      <th>13</th>
      <td>sub-20_ses-01_ses-01</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/sub-20_ses-01/fmriprep/sub-20/ses-01/func</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>sub-20_ses-01_task-rest_desc-coreg_boldref.nii.gz</td>
      <td>sub-20_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz</td>
      <td>sub-20_ses-01_task-rest_desc-brain_mask.nii.gz</td>
    </tr>
    <tr>
      <th>14</th>
      <td>sub-21_ses-01_ses-01</td>
      <td>/mnt/c/EEGFMRI/hmm/R01_rerun/01_raw/fmri_bids/sub-21_ses-01/fmriprep/sub-21/ses-01/func</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>sub-21_ses-01_task-rest_desc-coreg_boldref.nii.gz</td>
      <td>sub-21_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz</td>
      <td>sub-21_ses-01_task-rest_desc-brain_mask.nii.gz</td>
    </tr>
  </tbody>
</table>
</div>




```python
# “Run present” means at least one MNI boldref exists in that func dir
missing_mni = df[(df["n_boldref_MNI"] == 0) | (df["n_preproc_MNI"] == 0) | (df["n_mask_MNI"] == 0)]

print("func dirs with missing MNI outputs (boldref/preproc/mask):", len(missing_mni))
missing_mni[["subses","n_boldref_MNI","n_preproc_MNI","n_mask_MNI",
             "n_boldref_any","n_preproc_any","n_mask_any",
             "example_boldref_any","example_preproc_any"]]

```

    func dirs with missing MNI outputs (boldref/preproc/mask): 0





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
      <th>subses</th>
      <th>n_boldref_MNI</th>
      <th>n_preproc_MNI</th>
      <th>n_mask_MNI</th>
      <th>n_boldref_any</th>
      <th>n_preproc_any</th>
      <th>n_mask_any</th>
      <th>example_boldref_any</th>
      <th>example_preproc_any</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python

```
