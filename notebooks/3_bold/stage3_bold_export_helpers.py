"""Helper functions for the cleaned Stage-3 BOLD parcel-export workflow.

This module supports the public Stage-3 notebooks that:
- map the Schaefer atlas to each BOLD run grid
- export parcel PC1 time series after nuisance regression
- write atlas-preservation and exporter-side QC summaries

Main inputs:
- BIDS or derivatives folders containing BOLD reference images and masks
- the frozen Stage-3 Schaefer atlas files

Main outputs:
- atlas-on-grid NIfTI files, overlays, and QC tables
- exporter-side CSV, MAT, and NPY outputs written by the public notebooks

Important note:
- the standalone atlas-preservation branch and the authoritative exporter
  branch remain intentionally distinct in this stage
- this module is only helper logic; the public notebooks remain the main
  user-facing entry points
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img
from nilearn.plotting import plot_roi
from scipy.io import savemat
from templateflow.api import get as tf_get


DEFAULT_SPACE_TAG = "space-MNI152NLin2009cAsym"


def _assert_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _tf_get_single(*args, **kwargs) -> Path:
    out = tf_get(*args, **kwargs)
    if isinstance(out, (list, tuple)):
        out = out[0]
    return Path(out)


def _find_boldrefs(root: Path, space_tag: str) -> list[Path]:
    patterns = [
        f"**/func/*_{space_tag}_boldref.nii.gz",
        f"**/func/*_{space_tag}_desc-coreg_boldref.nii.gz",
        f"**/func/*_{space_tag}_desc-preproc_boldref.nii.gz",
    ]
    found: list[Path] = []
    for pattern in patterns:
        found.extend(root.glob(pattern))
    return sorted(set(found))


def _infer_run_tag_from_boldref(filename: str, space_tag: str) -> str:
    match = re.match(
        r"^(sub-[^_]+_ses-[^_]+(?:_task-[^_]+)?(?:_run-[^_]+)?)_.*$",
        filename,
    )
    if match:
        return match.group(1)
    return filename.split("_" + space_tag)[0].replace(".nii.gz", "")


def _find_mask_for_boldref(boldref: Path, space_tag: str) -> Path | None:
    folder = boldref.parent
    stem = boldref.name
    candidate = folder / stem.replace("_boldref.nii.gz", "_desc-brain_mask.nii.gz")
    if candidate.exists():
        return candidate
    prefix = stem.split("_" + space_tag)[0]
    masks = sorted(folder.glob(f"{prefix}_{space_tag}_desc-brain_mask.nii.gz"))
    return masks[0] if masks else None


def _find_raw_json_for_run(run_tag: str, fmri_root: Path, raw_bids_root: Path | None) -> Path | None:
    root = raw_bids_root if raw_bids_root is not None else fmri_root
    sub = re.search(r"(sub-[^_]+)", run_tag)
    ses = re.search(r"(ses-[^_]+)", run_tag)
    task = re.search(r"(task-[^_]+)", run_tag)
    run = re.search(r"(run-[^_]+)", run_tag)

    if not sub or not ses:
        return None

    base = f"{sub.group(1)}_{ses.group(1)}"
    patterns: list[str] = []
    if task:
        patterns.append(f"**/func/{base}_{task.group(1)}_bold.json")
        if run:
            patterns.append(f"**/func/{base}_{task.group(1)}_{run.group(1)}_bold.json")
    else:
        patterns.append(f"**/func/{base}_bold.json")

    for pattern in patterns:
        hits = list(root.glob(pattern))
        if hits:
            return sorted(hits)[0]
    return None


def _load_json_fields(json_path: Path | None) -> dict:
    if json_path is None or not json_path.exists():
        return {}
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    keep = [
        "TaskName",
        "RepetitionTime",
        "EchoTime",
        "PhaseEncodingDirection",
        "TotalReadoutTime",
        "SliceTiming",
    ]
    return {key: data.get(key, None) for key in keep}


def run_atlas_on_boldgrid(
    fmri_root: str | Path,
    out_root: str | Path,
    atlas_nii: str | Path,
    atlas_tsv: str | Path | None = None,
    space_tag: str = DEFAULT_SPACE_TAG,
    overwrite: bool = True,
    raw_bids_root: str | Path | None = None,
    make_overlay_pngs: bool = True,
) -> pd.DataFrame:
    """Resample the frozen Schaefer atlas onto each run's BOLD grid.

    This is the standalone atlas-preservation QC branch used by Step 30.
    It writes per-run atlas-on-grid NIfTI files plus a summary CSV, but it
    does not define the atlas used by the main parcel-export path in Step 31.
    """
    fmri_root = Path(fmri_root)
    out_root = Path(out_root)
    atlas_nii = Path(atlas_nii)
    atlas_tsv = Path(atlas_tsv) if atlas_tsv is not None else Path(str(atlas_nii).replace("_dseg.nii.gz", "_dseg.tsv"))
    raw_bids_root = Path(raw_bids_root) if raw_bids_root is not None else None

    _assert_exists(atlas_nii, "atlas_nii")
    out_root.mkdir(parents=True, exist_ok=True)

    boldrefs = _find_boldrefs(fmri_root, space_tag)
    if not boldrefs:
        raise RuntimeError(f"No boldref files found under {fmri_root} for {space_tag}.")

    atlas_img = nib.load(str(atlas_nii))
    atlas_labels = np.unique(atlas_img.get_fdata()).astype(int)
    atlas_labels = atlas_labels[atlas_labels != 0]

    rows: list[dict] = []
    for boldref in boldrefs:
        run_tag = _infer_run_tag_from_boldref(boldref.name, space_tag)
        out_dir = out_root / run_tag
        out_dir.mkdir(parents=True, exist_ok=True)

        out_atlas = out_dir / f"{run_tag}_{space_tag}_atlas-Schaefer2018_desc-200Parcels7Networks_dseg_onbold.nii.gz"
        out_png = out_dir / f"{run_tag}_{space_tag}_atlas_overlay.png"

        if out_atlas.exists() and not overwrite:
            print(f"[SKIP] {run_tag} (exists)")
            continue

        mask = _find_mask_for_boldref(boldref, space_tag)
        boldref_img = nib.load(str(boldref))
        atlas_on_grid = resample_to_img(atlas_img, boldref_img, interpolation="nearest")
        atlas_data = atlas_on_grid.get_fdata().astype(np.int32)

        if mask is not None and mask.exists():
            mask_data = (nib.load(str(mask)).get_fdata() > 0).astype(np.uint8)
            atlas_data = atlas_data * mask_data

        nib.save(nib.Nifti1Image(atlas_data, boldref_img.affine, boldref_img.header), str(out_atlas))

        if atlas_tsv.exists():
            out_tsv = out_dir / atlas_tsv.name
            if not out_tsv.exists() or overwrite:
                out_tsv.write_bytes(atlas_tsv.read_bytes())

        present = np.unique(atlas_data)
        present = present[present != 0]
        missing = np.setdiff1d(atlas_labels, present)

        if make_overlay_pngs:
            try:
                disp = plot_roi(
                    nib.load(str(out_atlas)),
                    bg_img=boldref_img,
                    title=run_tag,
                    display_mode="ortho",
                    cut_coords=(0, 0, 0),
                )
                disp.savefig(str(out_png))
                disp.close()
            except Exception as exc:
                print(f"[WARN] overlay failed for {run_tag}: {exc}")

        raw_json = _find_raw_json_for_run(run_tag, fmri_root, raw_bids_root)
        meta = _load_json_fields(raw_json)

        rows.append(
            {
                "runTag": run_tag,
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
            }
        )

        print(f"[OK] {run_tag}: {len(present)}/{len(atlas_labels)} labels -> {out_atlas.name}")

    qc = pd.DataFrame(rows).sort_values("runTag")
    qc.to_csv(out_root / "qc_atlas_on_boldgrid_summary.csv", index=False)
    return qc


def _discover_func_dirs(root: Path) -> list[Path]:
    return sorted(root.glob("sub-*_ses-*/fmriprep/sub-*/ses-*/func"))


def _infer_export_run_tag(func_dir: Path) -> str:
    sub = func_dir.parts[-3]
    ses = func_dir.parts[-2]
    return f"{sub}_{ses}_task-rest"


def _pick_bold_file(func_dir: Path, bold_preference: list[str]) -> Path | None:
    for pattern in bold_preference:
        candidates = sorted(func_dir.glob(pattern))
        if candidates:
            return candidates[0]
    return None


def _derive_stem_space_from_bold(bold_path: Path) -> str:
    match = re.match(r"^(.*)_desc-[^_]+_bold\.nii\.gz$", bold_path.name)
    if not match:
        raise ValueError(f"Unrecognized BOLD filename: {bold_path.name}")
    return match.group(1)


def _load_confounds(confounds_tsv: Path, confounds_json: Path | None) -> tuple[pd.DataFrame, dict]:
    confounds = pd.read_csv(confounds_tsv, sep="\t")
    meta: dict = {}
    if confounds_json and confounds_json.exists():
        meta = json.loads(confounds_json.read_text())
    return confounds, meta


def _build_label_map(tsv_path: Path) -> dict[int, str]:
    if not tsv_path.exists():
        return {}
    df = pd.read_csv(tsv_path, sep="\t")
    if df.shape[1] < 2:
        return {}
    key_col, value_col = df.columns[0], df.columns[1]
    return dict(zip(df[key_col].astype(int).tolist(), df[value_col].astype(str).tolist()))


def _collect_motion24(confounds: pd.DataFrame) -> list[str]:
    base = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    cols: list[str] = []
    for item in base:
        cols += [item, f"{item}_derivative1", f"{item}_power2", f"{item}_derivative1_power2"]
    return [col for col in cols if col in confounds.columns]


def _collect_wm_csf(confounds: pd.DataFrame) -> list[str]:
    base = ["white_matter", "csf"]
    cols: list[str] = []
    for item in base:
        cols += [item, f"{item}_derivative1", f"{item}_power2", f"{item}_derivative1_power2"]
    return [col for col in cols if col in confounds.columns]


def _collect_cosines(confounds: pd.DataFrame) -> list[str]:
    return sorted([col for col in confounds.columns if col.startswith("cosine")])


def _collect_nonsteady(confounds: pd.DataFrame) -> list[str]:
    return sorted([col for col in confounds.columns if col.startswith("non_steady_state_outlier")])


def _collect_acompcor_retained(confounds: pd.DataFrame, meta: dict, n_max: int) -> list[str]:
    cols = sorted([col for col in confounds.columns if col.startswith("a_comp_cor_")])
    if not cols:
        return []
    retained: list[str] = []
    for col in cols:
        if col in meta and isinstance(meta[col], dict) and meta[col].get("Retained", None) is True:
            retained.append(col)
    use = sorted(retained) if retained else cols
    return use[: min(n_max, len(use))]


def _detect_fd(confounds: pd.DataFrame) -> np.ndarray:
    if "framewise_displacement" in confounds.columns:
        fd = confounds["framewise_displacement"].to_numpy()
    else:
        fd = np.zeros(len(confounds))
    return np.nan_to_num(fd, nan=0.0).astype(np.float32)


def _detect_dvars_spikes(confounds: pd.DataFrame, z_thresh: float) -> tuple[np.ndarray, str]:
    outlier_cols = [col for col in confounds.columns if col.startswith("dvars_outlier")]
    if outlier_cols:
        mask = confounds[outlier_cols].fillna(0).to_numpy().sum(axis=1) > 0
        return mask.astype(bool), "dvars_outlier_cols"

    if "std_dvars" in confounds.columns:
        values = confounds["std_dvars"].to_numpy()
    elif "dvars" in confounds.columns:
        values = confounds["dvars"].to_numpy()
    else:
        return np.zeros(len(confounds), dtype=bool), "none"

    values = np.nan_to_num(values, nan=np.nanmedian(values))
    median = np.median(values)
    mad = np.median(np.abs(values - median)) + 1e-8
    values_z = (values - median) / mad
    return (values_z > z_thresh).astype(bool), "mad_z"


def _spikes_to_design(mask: np.ndarray, prefix: str, block_min_len: int) -> tuple[np.ndarray, list[str]]:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return np.zeros((mask.shape[0], 0), dtype=np.float32), []

    segments: list[tuple[int, int]] = []
    start = end = idx[0]
    for timepoint in idx[1:]:
        if timepoint == end + 1:
            end = timepoint
        else:
            segments.append((start, end))
            start = end = timepoint
    segments.append((start, end))

    cols: list[np.ndarray] = []
    names: list[str] = []
    for start, end in segments:
        seg_len = end - start + 1
        if seg_len >= block_min_len:
            col = np.zeros((mask.shape[0],), dtype=np.float32)
            col[start : end + 1] = 1.0
            cols.append(col)
            names.append(f"{prefix}_block_{start:04d}_{end:04d}")
        else:
            for timepoint in range(start, end + 1):
                col = np.zeros((mask.shape[0],), dtype=np.float32)
                col[timepoint] = 1.0
                cols.append(col)
                names.append(f"{prefix}_{timepoint:04d}")

    X = np.stack(cols, axis=1) if cols else np.zeros((mask.shape[0], 0), dtype=np.float32)
    return X, names


def run_bold_parcel_export(
    fmri_root: str | Path,
    out_root: str | Path,
    atlas_space: str = "MNI152NLin2009cAsym",
    atlas_name: str = "Schaefer2018",
    atlas_desc: str = "200Parcels7Networks",
    atlas_resolution: int = 2,
    n_parcels: int = 200,
    bold_preference: list[str] | None = None,
    use_motion24: bool = True,
    use_wm_csf: bool = True,
    use_cosines: bool = True,
    n_acompcor: int = 10,
    add_nonsteady: bool = True,
    add_fd_spikes: bool = True,
    add_dvars_spikes: bool = False,
    fd_threshold: float = 0.5,
    dvars_z_thresh: float = 2.5,
    block_min_len: int = 3,
    min_voxels_for_pca: int = 10,
    min_voxels_for_any: int = 1,
    zscore_before_pca: bool = True,
    save_npy: bool = True,
    save_mat: bool = True,
    save_atlas_on_grid: bool = True,
    save_qc_figs: bool = True,
    save_qc_sidecars: bool = True,
) -> pd.DataFrame:
    """Run the authoritative Stage-3 BOLD parcel export workflow.

    This function freezes the recovered atlas branch, builds the preserved
    nuisance-regression design, extracts one parcel PC1 time series per
    Schaefer parcel, and writes the run-level QC sidecars used later for
    Table S5 and Figure S5 support.
    """
    fmri_root = Path(fmri_root)
    out_root = Path(out_root)
    if bold_preference is None:
        bold_preference = [f"*_task-rest_*space-{atlas_space}_desc-preproc_bold.nii.gz"]

    out_root.mkdir(parents=True, exist_ok=True)
    for subdir in ["npy", "mat", "atlas_on_grid", "qc", "atlas_source"]:
        (out_root / subdir).mkdir(parents=True, exist_ok=True)

    def build_design_matrix(confounds: pd.DataFrame, meta: dict) -> tuple[np.ndarray, list[str], dict, np.ndarray]:
        """Build the preserved nuisance design for one run.

        The design keeps the recovered manuscript choices explicit:
        continuous confounds first, then FD/DVARS spike regressors, then any
        retained motion-outlier columns, plus the intercept.
        """
        def _expand_mask(mask: np.ndarray, pre: int = 0, post: int = 2) -> np.ndarray:
            n_tr = mask.shape[0]
            out = mask.copy()
            idx = np.where(mask)[0]
            for timepoint in idx:
                start = max(0, timepoint - pre)
                end = min(n_tr, timepoint + post + 1)
                out[start:end] = True
            return out

        def _collect_motion_outliers(confounds_local: pd.DataFrame) -> list[str]:
            return sorted([col for col in confounds_local.columns if col.startswith("motion_outlier")])

        fd_cat = 2.0
        expand_pre = 0
        expand_post = 2
        n_tr = len(confounds)
        cont_cols: list[str] = []

        if use_motion24:
            cont_cols += _collect_motion24(confounds)
        if use_wm_csf:
            cont_cols += _collect_wm_csf(confounds)
        if use_cosines:
            cont_cols += _collect_cosines(confounds)
        cont_cols += _collect_acompcor_retained(confounds, meta, n_acompcor)
        if add_nonsteady:
            cont_cols += _collect_nonsteady(confounds)

        seen: set[str] = set()
        cont_cols = [col for col in cont_cols if (col not in seen and not seen.add(col))]
        X_cont = confounds[cont_cols].copy() if cont_cols else pd.DataFrame(index=np.arange(n_tr))
        X_cont = X_cont.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        X_parts: list[np.ndarray] = []
        names: list[str] = []
        if X_cont.shape[1] > 0:
            Xc = X_cont.to_numpy(dtype=np.float32)
            mu = Xc.mean(axis=0, keepdims=True)
            sd = Xc.std(axis=0, keepdims=True)
            sd = np.where(sd < 1e-8, 1.0, sd)
            Xc = (Xc - mu) / sd
            X_parts.append(Xc)
            names += list(X_cont.columns)

        fd = _detect_fd(confounds)
        fd_base = fd > fd_threshold
        fd_cat_mask = fd > fd_cat
        fd_spikes = fd_base | _expand_mask(fd_cat_mask, pre=expand_pre, post=expand_post)
        dvars_spikes, dvars_mode = _detect_dvars_spikes(confounds, dvars_z_thresh)

        Xfd = np.zeros((n_tr, 0), dtype=np.float32)
        Xdv = np.zeros((n_tr, 0), dtype=np.float32)
        Xmo = np.zeros((n_tr, 0), dtype=np.float32)

        if add_fd_spikes:
            Xfd, fd_names = _spikes_to_design(fd_spikes, "fd", block_min_len)
            X_parts.append(Xfd)
            names += fd_names

        if add_dvars_spikes:
            Xdv, dvars_names = _spikes_to_design(dvars_spikes, "dvars", block_min_len)
            X_parts.append(Xdv)
            names += dvars_names

        motion_outlier_cols = _collect_motion_outliers(confounds)
        motion_out_any = np.zeros(n_tr, dtype=bool)
        if motion_outlier_cols:
            motion_matrix = confounds[motion_outlier_cols].fillna(0.0).to_numpy(dtype=np.float32)
            motion_out_any = motion_matrix.sum(axis=1) > 0
            kept_cols: list[str] = []
            for col in motion_outlier_cols:
                idx = np.where(confounds[col].fillna(0.0).to_numpy() > 0)[0]
                if len(idx) == 0:
                    continue
                if np.all(fd[idx] <= fd_threshold):
                    kept_cols.append(col)

            if kept_cols:
                Xmo = confounds[kept_cols].fillna(0.0).to_numpy(dtype=np.float32)
                keep = Xmo.sum(axis=0) > 0
                Xmo = Xmo[:, keep]
                kept_cols = [col for col, keep_col in zip(kept_cols, keep) if keep_col]
                if Xmo.shape[1] > 0:
                    X_parts.append(Xmo)
                    names += kept_cols

        X_parts.append(np.ones((n_tr, 1), dtype=np.float32))
        names += ["intercept"]
        X = np.concatenate(X_parts, axis=1) if X_parts else np.ones((n_tr, 1), dtype=np.float32)
        motion_out_kept_any = float((Xmo.sum(axis=1) > 0).mean()) if Xmo.shape[1] > 0 else 0.0

        qc = dict(
            qc_fd_mean=float(fd.mean()),
            qc_fd_p95=float(np.percentile(fd, 95)),
            qc_fd_max=float(fd.max()),
            pct_fd_spikes=float(fd_spikes.mean()),
            pct_dvars_spikes=float(dvars_spikes.mean()),
            n_fd_units=int(Xfd.shape[1]) if add_fd_spikes else 0,
            n_dvars_units=int(Xdv.shape[1]) if add_dvars_spikes else 0,
            dvars_mode=dvars_mode,
            fd_cat_threshold=float(fd_cat),
            fd_expand_pre=int(expand_pre),
            fd_expand_post=int(expand_post),
            pct_fd_cat=float(fd_cat_mask.mean()),
            n_motion_out_cols_total=int(len(motion_outlier_cols)),
            n_motion_out_units=int(Xmo.shape[1]),
            pct_motion_out_any_total=float(motion_out_any.mean()),
            pct_motion_out_any_kept=float(motion_out_kept_any),
            n_conf_cont=int(X_cont.shape[1]),
            n_total_regressors=int(X.shape[1]),
        )
        return X, names, qc, fd

    def regress_out(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        beta, *_ = np.linalg.lstsq(X.astype(np.float64), Y.astype(np.float64), rcond=None)
        return (Y - X @ beta).astype(np.float32)

    def parcel_pc1(Y_resid: np.ndarray, atlas_lbl_3d: np.ndarray, mask_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        """Extract one parcel summary time series per atlas label.

        The preserved fallback rule is:
        use the first principal component when enough voxels are available,
        otherwise fall back to the parcel mean, and mark truly undersupported
        parcels as all-NaN.
        """
        if atlas_lbl_3d.shape != mask_3d.shape:
            raise ValueError(f"atlas/mask mismatch {atlas_lbl_3d.shape} vs {mask_3d.shape}")

        mask_flat = mask_3d.reshape(-1)
        atlas_flat = atlas_lbl_3d.reshape(-1)[mask_flat]
        n_time = Y_resid.shape[0]
        pc = np.zeros((n_time, n_parcels), dtype=np.float32)
        vox_counts = np.zeros(n_parcels, dtype=int)
        n_nan = 0
        n_mean = 0

        for parcel_idx in range(1, n_parcels + 1):
            idx = np.where(atlas_flat == parcel_idx)[0]
            vox_counts[parcel_idx - 1] = len(idx)

            if len(idx) < min_voxels_for_any:
                pc[:, parcel_idx - 1] = np.nan
                n_nan += 1
                continue

            Xp = Y_resid[:, idx]
            Xp = np.nan_to_num(Xp, nan=0.0)
            Xp = Xp - Xp.mean(axis=0, keepdims=True)

            if zscore_before_pca:
                sd = Xp.std(axis=0, keepdims=True)
                sd = np.where(sd < 1e-8, 1.0, sd)
                Xp = Xp / sd

            if len(idx) < min_voxels_for_pca:
                ts = Xp.mean(axis=1)
                n_mean += 1
            else:
                U, S, _ = np.linalg.svd(Xp, full_matrices=False)
                ts = U[:, 0] * S[0]
                mean_ts = Xp.mean(axis=1)
                if np.corrcoef(ts, mean_ts)[0, 1] < 0:
                    ts = -ts

            pc[:, parcel_idx - 1] = ts.astype(np.float32)

        qc = dict(
            qc_pct_all_nan_parcels=float(n_nan / n_parcels),
            qc_n_mean_fallback=int(n_mean),
            min_voxels_per_parcel=int(vox_counts.min()),
            p10_voxels_per_parcel=float(np.percentile(vox_counts, 10)),
            median_voxels_per_parcel=float(np.median(vox_counts)),
        )
        return pc, vox_counts, qc

    atlas_nii = _tf_get_single(
        atlas_space,
        atlas=atlas_name,
        desc=atlas_desc,
        suffix="dseg",
        extension=".nii.gz",
        resolution=atlas_resolution,
    )
    atlas_tsv = _tf_get_single(
        atlas_space,
        atlas=atlas_name,
        desc=atlas_desc,
        suffix="dseg",
        extension=".tsv",
    )

    atlas_nii_frozen = out_root / "atlas_source" / atlas_nii.name
    atlas_tsv_frozen = out_root / "atlas_source" / atlas_tsv.name
    if not atlas_nii_frozen.exists():
        atlas_nii_frozen.write_bytes(atlas_nii.read_bytes())
    if not atlas_tsv_frozen.exists():
        atlas_tsv_frozen.write_bytes(atlas_tsv.read_bytes())

    atlas_img = nib.load(str(atlas_nii_frozen))
    label_map = _build_label_map(atlas_tsv_frozen)
    parcel_labels = np.arange(1, n_parcels + 1, dtype=np.int32)
    parcel_names = np.array([label_map.get(int(label), f"ROI_{int(label)}") for label in parcel_labels], dtype=object)

    print("Atlas frozen:", atlas_nii_frozen)
    print("TSV frozen:", atlas_tsv_frozen)

    rows: list[dict] = []
    # Process each run independently so the saved outputs and QC sidecars
    # stay easy to audit run by run.
    for func_dir in _discover_func_dirs(fmri_root):
        run_tag = _infer_export_run_tag(func_dir)
        bold_file = _pick_bold_file(func_dir, bold_preference)
        if bold_file is None:
            continue

        stem_space = _derive_stem_space_from_bold(bold_file)
        brain_mask = func_dir / f"{stem_space}_desc-brain_mask.nii.gz"
        boldref = func_dir / f"{stem_space}_boldref.nii.gz"
        confounds_tsv = func_dir / f"{run_tag}_desc-confounds_timeseries.tsv"
        confounds_json = func_dir / f"{run_tag}_desc-confounds_timeseries.json"

        rows.append(
            dict(
                runTag=run_tag,
                func_dir=str(func_dir),
                bold_file=str(bold_file),
                brain_mask=str(brain_mask),
                boldref=str(boldref) if boldref.exists() else "",
                confounds_tsv=str(confounds_tsv),
                confounds_json=str(confounds_json) if confounds_json.exists() else "",
                stem_space=stem_space,
            )
        )

    index = pd.DataFrame(rows).sort_values("runTag").reset_index(drop=True)
    print("Runs to export:", len(index))

    for col in ["bold_file", "brain_mask", "confounds_tsv"]:
        bad = index.loc[index[col].apply(lambda item: not Path(item).exists()), ["runTag", col, "stem_space", "func_dir"]]
        if len(bad):
            raise RuntimeError(f"Missing required {col}:\n{bad.to_string(index=False)}")

    qc_rows: list[dict] = []
    for idx, row in index.iterrows():
        run_tag = row["runTag"]
        print(f"[{idx + 1}/{len(index)}] {run_tag}")

        bold_img_run = nib.load(row["bold_file"])
        data = bold_img_run.get_fdata().astype(np.float32)
        mask_img = nib.load(row["brain_mask"])
        if mask_img.shape != data.shape[:3] or (not np.allclose(mask_img.affine, bold_img_run.affine, atol=1e-4)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mask_img = resample_to_img(
                    mask_img,
                    bold_img_run.slicer[:, :, :, 0],
                    interpolation="nearest",
                    force_resample=True,
                    copy_header=True,
                )
        mask = mask_img.get_fdata().astype(np.float32) > 0.5

        confounds, meta = _load_confounds(
            Path(row["confounds_tsv"]),
            Path(row["confounds_json"]) if row["confounds_json"] else None,
        )
        X, _, qc_motion, fd = build_design_matrix(confounds, meta)
        n_time = data.shape[3]
        if X.shape[0] != n_time:
            raise RuntimeError(f"TR mismatch for {run_tag}: bold T={n_time}, confounds T={X.shape[0]}")

        Y = data[mask].T
        Y_resid = regress_out(X, Y)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            atlas_res = resample_to_img(
                atlas_img,
                bold_img_run.slicer[:, :, :, 0],
                interpolation="nearest",
                force_resample=True,
                copy_header=True,
            )
        atlas_lbl = atlas_res.get_fdata().astype(np.int32)
        if save_atlas_on_grid:
            atlas_on_grid_path = out_root / "atlas_on_grid" / f"{run_tag}_atlas_on_grid.nii.gz"
            nib.save(atlas_res, str(atlas_on_grid_path))
        else:
            atlas_on_grid_path = ""

        pc, vox_counts, qc_parc = parcel_pc1(Y_resid, atlas_lbl, mask)
        out_pc1 = out_root / "npy" / f"{run_tag}_parcel_pc1.npy"
        out_lab = out_root / "npy" / f"{run_tag}_parcel_labels.npy"
        out_names = out_root / "npy" / f"{run_tag}_parcel_names.npy"
        out_nvox = out_root / "npy" / f"{run_tag}_parcel_nvox.npy"

        if save_npy:
            np.save(out_pc1, pc.astype(np.float32))
            np.save(out_lab, parcel_labels)
            np.save(out_names, parcel_names)
            np.save(out_nvox, vox_counts.astype(np.int32))

        out_mat: str | Path = ""
        if save_mat:
            out_mat = out_root / "mat" / f"{run_tag}_parcel_PCs_final.mat"
            savemat(
                out_mat,
                {
                    "runTag": run_tag,
                    "parcel_pc1": pc.astype(np.float32),
                    "labels": parcel_labels,
                    "parcel_names": parcel_names,
                    "parcel_nvox": vox_counts.astype(np.int32),
                },
                do_compression=True,
            )

        qc_fig: str | Path = ""
        if save_qc_figs:
            qc_fig = out_root / "qc" / f"{run_tag}_qc_pc1_fd.png"
            plt.figure(figsize=(10, 5))
            ax1 = plt.subplot(2, 1, 1)
            ax1.plot(pc[:, :5])
            ax1.set_title(f"{run_tag} — first 5 parcel PC1 traces (after nuisance regression)")
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

        qc_rows.append(
            {
                "runTag": run_tag,
                "n_volumes": int(pc.shape[0]),
                "n_parcels": int(pc.shape[1]),
                **qc_motion,
                **qc_parc,
                "bold_file": row["bold_file"],
                "brain_mask": row["brain_mask"],
                "confounds_tsv": row["confounds_tsv"],
                "confounds_json": row["confounds_json"],
                "atlas_source_nii": str(atlas_nii_frozen),
                "atlas_source_tsv": str(atlas_tsv_frozen),
                "atlas_on_grid": str(atlas_on_grid_path) if atlas_on_grid_path else "",
                "out_npy_pc1": str(out_pc1),
                "out_npy_labels": str(out_lab),
                "out_npy_names": str(out_names),
                "out_npy_nvox": str(out_nvox),
                "out_mat": str(out_mat) if out_mat else "",
                "qc_fig": str(qc_fig) if qc_fig else "",
            }
        )

    qc_df = pd.DataFrame(qc_rows).sort_values("runTag").reset_index(drop=True)
    qc_df.to_csv(out_root / "dataset_index.csv", index=False)

    if save_qc_sidecars:
        write_motion_qc_flags(out_root)
        write_qc_motion_to_pc(out_root)
        write_qc_parcel_blowups(out_root)

    return qc_df


def write_motion_qc_flags(
    out_root: str | Path,
    warn_pct_spikes: float = 0.10,
    flag_pct_spikes: float = 0.20,
    warn_fd_max: float = 2.0,
    flag_fd_max: float = 5.0,
) -> pd.DataFrame:
    out_root = Path(out_root)
    df = pd.read_csv(out_root / "dataset_index.csv")

    def label_row(row: pd.Series) -> str:
        worst = 0
        if row["pct_fd_spikes"] >= flag_pct_spikes:
            worst = max(worst, 2)
        elif row["pct_fd_spikes"] >= warn_pct_spikes:
            worst = max(worst, 1)

        if row["qc_fd_max"] >= flag_fd_max:
            worst = max(worst, 2)
        elif row["qc_fd_max"] >= warn_fd_max:
            worst = max(worst, 1)

        return ["OK", "WARN", "FLAG"][worst]

    df["motion_flag"] = df.apply(label_row, axis=1)
    show = df[["runTag", "qc_fd_mean", "qc_fd_p95", "qc_fd_max", "pct_fd_spikes", "pct_dvars_spikes", "motion_flag"]]
    show = show.sort_values(["motion_flag", "pct_fd_spikes", "qc_fd_max"], ascending=[True, False, False])
    show.to_csv(out_root / "motion_qc_flags.csv", index=False)
    return show


def write_qc_motion_to_pc(out_root: str | Path) -> pd.DataFrame:
    out_root = Path(out_root)
    df = pd.read_csv(out_root / "dataset_index.csv")

    rows: list[dict] = []
    for _, row in df.iterrows():
        pc = np.load(row["out_npy_pc1"])
        confounds = pd.read_csv(row["confounds_tsv"], sep="\t")
        fd = confounds["framewise_displacement"].fillna(0).to_numpy()

        pc_z = (pc - pc.mean(axis=0, keepdims=True)) / (pc.std(axis=0, keepdims=True) + 1e-8)
        fd_z = (fd - fd.mean()) / (fd.std() + 1e-8)
        corr = (fd_z[:, None] * pc_z).mean(axis=0)

        rows.append(
            {
                "runTag": row["runTag"],
                "fd_mean": row["qc_fd_mean"],
                "fd_max": row["qc_fd_max"],
                "pct_fd_spikes": row["pct_fd_spikes"],
                "median_abs_corr_fd_pc": float(np.median(np.abs(corr))),
                "p95_abs_corr_fd_pc": float(np.percentile(np.abs(corr), 95)),
                "max_abs_corr_fd_pc": float(np.max(np.abs(corr))),
            }
        )

    qc = pd.DataFrame(rows).sort_values(["max_abs_corr_fd_pc", "p95_abs_corr_fd_pc"], ascending=False)
    qc.to_csv(out_root / "qc_motion_to_pc.csv", index=False)
    return qc


def write_qc_parcel_blowups(
    out_root: str | Path,
    z_thr: float = 3.0,
    pct_thr_mark: float = 0.30,
    save_plots: bool = True,
) -> pd.DataFrame:
    out_root = Path(out_root)
    df = pd.read_csv(out_root / "dataset_index.csv")

    qc_rows: list[dict] = []
    for _, row in df.iterrows():
        run_tag = row["runTag"]
        pc = np.load(row["out_npy_pc1"]).astype(np.float32)
        n_time, n_parcels = pc.shape

        mu = pc.mean(axis=0, keepdims=True)
        sd = pc.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)
        pc_z = (pc - mu) / sd
        frac = (np.abs(pc_z) > z_thr).mean(axis=1)

        confounds = pd.read_csv(row["confounds_tsv"], sep="\t")
        fd = confounds["framewise_displacement"].fillna(0).to_numpy().astype(np.float32)
        if len(fd) != n_time:
            raise RuntimeError(f"TR mismatch for {run_tag}: pc T={n_time} vs FD T={len(fd)}")

        qc_rows.append(
            {
                "runTag": run_tag,
                "T": int(n_time),
                "P": int(n_parcels),
                "max_frac_absz_gt3": float(frac.max()),
                "p95_frac_absz_gt3": float(np.percentile(frac, 95)),
                "mean_frac_absz_gt3": float(frac.mean()),
                "pct_TR_frac_gt_0p30": float((frac >= pct_thr_mark).mean()),
                "fd_mean": float(fd.mean()),
                "fd_p95": float(np.percentile(fd, 95)),
                "fd_max": float(fd.max()),
                "pct_fd_gt_0p5": float((fd > 0.5).mean()),
            }
        )

        if save_plots:
            fig_path = out_root / "qc" / f"qc_parcel_blowups_{run_tag}.png"
            fig_path.parent.mkdir(parents=True, exist_ok=True)

            x = np.arange(n_time)
            plt.figure(figsize=(12, 5))
            ax1 = plt.gca()
            ax1.plot(x, frac)
            ax1.set_ylim(0, 1)
            ax1.set_ylabel(f"Fraction parcels with |z|>{z_thr}")
            ax1.set_xlabel("TR")
            ax1.set_title(f"{run_tag} — parcel blowup fraction (|z|>{z_thr}) with FD overlay")

            bad = np.where(frac >= pct_thr_mark)[0]
            if len(bad) > 0:
                ax1.scatter(bad, frac[bad], marker="x")

            ax2 = ax1.twinx()
            ax2.plot(x, fd)
            ax2.set_ylabel("FD (mm)")

            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()

    qc = pd.DataFrame(qc_rows).sort_values(
        ["max_frac_absz_gt3", "p95_frac_absz_gt3", "pct_TR_frac_gt_0p30"],
        ascending=False,
    ).reset_index(drop=True)
    qc.to_csv(out_root / "qc" / "qc_parcel_blowups.csv", index=False)
    return qc
