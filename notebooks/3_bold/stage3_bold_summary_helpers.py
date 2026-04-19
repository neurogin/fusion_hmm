"""Helper functions for the cleaned Stage-3 BOLD QC and summary notebooks.

This module supports the public Stage-3 notebooks that build Table S4,
Table S5, and the reconstructed Figure S5 support outputs from saved BOLD
parcel-export sidecars.

Main inputs:
- exporter `dataset_index.csv`
- atlas QC CSVs
- exporter QC sidecars such as motion and parcel-blowup summaries

Main outputs:
- manuscript-facing table-support CSVs
- reconstructed BOLD QC summary figures

Important note:
- these helpers summarize saved Stage-3 outputs; they do not replace the
  upstream parcel-export notebook that generates those outputs
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _assert_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def build_table_s4_bold_parcel_atlas_summary(
    dataset_index_csv: str | Path,
    atlas_qc_csv: str | Path,
    out_csv: str | Path,
) -> pd.DataFrame:
    """Join exporter metadata with atlas-preservation QC for Table S4."""
    dataset_index_csv = Path(dataset_index_csv)
    atlas_qc_csv = Path(atlas_qc_csv)
    out_csv = Path(out_csv)

    idx = pd.read_csv(dataset_index_csv)
    atlas = pd.read_csv(atlas_qc_csv)

    idx_cols = ["runTag", "n_parcels", "qc_pct_all_nan_parcels", "qc_n_mean_fallback", "median_voxels_per_parcel"]
    atlas_cols = ["runTag", "n_labels_expected", "n_labels_present", "n_labels_missing"]

    merged = idx[idx_cols].merge(atlas[atlas_cols], on="runTag", how="inner", validate="one_to_one")
    if len(merged) != len(idx) or len(merged) != len(atlas):
        raise RuntimeError("Table S4 inputs do not have a one-to-one run match between dataset_index and atlas QC.")

    table = pd.DataFrame(
        {
            "Run": merged["runTag"],
            "Parcels": merged["n_parcels"].astype(int),
            "All-NaN parcels (%)": np.round(100.0 * merged["qc_pct_all_nan_parcels"].astype(float), 1),
            "Mean-fallback parcels (n)": merged["qc_n_mean_fallback"].astype(int),
            "Median voxels/parcel": np.round(merged["median_voxels_per_parcel"].astype(float), 1),
            "Atlas labels expected": merged["n_labels_expected"].astype(int),
            "Atlas labels present": merged["n_labels_present"].astype(int),
            "Atlas labels missing": merged["n_labels_missing"].astype(int),
        }
    ).sort_values("Run").reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    return table


def build_table_s5_bold_motion_nuisance_summary(
    dataset_index_csv: str | Path,
    out_csv: str | Path,
    manual_notes_by_run: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build the manuscript-facing motion and nuisance summary table."""
    dataset_index_csv = Path(dataset_index_csv)
    out_csv = Path(out_csv)
    manual_notes_by_run = manual_notes_by_run or {}

    idx = pd.read_csv(dataset_index_csv).sort_values("runTag").reset_index(drop=True)
    notes = [manual_notes_by_run.get(run_tag, "—") for run_tag in idx["runTag"]]

    table = pd.DataFrame(
        {
            "Run": idx["runTag"],
            "Volumes": idx["n_volumes"].astype(int),
            "FD mean (mm)": np.round(idx["qc_fd_mean"].astype(float), 3),
            "FD p95 (mm)": np.round(idx["qc_fd_p95"].astype(float), 3),
            "FD max (mm)": np.round(idx["qc_fd_max"].astype(float), 3),
            "FD spikes (%)": np.round(100.0 * idx["pct_fd_spikes"].astype(float), 3),
            "Motion-outlier TRs kept (%)": np.round(100.0 * idx["pct_motion_out_any_kept"].astype(float), 3),
            "Total regressors": idx["n_total_regressors"].astype(int),
            "Note": notes,
        }
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    return table


def build_figure_s5_reconstructed_from_exporter_qc(
    out_root: str | Path,
    out_png: str | Path,
) -> dict[str, str]:
    """Rebuild the available Figure-S5-style QC figure from saved sidecars."""
    out_root = Path(out_root)
    out_png = Path(out_png)

    motion_to_pc_csv = out_root / "qc_motion_to_pc.csv"
    blowups_csv = out_root / "qc" / "qc_parcel_blowups.csv"
    _assert_exists(motion_to_pc_csv, "qc_motion_to_pc.csv")
    _assert_exists(blowups_csv, "qc_parcel_blowups.csv")

    motion = pd.read_csv(motion_to_pc_csv).sort_values("max_abs_corr_fd_pc", ascending=False).reset_index(drop=True)
    blowups = pd.read_csv(blowups_csv).sort_values("max_frac_absz_gt3", ascending=False).reset_index(drop=True)

    worst_blowup_run = str(blowups.loc[0, "runTag"])
    worst_blowup_png = out_root / "qc" / f"qc_parcel_blowups_{worst_blowup_run}.png"
    _assert_exists(worst_blowup_png, "worst-run qc_parcel_blowups PNG")

    image = mpimg.imread(str(worst_blowup_png))

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.6])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(np.arange(len(motion)), motion["max_abs_corr_fd_pc"].to_numpy(dtype=float))
    ax1.axhline(0.1, linestyle="--")
    ax1.set_xticks(np.arange(len(motion)))
    ax1.set_xticklabels(motion["runTag"].tolist(), rotation=60, ha="right")
    ax1.set_ylabel("max |corr(FD, parcel PC1)|")
    ax1.set_title("Residual FD-to-parcel coupling by run")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(image)
    ax2.axis("off")
    ax2.set_title(f"Worst remaining parcel-blowup overlay from exporter QC: {worst_blowup_run}")

    fig.suptitle(
        "Figure S5 reconstructed from available exporter QC sidecars\n"
        "Exact original panel composition was not recovered from the repo snapshot.",
        y=0.98,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "motion_to_pc_csv": str(motion_to_pc_csv),
        "blowups_csv": str(blowups_csv),
        "worst_blowup_run": worst_blowup_run,
        "worst_blowup_png": str(worst_blowup_png),
        "out_png": str(out_png),
    }


def build_table_s5_and_figure_s5_bold_qc(
    out_root: str | Path,
    table_s5_csv: str | Path,
    figure_s5_png: str | Path,
    manual_notes_by_run: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Write both the Table S5 support CSV and the reconstructed Figure S5."""
    out_root = Path(out_root)
    table = build_table_s5_bold_motion_nuisance_summary(
        dataset_index_csv=out_root / "dataset_index.csv",
        out_csv=table_s5_csv,
        manual_notes_by_run=manual_notes_by_run,
    )
    figure_info = build_figure_s5_reconstructed_from_exporter_qc(
        out_root=out_root,
        out_png=figure_s5_png,
    )
    return table, figure_info
