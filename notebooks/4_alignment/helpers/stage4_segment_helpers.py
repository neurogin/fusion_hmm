"""Helper functions for the cleaned Stage-4 retained-segment workflow.

This module supports the public Stage-4 notebooks that convert per-run
alignment products into the final retained observation segments for the
fusion HMM pipeline.

Main inputs:
- per-run aligned BOLD and EEG feature arrays
- TR-edge files
- per-run keep masks for the requested feature mode and minimum segment
  length

Main outputs:
- retained segment arrays
- segment manifests
- per-run segment QC summaries and support plots

Important note:
- the public manuscript path is `FEATURE_MODE="nolags"` with `MINLEN=15`
- optional lagged branches remain available only as provenance support
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def contiguous_true_segments(mask_bool: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask_bool).astype(bool)
    if mask.ndim != 1:
        raise ValueError("mask must be 1D")
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.where(np.diff(idx) != 1)[0] + 1
    blocks = np.split(idx, breaks)
    return [(int(block[0]), int(block[-1] + 1)) for block in blocks]


def filter_segments_by_minlen(segs: list[tuple[int, int]], minlen: int) -> list[tuple[int, int]]:
    return [(start, end) for (start, end) in segs if (end - start) >= int(minlen)]


def safe_load_npy(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    return np.load(path, allow_pickle=False)


def ensure_2d_time_by_features(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D (T x F). got {arr.shape}")
    return arr


def finite_rows_mask(*arrays_2d: np.ndarray) -> np.ndarray:
    mask = None
    for arr in arrays_2d:
        arr = np.asarray(arr)
        rows_ok = np.isfinite(arr).all(axis=1)
        mask = rows_ok if mask is None else (mask & rows_ok)
    if mask is None:
        raise ValueError("finite_rows_mask needs at least one array")
    return mask


def resolve_mode_spec(feature_mode: str) -> dict[str, object]:
    mode = feature_mode.lower()
    if mode == "lags":
        return {"feature_mode": "lags", "lags_tr": (-1, 0, 1), "ltag": "lags-1_0_1", "eeg_file": "eeg_power_tr_lags.npy"}
    if mode == "nolags":
        return {"feature_mode": "nolags", "lags_tr": (0,), "ltag": "lags0", "eeg_file": "eeg_power_tr.npy"}
    raise ValueError("FEATURE_MODE must be 'lags' or 'nolags'")


def audit_segment_inputs(per_run_dir: str | Path, feature_mode: str, minlen: int) -> pd.DataFrame:
    per_run_dir = Path(per_run_dir)
    spec = resolve_mode_spec(feature_mode)
    runs = sorted([p.name for p in per_run_dir.iterdir() if p.is_dir()])
    rows = []
    for run in runs:
        run_dir = per_run_dir / run
        req = {
            "bold": run_dir / "bold_pc1.npy",
            "eeg": run_dir / str(spec["eeg_file"]),
            "tr_edges": run_dir / "tr_edges_sec.npy",
            "keep_mask": run_dir / f"keep_center_minlen{int(minlen)}_{spec['ltag']}.npy",
        }
        missing = [name for name, path in req.items() if not path.exists()]
        row = {
            "run": run,
            "feature_mode": str(spec["feature_mode"]),
            "minlen_tr": int(minlen),
            "ready": len(missing) == 0,
            "missing_inputs": ",".join(missing),
        }
        for name, path in req.items():
            row[f"{name}_path"] = str(path) if path.exists() else ""
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["ready", "run"], ascending=[False, True]).reset_index(drop=True)


def build_segments_dataset(
    per_run_dir: str | Path,
    feature_mode: str,
    minlen: int,
    tr_sec: float,
    out_root: str | Path | None = None,
    make_plots: bool = True,
) -> dict[str, object]:
    per_run_dir = Path(per_run_dir)
    spec = resolve_mode_spec(feature_mode)
    if out_root is None:
        out_root = per_run_dir.parent / f"hmm_segments_minlen{int(minlen)}_{spec['feature_mode']}"
    out_root = Path(out_root)
    out_seg = out_root / "segments"
    out_qc = out_root / "qc"
    out_plots = out_qc / "plots"
    out_seg.mkdir(parents=True, exist_ok=True)
    out_qc.mkdir(parents=True, exist_ok=True)
    out_plots.mkdir(parents=True, exist_ok=True)

    audit_df = audit_segment_inputs(per_run_dir, spec["feature_mode"], minlen)
    audit_csv = out_qc / f"segment_input_audit_minlen{int(minlen)}.csv"
    audit_df.to_csv(audit_csv, index=False)

    ok_runs = audit_df.loc[audit_df["ready"], "run"].tolist()
    manifest_rows = []
    per_run_rows = []
    for run in ok_runs:
        run_dir = per_run_dir / run
        bold = ensure_2d_time_by_features(safe_load_npy(run_dir / "bold_pc1.npy"), "bold_pc1")
        eeg = ensure_2d_time_by_features(safe_load_npy(run_dir / str(spec["eeg_file"])), str(spec["eeg_file"]))
        edges = np.asarray(safe_load_npy(run_dir / "tr_edges_sec.npy")).astype(float).ravel()
        keep = np.asarray(safe_load_npy(run_dir / f"keep_center_minlen{int(minlen)}_{spec['ltag']}.npy")).astype(bool).ravel()

        n_tr = min(bold.shape[0], eeg.shape[0], keep.size, edges.size - 1)
        bold = bold[:n_tr]
        eeg = eeg[:n_tr]
        keep = keep[:n_tr]
        edges = edges[: n_tr + 1]

        finite = finite_rows_mask(bold, eeg)
        valid = keep & finite
        segs = contiguous_true_segments(valid)
        segs = filter_segments_by_minlen(segs, minlen)

        n_written = 0
        total_tr = 0
        max_len = 0
        for idx, (start, end) in enumerate(segs):
            x = np.concatenate([bold[start:end], eeg[start:end]], axis=1).astype(np.float32)
            seg_id = f"{run}__seg{idx:04d}"
            seg_path = out_seg / f"{seg_id}.npy"
            np.save(seg_path, x)
            dur_sec = (end - start) * tr_sec
            manifest_rows.append(
                {
                    "run": run,
                    "feature_mode": str(spec["feature_mode"]),
                    "lags_tr": ",".join([str(int(x)) for x in spec["lags_tr"]]),
                    "seg_id": seg_id,
                    "start_TR": int(start),
                    "end_TR": int(end),
                    "len_TR": int(end - start),
                    "start_sec": float(edges[start]),
                    "end_sec": float(edges[end]),
                    "dur_sec": float(dur_sec),
                    "n_features": int(x.shape[1]),
                    "seg_path": str(seg_path),
                }
            )
            n_written += 1
            total_tr += int(end - start)
            max_len = max(max_len, int(end - start))

        per_run_rows.append(
            {
                "run": run,
                "feature_mode": str(spec["feature_mode"]),
                "lags_tr": ",".join([str(int(x)) for x in spec["lags_tr"]]),
                "T_total_TR": int(n_tr),
                f"n_segments_minlen{int(minlen)}": int(n_written),
                f"kept_TR_minlen{int(minlen)}": int(total_tr),
                f"usable_min_minlen{int(minlen)}": float(total_tr * tr_sec / 60.0),
                f"maxSeg_TR_minlen{int(minlen)}": int(max_len),
                "finite_drop_TR": int((keep & ~finite).sum()),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_root / "segments_manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)

    per_run_qc = pd.DataFrame(per_run_rows)
    if not per_run_qc.empty:
        per_run_qc = per_run_qc.sort_values(
            [f"usable_min_minlen{int(minlen)}", f"maxSeg_TR_minlen{int(minlen)}"],
            ascending=[False, False],
        )
    per_run_qc_path = out_qc / f"per_run_segments_minlen{int(minlen)}.csv"
    per_run_qc.to_csv(per_run_qc_path, index=False)

    if make_plots and not per_run_qc.empty and not manifest.empty:
        plt.figure()
        plt.bar(per_run_qc["run"], per_run_qc[f"usable_min_minlen{int(minlen)}"])
        plt.xticks(rotation=60, ha="right")
        plt.ylabel(f"usable minutes (minlen{int(minlen)})")
        plt.title(f"Usable minutes per run after minlen{int(minlen)}")
        plt.tight_layout()
        plt.savefig(out_plots / f"usable_minutes_minlen{int(minlen)}.png", dpi=180)
        plt.close()

        plt.figure()
        plt.bar(per_run_qc["run"], per_run_qc[f"n_segments_minlen{int(minlen)}"])
        plt.xticks(rotation=60, ha="right")
        plt.ylabel(f"segment count (minlen{int(minlen)})")
        plt.title(f"Segment count per run after minlen{int(minlen)}")
        plt.tight_layout()
        plt.savefig(out_plots / f"segment_count_minlen{int(minlen)}.png", dpi=180)
        plt.close()

        plt.figure()
        plt.hist(manifest["len_TR"], bins=30)
        plt.xlabel("segment length (TR)")
        plt.ylabel("count")
        plt.title(f"Distribution of segment lengths (minlen{int(minlen)})")
        plt.tight_layout()
        plt.savefig(out_plots / f"segment_length_distribution_minlen{int(minlen)}.png", dpi=180)
        plt.close()

    return {
        "audit_df": audit_df,
        "manifest_df": manifest,
        "per_run_qc_df": per_run_qc,
        "audit_csv": str(audit_csv),
        "manifest_tsv": str(manifest_path),
        "per_run_qc_csv": str(per_run_qc_path),
        "out_root": str(out_root),
        "feature_mode": str(spec["feature_mode"]),
        "lags_tr": list(spec["lags_tr"]),
        "minlen": int(minlen),
    }


def build_table_s6_alignment_parameters(
    alignment_parameters_json: str | Path,
    feature_mode: str,
    minlen: int,
    out_csv: str | Path,
) -> pd.DataFrame:
    with open(alignment_parameters_json, "r", encoding="utf-8") as handle:
        params = json.load(handle)

    rows = [
        {
            "Parameter": "Timeline reconciliation",
            "Final value": "Raw-to-preprocessed EEG mapping",
            "Notes": "Run-specific mapping estimated from matched recurring R128 events; anchored to first raw S1 event.",
        },
        {"Parameter": "BOLD TR", "Final value": f"{params['TR_SEC']} s", "Notes": "Common temporal grid used for EEG-BOLD fusion."},
        {"Parameter": "EEG sampling rate", "Final value": f"{params['EEG_FS_HZ']} Hz", "Notes": "Used to assign EEG samples to TR bins and evaluate counts-per-TR."},
        {"Parameter": "TR-mask policy", "Final value": "Intermediate-strict", "Notes": "Final retained-mask rule used in the manuscript."},
        {"Parameter": "Base EEG coverage threshold", "Final value": f"\u2265 {params['BASE_COVERAGE_THR']:.2f} of TR", "Notes": "TR retained directly when this much of the bin is usable."},
        {"Parameter": "Hybrid rescue threshold", "Final value": f"\u2265 {params['HYBRID_MIN_COVERAGE']:.2f} of TR", "Notes": "Partially contaminated TR can still be retained if rescue criteria are met."},
        {
            "Parameter": "Hybrid rescue continuity rule",
            "Final value": f"Single contiguous retained block spanning \u2265 {params['HYBRID_MIN_GOOD_BLOCK_FRAC']:.2f} of the TR",
            "Notes": "Prevents fragmented EEG support inside a retained TR.",
        },
        {
            "Parameter": "Sample-completeness gate",
            "Final value": f"max({int(params['EEG_MIN_SAMPLES_MIN'])}, {float(params['EEG_MIN_SAMPLES_FRAC']):.2f} \u00d7 expected EEG samples/TR)",
            "Notes": "Applied after interval masking to prevent sparse or NaN-prone TR bins.",
        },
        {
            "Parameter": "Trigger dependency",
            "Final value": "Recurring R128 in raw and preprocessed EEG events; raw S1 anchor",
            "Notes": "Accepted label columns: trial_type, value, type. Accepted time columns: onset, start, start_sec, time, latency_sec, latency.",
        },
        {
            "Parameter": "Offset jump threshold exposure",
            "Final value": f"Notebook-exposed {float(params['OFFSET_JUMP_THR_EXPOSED_BUT_UNUSED']):.2f} s; active split uses {float(params['OFFSET_JUMP_THR_ACTIVE_IN_MATCH_SPLIT']):.1f} s",
            "Notes": "Preserved as-is from the original notebook; this mismatch was not harmonized silently.",
        },
        {"Parameter": "Feature mode", "Final value": str(feature_mode), "Notes": "Final manuscript dataset uses the no-lag branch."},
        {"Parameter": "Minimum retained segment length", "Final value": f"{int(minlen)} TR", "Notes": "Only contiguous retained intervals of at least this length were exported."},
    ]
    table = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    return table


def build_table_s7_run_level_summary(
    per_run_segments_csv: str | Path,
    out_csv: str | Path,
    minlen: int,
) -> pd.DataFrame:
    df = pd.read_csv(per_run_segments_csv).sort_values("run").reset_index(drop=True)
    table = pd.DataFrame(
        {
            "Run": df["run"],
            "Feature mode": df["feature_mode"],
            "Lags (TR)": df["lags_tr"],
            "Total TRs": df["T_total_TR"].astype(int),
            f"Segments (minlen{int(minlen)})": df[f"n_segments_minlen{int(minlen)}"].astype(int),
            "Retained TRs": df[f"kept_TR_minlen{int(minlen)}"].astype(int),
            "Usable min": np.round(df[f"usable_min_minlen{int(minlen)}"].astype(float), 3),
            "Max segment (TR)": df[f"maxSeg_TR_minlen{int(minlen)}"].astype(int),
            "Retained (%)": np.round(100.0 * df[f"kept_TR_minlen{int(minlen)}"].astype(float) / df["T_total_TR"].astype(float), 1),
        }
    )
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    return table


def build_final_dataset_summary(
    manifest_tsv: str | Path,
    out_json: str | Path,
) -> dict[str, object]:
    manifest = pd.read_csv(manifest_tsv, sep="\t")
    if manifest.empty:
        summary = {"n_runs": 0, "n_segments": 0, "retained_TRs": 0, "usable_minutes": 0.0, "n_features": 0}
    else:
        summary = {
            "n_runs": int(manifest["run"].nunique()),
            "n_segments": int(len(manifest)),
            "retained_TRs": int(manifest["len_TR"].sum()),
            "usable_minutes": float(manifest["dur_sec"].sum() / 60.0),
            "n_features": int(manifest["n_features"].iloc[0]),
            "feature_mode": str(manifest["feature_mode"].iloc[0]),
            "lags_tr": str(manifest["lags_tr"].iloc[0]),
        }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def build_figure1_support_outputs(
    alignment_output_dir: str | Path,
    segment_root: str | Path,
    representative_run: str | None,
    out_dir: str | Path,
    minlen: int,
) -> dict[str, str]:
    alignment_output_dir = Path(alignment_output_dir)
    segment_root = Path(segment_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run_qc = pd.read_csv(segment_root / "qc" / f"per_run_segments_minlen{int(minlen)}.csv")
    manifest = pd.read_csv(segment_root / "segments_manifest.tsv", sep="\t")

    retained_png = out_dir / "figure1_retained_tr_fraction_by_run.png"
    plt.figure(figsize=(10, 4))
    retained_pct = 100.0 * per_run_qc[f"kept_TR_minlen{int(minlen)}"].astype(float) / per_run_qc["T_total_TR"].astype(float)
    plt.bar(per_run_qc["run"], retained_pct)
    plt.xticks(rotation=90, fontsize=8)
    plt.ylabel("Retained TR (%)")
    plt.title(f"Final no-lag minlen{int(minlen)} retained fraction by run")
    plt.tight_layout()
    plt.savefig(retained_png, dpi=180)
    plt.close()

    length_png = out_dir / "figure1_segment_length_distribution.png"
    plt.figure(figsize=(6, 4))
    plt.hist(manifest["len_TR"], bins=30)
    plt.xlabel("Segment length (TR)")
    plt.ylabel("Count")
    plt.title(f"Segment length distribution (minlen{int(minlen)})")
    plt.tight_layout()
    plt.savefig(length_png, dpi=180)
    plt.close()

    available_runs = sorted([p.name for p in (alignment_output_dir / "per_run").iterdir() if p.is_dir()])
    if representative_run is None or representative_run not in available_runs:
        representative_run = available_runs[0] if available_runs else None

    representative_png = out_dir / "figure1_representative_run_keep_mask.png"
    if representative_run is not None:
        run_dir = alignment_output_dir / "per_run" / representative_run
        coverage = np.load(run_dir / "coverage.npy")
        keep_hybrid = np.load(run_dir / "keep_hybridG2.npy").astype(bool)
        keep_center = np.load(run_dir / f"keep_center_minlen{int(minlen)}_lags0.npy").astype(bool)
        plt.figure(figsize=(11, 3.5))
        plt.plot(coverage, label="coverage", linewidth=1)
        plt.plot(keep_hybrid.astype(int), label="keep_hybridG2", linewidth=1)
        plt.plot(keep_center.astype(int), label=f"keep_center_minlen{int(minlen)}_lags0", linewidth=1)
        plt.xlabel("TR index")
        plt.ylabel("coverage / keep")
        plt.title(f"Representative aligned run: {representative_run}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(representative_png, dpi=180)
        plt.close()

    info = {
        "retained_fraction_png": str(retained_png),
        "segment_length_png": str(length_png),
        "representative_run": representative_run or "",
        "representative_png": str(representative_png) if representative_run is not None else "",
        "note": "Figure 1 still requires hybrid/manual final assembly around these scripted support plots.",
    }
    info_json = out_dir / "figure1_support_manifest.json"
    with open(info_json, "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)
    info["manifest_json"] = str(info_json)
    return info
