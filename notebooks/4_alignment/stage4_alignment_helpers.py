"""Helper functions for the cleaned Stage-4 EEG-BOLD alignment workflow.

This module supports the public Stage-4 notebooks that reconcile raw and
preprocessed EEG timelines, project usable EEG intervals onto the BOLD TR
axis, and write per-run alignment products for the final fusion dataset.

Main inputs:
- raw EEG event TSVs
- preprocessed EEG event TSVs
- exclusion-union TSVs
- EEG parcel NPY exports including `*_time_sec.npy`
- BOLD parcel PC1 NPY exports

Main outputs:
- per-run alignment products such as keep masks, aligned TR summaries, and
  run-input audit tables

Important note:
- the preserved trigger logic remains explicit here:
  recurring `R128` events anchor raw-to-preprocessed reconciliation, with
  raw `S1` used as the absolute anchor
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _find_col(df: pd.DataFrame, candidates: list[str] | tuple[str, ...]) -> str | None:
    cols = {str(c).lower(): str(c) for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def ensure_2d_time_major(x: np.ndarray, n_parcels_expected: int = 200) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")
    n_time, n_parcels = x.shape
    if n_parcels == n_parcels_expected:
        return x
    if n_time == n_parcels_expected:
        return x.T
    print(
        f"[WARN] array shape {x.shape} does not match expected parcels={n_parcels_expected}. "
        "Using as-is."
    )
    return x


def run_id_from_fname(fname: str) -> str | None:
    match = re.search(r"(sub-\d+_ses-\d+)", fname)
    return match.group(1) if match else None


def _map_files_by_run(directory: Path, pattern: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in directory.glob(pattern):
        run_id = run_id_from_fname(path.name)
        if run_id is not None:
            out[run_id] = path
    return out


def discover_input_maps(
    raw_events_dir: str | Path,
    preproc_events_dir: str | Path,
    excl_union_dir: str | Path,
    eeg_parcel_npy_dir: str | Path,
    bold_parcel_npy_dir: str | Path,
) -> dict[str, dict[str, Path]]:
    raw_events_dir = Path(raw_events_dir)
    preproc_events_dir = Path(preproc_events_dir)
    excl_union_dir = Path(excl_union_dir)
    eeg_parcel_npy_dir = Path(eeg_parcel_npy_dir)
    bold_parcel_npy_dir = Path(bold_parcel_npy_dir)

    return {
        "bold": _map_files_by_run(bold_parcel_npy_dir, "sub-*_ses-*_task-rest_parcel_pc1.npy"),
        "eeg_pc1": _map_files_by_run(eeg_parcel_npy_dir, "sub-*_ses-*_desc-ICRej70_clean_PC1_gnorm.npy"),
        "eeg_time": _map_files_by_run(eeg_parcel_npy_dir, "sub-*_ses-*_desc-ICRej70_clean_time_sec.npy"),
        "raw_events": _map_files_by_run(raw_events_dir, "sub-*_ses-*_task-rest_events.tsv"),
        "preproc_events": _map_files_by_run(preproc_events_dir, "sub-*_ses-*_task-rest_events.tsv"),
        "excl_union": _map_files_by_run(excl_union_dir, "sub-*_ses-*_desc-ICRej70_clean_excl_union.tsv"),
    }


def build_run_input_audit(input_maps: dict[str, dict[str, Path]]) -> pd.DataFrame:
    all_runs = sorted({run for mapping in input_maps.values() for run in mapping.keys()})
    rows: list[dict[str, object]] = []
    required = ["bold", "eeg_pc1", "eeg_time", "raw_events", "preproc_events", "excl_union"]
    for run in all_runs:
        row: dict[str, object] = {"run": run}
        missing: list[str] = []
        for name in required:
            present = run in input_maps.get(name, {})
            row[f"has_{name}"] = bool(present)
            row[f"{name}_path"] = str(input_maps[name][run]) if present else ""
            if not present:
                missing.append(name)
        row["ready"] = len(missing) == 0
        row["missing_inputs"] = ",".join(missing)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["ready", "run"], ascending=[False, True]).reset_index(drop=True)


def extract_event_times(
    df: pd.DataFrame,
    key_regex: str,
    label_cols: tuple[str, ...] = ("trial_type", "value", "type"),
    time_cols: list[str] | None = None,
) -> np.ndarray:
    if time_cols is None:
        time_cols = ["onset", "start", "start_sec", "time", "latency_sec", "latency"]

    col_t = _find_col(df, time_cols)
    if col_t is None:
        raise ValueError(f"Could not find a time column among {time_cols}. Columns: {list(df.columns)}")

    times_list = []
    for col in label_cols:
        if col not in df.columns:
            continue
        series = df[col].astype(str)
        mask = series.str.contains(key_regex, regex=True, na=False)
        if mask.any():
            times_list.append(df.loc[mask, col_t].astype(float).to_numpy())

    if len(times_list) == 0:
        present = [c for c in label_cols if c in df.columns]
        examples: dict[str, list[str]] = {}
        for col in present:
            values = df[col].astype(str).unique()
            examples[col] = values[:10].tolist()
        raise ValueError(
            f"No events matched /{key_regex}/ in columns {present}. "
            f"Examples: {examples}"
        )

    times = np.concatenate(times_list)
    times = np.unique(times)
    return np.sort(times)


def greedy_match_intervals(dt_raw: np.ndarray, dt_pre: np.ndarray, tol: float = 0.15) -> list[tuple[int, int]]:
    matches: list[tuple[int, int]] = []
    j = 0
    for i in range(len(dt_raw)):
        if j >= len(dt_pre):
            break
        if abs(dt_raw[i] - dt_pre[j]) <= tol:
            matches.append((i, j))
            j += 1
    return matches


def dp_monotone_match_intervals(dt_raw: np.ndarray, dt_pre: np.ndarray, tol: float = 0.15) -> list[tuple[int, int]]:
    n_raw = len(dt_raw)
    n_pre = len(dt_pre)
    if n_raw == 0 or n_pre == 0:
        return []

    dp = np.zeros((n_raw + 1, n_pre + 1), dtype=int)
    back = np.zeros((n_raw + 1, n_pre + 1, 2), dtype=int)

    for i in range(n_raw):
        for j in range(n_pre):
            if dp[i + 1, j] >= dp[i, j + 1]:
                best = dp[i + 1, j]
                pi, pj = i + 1, j
            else:
                best = dp[i, j + 1]
                pi, pj = i, j + 1

            if abs(dt_raw[i] - dt_pre[j]) <= tol:
                cand = dp[i, j] + 1
                if cand > best:
                    best = cand
                    pi, pj = i, j

            dp[i + 1, j + 1] = best
            back[i + 1, j + 1] = (pi, pj)

    i, j = n_raw, n_pre
    matches: list[tuple[int, int]] = []
    while i > 0 and j > 0:
        pi, pj = back[i, j]
        if pi == i - 1 and pj == j - 1 and abs(dt_raw[i - 1] - dt_pre[j - 1]) <= tol:
            matches.append((i - 1, j - 1))
        i, j = pi, pj

    matches.reverse()
    return matches


def build_segment_offsets_from_matches(
    raw_r128: np.ndarray,
    pre_r128: np.ndarray,
    interval_matches: list[tuple[int, int]],
    min_matches: int = 5,
) -> tuple[list[dict[str, float]], dict[str, float | int]]:
    if len(interval_matches) < min_matches:
        raise ValueError("Too few interval matches to build a stable mapping.")

    i_idx = np.array([i + 1 for (i, _) in interval_matches], dtype=int)
    j_idx = np.array([j + 1 for (_, j) in interval_matches], dtype=int)

    offsets = raw_r128[i_idx] - pre_r128[j_idx]

    # Preserve the original active split threshold exactly.
    jumps = np.where(np.abs(np.diff(offsets)) > 0.5)[0]
    seg_starts = np.r_[0, jumps + 1]
    seg_ends = np.r_[jumps + 1, len(offsets)]

    segments: list[dict[str, float]] = []
    for start, end in zip(seg_starts, seg_ends):
        offset = float(np.median(offsets[start:end]))
        pre_start = float(pre_r128[j_idx[start]])
        pre_end = float(pre_r128[j_idx[end - 1]])
        segments.append({"pre_start": pre_start, "pre_end": pre_end, "offset": offset})

    segments[0]["pre_start"] = 0.0
    segments[-1]["pre_end"] = float("inf")

    pred = np.zeros_like(offsets)
    for seg in segments:
        mask = (pre_r128[j_idx] >= seg["pre_start"]) & (pre_r128[j_idx] <= seg["pre_end"])
        pred[mask] = pre_r128[j_idx[mask]] + seg["offset"]

    err = pred - raw_r128[i_idx]
    qc: dict[str, float | int] = {
        "med_abs_sec": float(np.median(np.abs(err))),
        "p95_abs_sec": float(np.percentile(np.abs(err), 95)),
        "mono_viol": int(np.sum(np.diff(j_idx) <= 0)),
        "n_pre_r128": int(len(pre_r128)),
        "n_raw_r128": int(len(raw_r128)),
        "n_boundaries_used": int(len(segments) - 1),
        "TR_hat_sec": float(np.median(np.diff(raw_r128))) if len(raw_r128) > 3 else np.nan,
    }
    return segments, qc


def align_raw_preproc(raw_events_tsv: str | Path, preproc_events_tsv: str | Path, dt_tol_sec: float = 0.15):
    raw_df = _read_tsv(Path(raw_events_tsv))
    pre_df = _read_tsv(Path(preproc_events_tsv))

    raw_r128 = extract_event_times(raw_df, r"R128")
    pre_r128 = extract_event_times(pre_df, r"R128")

    if len(raw_r128) < 10 or len(pre_r128) < 10:
        raise ValueError(
            f"R128 too few: raw={len(raw_r128)} pre={len(pre_r128)} "
            f"({Path(raw_events_tsv).name} vs {Path(preproc_events_tsv).name})"
        )

    dt_raw = np.diff(raw_r128)
    dt_pre = np.diff(pre_r128)

    matches = greedy_match_intervals(dt_raw, dt_pre, tol=dt_tol_sec)
    method = "greedy"
    ok = False

    try:
        segments, qc = build_segment_offsets_from_matches(raw_r128, pre_r128, matches)
        ok = bool((qc["p95_abs_sec"] < 1e-6) and (qc["mono_viol"] == 0))
    except Exception:
        ok = False

    if not ok:
        matches = dp_monotone_match_intervals(dt_raw, dt_pre, tol=dt_tol_sec)
        segments, qc = build_segment_offsets_from_matches(raw_r128, pre_r128, matches)
        method = "DP_monotone"

    qc["method"] = method

    s1 = extract_event_times(raw_df, r"\bS\s*1\b|^S1$")
    if len(s1) == 0:
        s1 = extract_event_times(raw_df, r"S\s*1")
    s1_t0 = float(s1[0])

    for seg in segments:
        seg["offset"] = float(seg["offset"] - s1_t0)
    qc["raw_S1_t0_sec"] = s1_t0

    return segments, qc, raw_r128 - s1_t0


def load_union_intervals(union_tsv: str | Path) -> np.ndarray:
    df = _read_tsv(Path(union_tsv))
    col_s = _find_col(df, ["start_sec", "start", "onset"])
    col_e = _find_col(df, ["end_sec", "end"])
    if col_s is None or col_e is None:
        raise ValueError(f"Union TSV missing start/end columns: {df.columns}")
    intervals = df[[col_s, col_e]].astype(float).to_numpy()
    return intervals[np.argsort(intervals[:, 0])]


def map_intervals_pre_to_raw(intervals_pre: np.ndarray, segments: list[dict[str, float]]) -> np.ndarray:
    pieces = []
    for start_pre, end_pre in intervals_pre:
        if end_pre <= start_pre:
            continue
        for seg in segments:
            pre_start = seg["pre_start"]
            pre_end = seg["pre_end"]
            offset = seg["offset"]
            lo = max(start_pre, pre_start)
            hi = min(end_pre, pre_end)
            if hi > lo:
                pieces.append((lo + offset, hi + offset))
    pieces_arr = np.array(pieces, dtype=float)
    if len(pieces_arr):
        pieces_arr = pieces_arr[np.argsort(pieces_arr[:, 0])]
    return pieces_arr


def canonicalize_union(intervals: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, float]:
    if len(intervals) == 0:
        return intervals, 0.0

    intervals = intervals[np.argsort(intervals[:, 0])]
    overlap = 0.0
    for i in range(1, len(intervals)):
        overlap += max(0.0, intervals[i - 1, 1] - intervals[i, 0])

    merged = [intervals[0].tolist()]
    for start_sec, end_sec in intervals[1:]:
        if start_sec <= merged[-1][1] + eps:
            merged[-1][1] = max(merged[-1][1], end_sec)
        else:
            merged.append([start_sec, end_sec])
    return np.array(merged, dtype=float), float(overlap)


def union_duration(intervals: np.ndarray) -> float:
    if len(intervals) == 0:
        return 0.0
    return float(np.sum(intervals[:, 1] - intervals[:, 0]))


def save_intervals_tsv(intervals: np.ndarray, out_path: str | Path, label: str = "excl_union_raw") -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "label": label,
            "start_sec": intervals[:, 0] if len(intervals) else [],
            "end_sec": intervals[:, 1] if len(intervals) else [],
        }
    )
    df.to_csv(out_path, sep="\t", index=False)


def estimate_tr_offset_from_r128(raw_r128_shifted: np.ndarray, tr_sec: float, n_tr: int) -> float:
    k = np.rint(raw_r128_shifted / tr_sec).astype(int)
    mask = (k >= 0) & (k <= n_tr)
    if not np.any(mask):
        return 0.0
    return float(np.median(raw_r128_shifted[mask] - k[mask] * tr_sec))


def tr_edges_from_bold(n_tr: int, tr_sec: float, offset: float = 0.0) -> np.ndarray:
    return offset + np.arange(n_tr + 1, dtype=float) * tr_sec


def intersect_len(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def compute_tr_excl_dur(tr_edges: np.ndarray, union_raw: np.ndarray) -> np.ndarray:
    n_tr = len(tr_edges) - 1
    excl = np.zeros(n_tr, dtype=float)
    if len(union_raw) == 0:
        return excl

    j = 0
    for k in range(n_tr):
        t0, t1 = tr_edges[k], tr_edges[k + 1]
        while j < len(union_raw) and union_raw[j, 1] <= t0:
            j += 1
        jj = j
        while jj < len(union_raw) and union_raw[jj, 0] < t1:
            excl[k] += intersect_len(t0, t1, union_raw[jj, 0], union_raw[jj, 1])
            jj += 1
    return excl


def complement_intervals_in_window(t0: float, t1: float, union_raw: np.ndarray) -> list[tuple[float, float]]:
    kept = []
    cur = t0
    for start_sec, end_sec in union_raw:
        if end_sec <= t0:
            continue
        if start_sec >= t1:
            break
        if start_sec > cur:
            kept.append((cur, min(start_sec, t1)))
        cur = max(cur, end_sec)
        if cur >= t1:
            break
    if cur < t1:
        kept.append((cur, t1))
    return [(a, b) for (a, b) in kept if b > a]


def build_keep_masks(
    tr_edges: np.ndarray,
    union_raw: np.ndarray,
    tr_sec: float,
    base_thr: float,
    hybrid_min_cov: float,
    hybrid_min_block_frac: float,
):
    n_tr = len(tr_edges) - 1
    excl_dur = compute_tr_excl_dur(tr_edges, union_raw)
    excl_frac = excl_dur / tr_sec
    coverage = 1.0 - excl_frac

    keep_base = coverage >= base_thr
    keep_hybrid = np.zeros(n_tr, dtype=bool)
    for k in range(n_tr):
        cov = coverage[k]
        if cov >= base_thr:
            keep_hybrid[k] = True
            continue
        if cov < hybrid_min_cov:
            keep_hybrid[k] = False
            continue

        t0, t1 = tr_edges[k], tr_edges[k + 1]
        kept_int = complement_intervals_in_window(t0, t1, union_raw)
        if len(kept_int) == 1:
            good_len = kept_int[0][1] - kept_int[0][0]
            if good_len >= hybrid_min_block_frac * tr_sec:
                keep_hybrid[k] = True

    def seg_stats(mask: np.ndarray) -> tuple[int, int]:
        best = 0
        cur = 0
        n_seg = 0
        in_seg = False
        for value in mask:
            if value:
                cur += 1
                if not in_seg:
                    n_seg += 1
                    in_seg = True
                best = max(best, cur)
            else:
                cur = 0
                in_seg = False
        return int(best), int(n_seg)

    max_seg_base, n_seg_base = seg_stats(keep_base)
    max_seg_hybrid, n_seg_hybrid = seg_stats(keep_hybrid)

    stats = {
        "N_TR": int(n_tr),
        "kept_TR_base": int(np.sum(keep_base)),
        "kept_TR_hybridG2": int(np.sum(keep_hybrid)),
        "maxSeg_TR_base": int(max_seg_base),
        "nSeg_base": int(n_seg_base),
        "maxSeg_TR_hybridG2": int(max_seg_hybrid),
        "nSeg_hybridG2": int(n_seg_hybrid),
    }
    return excl_dur, excl_frac, coverage, keep_base, keep_hybrid, stats


def segments_from_mask(mask: np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    segs: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        if (not value) and start is not None:
            if idx - start >= min_len:
                segs.append((start, idx))
            start = None
    if start is not None and (len(mask) - start) >= min_len:
        segs.append((start, len(mask)))
    return segs


def keep_center_for_lags(keep_bool: np.ndarray, lags: tuple[int, ...] | list[int]) -> np.ndarray:
    keep = np.asarray(keep_bool).astype(bool)
    n_tr = len(keep)
    center = np.ones(n_tr, dtype=bool)
    for lag in lags:
        idx = np.arange(n_tr) + lag
        valid = (idx >= 0) & (idx < n_tr)
        tmp = np.zeros(n_tr, dtype=bool)
        tmp[valid] = keep[idx[valid]]
        center &= tmp
    return center


def mask_from_segments(n: int, segs: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for start, end in segs:
        mask[start:end] = True
    return mask


def segments_to_tsv(segs: list[tuple[int, int]], tr_edges: np.ndarray, out_path: str | Path) -> pd.DataFrame:
    rows = []
    for start, end in segs:
        rows.append(
            {
                "start_TR": int(start),
                "end_TR": int(end),
                "len_TR": int(end - start),
                "start_sec": float(tr_edges[start]),
                "end_sec": float(tr_edges[end]),
                "dur_sec": float(tr_edges[end] - tr_edges[start]),
            }
        )
    df = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    return df


def load_eeg_pc_and_time(run: str, eeg_pc_map: dict[str, Path], eeg_time_map: dict[str, Path]) -> tuple[np.ndarray, np.ndarray]:
    pc = np.load(eeg_pc_map[run])
    pc = ensure_2d_time_major(pc)
    time_sec = np.load(eeg_time_map[run]).astype(float).reshape(-1)
    if pc.shape[0] != time_sec.shape[0]:
        raise ValueError(f"EEG PC and time mismatch for {run}: pc {pc.shape}, t {time_sec.shape}")
    return pc, time_sec


def eeg_power_per_tr(
    pc: np.ndarray,
    t_sec: np.ndarray,
    tr_edges: np.ndarray,
    tr_sec: float,
    min_samples_per_tr: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    n_tr = len(tr_edges) - 1
    _, n_parcels = pc.shape
    power = np.full((n_tr, n_parcels), np.nan, dtype=float)
    counts = np.zeros(n_tr, dtype=int)

    mask = (t_sec >= tr_edges[0]) & (t_sec < tr_edges[-1])
    if not np.any(mask):
        return power, counts

    t = t_sec[mask]
    x2 = pc[mask, :] ** 2

    bin_idx = np.floor((t - tr_edges[0]) / tr_sec).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_tr - 1)
    counts = np.bincount(bin_idx, minlength=n_tr).astype(int)

    ok = counts >= int(min_samples_per_tr)
    denom = counts.astype(float)
    denom[~ok] = np.nan

    chunk = 25
    for p0 in range(0, n_parcels, chunk):
        p1 = min(n_parcels, p0 + chunk)
        s = np.zeros((n_tr, p1 - p0), dtype=float)
        np.add.at(s, bin_idx, x2[:, p0:p1])
        power[:, p0:p1] = s / denom[:, None]

    return power, counts


def build_lagged_features(x: np.ndarray, lags: tuple[int, ...] | list[int], pad: float = np.nan) -> np.ndarray:
    n_rows, n_features = x.shape
    out = np.full((n_rows, n_features * len(lags)), pad, dtype=float)
    for li, lag in enumerate(lags):
        if lag == 0:
            out[:, li * n_features : (li + 1) * n_features] = x
        elif lag < 0:
            out[-lag:, li * n_features : (li + 1) * n_features] = x[: n_rows + lag, :]
        else:
            out[: n_rows - lag, li * n_features : (li + 1) * n_features] = x[lag:, :]
    return out


def _lags_tag(lags: tuple[int, ...] | list[int]) -> str:
    return "lags" + "_".join([str(int(x)) for x in lags])


def _minlen_summary(
    run: str,
    keep_hybrid: np.ndarray,
    keep_center: np.ndarray,
    segs_center_all: list[tuple[int, int]],
    segs_center_minlen: list[tuple[int, int]],
    tr_sec: float,
    minlen: int,
    lags_used: tuple[int, ...] | list[int],
) -> dict[str, object]:
    n_tr = len(keep_hybrid)
    kept_tr_center_minlen = int(sum((end - start) for (start, end) in segs_center_minlen))
    duration_sec = float(kept_tr_center_minlen * tr_sec)
    return {
        "run": run,
        "TR_SEC": float(tr_sec),
        "lags_used": [int(x) for x in lags_used],
        "minlen_TR": int(minlen),
        "N_TR": int(n_tr),
        "kept_TR_hybridG2": int(np.sum(keep_hybrid)),
        "kept_TR_center": int(np.sum(keep_center)),
        "keep_frac_center": float(np.mean(keep_center)) if n_tr else np.nan,
        "nSeg_center_all": int(len(segs_center_all)),
        "maxSeg_center_all": int(max([end - start for (start, end) in segs_center_all], default=0)),
        "nSeg_center_minlen": int(len(segs_center_minlen)),
        "maxSeg_center_minlen": int(max([end - start for (start, end) in segs_center_minlen], default=0)),
        "kept_TR_center_minlen": kept_tr_center_minlen,
        "duration_center_minlen_sec": duration_sec,
        "duration_center_minlen_min": duration_sec / 60.0,
    }


def plot_run_diagnostics(
    run: str,
    coverage: np.ndarray,
    keep_hybrid: np.ndarray,
    bold_pc: np.ndarray,
    out_dir: str | Path,
    base_coverage_thr: float,
    hybrid_min_coverage: float,
    example_parcel_index: int,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    plt.plot(coverage, marker="o", linestyle="-", linewidth=1, markersize=3)
    dropped = np.where(~keep_hybrid)[0]
    plt.scatter(dropped, coverage[dropped], s=18)
    plt.axhline(base_coverage_thr, linestyle="--")
    plt.axhline(hybrid_min_coverage, linestyle="--")
    plt.title(f"{run}: TR coverage + keep mask (all TRs)")
    plt.xlabel("TR index")
    plt.ylabel("Coverage (1 - excl_frac)")
    fig.savefig(out_dir / f"{run}_coverage_keep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    segs = segments_from_mask(keep_hybrid, min_len=1)
    lengths = [end - start for (start, end) in segs]
    fig = plt.figure()
    plt.hist(lengths, bins=30)
    plt.title(f"{run}: contiguous kept segment lengths (Hybrid-G2)")
    plt.xlabel("segment length (TR)")
    plt.ylabel("count")
    fig.savefig(out_dir / f"{run}_kept_segment_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    y = bold_pc[:, example_parcel_index].astype(float).copy()
    y[~keep_hybrid] = np.nan
    fig = plt.figure()
    plt.plot(y)
    plt.title(f"{run}: BOLD PC1 parcel[{example_parcel_index}] (dropped TRs -> NaN)")
    plt.xlabel("TR index")
    plt.ylabel("PC1 (a.u.)")
    fig.savefig(out_dir / f"{run}_bold_example_nan.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def process_one_run_alignment(
    run: str,
    input_maps: dict[str, dict[str, Path]],
    out_runs: Path,
    tr_sec: float,
    eeg_fs_hz: float,
    lags_tr: tuple[int, ...] | list[int],
    lags_tr_no: tuple[int, ...] | list[int],
    base_coverage_thr: float,
    hybrid_min_coverage: float,
    hybrid_min_good_block_frac: float,
    eeg_min_samples_frac: float,
    eeg_min_samples_min: int,
    dt_tol_sec: float,
    report_r128_tr_offset: bool,
    apply_r128_tr_offset: bool,
    make_plots: bool,
    plot_run_set: set[str],
    example_parcel_index: int,
) -> dict[str, object]:
    bold_pc = np.load(input_maps["bold"][run])
    bold_pc = ensure_2d_time_major(bold_pc)
    n_tr = bold_pc.shape[0]

    segments, qc_align, raw_r128_shifted = align_raw_preproc(
        input_maps["raw_events"][run],
        input_maps["preproc_events"][run],
        dt_tol_sec=dt_tol_sec,
    )

    tr_offset = 0.0
    if report_r128_tr_offset:
        tr_offset = estimate_tr_offset_from_r128(raw_r128_shifted, tr_sec, n_tr)
        qc_align["r128_tr_offset_est_sec"] = tr_offset

    if apply_r128_tr_offset:
        tr_edges = tr_edges_from_bold(n_tr, tr_sec, offset=tr_offset)
    else:
        tr_edges = tr_edges_from_bold(n_tr, tr_sec, offset=0.0)

    union_pre = load_union_intervals(input_maps["excl_union"][run])
    pieces_raw = map_intervals_pre_to_raw(union_pre, segments)
    union_raw, overlap_before = canonicalize_union(pieces_raw)
    map_stats = {
        "dur_preproc_sec": union_duration(union_pre),
        "dur_mapped_pieces_sec": union_duration(pieces_raw),
        "dur_mapped_union_sec": union_duration(union_raw),
        "overlap_sec_canonical": float(overlap_before),
        "n_pieces": int(len(pieces_raw)),
        "n_union": int(len(union_raw)),
    }

    excl_dur, _, coverage, keep_base, keep_hybrid, _ = build_keep_masks(
        tr_edges=tr_edges,
        union_raw=union_raw,
        tr_sec=tr_sec,
        base_thr=base_coverage_thr,
        hybrid_min_cov=hybrid_min_coverage,
        hybrid_min_block_frac=hybrid_min_good_block_frac,
    )

    eeg_pc, eeg_t = load_eeg_pc_and_time(run, input_maps["eeg_pc1"], input_maps["eeg_time"])
    eeg_t0 = float(np.min(eeg_t))
    eeg_t1 = float(np.max(eeg_t))
    tr0 = float(tr_edges[0])
    tr1 = float(tr_edges[-1])
    overlap_sec = max(0.0, min(eeg_t1, tr1) - max(eeg_t0, tr0))
    tr_span_sec = max(1e-9, tr1 - tr0)
    eeg_overlap_frac = overlap_sec / tr_span_sec
    eeg_pc_std = float(np.std(eeg_pc))
    eeg_pc_max = float(np.max(np.abs(eeg_pc)))

    expected_samples_per_tr = int(round(tr_sec * eeg_fs_hz))
    min_eeg_samples_per_tr = int(max(eeg_min_samples_min, eeg_min_samples_frac * expected_samples_per_tr))
    eeg_pow, eeg_counts = eeg_power_per_tr(
        eeg_pc,
        eeg_t,
        tr_edges,
        tr_sec=tr_sec,
        min_samples_per_tr=min_eeg_samples_per_tr,
    )

    keep_eeg_counts = eeg_counts >= min_eeg_samples_per_tr
    keep_base = keep_base & keep_eeg_counts
    keep_hybrid = keep_hybrid & keep_eeg_counts

    keep_center = keep_center_for_lags(keep_hybrid, lags_tr)
    segs_hybrid = segments_from_mask(keep_hybrid, min_len=1)
    segs_base = segments_from_mask(keep_base, min_len=1)
    segs_center_all = segments_from_mask(keep_center, min_len=1)
    segs_center_minlen10 = segments_from_mask(keep_center, min_len=10)
    segs_center_minlen15 = segments_from_mask(keep_center, min_len=15)
    keep_center_minlen10 = mask_from_segments(len(keep_center), segs_center_minlen10)
    keep_center_minlen15 = mask_from_segments(len(keep_center), segs_center_minlen15)

    keep_center0 = keep_center_for_lags(keep_hybrid, lags_tr_no)
    segs_center0_all = segments_from_mask(keep_center0, min_len=1)
    segs_center0_minlen10 = segments_from_mask(keep_center0, min_len=10)
    segs_center0_minlen15 = segments_from_mask(keep_center0, min_len=15)
    keep_center0_minlen10 = mask_from_segments(len(keep_center0), segs_center0_minlen10)
    keep_center0_minlen15 = mask_from_segments(len(keep_center0), segs_center0_minlen15)

    eeg_pow_lags = build_lagged_features(eeg_pow, lags_tr)
    eeg_pow_std = float(np.nanstd(eeg_pow))
    eeg_pow_p99 = float(np.nanpercentile(np.abs(eeg_pow), 99))
    eeg_pow_nan_frac = float(np.mean(~np.isfinite(eeg_pow)))

    eeg_flat = (eeg_pc_std < 1e-8) or (eeg_pc_max < 1e-6) or (eeg_pow_std < 1e-12)
    eeg_misaligned = eeg_overlap_frac < 0.50
    if eeg_flat:
        raise RuntimeError(f"FAIL_EEG_FLAT: eeg_pc_std={eeg_pc_std:.2e}, eeg_pow_std={eeg_pow_std:.2e}")
    if eeg_misaligned:
        raise RuntimeError(f"FAIL_EEG_TIME_MISMATCH: overlap_frac={eeg_overlap_frac:.2f}")

    bold_masked = bold_pc.astype(float).copy()
    bold_masked[~keep_hybrid, :] = np.nan
    eeg_pow_masked = eeg_pow.copy()
    eeg_pow_masked[~keep_hybrid, :] = np.nan
    eeg_pow_lags_masked = eeg_pow_lags.copy()
    eeg_pow_lags_masked[~keep_hybrid, :] = np.nan

    out_dir = out_runs / run
    out_dir.mkdir(parents=True, exist_ok=True)

    ltag = _lags_tag(tuple(lags_tr))
    ltag0 = _lags_tag(tuple(lags_tr_no))

    np.save(out_dir / "tr_edges_sec.npy", tr_edges)
    np.save(out_dir / "coverage.npy", coverage)
    np.save(out_dir / "excl_dur_sec.npy", excl_dur)
    np.save(out_dir / "keep_base.npy", keep_base.astype(np.uint8))
    np.save(out_dir / "keep_hybridG2.npy", keep_hybrid.astype(np.uint8))
    np.save(out_dir / f"keep_center_{ltag}.npy", keep_center.astype(np.uint8))
    np.save(out_dir / f"keep_center_minlen10_{ltag}.npy", keep_center_minlen10.astype(np.uint8))
    np.save(out_dir / f"keep_center_minlen15_{ltag}.npy", keep_center_minlen15.astype(np.uint8))
    np.save(out_dir / f"keep_center_{ltag0}.npy", keep_center0.astype(np.uint8))
    np.save(out_dir / f"keep_center_minlen10_{ltag0}.npy", keep_center0_minlen10.astype(np.uint8))
    np.save(out_dir / f"keep_center_minlen15_{ltag0}.npy", keep_center0_minlen15.astype(np.uint8))
    np.save(out_dir / "bold_pc1.npy", bold_pc)
    np.save(out_dir / "bold_pc1_masked.npy", bold_masked)
    np.save(out_dir / "eeg_power_tr.npy", eeg_pow)
    np.save(out_dir / "eeg_power_tr_masked.npy", eeg_pow_masked)
    np.save(out_dir / "eeg_power_tr_lags.npy", eeg_pow_lags)
    np.save(out_dir / "eeg_power_tr_lags_masked.npy", eeg_pow_lags_masked)
    np.save(out_dir / "eeg_counts_per_tr.npy", eeg_counts.astype(np.int32))

    save_intervals_tsv(union_raw, out_dir / "excl_union_mapped_raw.tsv")
    pd.DataFrame(segments).to_csv(out_dir / "segment_offsets.tsv", sep="\t", index=False)
    _write_json(out_dir / "segments_hybridG2.json", {"segments": [list(seg) for seg in segs_hybrid]})
    _write_json(out_dir / "segments_base.json", {"segments": [list(seg) for seg in segs_base]})

    segments_to_tsv(segs_center_all, tr_edges, out_dir / f"segments_center_all_{ltag}.tsv")
    segments_to_tsv(segs_center_minlen10, tr_edges, out_dir / "segments_center_minlen10.tsv")
    segments_to_tsv(segs_center_minlen15, tr_edges, out_dir / "segments_center_minlen15.tsv")
    segments_to_tsv(segs_center0_all, tr_edges, out_dir / f"segments_center_all_{ltag0}.tsv")
    segments_to_tsv(segs_center0_minlen10, tr_edges, out_dir / "segments_center0_minlen10.tsv")
    segments_to_tsv(segs_center0_minlen15, tr_edges, out_dir / "segments_center0_minlen15.tsv")

    _write_json(out_dir / "summary_minlen10.json", _minlen_summary(run, keep_hybrid, keep_center, segs_center_all, segs_center_minlen10, tr_sec, 10, lags_tr))
    _write_json(out_dir / "summary_minlen15.json", _minlen_summary(run, keep_hybrid, keep_center, segs_center_all, segs_center_minlen15, tr_sec, 15, lags_tr))
    _write_json(out_dir / "summary0_minlen10.json", _minlen_summary(run, keep_hybrid, keep_center0, segs_center0_all, segs_center0_minlen10, tr_sec, 10, lags_tr_no))
    _write_json(out_dir / "summary0_minlen15.json", _minlen_summary(run, keep_hybrid, keep_center0, segs_center0_all, segs_center0_minlen15, tr_sec, 15, lags_tr_no))

    if make_plots and (len(plot_run_set) == 0 or run in plot_run_set):
        plot_run_diagnostics(run, coverage, keep_hybrid, bold_pc, out_dir / "plots", base_coverage_thr, hybrid_min_coverage, example_parcel_index)

    status = "OK_strict" if qc_align.get("method", "") == "greedy" else "OK_dp_monotone"
    return {
        "run": run,
        "status": status,
        "method": qc_align.get("method"),
        "N_TR": int(n_tr),
        "kept_TR_hybridG2": int(np.sum(keep_hybrid)),
        "keep_frac_hybridG2": float(np.mean(keep_hybrid)),
        "nSeg_hybridG2": int(len(segs_hybrid)),
        "maxSeg_TR_hybridG2": int(max([end - start for (start, end) in segs_hybrid], default=0)),
        "kept_TR_center": int(np.sum(keep_center)),
        "keep_frac_center": float(np.mean(keep_center)),
        "nSeg_center_all": int(len(segs_center_all)),
        "maxSeg_center_all": int(max([end - start for (start, end) in segs_center_all], default=0)),
        "kept_TR_center_minlen10": int(sum((end - start) for (start, end) in segs_center_minlen10)),
        "nSeg_center_minlen10": int(len(segs_center_minlen10)),
        "maxSeg_center_minlen10": int(max([end - start for (start, end) in segs_center_minlen10], default=0)),
        "usable_min_center_minlen10": float(sum((end - start) for (start, end) in segs_center_minlen10) * tr_sec / 60.0),
        "kept_TR_center_minlen15": int(sum((end - start) for (start, end) in segs_center_minlen15)),
        "nSeg_center_minlen15": int(len(segs_center_minlen15)),
        "maxSeg_center_minlen15": int(max([end - start for (start, end) in segs_center_minlen15], default=0)),
        "usable_min_center_minlen15": float(sum((end - start) for (start, end) in segs_center_minlen15) * tr_sec / 60.0),
        "union_overlap_sec_canonical": map_stats["overlap_sec_canonical"],
        "dur_preproc_sec": map_stats["dur_preproc_sec"],
        "dur_mapped_union_sec": map_stats["dur_mapped_union_sec"],
        "med_abs_sec": qc_align["med_abs_sec"],
        "p95_abs_sec": qc_align["p95_abs_sec"],
        "n_boundaries_used": qc_align["n_boundaries_used"],
        "r128_tr_offset_est_sec": qc_align.get("r128_tr_offset_est_sec", np.nan),
        "eeg_t0_sec": eeg_t0,
        "eeg_t1_sec": eeg_t1,
        "eeg_overlap_frac": eeg_overlap_frac,
        "eeg_pc_std": eeg_pc_std,
        "eeg_pc_max": eeg_pc_max,
        "eeg_pow_std": eeg_pow_std,
        "eeg_pow_p99": eeg_pow_p99,
        "eeg_pow_nan_frac": eeg_pow_nan_frac,
        "eeg_counts_min": int(np.min(eeg_counts)),
        "eeg_counts_p05": int(np.percentile(eeg_counts, 5)),
        "eeg_expected_samples_per_tr": int(expected_samples_per_tr),
        "eeg_min_samples_per_tr": int(min_eeg_samples_per_tr),
        "eeg_min_samples_frac": float(eeg_min_samples_frac),
        "kept_TR_eeg_counts": int(np.sum(keep_eeg_counts)),
        "keep_frac_eeg_counts": float(np.mean(keep_eeg_counts)),
    }


def run_alignment_batch(
    raw_events_dir: str | Path,
    preproc_events_dir: str | Path,
    excl_union_dir: str | Path,
    eeg_parcel_npy_dir: str | Path,
    bold_parcel_npy_dir: str | Path,
    alignment_output_dir: str | Path,
    tr_sec: float,
    eeg_fs_hz: float,
    lags_tr: tuple[int, ...] | list[int],
    lags_tr_no: tuple[int, ...] | list[int],
    base_coverage_thr: float,
    hybrid_min_coverage: float,
    hybrid_min_good_block_frac: float,
    eeg_min_samples_frac: float,
    eeg_min_samples_min: int,
    dt_tol_sec: float,
    offset_jump_thr: float,
    report_r128_tr_offset: bool,
    apply_r128_tr_offset: bool,
    make_plots: bool,
    plot_runs: set[str] | list[str] | tuple[str, ...],
    example_parcel_index: int,
) -> dict[str, object]:
    alignment_output_dir = Path(alignment_output_dir)
    out_runs = alignment_output_dir / "per_run"
    out_qc = alignment_output_dir / "qc"
    out_runs.mkdir(parents=True, exist_ok=True)
    out_qc.mkdir(parents=True, exist_ok=True)

    input_maps = discover_input_maps(
        raw_events_dir=raw_events_dir,
        preproc_events_dir=preproc_events_dir,
        excl_union_dir=excl_union_dir,
        eeg_parcel_npy_dir=eeg_parcel_npy_dir,
        bold_parcel_npy_dir=bold_parcel_npy_dir,
    )
    audit_df = build_run_input_audit(input_maps)
    audit_csv = out_qc / "run_input_audit.csv"
    audit_df.to_csv(audit_csv, index=False)

    ready_runs = audit_df.loc[audit_df["ready"], "run"].tolist()
    plot_run_set = set(plot_runs)
    qc_rows = []
    for run in ready_runs:
        try:
            row = process_one_run_alignment(
                run=run,
                input_maps=input_maps,
                out_runs=out_runs,
                tr_sec=tr_sec,
                eeg_fs_hz=eeg_fs_hz,
                lags_tr=lags_tr,
                lags_tr_no=lags_tr_no,
                base_coverage_thr=base_coverage_thr,
                hybrid_min_coverage=hybrid_min_coverage,
                hybrid_min_good_block_frac=hybrid_min_good_block_frac,
                eeg_min_samples_frac=eeg_min_samples_frac,
                eeg_min_samples_min=eeg_min_samples_min,
                dt_tol_sec=dt_tol_sec,
                report_r128_tr_offset=report_r128_tr_offset,
                apply_r128_tr_offset=apply_r128_tr_offset,
                make_plots=make_plots,
                plot_run_set=plot_run_set,
                example_parcel_index=example_parcel_index,
            )
            qc_rows.append(row)
            print(f"[OK] {run}: {row['status']} ({row['method']}) kept_hybridG2={row['kept_TR_hybridG2']}/{row['N_TR']}")
        except Exception as exc:
            print(f"[FAIL] {run}: {exc}")
            qc_rows.append({"run": run, "status": "FAIL", "error": str(exc)})

    qc_df = pd.DataFrame(qc_rows)
    qc_csv = out_qc / "align_trmask_lags_summary.csv"
    qc_df.to_csv(qc_csv, index=False)

    alignment_parameters = {
        "TR_SEC": float(tr_sec),
        "EEG_FS_HZ": float(eeg_fs_hz),
        "LAGS_TR": [int(x) for x in lags_tr],
        "LAGS_TR_NO": [int(x) for x in lags_tr_no],
        "BASE_COVERAGE_THR": float(base_coverage_thr),
        "HYBRID_MIN_COVERAGE": float(hybrid_min_coverage),
        "HYBRID_MIN_GOOD_BLOCK_FRAC": float(hybrid_min_good_block_frac),
        "EEG_MIN_SAMPLES_FRAC": float(eeg_min_samples_frac),
        "EEG_MIN_SAMPLES_MIN": int(eeg_min_samples_min),
        "DT_TOL_SEC": float(dt_tol_sec),
        "OFFSET_JUMP_THR_EXPOSED_BUT_UNUSED": float(offset_jump_thr),
        "OFFSET_JUMP_THR_ACTIVE_IN_MATCH_SPLIT": 0.5,
        "REPORT_R128_TR_OFFSET": bool(report_r128_tr_offset),
        "APPLY_R128_TR_OFFSET": bool(apply_r128_tr_offset),
        "raw_events_dir": str(Path(raw_events_dir)),
        "preproc_events_dir": str(Path(preproc_events_dir)),
        "excl_union_dir": str(Path(excl_union_dir)),
        "eeg_parcel_npy_dir": str(Path(eeg_parcel_npy_dir)),
        "bold_parcel_npy_dir": str(Path(bold_parcel_npy_dir)),
        "alignment_output_dir": str(alignment_output_dir),
    }
    params_json = out_qc / "alignment_parameters_used.json"
    _write_json(params_json, alignment_parameters)

    return {
        "audit_df": audit_df,
        "qc_df": qc_df,
        "audit_csv": str(audit_csv),
        "qc_csv": str(qc_csv),
        "alignment_parameters_json": str(params_json),
        "ready_runs": ready_runs,
    }
