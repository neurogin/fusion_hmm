# R01 Pipeline A (STRICT-INTERMEDIATE) — Align BOLD/EEG, build TR masks, lag-centering (v3 GNORM EEG PCs)

This notebook rebuilds **BOLD–EEG alignment + TR masks + lag-centering outputs** using the **v3 gain-corrected EEG parcel PCs** (`*_PC1_gnorm.npy`), then writes per-run artifacts used by the fusion HMM.

**Main outputs (per run):**
- `eeg_power_tr_lags.npy` (TR × (parcels × lags))
- `bold_tr.npy` (TR × parcels)
- `keep_base.npy`, `keep_hybridG2.npy`
- `keep_center_<LTAG>.npy`, `keep_center_minlen10_<LTAG>.npy`, `keep_center_minlen15_<LTAG>.npy`
- diagnostics: `segment_offsets.tsv`, `segments_center_minlen*.tsv`, QC plots + a run-level summary CSV

You should only edit **Cell 0** unless you are changing methods.


```python
# Cell 0 — USER INPUTS
# =========================
from pathlib import Path

# --- Core constants ---
TR_SEC = 2.1
EEG_FS_HZ = 250.0
LAGS_TR = (-1, 0, +1)

# --- Design variants supported from the same per_run outputs ---
# We always compute EEG power per TR (no-lag) AND lagged EEG features.
# We also export lag-usable center masks for both designs:
#   (a) with lags: LAGS_TR = (-1,0,+1)  -> tag lags-1_0_1
#   (b) no lags:  LAGS_TR_NO = (0,)    -> tag lags0
LAGS_TR_NO = (0,)

# --- WSL base ---
BASE = Path("/mnt/c/EEGFMRI/hmm/R01_rerun")

# --- Input folders ---
RAW_EVENTS_DIR     = BASE / "01_raw/eeg_eeglab/events_raw"      # raw TSVs
PREPROC_EVENTS_DIR = BASE / "01_raw/eeg_eeglab/events_preproc"  # preproc TSVs (compressed time)

# Exclusion union TSVs (from Brainstorm/Matlab union merge)
EXCL_UNION_DIR = BASE / "02_derivatives/masks/bst_exports"

# Parcel PC data
EEG_PC_DIR  = BASE / "02_derivatives/eeg_source/parcel_pc1/npy"  # (expects *_PC1_gnorm.npy + *_time_sec.npy)
BOLD_PC_DIR = BASE / "02_derivatives/bold_parcel/parcel_pc1_v6/npy"

# --- Output folder ---
OUT_ROOT = BASE / "02_derivatives/fusion_prep/align_trmask_lags/FINAL_v3_gnorm_allTR/intermediate"
OUT_RUNS = OUT_ROOT / "per_run"
OUT_QC   = OUT_ROOT / "qc"

# --- Mask rules ---
# Base/strict-like retention rule (configurable):
# Base used previously behaved like coverage >= 0.5.
BASE_COVERAGE_THR = 0.70  # STRICT-INTERMEDIATE

# Hybrid-G2: allow down to coverage >= 0.25, BUT only if
# the kept portion inside the TR is one contiguous block with length >= MIN_GOOD_BLOCK_FRAC * TR
HYBRID_MIN_COVERAGE = 0.50  # STRICT-INTERMEDIATE
HYBRID_MIN_GOOD_BLOCK_FRAC = 0.50  # STRICT-INTERMEDIATE

# --- STRICT EEG completeness gating ---
# Drop TRs that do not contain enough EEG samples (prevents NaN holes and tail TRs beyond EEG)
EEG_MIN_SAMPLES_FRAC = 0.65   # require >=80% of expected EEG samples per TR bin
EEG_MIN_SAMPLES_MIN  = 50     # absolute floor (safety) in case of unusual TR/FS


# --- Alignment tolerances ---
DT_TOL_SEC = 0.15        # tolerance for matching inter-trigger intervals (R128)
OFFSET_JUMP_THR = 0.20   # detect segment boundary when offset changes by > this

# --- Optional: TR offset check from R128 (report only unless APPLY=True) ---
REPORT_R128_TR_OFFSET = True
APPLY_R128_TR_OFFSET  = False  # keep False unless you decide otherwise

# --- Plotting ---
MAKE_PLOTS = True
PLOT_RUNS = {"sub-16_ses-01"}  # set() for none; or include runs you want
EXAMPLE_PARCEL_INDEX = 0       # for BOLD trace plot

# --- Safety ---
OUT_ROOT.mkdir(parents=True, exist_ok=True)
OUT_RUNS.mkdir(parents=True, exist_ok=True)
OUT_QC.mkdir(parents=True, exist_ok=True)

print("OUT_ROOT:", OUT_ROOT)

# =========================
```


```python
# Cell 1 — Imports + Utilities
# =========================
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _read_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    # normalize column names a bit
    df.columns = [c.strip() for c in df.columns]
    return df

def _find_col(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def extract_event_times(df: pd.DataFrame, key_regex: str):
    """
    Returns onset times (seconds) for events whose 'trial_type' or 'value' matches key_regex.
    Works across slightly different TSV schemas.
    """
    col_tt = _find_col(df, ["trial_type", "type", "value", "event", "label"])
    col_on = _find_col(df, ["onset", "start", "start_sec", "time", "latency_sec"])
    if col_tt is None or col_on is None:
        raise ValueError(f"TSV missing expected columns. Have {df.columns}")

    mask = df[col_tt].astype(str).str.contains(key_regex, regex=True, na=False)
    times = df.loc[mask, col_on].astype(float).to_numpy()
    times.sort()
    return times

def extract_boundaries(df: pd.DataFrame):
    """
    Boundary events used only as diagnostics; the main mapping uses R128 matching.
    """
    col_tt = _find_col(df, ["trial_type", "type", "value", "event", "label"])
    col_dur = _find_col(df, ["duration", "dur", "duration_sec"])
    if col_tt is None:
        return np.array([])

    mask = df[col_tt].astype(str).str.contains(r"boundary", regex=True, na=False)
    if col_dur is None:
        return np.zeros(mask.sum(), dtype=float)
    return df.loc[mask, col_dur].astype(float).to_numpy()

def ensure_2d_time_major(x: np.ndarray, n_parcels_expected=200) -> np.ndarray:
    """
    Accepts (T, P) or (P, T) and returns (T, P).
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")
    T, P = x.shape
    if P == n_parcels_expected:
        return x
    if T == n_parcels_expected:
        return x.T
    # if neither dimension is 200, keep as-is but warn
    print(f"[WARN] array shape {x.shape} doesn't match expected parcels={n_parcels_expected}. Using as-is.")
    return x

def run_id_from_fname(fname: str) -> str:
    m = re.search(r"(sub-\d+_ses-\d+)", fname)
    return m.group(1) if m else None

def discover_runs():
    bold = {run_id_from_fname(p.name): p for p in BOLD_PC_DIR.glob("sub-*_ses-*_task-rest_parcel_pc1.npy")}
    eeg  = {run_id_from_fname(p.name): p for p in EEG_PC_DIR.glob("sub-*_ses-*_desc-ICRej70_clean_PC1_gnorm.npy")}
    raw  = {run_id_from_fname(p.name): p for p in RAW_EVENTS_DIR.glob("sub-*_ses-*_task-rest_events.tsv")}
    pre  = {run_id_from_fname(p.name): p for p in PREPROC_EVENTS_DIR.glob("sub-*_ses-*_task-rest_events.tsv")}
    uni  = {run_id_from_fname(p.name): p for p in EXCL_UNION_DIR.glob("sub-*_ses-*_desc-ICRej70_clean_excl_union.tsv")}

    runs = sorted(set(bold) & set(eeg) & set(raw) & set(pre) & set(uni))
    return runs, bold, eeg, raw, pre, uni

runs, bold_map, eeg_map, raw_map, pre_map, union_map = discover_runs()
print(f"Discovered {len(runs)} runnable runs:\n{runs}")

# =========================
```


```python
# Cell 2 — Alignment (raw vs preproc) using R128, anchored to raw S1
# PATCHED: robust event-code extraction (value vs type)
# =========================
import re

def extract_event_times(df, key_regex, label_cols=("trial_type", "value", "type"), time_cols=None):
    """
    Robustly extract event times where ANY of label_cols matches key_regex.
    This fixes the common EEGLAB/BIDS-export case where event codes live in `value`
    and `type` only has generic categories (Stimulus/Response/etc).
    """
    if time_cols is None:
        time_cols = ["onset", "start", "start_sec", "time", "latency_sec", "latency"]

    col_t = _find_col(df, time_cols)
    if col_t is None:
        raise ValueError(f"Could not find a time column among {time_cols}. Columns: {list(df.columns)}")

    times_list = []
    for col in label_cols:
        if col not in df.columns:
            continue
        s = df[col].astype(str)
        m = s.str.contains(key_regex, regex=True, na=False)
        if m.any():
            times_list.append(df.loc[m, col_t].astype(float).to_numpy())

    if len(times_list) == 0:
        # Helpful debug: show what label columns exist + a few unique examples
        present = [c for c in label_cols if c in df.columns]
        examples = {}
        for c in present:
            vals = df[c].astype(str).unique()
            examples[c] = vals[:10].tolist()
        raise ValueError(
            f"No events matched /{key_regex}/ in columns {present}. "
            f"Examples: {examples}"
        )

    t = np.concatenate(times_list)
    # de-dup + sort
    t = np.unique(t)
    t = np.sort(t)
    return t

def greedy_match_intervals(dt_raw, dt_pre, tol=0.15):
    """
    Greedy monotone matching of interval sequences by similarity.
    Returns list of matched interval indices (i_raw, j_pre).
    """
    matches = []
    j = 0
    for i in range(len(dt_raw)):
        if j >= len(dt_pre):
            break
        if abs(dt_raw[i] - dt_pre[j]) <= tol:
            matches.append((i, j))
            j += 1
        else:
            # allow skipping raw intervals (raw has more, preproc is "compressed")
            continue
    return matches

def dp_monotone_match_intervals(dt_raw, dt_pre, tol=0.15):
    """
    Dynamic-programming monotone match: maximize number of matches under |dt_raw-dt_pre|<=tol.
    Returns list of matched interval indices (i_raw, j_pre).
    """
    R, P = len(dt_raw), len(dt_pre)
    if R == 0 or P == 0:
        return []

    # dp[i+1][j+1] = best up to i,j
    dp = np.zeros((R+1, P+1), dtype=int)
    back = np.zeros((R+1, P+1, 2), dtype=int)  # previous (pi,pj)

    for i in range(R):
        for j in range(P):
            # option 1: skip raw interval
            if dp[i+1, j] >= dp[i, j+1]:
                best = dp[i+1, j]
                pi, pj = i+1, j
            else:
                best = dp[i, j+1]
                pi, pj = i, j+1

            # option 2: match
            if abs(dt_raw[i] - dt_pre[j]) <= tol:
                cand = dp[i, j] + 1
                if cand > best:
                    best = cand
                    pi, pj = i, j

            dp[i+1, j+1] = best
            back[i+1, j+1] = (pi, pj)

    # traceback
    i, j = R, P
    matches = []
    while i > 0 and j > 0:
        pi, pj = back[i, j]
        # if we moved diagonally by (1,1) and it was a match, record it
        if pi == i-1 and pj == j-1 and abs(dt_raw[i-1] - dt_pre[j-1]) <= tol:
            matches.append((i-1, j-1))
        i, j = pi, pj

    matches.reverse()
    return matches

def build_segment_offsets_from_matches(raw_r128, pre_r128, interval_matches, min_matches=5):
    """
    Convert matched INTERVALS into matched EVENT indices.
    Each interval match (i,j) corresponds to matching:
      raw events at (i,i+1) with pre events at (j,j+1)
    We use per-event offsets raw_t - pre_t, then split into segments at jumps.
    """
    if len(interval_matches) < min_matches:
        raise ValueError("Too few interval matches to build a stable mapping.")

    i_idx = np.array([i+1 for (i, _) in interval_matches], dtype=int)  # event index in raw
    j_idx = np.array([j+1 for (_, j) in interval_matches], dtype=int)  # event index in pre

    offsets = raw_r128[i_idx] - pre_r128[j_idx]

    # detect segment boundaries where offset jumps (compressed gaps)
    jumps = np.where(np.abs(np.diff(offsets)) > 0.5)[0]  # 0.5s jump threshold
    seg_starts = np.r_[0, jumps+1]
    seg_ends   = np.r_[jumps+1, len(offsets)]

    segments = []
    for a, b in zip(seg_starts, seg_ends):
        off = float(np.median(offsets[a:b]))
        pre_start = float(pre_r128[j_idx[a]])
        pre_end   = float(pre_r128[j_idx[b-1]])
        segments.append({"pre_start": pre_start, "pre_end": pre_end, "offset": off})

    # cover whole axis
    segments[0]["pre_start"] = 0.0
    segments[-1]["pre_end"] = float("inf")

    # QC prediction error on matched points
    pred = np.zeros_like(offsets)
    for s in segments:
        mask = (pre_r128[j_idx] >= s["pre_start"]) & (pre_r128[j_idx] <= s["pre_end"])
        pred[mask] = pre_r128[j_idx[mask]] + s["offset"]

    err = pred - raw_r128[i_idx]
    qc = {
        "med_abs_sec": float(np.median(np.abs(err))),
        "p95_abs_sec": float(np.percentile(np.abs(err), 95)),
        "mono_viol": int(np.sum(np.diff(j_idx) <= 0)),
        "n_pre_r128": int(len(pre_r128)),
        "n_raw_r128": int(len(raw_r128)),
        "n_boundaries_used": int(len(segments)-1),
        "TR_hat_sec": float(np.median(np.diff(raw_r128))) if len(raw_r128) > 3 else np.nan,
    }
    return segments, qc

def align_raw_preproc(raw_events_tsv: Path, preproc_events_tsv: Path):
    raw_df = _read_tsv(raw_events_tsv)
    pre_df = _read_tsv(preproc_events_tsv)

    # R128 from VALUE/TYPE/TRIAL_TYPE (robust)
    raw_r128 = extract_event_times(raw_df, r"R128")
    pre_r128 = extract_event_times(pre_df, r"R128")

    if len(raw_r128) < 10 or len(pre_r128) < 10:
        raise ValueError(f"R128 too few: raw={len(raw_r128)} pre={len(pre_r128)} "
                         f"({raw_events_tsv.name} vs {preproc_events_tsv.name})")

    dt_raw = np.diff(raw_r128)
    dt_pre = np.diff(pre_r128)

    # try greedy, else DP
    matches = greedy_match_intervals(dt_raw, dt_pre)
    method = "greedy"

    ok = False
    try:
        segments, qc = build_segment_offsets_from_matches(raw_r128, pre_r128, matches)
        ok = (qc["p95_abs_sec"] < 1e-6) and (qc["mono_viol"] == 0)
    except Exception:
        ok = False

    if not ok:
        matches = dp_monotone_match_intervals(dt_raw, dt_pre)
        segments, qc = build_segment_offsets_from_matches(raw_r128, pre_r128, matches)
        method = "DP_monotone"

    qc["method"] = method

    # raw anchor: S1 in RAW (also stored in `value` in your files)
    s1 = extract_event_times(raw_df, r"\bS\s*1\b|^S1$")
    if len(s1) == 0:
        s1 = extract_event_times(raw_df, r"S\s*1")
    s1_t0 = float(s1[0])

    # shift model so raw time=0 at S1
    for s in segments:
        s["offset"] = float(s["offset"] - s1_t0)
    qc["raw_S1_t0_sec"] = s1_t0

    return segments, qc, raw_r128 - s1_t0

# =========================
```


```python
# Cell 3 — Map exclusion union (preproc -> raw) and canonicalize
# =========================
def load_union_intervals(union_tsv: Path):
    df = _read_tsv(union_tsv)
    # expect columns: label, start_sec, end_sec (your files look like that)
    col_s = _find_col(df, ["start_sec", "start", "onset"])
    col_e = _find_col(df, ["end_sec", "end"])
    if col_s is None or col_e is None:
        raise ValueError(f"Union TSV missing start/end columns: {df.columns}")
    intervals = df[[col_s, col_e]].astype(float).to_numpy()
    intervals = intervals[np.argsort(intervals[:,0])]
    return intervals

def map_intervals_pre_to_raw(intervals_pre, segments):
    pieces = []
    for (a, b) in intervals_pre:
        if b <= a:
            continue
        for s in segments:
            ps, pe, off = s["pre_start"], s["pre_end"], s["offset"]
            # overlap with segment in pre-time
            lo = max(a, ps)
            hi = min(b, pe)
            if hi > lo:
                pieces.append((lo + off, hi + off))
    pieces = np.array(pieces, dtype=float)
    pieces = pieces[np.argsort(pieces[:,0])] if len(pieces) else pieces
    return pieces

def canonicalize_union(intervals, eps=1e-12):
    """
    Merge overlapping/adjacent intervals.
    Also compute overlap before merge (diagnostic).
    """
    if len(intervals) == 0:
        return intervals, 0.0

    intervals = intervals[np.argsort(intervals[:,0])]
    # overlap diagnostic
    overlap = 0.0
    for i in range(1, len(intervals)):
        overlap += max(0.0, intervals[i-1,1] - intervals[i,0])

    merged = [intervals[0].tolist()]
    for (a,b) in intervals[1:]:
        if a <= merged[-1][1] + eps:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a,b])
    return np.array(merged, dtype=float), float(overlap)

def union_duration(intervals):
    if len(intervals) == 0:
        return 0.0
    return float(np.sum(intervals[:,1] - intervals[:,0]))

def save_intervals_tsv(intervals, out_path: Path, label="excl_union_raw"):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"label": label,
                       "start_sec": intervals[:,0] if len(intervals) else [],
                       "end_sec": intervals[:,1] if len(intervals) else []})
    df.to_csv(out_path, sep="\t", index=False)

# =========================
```


```python
# Cell 4 — TR coverage + masks (BASE + Hybrid-G2) + segment extraction
# + keep_center + minlen helpers
# =========================
def estimate_tr_offset_from_r128(raw_r128_shifted, TR, N_TR):
    """
    Estimate constant offset such that raw_r128 ~= offset + k*TR.
    Uses nearest integer k mapping.
    """
    k = np.rint(raw_r128_shifted / TR).astype(int)
    m = (k >= 0) & (k <= N_TR)
    if not np.any(m):
        return 0.0
    off = np.median(raw_r128_shifted[m] - k[m]*TR)
    return float(off)

def tr_edges_from_bold(N_TR, TR, offset=0.0):
    return offset + np.arange(N_TR + 1, dtype=float) * TR

def intersect_len(a0, a1, b0, b1):
    return max(0.0, min(a1,b1) - max(a0,b0))

def compute_tr_excl_dur(tr_edges, union_raw):
    """
    Sweep-line intersection of TR bins with union intervals.
    Returns excl_dur per TR.
    """
    N_TR = len(tr_edges) - 1
    excl = np.zeros(N_TR, dtype=float)
    if len(union_raw) == 0:
        return excl

    j = 0
    for k in range(N_TR):
        t0, t1 = tr_edges[k], tr_edges[k+1]
        while j < len(union_raw) and union_raw[j,1] <= t0:
            j += 1
        jj = j
        while jj < len(union_raw) and union_raw[jj,0] < t1:
            excl[k] += intersect_len(t0, t1, union_raw[jj,0], union_raw[jj,1])
            jj += 1
    return excl

def complement_intervals_in_window(t0, t1, union_raw):
    """
    Return list of kept intervals inside [t0,t1] after subtracting union_raw.
    Assumes union_raw is canonicalized (sorted, non-overlapping).
    """
    kept = []
    cur = t0
    for (a,b) in union_raw:
        if b <= t0:
            continue
        if a >= t1:
            break
        if a > cur:
            kept.append((cur, min(a, t1)))
        cur = max(cur, b)
        if cur >= t1:
            break
    if cur < t1:
        kept.append((cur, t1))
    return [(a,b) for (a,b) in kept if b > a]

def build_keep_masks(tr_edges, union_raw, base_thr=BASE_COVERAGE_THR,
                     hybrid_min_cov=HYBRID_MIN_COVERAGE,
                     hybrid_min_block_frac=HYBRID_MIN_GOOD_BLOCK_FRAC):
    N_TR = len(tr_edges) - 1
    excl_dur = compute_tr_excl_dur(tr_edges, union_raw)
    excl_frac = excl_dur / TR_SEC
    coverage = 1.0 - excl_frac

    keep_base = coverage >= base_thr

    # Hybrid-G2
    keep_hybrid = np.zeros(N_TR, dtype=bool)
    for k in range(N_TR):
        cov = coverage[k]
        if cov >= base_thr:
            keep_hybrid[k] = True
            continue
        if cov < hybrid_min_cov:
            keep_hybrid[k] = False
            continue

        t0, t1 = tr_edges[k], tr_edges[k+1]
        kept_int = complement_intervals_in_window(t0, t1, union_raw)

        # G2: exactly one contiguous "good" block, long enough
        if len(kept_int) == 1:
            good_len = kept_int[0][1] - kept_int[0][0]
            if good_len >= hybrid_min_block_frac * TR_SEC:
                keep_hybrid[k] = True

    def seg_stats(mask):
        best = cur = 0
        nseg = 0
        in_seg = False
        for v in mask:
            if v:
                cur += 1
                if not in_seg:
                    nseg += 1
                    in_seg = True
                best = max(best, cur)
            else:
                cur = 0
                in_seg = False
        return int(best), int(nseg)

    maxSeg_base, nSeg_base = seg_stats(keep_base)
    maxSeg_hyb,  nSeg_hyb  = seg_stats(keep_hybrid)

    stats = {
        "N_TR": int(N_TR),
        "kept_TR_base": int(np.sum(keep_base)),
        "kept_TR_hybridG2": int(np.sum(keep_hybrid)),
        "maxSeg_TR_base": int(maxSeg_base),
        "nSeg_base": int(nSeg_base),
        "maxSeg_TR_hybridG2": int(maxSeg_hyb),
        "nSeg_hybridG2": int(nSeg_hyb),
    }
    return excl_dur, excl_frac, coverage, keep_base, keep_hybrid, stats

def segments_from_mask(mask, min_len=1):
    segs = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if (not v) and start is not None:
            if i - start >= min_len:
                segs.append((start, i))  # [start, i)
            start = None
    if start is not None and (len(mask) - start) >= min_len:
        segs.append((start, len(mask)))
    return segs

def keep_center_for_lags(keep_bool, lags=LAGS_TR):
    """
    Center TR is usable iff keep[t+lag] is True for ALL lags.
    Example lags=(-1,0,+1): need (t-1,t,t+1) all kept.
    """
    keep = np.asarray(keep_bool).astype(bool)
    N = len(keep)
    center = np.ones(N, dtype=bool)
    for lag in lags:
        idx = np.arange(N) + lag
        valid = (idx >= 0) & (idx < N)
        tmp = np.zeros(N, dtype=bool)
        tmp[valid] = keep[idx[valid]]
        center &= tmp
    return center

def mask_from_segments(N, segs):
    m = np.zeros(N, dtype=bool)
    for a,b in segs:
        m[a:b] = True
    return m

def segments_to_tsv(segs, tr_edges, out_path):
    """
    Write TR segments with both TR indices and seconds.
    segs are (start_TR, end_TR) with end exclusive.
    """
    rows = []
    for a,b in segs:
        rows.append(dict(
            start_TR=int(a),
            end_TR=int(b),
            len_TR=int(b-a),
            start_sec=float(tr_edges[a]),
            end_sec=float(tr_edges[b]),
            dur_sec=float(tr_edges[b]-tr_edges[a]),
        ))
    df = pd.DataFrame(rows)
    df.to_csv(out_path, sep="\t", index=False)
    return df

# =========================
```


```python
# Cell 5 — EEG power per TR + lag builder
# =========================
def load_eeg_pc_and_time(run: str):
    pc_path = eeg_map[run]
    pc = np.load(pc_path)
    pc = ensure_2d_time_major(pc)

    # time_sec is useful if present
    t_path = EEG_PC_DIR / f"{run}_desc-ICRej70_clean_time_sec.npy"
    if not t_path.exists():
        raise FileNotFoundError(f"Missing EEG time_sec for {run}: {t_path}")
    t = np.load(t_path).astype(float).reshape(-1)

    # sanity
    if pc.shape[0] != t.shape[0]:
        raise ValueError(f"EEG PC and time mismatch for {run}: pc {pc.shape}, t {t.shape}")

    return pc, t

def eeg_power_per_tr(pc, t_sec, tr_edges, min_samples_per_tr=1):
    """
    Power = mean(x^2) within each TR bin, per parcel.
    Returns:
      power: (N_TR, P) float, NaN where <min_samples_per_tr samples
      counts: (N_TR,) int number of EEG samples in each TR
    """
    N_TR = len(tr_edges) - 1
    T, P = pc.shape
    power = np.full((N_TR, P), np.nan, dtype=float)
    counts = np.zeros(N_TR, dtype=int)

    # keep samples within [tr_edges[0], tr_edges[-1])
    m = (t_sec >= tr_edges[0]) & (t_sec < tr_edges[-1])
    if not np.any(m):
        return power, counts

    t = t_sec[m]
    x2 = (pc[m, :] ** 2)

    bin_idx = np.floor((t - tr_edges[0]) / TR_SEC).astype(int)
    bin_idx = np.clip(bin_idx, 0, N_TR - 1)

    counts = np.bincount(bin_idx, minlength=N_TR).astype(int)

    # only compute power for TRs with enough samples
    ok = counts >= int(min_samples_per_tr)
    denom = counts.astype(float)
    denom[~ok] = np.nan

    chunk = 25
    for p0 in range(0, P, chunk):
        p1 = min(P, p0 + chunk)
        s = np.zeros((N_TR, p1 - p0), dtype=float)
        np.add.at(s, bin_idx, x2[:, p0:p1])
        power[:, p0:p1] = s / denom[:, None]

    return power, counts

def build_lagged_features(X, lags=LAGS_TR, pad=np.nan):
    """
    X: (N, D) -> lagged: (N, D*len(lags)) with NaN padding at edges.
    """
    N, D = X.shape
    out = np.full((N, D*len(lags)), pad, dtype=float)
    for li, lag in enumerate(lags):
        if lag == 0:
            out[:, li*D:(li+1)*D] = X
        elif lag < 0:
            out[-lag:, li*D:(li+1)*D] = X[:N+lag, :]
        else:
            out[:N-lag, li*D:(li+1)*D] = X[lag:, :]
    return out

# =========================
```


```python
# Cell 6 — Per-run processing (alignment + mapping + masks + features + plots)
# + keep_center + minlen10/minlen15 exports
# =========================
def plot_run_diagnostics(run, tr_edges, coverage, keep_hybrid, bold_pc, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Coverage plot
    fig = plt.figure()
    plt.plot(coverage, marker="o", linestyle="-", linewidth=1, markersize=3)
    dropped = np.where(~keep_hybrid)[0]
    plt.scatter(dropped, coverage[dropped], s=18)
    plt.axhline(BASE_COVERAGE_THR, linestyle="--")
    plt.axhline(HYBRID_MIN_COVERAGE, linestyle="--")
    plt.title(f"{run}: TR coverage + keep mask (all TRs)")
    plt.xlabel("TR index")
    plt.ylabel("Coverage (1 - excl_frac)")
    fig.savefig(out_dir / f"{run}_coverage_keep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Segment length histogram (kept)
    segs = segments_from_mask(keep_hybrid, min_len=1)
    lens = [b-a for (a,b) in segs]
    fig = plt.figure()
    plt.hist(lens, bins=30)
    plt.title(f"{run}: contiguous kept segment lengths (Hybrid-G2)")
    plt.xlabel("segment length (TR)")
    plt.ylabel("count")
    fig.savefig(out_dir / f"{run}_kept_segment_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Example BOLD parcel with drops set to NaN
    y = bold_pc[:, EXAMPLE_PARCEL_INDEX].astype(float).copy()
    y[~keep_hybrid] = np.nan
    fig = plt.figure()
    plt.plot(y)
    plt.title(f"{run}: BOLD PC1 parcel[{EXAMPLE_PARCEL_INDEX}] (dropped TRs -> NaN)")
    plt.xlabel("TR index")
    plt.ylabel("PC1 (a.u.)")
    fig.savefig(out_dir / f"{run}_bold_example_nan.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def _lags_tag(lags):
    # lags=(-1,0,1) -> "lags-1_0_1"
    return "lags" + "_".join([str(int(x)) for x in lags])

def _minlen_summary(run, keep_hyb, keep_center, segs_center_all, segs_center_minlen, tr_edges, minlen, lags_used):
    """Summary for a given center mask (either lagged centers or no-lag centers)."""
    N_TR = len(keep_hyb)
    kept_TR_hybridG2 = int(np.sum(keep_hyb))

    kept_TR_center = int(np.sum(keep_center))
    keep_frac_center = float(np.mean(keep_center)) if N_TR else np.nan

    nSeg_center_all = int(len(segs_center_all))
    maxSeg_center_all = int(max([b-a for (a,b) in segs_center_all], default=0))

    nSeg_center_minlen = int(len(segs_center_minlen))
    maxSeg_center_minlen = int(max([b-a for (a,b) in segs_center_minlen], default=0))

    kept_TR_center_minlen = int(sum((b-a) for (a,b) in segs_center_minlen))
    duration_center_minlen_sec = float(kept_TR_center_minlen * TR_SEC)

    return {
        "run": run,
        "TR_SEC": float(TR_SEC),
        "lags_used": [int(x) for x in lags_used],
        "minlen_TR": int(minlen),

        "N_TR": int(N_TR),
        "kept_TR_hybridG2": kept_TR_hybridG2,

        "kept_TR_center": kept_TR_center,
        "keep_frac_center": keep_frac_center,
        "nSeg_center_all": nSeg_center_all,
        "maxSeg_center_all": maxSeg_center_all,

        "nSeg_center_minlen": nSeg_center_minlen,
        "maxSeg_center_minlen": maxSeg_center_minlen,
        "kept_TR_center_minlen": kept_TR_center_minlen,
        "duration_center_minlen_sec": duration_center_minlen_sec,
        "duration_center_minlen_min": duration_center_minlen_sec / 60.0,
    }


def process_one_run(run: str):
    # --- Load BOLD PC1 ---
    bold_pc = np.load(bold_map[run])
    bold_pc = ensure_2d_time_major(bold_pc)
    N_TR = bold_pc.shape[0]

    # --- Alignment (raw vs preproc) ---
    segments, qc_align, raw_r128_shifted = align_raw_preproc(raw_map[run], pre_map[run])

    # --- Optional TR offset reporting ---
    tr_offset = 0.0
    if REPORT_R128_TR_OFFSET:
        tr_offset = estimate_tr_offset_from_r128(raw_r128_shifted, TR_SEC, N_TR)
        qc_align["r128_tr_offset_est_sec"] = tr_offset

    # define TR edges (S1 anchored)
    if APPLY_R128_TR_OFFSET:
        tr_edges = tr_edges_from_bold(N_TR, TR_SEC, offset=tr_offset)
    else:
        tr_edges = tr_edges_from_bold(N_TR, TR_SEC, offset=0.0)

    # --- Load union exclusions in preproc time, map to raw, canonicalize ---
    union_pre = load_union_intervals(union_map[run])
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

    # --- Masks from union_raw ---
    excl_dur, excl_frac, coverage, keep_base, keep_hybrid, stats = build_keep_masks(tr_edges, union_raw)

    # --- keep_center for lagged design ---
    keep_center = keep_center_for_lags(keep_hybrid, lags=LAGS_TR)

    # segments
    segs_hybrid = segments_from_mask(keep_hybrid, min_len=1)
    segs_base   = segments_from_mask(keep_base, min_len=1)
    segs_center_all = segments_from_mask(keep_center, min_len=1)

    # minlen filters (you asked for 10 and 15)
    segs_center_minlen10 = segments_from_mask(keep_center, min_len=10)
    segs_center_minlen15 = segments_from_mask(keep_center, min_len=15)

    keep_center_minlen10 = mask_from_segments(len(keep_center), segs_center_minlen10)
    keep_center_minlen15 = mask_from_segments(len(keep_center), segs_center_minlen15)

    # --- EEG power per TR + counts + lag builder ---
    eeg_pc, eeg_t = load_eeg_pc_and_time(run)
    
    # --- EEG time overlap sanity ---
    eeg_t0, eeg_t1 = float(np.min(eeg_t)), float(np.max(eeg_t))
    tr0, tr1 = float(tr_edges[0]), float(tr_edges[-1])
    overlap_sec = max(0.0, min(eeg_t1, tr1) - max(eeg_t0, tr0))
    tr_span_sec = max(1e-9, tr1 - tr0)
    eeg_overlap_frac = overlap_sec / tr_span_sec
    
    # --- EEG PC scale sanity (raw sample domain) ---
    eeg_pc_std = float(np.std(eeg_pc))
    eeg_pc_max = float(np.max(np.abs(eeg_pc)))
    
    # --- EEG power per TR + counts (TR domain) ---
    EXPECTED_EEG_SAMPLES_PER_TR = int(round(TR_SEC * EEG_FS_HZ))
    MIN_EEG_SAMPLES_PER_TR = int(max(EEG_MIN_SAMPLES_MIN, EEG_MIN_SAMPLES_FRAC * EXPECTED_EEG_SAMPLES_PER_TR))  # STRICT
    eeg_pow, eeg_counts = eeg_power_per_tr(
        eeg_pc, eeg_t, tr_edges, min_samples_per_tr=MIN_EEG_SAMPLES_PER_TR
    )

    # --- STRICT: EEG completeness gate (counts-based) ---
    # Note: eeg_counts is (# EEG samples) in each TR bin. This naturally drops tail TRs beyond EEG recording.
    keep_eeg_counts = (eeg_counts >= MIN_EEG_SAMPLES_PER_TR)
    keep_eeg_time   = (eeg_counts > 0)
    keep_eeg_strict = keep_eeg_counts  # main strict gate

    # Apply strict gate to union-derived masks (do not delete samples; just drop TRs)
    keep_base   = keep_base & keep_eeg_strict
    keep_hybrid = keep_hybrid & keep_eeg_strict

    # Recompute lag-center and segments after strict gating
    keep_center = keep_center_for_lags(keep_hybrid, lags=LAGS_TR)

    segs_hybrid = segments_from_mask(keep_hybrid, min_len=1)
    segs_base   = segments_from_mask(keep_base, min_len=1)
    segs_center_all = segments_from_mask(keep_center, min_len=1)

    segs_center_minlen10 = segments_from_mask(keep_center, min_len=10)
    segs_center_minlen15 = segments_from_mask(keep_center, min_len=15)

    keep_center_minlen10 = mask_from_segments(len(keep_center), segs_center_minlen10)
    keep_center_minlen15 = mask_from_segments(len(keep_center), segs_center_minlen15)

    # --- ALSO: no-lag center masks (for the "no lags" fusion variant) ---
    keep_center0 = keep_center_for_lags(keep_hybrid, lags=LAGS_TR_NO)  # effectively == keep_hybrid
    segs_center0_all = segments_from_mask(keep_center0, min_len=1)
    segs_center0_minlen10 = segments_from_mask(keep_center0, min_len=10)
    segs_center0_minlen15 = segments_from_mask(keep_center0, min_len=15)
    keep_center0_minlen10 = mask_from_segments(len(keep_center0), segs_center0_minlen10)
    keep_center0_minlen15 = mask_from_segments(len(keep_center0), segs_center0_minlen15)


    
    eeg_pow_lags = build_lagged_features(eeg_pow, lags=LAGS_TR)
    
    # --- EEG power scale sanity ---
    eeg_pow_std = float(np.nanstd(eeg_pow))
    eeg_pow_p99 = float(np.nanpercentile(np.abs(eeg_pow), 99))
    eeg_pow_nan_frac = float(np.mean(~np.isfinite(eeg_pow)))
    
    # --- decide if EEG is unusable ---
    EEG_FLAT = (eeg_pc_std < 1e-8) or (eeg_pc_max < 1e-6) or (eeg_pow_std < 1e-12)
    EEG_MISALIGNED = (eeg_overlap_frac < 0.50)
    
    if EEG_FLAT:
        raise RuntimeError(f"FAIL_EEG_FLAT: eeg_pc_std={eeg_pc_std:.2e}, eeg_pow_std={eeg_pow_std:.2e}")
    if EEG_MISALIGNED:
        raise RuntimeError(f"FAIL_EEG_TIME_MISMATCH: overlap_frac={eeg_overlap_frac:.2f}")

    
    # --- Apply mask as NaN padding (do NOT delete) ---
    bold_masked = bold_pc.astype(float).copy()
    bold_masked[~keep_hybrid, :] = np.nan

    eeg_pow_masked = eeg_pow.copy()
    eeg_pow_masked[~keep_hybrid, :] = np.nan

    eeg_pow_lags_masked = eeg_pow_lags.copy()
    eeg_pow_lags_masked[~keep_hybrid, :] = np.nan

    # --- Output paths ---
    out_dir = OUT_RUNS / run
    out_dir.mkdir(parents=True, exist_ok=True)

    ltag = _lags_tag(LAGS_TR)
    ltag0 = _lags_tag(LAGS_TR_NO)

    # Save key arrays
    np.save(out_dir / "tr_edges_sec.npy", tr_edges)
    np.save(out_dir / "coverage.npy", coverage)
    np.save(out_dir / "excl_dur_sec.npy", excl_dur)
    np.save(out_dir / "keep_base.npy", keep_base.astype(np.uint8))
    np.save(out_dir / "keep_hybridG2.npy", keep_hybrid.astype(np.uint8))

    # NEW: center masks
    np.save(out_dir / f"keep_center_{ltag}.npy", keep_center.astype(np.uint8))
    np.save(out_dir / f"keep_center_minlen10_{ltag}.npy", keep_center_minlen10.astype(np.uint8))
    np.save(out_dir / f"keep_center_minlen15_{ltag}.npy", keep_center_minlen15.astype(np.uint8))

    # ALSO export no-lag center masks (tag lags0) for the no-lags fusion variant
    np.save(out_dir / f"keep_center_{ltag0}.npy", keep_center0.astype(np.uint8))
    np.save(out_dir / f"keep_center_minlen10_{ltag0}.npy", keep_center0_minlen10.astype(np.uint8))
    np.save(out_dir / f"keep_center_minlen15_{ltag0}.npy", keep_center0_minlen15.astype(np.uint8))

    np.save(out_dir / "bold_pc1.npy", bold_pc)                # raw
    np.save(out_dir / "bold_pc1_masked.npy", bold_masked)     # masked with NaNs

    np.save(out_dir / "eeg_power_tr.npy", eeg_pow)
    np.save(out_dir / "eeg_power_tr_masked.npy", eeg_pow_masked)
    np.save(out_dir / "eeg_power_tr_lags.npy", eeg_pow_lags)
    np.save(out_dir / "eeg_power_tr_lags_masked.npy", eeg_pow_lags_masked)
    np.save(out_dir / "eeg_counts_per_tr.npy", eeg_counts.astype(np.int32))


    # Save mapped union + mapping segments
    save_intervals_tsv(union_raw, out_dir / "excl_union_mapped_raw.tsv")
    pd.DataFrame(segments).to_csv(out_dir / "segment_offsets.tsv", sep="\t", index=False)

    # Save segment JSONs (kept/base)
    with open(out_dir / "segments_hybridG2.json", "w") as f:
        json.dump({"segments":[list(s) for s in segs_hybrid]}, f, indent=2)
    with open(out_dir / "segments_base.json", "w") as f:
        json.dump({"segments":[list(s) for s in segs_base]}, f, indent=2)

    # NEW: center segments TSVs (all + minlen)
    segments_to_tsv(segs_center_all, tr_edges, out_dir / f"segments_center_all_{ltag}.tsv")
    segments_to_tsv(segs_center_minlen10, tr_edges, out_dir / "segments_center_minlen10.tsv")
    segments_to_tsv(segs_center_minlen15, tr_edges, out_dir / "segments_center_minlen15.tsv")

    # ALSO: no-lag center segments TSVs (mostly identical to keep_hybrid) for convenience
    segments_to_tsv(segs_center0_all, tr_edges, out_dir / f"segments_center_all_{ltag0}.tsv")
    segments_to_tsv(segs_center0_minlen10, tr_edges, out_dir / "segments_center0_minlen10.tsv")
    segments_to_tsv(segs_center0_minlen15, tr_edges, out_dir / "segments_center0_minlen15.tsv")

    # NEW: per-run minlen summaries
    summ10 = _minlen_summary(run, keep_hybrid, keep_center, segs_center_all, segs_center_minlen10, tr_edges, minlen=10, lags_used=LAGS_TR)
    summ15 = _minlen_summary(run, keep_hybrid, keep_center, segs_center_all, segs_center_minlen15, tr_edges, minlen=15, lags_used=LAGS_TR)
    with open(out_dir / "summary_minlen10.json", "w") as f:
        json.dump(summ10, f, indent=2)
    with open(out_dir / "summary_minlen15.json", "w") as f:
        json.dump(summ15, f, indent=2)

    # No-lag minlen summaries (useful for the no-lags variant)
    summ10_0 = _minlen_summary(run, keep_hybrid, keep_center0, segs_center0_all, segs_center0_minlen10, tr_edges, minlen=10, lags_used=LAGS_TR_NO)
    summ15_0 = _minlen_summary(run, keep_hybrid, keep_center0, segs_center0_all, segs_center0_minlen15, tr_edges, minlen=15, lags_used=LAGS_TR_NO)
    with open(out_dir / "summary0_minlen10.json", "w") as f:
        json.dump(summ10_0, f, indent=2)
    with open(out_dir / "summary0_minlen15.json", "w") as f:
        json.dump(summ15_0, f, indent=2)

    # Plots
    if MAKE_PLOTS and (len(PLOT_RUNS) == 0 or run in PLOT_RUNS):
        plot_run_diagnostics(run, tr_edges, coverage, keep_hybrid, bold_pc, out_dir / "plots")

    # Status decision
    status = "OK_strict" if qc_align.get("method","") == "greedy" else "OK_dp_monotone"

    # QC row (add center + minlen10 + minlen15)
    qc_row = {
        "run": run,
        "status": status,
        "method": qc_align.get("method"),
        "N_TR": int(N_TR),

        "kept_TR_hybridG2": int(np.sum(keep_hybrid)),
        "keep_frac_hybridG2": float(np.mean(keep_hybrid)),
        "nSeg_hybridG2": int(len(segs_hybrid)),
        "maxSeg_TR_hybridG2": int(max([b-a for (a,b) in segs_hybrid], default=0)),

        "kept_TR_center": int(np.sum(keep_center)),
        "keep_frac_center": float(np.mean(keep_center)),
        "nSeg_center_all": int(len(segs_center_all)),
        "maxSeg_center_all": int(max([b-a for (a,b) in segs_center_all], default=0)),

        "kept_TR_center_minlen10": int(sum((b-a) for (a,b) in segs_center_minlen10)),
        "nSeg_center_minlen10": int(len(segs_center_minlen10)),
        "maxSeg_center_minlen10": int(max([b-a for (a,b) in segs_center_minlen10], default=0)),
        "usable_min_center_minlen10": float(sum((b-a) for (a,b) in segs_center_minlen10) * TR_SEC / 60.0),

        "kept_TR_center_minlen15": int(sum((b-a) for (a,b) in segs_center_minlen15)),
        "nSeg_center_minlen15": int(len(segs_center_minlen15)),
        "maxSeg_center_minlen15": int(max([b-a for (a,b) in segs_center_minlen15], default=0)),
        "usable_min_center_minlen15": float(sum((b-a) for (a,b) in segs_center_minlen15) * TR_SEC / 60.0),

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
        "eeg_expected_samples_per_tr": int(EXPECTED_EEG_SAMPLES_PER_TR),
        "eeg_min_samples_per_tr": int(MIN_EEG_SAMPLES_PER_TR),
        "eeg_min_samples_frac": float(EEG_MIN_SAMPLES_FRAC),
        "kept_TR_eeg_counts": int(np.sum(keep_eeg_counts)),
        "keep_frac_eeg_counts": float(np.mean(keep_eeg_counts)),

    }
    return qc_row

# =========================

```


```python
# Cell 7 — Run ALL runs + write QC summary
# =========================
qc_rows = []
for run in runs:
    try:
        row = process_one_run(run)
        qc_rows.append(row)
        print(f"[OK] {run}: {row['status']} ({row['method']}) kept_hybridG2={row['kept_TR_hybridG2']}/{row['N_TR']}")
    except Exception as e:
        print(f"[FAIL] {run}: {e}")
        qc_rows.append({"run": run, "status": "FAIL", "error": str(e)})

qc_df = pd.DataFrame(qc_rows)
qc_path = OUT_QC / "align_trmask_lags_summary.csv"
qc_df.to_csv(qc_path, index=False)
print("Saved QC summary:", qc_path)

# Safe display even if some runs failed early
if "kept_TR_hybridG2" in qc_df.columns:
    qc_df.sort_values(["status", "kept_TR_hybridG2"], ascending=[True, False]).head(30)
else:
    qc_df.head(30)


import numpy as np
from pathlib import Path

bold_dir = Path("/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/bold_parcel/parcel_pc1_v6/npy")
eeg_dir  = Path("/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/eeg_source/parcel_pc1/npy")

# BOLD lengths (TR domain)
for f in sorted(bold_dir.glob("sub-*_ses-*_task-rest_parcel_pc1.npy")):
    x = np.load(f)
    T = x.shape[0]
    print(f.name, "shape", x.shape, "T", T)

# EEG lengths (sample domain)
for f in sorted(eeg_dir.glob("sub-*_ses-*_desc-ICRej70_clean_time_sec.npy")):
    t = np.load(f).reshape(-1)
    print(f.name, "N", t.size, "t_end", float(t[-1]))


# =========================
```


```python
# Cell 0 — USER INPUTS
# =========================
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Your actual output root
OUT_ROOT = Path("/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/align_trmask_lags/FINAL_v3_gnorm_allTR/strict")
PER_RUN = OUT_ROOT / "per_run"
QC_DIR  = OUT_ROOT / "qc"
PLOT_DIR = QC_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

TR_SEC = 2.1
LAGS = (-1, 0, +1)  # your temporal context

# =========================
```


```python
# Cell 1 — LOAD SUMMARY + QUICK GLOBAL PLOTS
# =========================
runs = sorted([p.name for p in PER_RUN.iterdir() if p.is_dir()])
print("Discovered runs:", len(runs))
print(runs[:10], "..." if len(runs) > 10 else "")

# quick check
missing = []
for r in runs:
    if not (PER_RUN / r / "keep_hybridG2.npy").exists():
        missing.append(r)
print("Missing keep_hybridG2.npy:", len(missing))
if missing:
    print(missing)

# =========================
# QC Cell 2 — HELPERS: segments + lag center + minlen filtering
# =========================
def contiguous_segments(mask_bool):
    m = np.asarray(mask_bool).astype(bool)
    if m.size == 0:
        return []
    dm = np.diff(m.astype(int))
    starts = list(np.where(dm == 1)[0] + 1)
    ends   = list(np.where(dm == -1)[0] + 1)
    if m[0]:  starts = [0] + starts
    if m[-1]: ends = ends + [len(m)]
    return [(s, e, e - s) for s, e in zip(starts, ends)]

def keep_center_for_lags(keep_bool, lags=(-1,0,1)):
    keep = np.asarray(keep_bool).astype(bool)
    N = len(keep)
    center = np.ones(N, dtype=bool)
    for lag in lags:
        idx = np.arange(N) + lag
        valid = (idx >= 0) & (idx < N)
        tmp = np.zeros(N, dtype=bool)
        tmp[valid] = keep[idx[valid]]
        center &= tmp
    return center

def segs_minlen_from_mask(mask_bool, minlen):
    segs = contiguous_segments(mask_bool)
    return [(s,e,L) for (s,e,L) in segs if L >= minlen]

def kept_tr_from_segs(segs):
    return int(sum(L for (_,_,L) in segs))

def find_mask_path(run_dir: Path):
    # be forgiving on filenames
    candidates = [
        run_dir / "keep_hybridG2.npy",
        run_dir / "keep_hybridG2.npy",  # same, but keep pattern explicit
        run_dir / "keep_hybrid.npy",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

# =========================
# QC Cell 3 — PER-RUN QC + minlen10/minlen15 summaries + plots
# =========================
rows = []
MINLEN_LIST = [10, 15]

for run in runs:
    run_dir = PER_RUN / run
    mask_path = find_mask_path(run_dir)
    if mask_path is None:
        print(f"[SKIP] {run}: missing keep_hybridG2.npy")
        rows.append(dict(run=run, status="SKIP_missing_mask"))
        continue

    keep = np.load(mask_path).reshape(-1)
    keep = keep.astype(bool) if keep.dtype == bool else (keep != 0)
    N_TR = len(keep)

    keep_center = keep_center_for_lags(keep, lags=LAGS)

    segs_keep = contiguous_segments(keep)
    segs_center_all = contiguous_segments(keep_center)

    # base center stats
    kept_TR = int(keep.sum())
    kept_center = int(keep_center.sum())
    maxSeg_keep = int(max([L for _,_,L in segs_keep], default=0))
    maxSeg_center = int(max([L for _,_,L in segs_center_all], default=0))

    row = dict(
        run=run,
        status="OK",
        N_TR=N_TR,

        kept_TR_hG2=kept_TR,
        keep_frac_hG2=kept_TR / N_TR if N_TR else np.nan,
        nSeg_hG2=len(segs_keep),
        maxSeg_hG2=maxSeg_keep,

        kept_TR_center=kept_center,
        keep_frac_center=kept_center / N_TR if N_TR else np.nan,
        nSeg_center=len(segs_center_all),
        maxSeg_center=maxSeg_center,
        mask_path=str(mask_path),
    )

    # minlen summaries
    for minlen in MINLEN_LIST:
        segs_center_minlen = segs_minlen_from_mask(keep_center, minlen=minlen)
        kept_minlen = kept_tr_from_segs(segs_center_minlen)
        maxSeg_minlen = int(max([L for _,_,L in segs_center_minlen], default=0))
        row[f"nSeg_center_minlen{minlen}"] = int(len(segs_center_minlen))
        row[f"kept_TR_center_minlen{minlen}"] = int(kept_minlen)
        row[f"usable_min_center_minlen{minlen}"] = float(kept_minlen * TR_SEC / 60.0)
        row[f"maxSeg_center_minlen{minlen}"] = int(maxSeg_minlen)

    rows.append(row)

    # ---- Plot 1: keep vs center ----
    plt.figure(figsize=(11, 2.8))
    plt.plot(keep.astype(int), label="keep_hybridG2", lw=1)
    plt.plot(keep_center.astype(int), label=f"keep_center (lags={LAGS})", lw=1)
    plt.ylim(-0.1, 1.1)
    plt.xlabel("TR index")
    plt.ylabel("keep")
    plt.title(f"{run}: kept TRs and lag-usable centers")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{run}_keep_vs_center.png", dpi=180)
    plt.close()

    # ---- Plot 2: segment lengths (kept vs center) ----
    keep_lengths = [L for _,_,L in segs_keep]
    center_lengths = [L for _,_,L in segs_center_all]
    plt.figure(figsize=(8, 3))
    plt.hist(keep_lengths, bins=30, alpha=0.7, label="kept segments")
    plt.hist(center_lengths, bins=30, alpha=0.7, label="center-usable segments")
    plt.xlabel("segment length (TR)")
    plt.ylabel("count")
    plt.title(f"{run}: segment length distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{run}_segment_lengths.png", dpi=180)
    plt.close()

qc2 = pd.DataFrame(rows)

out_csv = QC_DIR / "viable_TR_for_lags_summary.csv"
qc2.to_csv(out_csv, index=False)
print("Saved:", out_csv)

# Safe display
qc2.head(30)

# =========================
# QC Cell 4 — Global plots: usable minutes after minlen + segment counts
# =========================
if len(qc2) == 0:
    print("[WARN] qc2 is empty — nothing to plot.")
else:
    # Filter to OK rows only (drop SKIPs)
    ok = qc2[qc2["status"] == "OK"].copy()

    # 1) usable minutes after minlen10 / minlen15
    for minlen in [10, 15]:
        col = f"usable_min_center_minlen{minlen}"
        if col not in ok.columns:
            print(f"[WARN] Missing column {col}; did Cell 3 run?")
            continue

        plt.figure(figsize=(10,4))
        plt.bar(ok["run"], ok[col])
        plt.xticks(rotation=90, fontsize=8)
        plt.ylabel("usable minutes")
        plt.title(f"Usable minutes (lag-centers) after minlen{minlen}")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"usable_minutes_minlen{minlen}_global.png", dpi=180)
        plt.show()

    # 2) segment count after minlen10 / minlen15
    for minlen in [10, 15]:
        col = f"nSeg_center_minlen{minlen}"
        if col not in ok.columns:
            print(f"[WARN] Missing column {col}; did Cell 3 run?")
            continue

        plt.figure(figsize=(10,4))
        plt.bar(ok["run"], ok[col])
        plt.xticks(rotation=90, fontsize=8)
        plt.ylabel("segment count")
        plt.title(f"Number of lag-usable segments after minlen{minlen}")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"segment_count_minlen{minlen}_global.png", dpi=180)
        plt.show()

    # 3) helpful scatter: fragmentation vs usable minutes (minlen10)
    if "nSeg_center_minlen10" in ok.columns and "usable_min_center_minlen10" in ok.columns:
        plt.figure(figsize=(6,5))
        plt.scatter(ok["nSeg_center_minlen10"], ok["usable_min_center_minlen10"])
        plt.xlabel("nSeg_center_minlen10")
        plt.ylabel("usable_min_center_minlen10")
        plt.title("Fragmentation vs usable minutes (minlen10)")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "fragmentation_vs_usable_minutes_minlen10.png", dpi=180)
        plt.show()

from pathlib import Path

OUT_ROOT = Path("/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/align_trmask_lags/FINAL_v3_gnorm_allTR/strict")

# show top-level structure
print("OUT_ROOT exists:", OUT_ROOT.exists())
print("Top-level dirs:", [p.name for p in OUT_ROOT.iterdir() if p.is_dir()][:20])

# find ANY hybrid mask files anywhere under OUT_ROOT
hits = list(OUT_ROOT.rglob("*hybrid*G2*.npy")) + list(OUT_ROOT.rglob("*Hybrid*G2*.npy"))
print("Found hybridG2 mask-ish npys:", len(hits))
for h in hits[:30]:
    print("  ", h)
```


```python
from pathlib import Path
import numpy as np

OUT_RUNS = Path("/mnt/c/EEGFMRI/hmm/R01_rerun/02_derivatives/fusion_prep/align_trmask_lags/FINAL_v3_gnorm_allTR/intermediate/per_run")

runs = sorted([p.name for p in OUT_RUNS.iterdir() if p.is_dir()])
need = [
    "keep_hybridG2.npy",
    "keep_center_lags-1_0_1.npy",
    "keep_center_lags0.npy",
    "keep_center_minlen10_lags-1_0_1.npy",
    "keep_center_minlen10_lags0.npy",
    "summary_minlen10.json",
    "summary0_minlen10.json",
]
missing = {}
for r in runs:
    rd = OUT_RUNS / r
    miss = [f for f in need if not (rd / f).exists()]
    if miss:
        missing[r] = miss

print("Runs:", len(runs))
print("Runs with missing artifacts:", len(missing))
for r, miss in missing.items():
    print(r, "->", miss)

```


```python

```
