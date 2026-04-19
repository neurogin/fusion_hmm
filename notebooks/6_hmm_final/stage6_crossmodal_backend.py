"""Backend functions for the public Stage-6 cross-modal reconstruction workflow.

This module holds the active runtime behind step63_reconstruct_crossmodal_state_blocks_and_ranked_contrasts.ipynb.
It keeps only the true cross-modal reconstruction half of the mixed provenance notebook as active backend logic."""

from __future__ import annotations


def _display_fallback(obj):
    try:
        from IPython.display import display as ipy_display
        ipy_display(obj)
    except Exception:
        print(obj)


def run_crossmodal_reconstruction_backend(
    *,
    final_model_root,
    templateflow_root,
    crossmodal_output_root,
    parcel_labels_file=None,
    reference_state_override=None,
):
    """Rebuild the cross-modal state-block summaries from saved final-fit artifacts."""
    from pathlib import Path
    from pathlib import Path
    import json, math, re, warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    
    display = _display_fallback
    
    print("Imports ready.")
    
    ALIGN_ROOT = Path(final_model_root) / "_alignment_provenance_not_used_by_public_path"
    RESULT_ROOT = Path(final_model_root)
    FINAL_DIR = RESULT_ROOT / "final"
    SCHAEFER_TEMPLATE_DIR = Path(templateflow_root) / "tpl-MNI152NLin2009cAsym"
    REPRESENTATIVE_RUN = None
    REPRESENTATIVE_PARCEL_MODE = "max_abs_corr_on_kept"
    PARCEL_INDEX_FIXED = 0
    KEEP_MASK_CANDIDATES = [
        "keep_center_minlen15_lags0.npy",
        "keep_center_lags0.npy",
        "keep_center_minlen15.npy",
    ]
    DEFAULT_SCHAEFER_TSV = SCHAEFER_TEMPLATE_DIR / "tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-200Parcels7Networks_dseg.tsv"
    DEFAULT_BRAINSTORM_TXT = SCHAEFER_TEMPLATE_DIR / "tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-200Parcels7Networks_dseg.txt"
    PARCEL_LABELS_FILE = Path(parcel_labels_file) if parcel_labels_file is not None else None
    NETWORK_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
    OUT_DIR = Path(crossmodal_output_root)
    FIG_DIR = OUT_DIR / "figures"
    TABLE_DIR = OUT_DIR / "tables"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    ABS_VMAX_QUANTILE = 0.98
    DIFF_VMAX_QUANTILE = 0.98
    REFERENCE_STATE_OVERRIDE = reference_state_override
    
    print("ALIGN_ROOT :", ALIGN_ROOT)
    print("RESULT_ROOT:", RESULT_ROOT)
    print("FINAL_DIR  :", FINAL_DIR)
    print("OUT_DIR    :", OUT_DIR)
    # ---- notebook cell 3 ----
    
    # =========================
    # HELPERS
    # =========================
    
    def resolve_existing(*candidates):
        for c in candidates:
            if c is None:
                continue
            p = Path(c)
            if p.exists():
                return p
        return None
    
    def zscore_keep(x, keep):
        x = np.asarray(x, dtype=float)
        keep = np.asarray(keep).astype(bool)
        out = np.full_like(x, np.nan, dtype=float)
        if keep.sum() < 2:
            return out
        mu = np.nanmean(x[keep])
        sd = np.nanstd(x[keep], ddof=0)
        if not np.isfinite(sd) or sd == 0:
            out[keep] = x[keep] - mu
        else:
            out[keep] = (x[keep] - mu) / sd
        return out
    
    def compute_symmetric_limits(arrays, quantile=0.98, floor=1e-8):
        vals = []
        for a in arrays:
            a = np.asarray(a, dtype=float)
            a = a[np.isfinite(a)]
            if a.size:
                vals.append(np.abs(a))
        if not vals:
            return 1.0
        cat = np.concatenate(vals)
        vmax = np.quantile(cat, quantile)
        return float(max(vmax, floor))
    
    def savefig(fig, name):
        fig.savefig(FIG_DIR / name, bbox_inches="tight")
        plt.show()
    
    def load_json(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def infer_result_file(name):
        return resolve_existing(RESULT_ROOT / name, FINAL_DIR / name, Path(name))
    
    def ensure_cov_3d(c):
        c = np.asarray(c)
        if c.ndim == 2:
            c = c[None, ...]
        return c
    
    # --- Schaefer label parsing ---
    NETWORK_ALIASES = {
        "Vis": "Vis",
        "SomMot": "SomMot",
        "DorsAttn": "DorsAttn",
        "SalVentAttn": "SalVentAttn",
        "SalVent": "SalVentAttn",
        "VentAttn": "SalVentAttn",
        "Limbic": "Limbic",
        "Cont": "Cont",
        "Control": "Cont",
        "Default": "Default",
        "DefaultMode": "Default",
    }
    
    def parse_schaefer_name(name):
        name = str(name)
        hemi = None
        m_hemi = re.search(r'_(LH|RH)_', name)
        if m_hemi:
            hemi = "L" if m_hemi.group(1) == "LH" else "R"
        net = None
        for token in re.split(r"[_\-]", name):
            if token in NETWORK_ALIASES:
                net = NETWORK_ALIASES[token]
                break
        if net is None:
            m = re.search(r'7Networks_(LH|RH)_([^_]+)', name)
            if m:
                net = NETWORK_ALIASES.get(m.group(2), m.group(2))
        return hemi, net
    
    def load_schaefer_table(label_file=None):
        label_path = resolve_existing(label_file, DEFAULT_SCHAEFER_TSV, DEFAULT_BRAINSTORM_TXT)
        if label_path is None:
            raise FileNotFoundError("Could not find Schaefer label table (.tsv or .txt).")
        if label_path.suffix.lower() == ".tsv":
            df = pd.read_csv(label_path, sep="\t")
            id_col = "index" if "index" in df.columns else "id" if "id" in df.columns else None
            if id_col is None:
                ids = np.arange(1, len(df) + 1, dtype=int)
            else:
                ids_series = pd.to_numeric(df[id_col], errors="coerce")
                fallback = pd.Series(np.arange(1, len(df) + 1), index=df.index, dtype="int64")
                ids = ids_series.where(ids_series.notna(), fallback).astype(int).to_numpy()
            name_col = "name" if "name" in df.columns else "label" if "label" in df.columns else df.columns[-1]
            names = df[name_col].astype(str).to_numpy()
        else:
            raw = pd.read_csv(label_path, sep="\t", header=None)
            if raw.shape[1] == 1:
                raw = pd.read_csv(label_path, sep=None, engine="python", header=None)
            raw = raw.iloc[:, :2]
            raw.columns = ["atlas_id", "name"]
            ids = pd.to_numeric(raw["atlas_id"], errors="coerce").fillna(pd.Series(np.arange(1, len(raw) + 1), index=raw.index)).astype(int).to_numpy()
            names = raw["name"].astype(str).to_numpy()
    
        out = pd.DataFrame({"atlas_id": ids, "label": names})
        out = out[out["atlas_id"] > 0].copy()
        parsed = out["label"].apply(parse_schaefer_name)
        out["hemi"] = [p[0] for p in parsed]
        out["network"] = [p[1] for p in parsed]
        # Keep the first 200 positive labels if a background row sneaks in
        out = out.iloc[:200].copy()
        out["parcel_idx_1based"] = np.arange(1, len(out) + 1)
        out["parcel_idx_0based"] = np.arange(len(out))
        return out
    
    def aggregate_block_matrix(mat, labels_row, labels_col=None, order_row=None, order_col=None):
        mat = np.asarray(mat, dtype=float)
        labels_row = np.asarray(labels_row)
        if labels_col is None:
            labels_col = labels_row
        else:
            labels_col = np.asarray(labels_col)
        if order_row is None:
            order_row = list(pd.unique(labels_row))
        if order_col is None:
            order_col = list(pd.unique(labels_col))
        out = np.full((len(order_row), len(order_col)), np.nan, dtype=float)
        for i, nr in enumerate(order_row):
            idx_r = np.where(labels_row == nr)[0]
            for j, nc in enumerate(order_col):
                idx_c = np.where(labels_col == nc)[0]
                if len(idx_r) == 0 or len(idx_c) == 0:
                    continue
                block = mat[np.ix_(idx_r, idx_c)]
                out[i, j] = np.nanmean(block)
        return pd.DataFrame(out, index=order_row, columns=order_col)
    
    def rank_block_contrasts(diff_df, top_n=15):
        rows = []
        for r in diff_df.index:
            for c in diff_df.columns:
                rows.append((r, c, float(diff_df.loc[r, c])))
        out = pd.DataFrame(rows, columns=["bold_network", "eeg_network", "delta_value"])
        out["abs_delta"] = out["delta_value"].abs()
        out = out.sort_values(["abs_delta", "delta_value"], ascending=[False, False]).head(top_n).reset_index(drop=True)
        return out
    
    print("Helpers ready.")
    
    # ---- notebook cell 4 ----
    # =========================
    # RESOLVE REQUIRED FILES (corrected)
    # =========================
    
    required = {
        # Core cross-modal backprojection files
        "covs_pca": infer_result_file("covs_pca.npy"),
        "preproc_params": resolve_existing(
            RESULT_ROOT / "preproc_params.npz",
            FINAL_DIR / "preproc_params.npz",
        ),
    
        # Correct source for FINAL model-level FO vector / reference state
        "best_seed": resolve_existing(
            FINAL_DIR / "best_seed.json",
            RESULT_ROOT / "best_seed.json",
        ),
    
        # Useful fallbacks / summaries
        "run_metrics": resolve_existing(RESULT_ROOT / "run_metrics.tsv"),
        "subject_metrics": resolve_existing(RESULT_ROOT / "subject_metrics.tsv"),
        "qc_summary": resolve_existing(
            RESULT_ROOT / "qc_summary.json",
            FINAL_DIR / "qc_summary.json",
        ),
    
        # Optional manuscript-derived summary table if present
        "state_summary_table": resolve_existing(
            RESULT_ROOT / "manuscript_figures" / "tables" / "state_summary_table.tsv",
            RESULT_ROOT / "physiology_review_schaefer" / "tables" / "state_summary_table.tsv",
            Path("/mnt/data/state_summary_table.tsv"),
        ),
    }
    
    for k, v in required.items():
        print(f"{k:18s} -> {v}")
    
    if required["covs_pca"] is None or required["preproc_params"] is None:
        raise FileNotFoundError(
            "Need covs_pca.npy and preproc_params.npz for cross-modal backprojection."
        )
    
    if required["best_seed"] is None:
        print("WARNING: best_seed.json not found. Reference-state selection will use fallbacks.")
    
    schaefer_df = load_schaefer_table(PARCEL_LABELS_FILE)
    
    print("\nLabel table head:")
    print(schaefer_df.head())
    
    print("\nNetwork counts:")
    print(schaefer_df["network"].value_counts(dropna=False).sort_index())
    # ---- notebook cell 8 ----
    # =========================
    # PART B — CROSS-MODAL BOLD–EEG BLOCK MAPS (corrected reference-state logic)
    # =========================
    
    # Optional manual override
    covs_pca = ensure_cov_3d(np.load(required["covs_pca"]))
    pp = np.load(required["preproc_params"])
    
    Vb = pp["Vb"]
    Ve = pp["Ve"]
    nb = Vb.shape[1]
    ne = Ve.shape[1]
    
    print("covs_pca shape:", covs_pca.shape)
    print("Vb shape      :", Vb.shape)
    print("Ve shape      :", Ve.shape)
    
    if covs_pca.shape[1] < nb + ne or covs_pca.shape[2] < nb + ne:
        raise ValueError("covs_pca dimensions do not match nb+ne from preproc_params.npz")
    
    # -------------------------------------------------
    # Correct reference state determination
    # Priority:
    # 1) explicit manual override
    # 2) FINAL_DIR / best_seed.json (correct final model-level FO vector)
    # 3) state_summary_table.tsv
    # 4) weighted subject/run FO summaries
    # 5) fallback to S2
    # -------------------------------------------------
    final_fo = None
    reference_state_idx = None
    fo_source = None
    
    if REFERENCE_STATE_OVERRIDE is not None:
        reference_state_idx = int(str(REFERENCE_STATE_OVERRIDE).replace("S", "")) - 1
        fo_source = f"manual override ({REFERENCE_STATE_OVERRIDE})"
    
    elif required["best_seed"] is not None:
        best_info = load_json(required["best_seed"])
        if isinstance(best_info, dict) and "fo" in best_info:
            final_fo = np.asarray(best_info["fo"], dtype=float)
            reference_state_idx = int(np.nanargmax(final_fo))
            fo_source = str(required["best_seed"])
    
    elif required["state_summary_table"] is not None:
        ss = pd.read_csv(required["state_summary_table"], sep="\t")
        if "is_reference" in ss.columns and ss["is_reference"].astype(int).sum() == 1:
            reference_state_idx = int(np.flatnonzero(ss["is_reference"].astype(int).to_numpy())[0])
            fo_source = str(required["state_summary_table"]) + " (is_reference)"
            if "final_FO" in ss.columns:
                final_fo = ss["final_FO"].astype(float).to_numpy()
        elif "is_dominant" in ss.columns and ss["is_dominant"].astype(int).sum() == 1:
            reference_state_idx = int(np.flatnonzero(ss["is_dominant"].astype(int).to_numpy())[0])
            fo_source = str(required["state_summary_table"]) + " (is_dominant)"
            if "final_FO" in ss.columns:
                final_fo = ss["final_FO"].astype(float).to_numpy()
        elif "final_FO" in ss.columns:
            final_fo = ss["final_FO"].astype(float).to_numpy()
            reference_state_idx = int(np.nanargmax(final_fo))
            fo_source = str(required["state_summary_table"]) + " (final_FO)"
    
    elif required["subject_metrics"] is not None:
        sm = pd.read_csv(required["subject_metrics"], sep="\t")
        fo_cols = [c for c in sm.columns if c.startswith("FO_s")]
        if len(fo_cols):
            final_fo = sm[fo_cols].mean(axis=0).to_numpy(dtype=float)
            reference_state_idx = int(np.nanargmax(final_fo))
            fo_source = str(required["subject_metrics"]) + " (mean subject FO)"
    
    elif required["run_metrics"] is not None:
        rm = pd.read_csv(required["run_metrics"], sep="\t")
        fo_cols = [c for c in rm.columns if c.startswith("FO_s")]
        if len(fo_cols):
            final_fo = rm[fo_cols].mean(axis=0).to_numpy(dtype=float)
            reference_state_idx = int(np.nanargmax(final_fo))
            fo_source = str(required["run_metrics"]) + " (mean run FO)"
    
    else:
        reference_state_idx = 1
        fo_source = "fallback default S2"
    
    if reference_state_idx is None:
        reference_state_idx = 1
        fo_source = "fallback default S2"
    
    ref_state = f"S{reference_state_idx + 1}"
    
    print("Reference-state source:", fo_source)
    if final_fo is not None:
        print("Final FO vector used:", np.round(final_fo, 6))
    print("Reference state:", ref_state)
    
    # -------------------------------------------------
    # Backproject state-wise BOLD, EEG, and cross-modal blocks
    # -------------------------------------------------
    def backproject_modal_blocks(covs_pca, Vb, Ve):
        covs_pca = ensure_cov_3d(covs_pca)
        K = covs_pca.shape[0]
        out_bb, out_ee, out_be = [], [], []
    
        for k in range(K):
            Cbb_p = covs_pca[k, :Vb.shape[1], :Vb.shape[1]]
            Cee_p = covs_pca[k, Vb.shape[1]:Vb.shape[1]+Ve.shape[1], Vb.shape[1]:Vb.shape[1]+Ve.shape[1]]
            Cbe_p = covs_pca[k, :Vb.shape[1], Vb.shape[1]:Vb.shape[1]+Ve.shape[1]]
    
            Cbb = (Vb @ Cbb_p @ Vb.T).astype(np.float32)
            Cee = (Ve @ Cee_p @ Ve.T).astype(np.float32)
            Cbe = (Vb @ Cbe_p @ Ve.T).astype(np.float32)
    
            out_bb.append(Cbb)
            out_ee.append(Cee)
            out_be.append(Cbe)
    
        return np.stack(out_bb), np.stack(out_ee), np.stack(out_be)
    
    def cov_to_crosscorr(Cbb, Cee, Cbe, eps=1e-8):
        vb = np.sqrt(np.clip(np.diag(Cbb), eps, None))
        ve = np.sqrt(np.clip(np.diag(Cee), eps, None))
        denom = np.outer(vb, ve)
        return Cbe / denom
    
    cov_bb, cov_ee, cov_be = backproject_modal_blocks(covs_pca, Vb, Ve)
    crosscorr = np.stack([cov_to_crosscorr(cov_bb[k], cov_ee[k], cov_be[k]) for k in range(cov_be.shape[0])])
    
    print("Backprojected shapes:")
    print("  cov_bb   :", cov_bb.shape)
    print("  cov_ee   :", cov_ee.shape)
    print("  cov_be   :", cov_be.shape)
    print("  crosscorr:", crosscorr.shape)
    
    np.save(TABLE_DIR / "state_crossmodal_crosscorr_parcel.npy", crosscorr)
    
    # ---- notebook cell 9 ----
    
    # Backproject state-wise BOLD, EEG, and cross-modal blocks into parcel space
    
    def backproject_modal_blocks(covs_pca, Vb, Ve):
        covs_pca = ensure_cov_3d(covs_pca)
        K = covs_pca.shape[0]
        out_bb, out_ee, out_be = [], [], []
        for k in range(K):
            Cbb_p = covs_pca[k, :Vb.shape[1], :Vb.shape[1]]
            Cee_p = covs_pca[k, Vb.shape[1]:Vb.shape[1]+Ve.shape[1], Vb.shape[1]:Vb.shape[1]+Ve.shape[1]]
            Cbe_p = covs_pca[k, :Vb.shape[1], Vb.shape[1]:Vb.shape[1]+Ve.shape[1]]
    
            Cbb = (Vb @ Cbb_p @ Vb.T).astype(np.float32)
            Cee = (Ve @ Cee_p @ Ve.T).astype(np.float32)
            Cbe = (Vb @ Cbe_p @ Ve.T).astype(np.float32)
    
            out_bb.append(Cbb)
            out_ee.append(Cee)
            out_be.append(Cbe)
        return np.stack(out_bb), np.stack(out_ee), np.stack(out_be)
    
    def cov_to_crosscorr(Cbb, Cee, Cbe, eps=1e-8):
        vb = np.sqrt(np.clip(np.diag(Cbb), eps, None))
        ve = np.sqrt(np.clip(np.diag(Cee), eps, None))
        denom = np.outer(vb, ve)
        return Cbe / denom
    
    cov_bb, cov_ee, cov_be = backproject_modal_blocks(covs_pca, Vb, Ve)
    crosscorr = np.stack([cov_to_crosscorr(cov_bb[k], cov_ee[k], cov_be[k]) for k in range(cov_be.shape[0])])
    
    print("Backprojected shapes:")
    print("  cov_bb   :", cov_bb.shape)
    print("  cov_ee   :", cov_ee.shape)
    print("  cov_be   :", cov_be.shape)
    print("  crosscorr:", crosscorr.shape)
    
    # Save raw parcel-level cross-modal matrices
    np.save(TABLE_DIR / "state_crossmodal_crosscorr_parcel.npy", crosscorr)
    
    # ---- notebook cell 10 ----
    
    # Build network-level BOLD-network × EEG-network block matrices
    
    labels = schaefer_df["network"].to_numpy()
    state_block_mats = {}
    for k in range(crosscorr.shape[0]):
        df_block = aggregate_block_matrix(
            crosscorr[k],
            labels_row=labels,
            labels_col=labels,
            order_row=NETWORK_ORDER,
            order_col=NETWORK_ORDER,
        )
        state_block_mats[f"S{k+1}"] = df_block
        df_block.to_csv(TABLE_DIR / f"crossmodal_block_matrix_S{k+1}.tsv", sep="\t")
    
    # Show the state block matrices
    for state, dfb in state_block_mats.items():
        print("\n", state)
        print(dfb.round(3))
    
    # ---- notebook cell 11 ----
    
    # Plot state-wise cross-modal block matrices
    
    K = len(state_block_mats)
    arrs = [df.values for df in state_block_mats.values()]
    abs_vmax = compute_symmetric_limits(arrs, quantile=ABS_VMAX_QUANTILE)
    
    fig, axes = plt.subplots(1, K, figsize=(5*K, 4.8), constrained_layout=True)
    if K == 1:
        axes = [axes]
    
    for ax, (state, dfb) in zip(axes, state_block_mats.items()):
        im = ax.imshow(dfb.values, cmap="coolwarm", vmin=-abs_vmax, vmax=abs_vmax)
        ax.set_title(f"{state}: BOLD–EEG cross-modal blocks")
        ax.set_xticks(range(len(dfb.columns)))
        ax.set_yticks(range(len(dfb.index)))
        ax.set_xticklabels(dfb.columns, rotation=45, ha="right")
        ax.set_yticklabels(dfb.index)
        ax.set_xlabel("EEG network")
        ax.set_ylabel("BOLD network")
    
    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label("Cross-modal correlation (unitless)")
    savefig(fig, "Fig_crossmodal_block_maps_states.png")
    
    # ---- notebook cell 12 ----
    
    # Differences versus the reference state
    ref_state = f"S{reference_state_idx+1}"
    contrast_blocks = {}
    for state, dfb in state_block_mats.items():
        if state == ref_state:
            continue
        diff = dfb - state_block_mats[ref_state]
        contrast_blocks[f"{state} - {ref_state}"] = diff
        diff.to_csv(TABLE_DIR / f"crossmodal_block_matrix_{state}_minus_{ref_state}.tsv", sep="\t")
    
    diff_vmax = compute_symmetric_limits([d.values for d in contrast_blocks.values()], quantile=DIFF_VMAX_QUANTILE)
    
    fig, axes = plt.subplots(1, len(contrast_blocks), figsize=(5*max(1, len(contrast_blocks)), 4.8), constrained_layout=True)
    if len(contrast_blocks) == 1:
        axes = [axes]
    
    for ax, (name, dfd) in zip(axes, contrast_blocks.items()):
        im = ax.imshow(dfd.values, cmap="coolwarm", vmin=-diff_vmax, vmax=diff_vmax)
        ax.set_title(name)
        ax.set_xticks(range(len(dfd.columns)))
        ax.set_yticks(range(len(dfd.index)))
        ax.set_xticklabels(dfd.columns, rotation=45, ha="right")
        ax.set_yticklabels(dfd.index)
        ax.set_xlabel("EEG network")
        ax.set_ylabel("BOLD network")
    
    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label("Δ cross-modal correlation (unitless)")
    savefig(fig, "Fig_crossmodal_block_maps_differences_vs_reference.png")
    
    # ---- notebook cell 13 ----
    
    # Ranked cross-modal network-pair contrasts
    rank_tables = []
    for name, dfd in contrast_blocks.items():
        rt = rank_block_contrasts(dfd, top_n=15)
        rt.insert(0, "contrast", name)
        rank_tables.append(rt)
        rt.to_csv(TABLE_DIR / f"top_crossmodal_network_contrasts_{name.replace(' ', '_').replace('-', 'minus')}.tsv", sep="\t", index=False)
    
    rank_all = pd.concat(rank_tables, ignore_index=True)
    rank_all.to_csv(TABLE_DIR / "top_crossmodal_network_contrasts_all.tsv", sep="\t", index=False)
    rank_all.head(30)
    
    # ---- notebook cell 14 ----
    
    # Plot ranked contrasts
    contrast_names = list(contrast_blocks.keys())
    fig, axes = plt.subplots(1, len(contrast_names), figsize=(6.2*max(1, len(contrast_names)), 5.4), constrained_layout=True)
    if len(contrast_names) == 1:
        axes = [axes]
    
    for ax, cname in zip(axes, contrast_names):
        sub = rank_all[rank_all["contrast"] == cname].copy().head(12)
        labels_plot = [f"{r} | {c}" for r, c in zip(sub["bold_network"], sub["eeg_network"])]
        y = np.arange(len(sub))
        ax.barh(y, sub["delta_value"].values)
        ax.set_yticks(y)
        ax.set_yticklabels(labels_plot)
        ax.invert_yaxis()
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(cname)
        ax.set_xlabel("Δ cross-modal correlation")
    
    savefig(fig, "Fig_crossmodal_ranked_network_contrasts.png")
    
    # ---- notebook cell 16 ----
    
    # Save a compact run manifest + session manifest
    run_manifest.to_csv(TABLE_DIR / "alignment_run_manifest.tsv", sep="\t", index=False)
    
    session_manifest = {
        "ALIGN_ROOT": str(ALIGN_ROOT),
        "RESULT_ROOT": str(RESULT_ROOT),
        "FINAL_DIR": str(FINAL_DIR),
        "chosen_run": chosen_run,
        "parcel_idx_0based": int(parcel_idx),
        "parcel_label": str(parcel_label),
        "parcel_network": str(parcel_network),
        "reference_state": f"S{reference_state_idx+1}",
        "n_states": int(crosscorr.shape[0]),
        "Vb_shape": list(Vb.shape),
        "Ve_shape": list(Ve.shape),
    }
    with open(TABLE_DIR / "session_manifest.json", "w", encoding="utf-8") as f:
        json.dump(session_manifest, f, indent=2)
    
    print("Saved outputs to:")
    print("  ", FIG_DIR)
    print("  ", TABLE_DIR)
    
    return {
        "result_root": str(RESULT_ROOT),
        "output_dir": str(OUT_DIR),
        "figure_dir": str(FIG_DIR),
        "table_dir": str(TABLE_DIR),
        "reference_state": f"S{reference_state_idx + 1}",
    }
