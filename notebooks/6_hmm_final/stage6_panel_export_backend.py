"""Backend functions for the public Stage-6 optional panel-export workflow.

This module holds the active runtime behind step65_optional_export_figure4_figure5_panels.ipynb.
It keeps the optional figure-panel export logic in normal Python code while the notebook remains a thin public entry point."""

from __future__ import annotations

from stage6_backend_common import load_json_file, require_result_file, resolve_existing_path
from stage6_crossmodal_utils import (
    aggregate_block_matrix,
    backproject_modal_blocks,
    cov_to_crosscorr,
    ensure_cov_3d,
    load_schaefer_table,
    rank_block_contrasts,
    reorder_by_network,
)
from stage6_matrix_utils import compute_symmetric_limits, ut_to_square

def run_panel_export_backend(
    *,
    final_model_root,
    templateflow_root,
    panel_export_root,
    parcel_labels_file=None,
    reference_state_override=None,
    top_n_bars: int = 10,
):
    """Export optional Figure-4 and Figure-5 panel files from saved final-fit outputs."""
    from pathlib import Path
    import warnings

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 14,
        "axes.titlesize": 17,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    
    pd.set_option("display.max_rows", 300)
    pd.set_option("display.max_columns", 300)
    
    print("Imports ready.")
    
    RESULT_ROOT = Path(final_model_root)
    FINAL_DIR = RESULT_ROOT / "final"
    TEMPLATEFLOW_HOME = Path(templateflow_root)
    TF_TPL_DIR = TEMPLATEFLOW_HOME / "tpl-MNI152NLin2009cAsym"
    DEFAULT_SCHAEFER_TSV = TF_TPL_DIR / "tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-200Parcels7Networks_dseg.tsv"
    DEFAULT_BRAINSTORM_TXT = TF_TPL_DIR / "tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-200Parcels7Networks_dseg.txt"
    PARCEL_LABELS_FILE = Path(parcel_labels_file) if parcel_labels_file is not None else None
    NETWORK_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
    REFERENCE_STATE_OVERRIDE = reference_state_override
    TOP_N_BARS = int(top_n_bars)
    OUT_DIR = Path(panel_export_root)
    FIG_DIR = OUT_DIR / "figures"
    TAB_DIR = OUT_DIR / "tables"
    for _p in (OUT_DIR, FIG_DIR, TAB_DIR):
        _p.mkdir(parents=True, exist_ok=True)
    
    print("RESULT_ROOT:", RESULT_ROOT)
    print("FINAL_DIR  :", FINAL_DIR)
    print("OUT_DIR    :", OUT_DIR)
    print("FIG_DIR    :", FIG_DIR)
    print("TAB_DIR    :", TAB_DIR)

    # Local plotting helpers stay in this file because they are specific to the
    # optional panel-export layout.
    def savefig(fig, name):
        fig.savefig(FIG_DIR / name, bbox_inches="tight")
        plt.show()

    def state_title(k, dominant_idx, ref_idx):
        title = f"S{k+1}"
        if k == dominant_idx:
            title += " (dominant"
        if k == ref_idx:
            title += " and reference)"
        return title
    
    def add_panel_letter(ax, letter, x=-0.30, y=1.02, size=30):
        ax.text(x, y, letter, transform=ax.transAxes,
                fontsize=size, fontweight="bold", va="top", ha="left")
    
    def plot_block_heatmap(ax, df, title, vlim, cmap="coolwarm"):
        im = ax.imshow(df.values, cmap=cmap, norm=TwoSlopeNorm(vcenter=0.0, vmin=-vlim, vmax=vlim))
        ax.set_title(title, pad=8)
        ax.set_xticks(range(len(df.columns)))
        ax.set_yticks(range(len(df.index)))
        ax.set_xticklabels(df.columns, rotation=40, ha="right")
        ax.set_yticklabels(df.index)
        return im
    
    def plot_ranked_bars(ax, rank_df, title, xlabel, top_n=10, label_fmt="{r} -- {c}", height=0.52,
                         x_decimals=2, max_xticks=5, xpad_frac=0.10):
        df = rank_df.head(top_n).copy().iloc[::-1]
        labels = [label_fmt.format(r=r, c=c) for r, c in zip(df["row_network"], df["col_network"])]
        vals = df["delta_value"].to_numpy(dtype=float)
        colors = ["#B2182B" if d > 0 else "#2166AC" for d in vals]
        y = np.arange(len(df))
        ax.barh(y, vals, color=colors, height=height)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(0, color="black", lw=0.9)
        ax.set_title(title, fontsize=14, pad=5)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.tick_params(axis="x", labelsize=10)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=max_xticks))
        ax.xaxis.set_major_formatter(FormatStrFormatter(f"%.{x_decimals}f"))
    
        if np.any(vals < 0) and np.any(vals > 0):
            lim = np.max(np.abs(vals)) * (1 + xpad_frac)
            ax.set_xlim(-lim, lim)
        elif np.all(vals >= 0):
            vmax = max(float(np.max(vals)), 1e-8)
            ax.set_xlim(0, vmax * (1 + xpad_frac))
        elif np.all(vals <= 0):
            vmin = min(float(np.min(vals)), -1e-8)
            ax.set_xlim(vmin * (1 + xpad_frac), 0)
    
        ax.margins(y=0.03)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        return ax
    
    print("Helpers ready.")

    # Load final-fit artifacts and rebuild the panel inputs used for optional
    # Figure 4 and Figure 5 export.
    required = {
        "state_signature": require_result_file(RESULT_ROOT, FINAL_DIR, "state_signature_ut_boldcorr.npy"),
        "covs_pca": require_result_file(RESULT_ROOT, FINAL_DIR, "covs_pca.npy"),
        "preproc_params": resolve_existing_path(RESULT_ROOT / "preproc_params.npz", FINAL_DIR / "preproc_params.npz"),
        "best_seed": resolve_existing_path(FINAL_DIR / "best_seed.json", RESULT_ROOT / "best_seed.json"),
    }
    for k, v in required.items():
        print(f"{k:14s} -> {v}")
    if required["preproc_params"] is None:
        raise FileNotFoundError("Could not find preproc_params.npz")

    labels_df_raw = load_schaefer_table(
        PARCEL_LABELS_FILE,
        DEFAULT_SCHAEFER_TSV,
        DEFAULT_BRAINSTORM_TXT,
    )
    ordered_labels_df, order_idx = reorder_by_network(labels_df_raw, NETWORK_ORDER)

    best_info = load_json_file(required["best_seed"]) if required["best_seed"] is not None else {}
    if isinstance(best_info, dict) and "fo" in best_info:
        final_fo = np.asarray(best_info["fo"], dtype=float)
    else:
        raise ValueError("best_seed.json with final FO vector is required.")
    
    dominant_state_idx = int(np.nanargmax(final_fo))
    if REFERENCE_STATE_OVERRIDE is not None:
        ref_idx = int(str(REFERENCE_STATE_OVERRIDE).replace("S", "")) - 1
    else:
        ref_idx = dominant_state_idx

    print("Final FO vector:", np.round(final_fo, 6))
    print("Dominant/reference state:", f"S{ref_idx+1}")

    # Build BOLD-only block summaries from the saved state signatures.
    sig_ut = np.load(required["state_signature"])
    parcel_mats = np.stack([ut_to_square(sig_ut[k], fill_diag=1.0) for k in range(sig_ut.shape[0])], axis=0)
    parcel_mats_ord = parcel_mats[:, order_idx][:, :, order_idx]

    bold_state_blocks = {}
    for k in range(parcel_mats_ord.shape[0]):
        dfb = aggregate_block_matrix(
            parcel_mats_ord[k],
            labels_row=ordered_labels_df["network"].to_numpy(),
            labels_col=ordered_labels_df["network"].to_numpy(),
            order_row=NETWORK_ORDER,
            order_col=NETWORK_ORDER,
        )
        bold_state_blocks[f"S{k+1}"] = dfb
        dfb.to_csv(TAB_DIR / f"bold_block_matrix_S{k+1}.tsv", sep="\t")
    
    bold_contrast_blocks = {}
    bold_rank_tables = {}
    for state, dfb in bold_state_blocks.items():
        if state == f"S{ref_idx+1}":
            continue
        diff = dfb - bold_state_blocks[f"S{ref_idx+1}"]
        cname = f"{state} - S{ref_idx+1}"
        bold_contrast_blocks[cname] = diff
        bold_rank_tables[cname] = rank_block_contrasts(diff, top_n=TOP_N_BARS)
        diff.to_csv(TAB_DIR / f"bold_block_matrix_{state}_minus_S{ref_idx+1}.tsv", sep="\t")
        bold_rank_tables[cname].to_csv(TAB_DIR / f"top_bold_network_contrasts_{cname.replace(' ', '_').replace('-', 'minus')}.tsv", sep="\t", index=False)

    # Build cross-modal block summaries from the saved PCA-space covariances.
    covs_pca = ensure_cov_3d(np.load(required["covs_pca"]))
    pp = np.load(required["preproc_params"])
    Vb = pp["Vb"]
    Ve = pp["Ve"]
    
    cov_bb, cov_ee, cov_be = backproject_modal_blocks(covs_pca, Vb, Ve)
    crosscorr = np.stack([cov_to_crosscorr(cov_bb[k], cov_ee[k], cov_be[k]) for k in range(cov_be.shape[0])])
    
    cross_state_blocks = {}
    for k in range(crosscorr.shape[0]):
        dfb = aggregate_block_matrix(
            crosscorr[k],
            labels_row=labels_df_raw["network"].to_numpy(),
            labels_col=labels_df_raw["network"].to_numpy(),
            order_row=NETWORK_ORDER,
            order_col=NETWORK_ORDER,
        )
        cross_state_blocks[f"S{k+1}"] = dfb
        dfb.to_csv(TAB_DIR / f"crossmodal_block_matrix_S{k+1}.tsv", sep="\t")
    
    cross_contrast_blocks = {}
    cross_rank_tables = {}
    for state, dfb in cross_state_blocks.items():
        if state == f"S{ref_idx+1}":
            continue
        diff = dfb - cross_state_blocks[f"S{ref_idx+1}"]
        cname = f"{state} - S{ref_idx+1}"
        cross_contrast_blocks[cname] = diff
        cross_rank_tables[cname] = rank_block_contrasts(diff, top_n=TOP_N_BARS)
        diff.to_csv(TAB_DIR / f"crossmodal_block_matrix_{state}_minus_S{ref_idx+1}.tsv", sep="\t")
        cross_rank_tables[cname].to_csv(TAB_DIR / f"top_crossmodal_network_contrasts_{cname.replace(' ', '_').replace('-', 'minus')}.tsv", sep="\t", index=False)
    
    print("\nBOLD state blocks:", list(bold_state_blocks.keys()))
    print("Cross-modal state blocks:", list(cross_state_blocks.keys()))

    # Figure 4 panel exports.
    state_keys = [f"S{i+1}" for i in range(len(bold_state_blocks))]
    contrast_keys = [f"S{i+1} - S{ref_idx+1}" for i in range(len(bold_state_blocks)) if i != ref_idx]

    bold_vmax_state = compute_symmetric_limits([bold_state_blocks[k].values for k in state_keys], quantile=0.98)
    bold_vmax_diff = compute_symmetric_limits([bold_contrast_blocks[k].values for k in contrast_keys], quantile=0.98)
    
    
    def export_fig4A_bold_states():
        fig = plt.figure(figsize=(15.8, 5.2), constrained_layout=False)
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.22)
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        cax = fig.add_subplot(gs[0, 3])
    
        im = None
        for ax, k in zip(axes, range(len(state_keys))):
            im = plot_block_heatmap(ax, bold_state_blocks[f"S{k+1}"], state_title(k, dominant_state_idx, ref_idx), bold_vmax_state)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("mean r")
        cb.ax.tick_params(labelsize=11)
    
        fig.suptitle("State-wise BOLD network organization", y=0.985, fontsize=19)
        fig.subplots_adjust(left=0.055, right=0.985, top=0.84, bottom=0.19, wspace=0.22)
        savefig(fig, "Fig4A_BOLD_states.png")
    
    
    def export_fig4B_bold_contrasts():
        fig = plt.figure(figsize=(11.4, 5.2), constrained_layout=False)
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.20)
        axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
        cax = fig.add_subplot(gs[0, 2])
    
        im = None
        for ax, cname in zip(axes, contrast_keys):
            im = plot_block_heatmap(ax, bold_contrast_blocks[cname], cname, bold_vmax_diff)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Δ mean r")
        cb.ax.tick_params(labelsize=11)
    
        fig.suptitle(f"BOLD network contrasts relative to S{ref_idx+1}", y=0.995, fontsize=19)
        fig.subplots_adjust(left=0.06, right=0.985, top=0.88, bottom=0.16, wspace=0.20)
        savefig(fig, "Fig4B_BOLD_contrasts.png")
    
    
    def export_fig4C1_bold_ranked():
        fig, ax = plt.subplots(figsize=(6.0, 4.7), constrained_layout=False)
        plot_ranked_bars(ax, bold_rank_tables[contrast_keys[0]],
                         title=f"Top ranked BOLD network contrasts: {contrast_keys[0].replace(' - ', ' vs ')}",
                         xlabel="Δ mean r", top_n=TOP_N_BARS, label_fmt="{r} -- {c}", height=0.42,
                         x_decimals=2, max_xticks=5)
        fig.subplots_adjust(left=0.33, right=0.98, top=0.88, bottom=0.15)
        savefig(fig, "Fig4C1_BOLD_ranked_S1_vs_S2.png")
    
    
    def export_fig4C2_bold_ranked():
        fig, ax = plt.subplots(figsize=(6.0, 4.7), constrained_layout=False)
        plot_ranked_bars(ax, bold_rank_tables[contrast_keys[1]],
                         title=f"Top ranked BOLD network contrasts: {contrast_keys[1].replace(' - ', ' vs ')}",
                         xlabel="Δ mean r", top_n=TOP_N_BARS, label_fmt="{r} -- {c}", height=0.42,
                         x_decimals=2, max_xticks=5)
        fig.subplots_adjust(left=0.33, right=0.98, top=0.88, bottom=0.15)
        savefig(fig, "Fig4C2_BOLD_ranked_S3_vs_S2.png")
    
    
    export_fig4A_bold_states()
    export_fig4B_bold_contrasts()
    export_fig4C1_bold_ranked()
    export_fig4C2_bold_ranked()

    # Figure 5 panel exports.
    state_keys_cross = [f"S{i+1}" for i in range(len(cross_state_blocks))]
    contrast_keys_cross = [f"S{i+1} - S{ref_idx+1}" for i in range(len(cross_state_blocks)) if i != ref_idx]
    
    cross_vmax_state = compute_symmetric_limits([cross_state_blocks[k].values for k in state_keys_cross], quantile=0.98)
    cross_vmax_diff = compute_symmetric_limits([cross_contrast_blocks[k].values for k in contrast_keys_cross], quantile=0.98)
    
    
    def export_fig5A_cross_states():
        fig = plt.figure(figsize=(16.8, 5.5), constrained_layout=False)
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.42)
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        cax = fig.add_subplot(gs[0, 3])
    
        im = None
        for idx, (ax, k) in enumerate(zip(axes, range(len(state_keys_cross)))):
            im = plot_block_heatmap(ax, cross_state_blocks[f"S{k+1}"], state_title(k, dominant_state_idx, ref_idx), cross_vmax_state)
            ax.set_xlabel("EEG network")
            ax.set_ylabel("BOLD network" if idx == 0 else "", labelpad=8)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Cross-modal correlation (unitless)")
        cb.ax.tick_params(labelsize=11)
    
        fig.suptitle("Descriptive cross-modal BOLD–EEG block structure", y=0.995, fontsize=19)
        fig.subplots_adjust(left=0.06, right=0.985, top=0.84, bottom=0.20, wspace=0.42)
        savefig(fig, "Fig5A_crossmodal_states.png")
    
    
    def export_fig5B_cross_contrasts():
        fig = plt.figure(figsize=(12.4, 5.6), constrained_layout=False)
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.40)
        axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
        cax = fig.add_subplot(gs[0, 2])
    
        im = None
        for idx, (ax, cname) in enumerate(zip(axes, contrast_keys_cross)):
            im = plot_block_heatmap(ax, cross_contrast_blocks[cname], cname, cross_vmax_diff)
            ax.set_xlabel("EEG network")
            ax.set_ylabel("BOLD network" if idx == 0 else "", labelpad=8)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Δ cross-modal correlation (unitless)")
        cb.ax.tick_params(labelsize=11)
    
        fig.suptitle(f"Cross-modal contrasts relative to S{ref_idx+1}", y=0.995, fontsize=19)
        fig.subplots_adjust(left=0.08, right=0.985, top=0.84, bottom=0.18, wspace=0.40)
        savefig(fig, "Fig5B_crossmodal_contrasts.png")
    
    
    def export_fig5C1_cross_ranked():
        fig, ax = plt.subplots(figsize=(6.1, 4.9), constrained_layout=False)
        plot_ranked_bars(ax, cross_rank_tables[contrast_keys_cross[0]],
                         title=f"Top ranked cross-modal contrasts: {contrast_keys_cross[0].replace(' - ', ' vs ')}",
                         xlabel="Δ cross-modal correlation", top_n=TOP_N_BARS, label_fmt="{r} | {c}", height=0.52,
                         x_decimals=2, max_xticks=5, xpad_frac=0.12)
        fig.subplots_adjust(left=0.28, right=0.98, top=0.88, bottom=0.15)
        savefig(fig, "Fig5C1_crossmodal_ranked_S1_vs_S2.png")
    
    
    def export_fig5C2_cross_ranked():
        fig, ax = plt.subplots(figsize=(6.4, 4.9), constrained_layout=False)
        plot_ranked_bars(ax, cross_rank_tables[contrast_keys_cross[1]],
                         title=f"Top ranked cross-modal contrasts: {contrast_keys_cross[1].replace(' - ', ' vs ')}",
                         xlabel="Δ cross-modal correlation", top_n=TOP_N_BARS, label_fmt="{r} | {c}", height=0.52,
                         x_decimals=2, max_xticks=5, xpad_frac=0.12)
        fig.subplots_adjust(left=0.40, right=0.98, top=0.88, bottom=0.15)
        savefig(fig, "Fig5C2_crossmodal_ranked_S3_vs_S2.png")
    
    
    export_fig5A_cross_states()
    export_fig5B_cross_contrasts()
    export_fig5C1_cross_ranked()
    export_fig5C2_cross_ranked()

    # Save a compact manifest of the optional panel files written here.
    manifest = pd.DataFrame([
        {"figure": "Fig4A", "panel": "BOLD states", "file": "Fig4A_BOLD_states.png"},
        {"figure": "Fig4B", "panel": "BOLD contrasts", "file": "Fig4B_BOLD_contrasts.png"},
        {"figure": "Fig4C1", "panel": "BOLD ranked S1 vs S2", "file": "Fig4C1_BOLD_ranked_S1_vs_S2.png"},
        {"figure": "Fig4C2", "panel": "BOLD ranked S3 vs S2", "file": "Fig4C2_BOLD_ranked_S3_vs_S2.png"},
        {"figure": "Fig5A", "panel": "Cross-modal states", "file": "Fig5A_crossmodal_states.png"},
        {"figure": "Fig5B", "panel": "Cross-modal contrasts", "file": "Fig5B_crossmodal_contrasts.png"},
        {"figure": "Fig5C1", "panel": "Cross-modal ranked S1 vs S2", "file": "Fig5C1_crossmodal_ranked_S1_vs_S2.png"},
        {"figure": "Fig5C2", "panel": "Cross-modal ranked S3 vs S2", "file": "Fig5C2_crossmodal_ranked_S3_vs_S2.png"},
    ])
    manifest.to_csv(TAB_DIR / "fig4_fig5_panel_manifest.tsv", sep="	", index=False)
    manifest
    
    return {
        "result_root": str(RESULT_ROOT),
        "output_dir": str(OUT_DIR),
        "figure_dir": str(FIG_DIR),
        "table_dir": str(TAB_DIR),
        "manifest_tsv": str(TAB_DIR / "fig4_fig5_panel_manifest.tsv"),
    }
