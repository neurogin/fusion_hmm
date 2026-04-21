"""Backend functions for the public Stage-6 BOLD-state reconstruction workflow.

This module holds the active runtime behind step62_reconstruct_bold_state_networks_and_ranked_contrasts.ipynb.
It preserves the saved-artifact reconstruction logic while keeping the public notebook compact."""

from __future__ import annotations

from stage6_backend_common import load_json_file, require_result_file, resolve_existing_path
from stage6_matrix_utils import corr_from_cov, square_to_ut, ut_to_square


def run_bold_state_reconstruction_backend(
    *,
    final_model_root,
    templateflow_root,
    state_summary_root,
    parcel_labels_file=None,
    preproc_params_file=None,
    reference_state=None,
):
    """Rebuild the BOLD-state network summaries from saved final-fit artifacts."""
    from pathlib import Path
    import json
    import re
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    pd.set_option("display.max_rows", 300)
    pd.set_option("display.max_columns", 300)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    RESULT_ROOT = Path(final_model_root)
    FINAL_DIR = RESULT_ROOT / "final"
    TEMPLATEFLOW_HOME = Path(templateflow_root)
    TF_TPL_DIR = TEMPLATEFLOW_HOME / "tpl-MNI152NLin2009cAsym"
    DEFAULT_SCHAEFER_TSV = TF_TPL_DIR / "tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-200Parcels7Networks_dseg.tsv"
    DEFAULT_SCHAEFER_NII = TF_TPL_DIR / "tpl-MNI152NLin2009cAsym_res-01_atlas-Schaefer2018_desc-200Parcels7Networks_dseg.nii.gz"
    DEFAULT_BRAINSTORM_TXT = TF_TPL_DIR / "tpl-MNI152NLin2009cAsym_res-01_atlas-Schaefer2018_desc-200Parcels7Networks_dseg.txt"
    PARCEL_LABELS_FILE = Path(parcel_labels_file) if parcel_labels_file is not None else None
    PREPROC_PARAMS_FILE = Path(preproc_params_file) if preproc_params_file is not None else None
    REFERENCE_STATE = reference_state
    USE_ABS_FOR_BLOCK_MEANS = False
    NETWORK_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
    FIG_DIR = Path(state_summary_root)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    print("RESULT_ROOT       :", RESULT_ROOT)
    print("FINAL_DIR         :", FINAL_DIR)
    print("DEFAULT_SCHAEFER_TSV :", DEFAULT_SCHAEFER_TSV)
    print("DEFAULT_BRAINSTORM_TXT:", DEFAULT_BRAINSTORM_TXT)
    print("FIG_DIR           :", FIG_DIR)
    # Resolve the saved final-fit outputs needed for the BOLD reconstruction.
    required = {
        "qc_summary": require_result_file(RESULT_ROOT, FINAL_DIR, "qc_summary.json"),
        "subject_metrics": require_result_file(RESULT_ROOT, FINAL_DIR, "subject_metrics.tsv"),
        "run_metrics": require_result_file(RESULT_ROOT, FINAL_DIR, "run_metrics.tsv"),
        "trans_prob": require_result_file(RESULT_ROOT, FINAL_DIR, "trans_prob.npy"),
        "state_signature": require_result_file(RESULT_ROOT, FINAL_DIR, "state_signature_ut_boldcorr.npy"),
        "covs_pca": require_result_file(RESULT_ROOT, FINAL_DIR, "covs_pca.npy"),
    }
    optional = {
        "refit_results": resolve_existing_path(FINAL_DIR / "refit_results.json", RESULT_ROOT / "refit_results.json"),
        "topM_seeds": resolve_existing_path(RESULT_ROOT / "topM_seeds.json"),
        "dwell_from_A": resolve_existing_path(RESULT_ROOT / "dwell_from_A.tsv"),
        "preproc_params": resolve_existing_path(PREPROC_PARAMS_FILE, RESULT_ROOT / "preproc_params.npz"),
        "atlas_tsv": resolve_existing_path(PARCEL_LABELS_FILE, DEFAULT_SCHAEFER_TSV),
        "atlas_txt": resolve_existing_path(PARCEL_LABELS_FILE, DEFAULT_BRAINSTORM_TXT),
    }
    
    print("Resolved files:")
    for k, v in required.items():
        print(f"  {k:16s} -> {v}")
    for k, v in optional.items():
        print(f"  {k:16s} -> {v}")
    
    # Load the saved result tables and arrays.
    qc = load_json_file(required["qc_summary"])
    subj = pd.read_csv(required["subject_metrics"], sep="\t")
    runs = pd.read_csv(required["run_metrics"], sep="\t")
    A = np.load(required["trans_prob"])
    sig_ut = np.load(required["state_signature"])
    covs_pca = np.load(required["covs_pca"])
    
    refit_results = load_json_file(optional["refit_results"]) if optional["refit_results"] else None
    topM = load_json_file(optional["topM_seeds"]) if optional["topM_seeds"] else None
    dwell_tbl = pd.read_csv(optional["dwell_from_A"], sep="\t") if optional["dwell_from_A"] else None
    
    print(pd.DataFrame({
        "n_runs": [qc.get("n_runs")],
        "collapsed_run_count": [qc.get("collapsed_run_count")],
        "collapsed_run_rate": [qc.get("collapsed_run_rate")],
        "seed_identifiability_median_mean_state_corr": [qc.get("seed_identifiability_median_mean_state_corr")],
        "seed_identifiability_min_mean_state_corr": [qc.get("seed_identifiability_min_mean_state_corr")],
        "final_seed": [qc.get("final_seed")],
        "final_seed_fe": [qc.get("final_seed_fe")],
        "final_seed_fo_max": [qc.get("final_seed_fo_max")],
        "final_seed_n_active": [qc.get("final_seed_n_active")],
        "final_seed_neff": [qc.get("final_seed_neff")],
    }))
    print("A shape                :", A.shape)
    print("state_signature shape  :", sig_ut.shape)
    print("covs_pca shape         :", covs_pca.shape)
    
    # Rebuild parcelwise correlation matrices from the saved upper-triangle vectors.
    corr_mats = np.stack([ut_to_square(sig_ut[k], fill_diag=1.0) for k in range(sig_ut.shape[0])], axis=0)
    K, P, _ = corr_mats.shape
    print(f"Recovered {K} state correlation matrices of size {P} x {P}")
    assert P == 200, f"Expected 200 parcels for Schaefer-200, but got {P}."
    
    # =========================
    # Determine final FO vector and dominant/reference state
    # =========================
    
    final_fo = None
    if refit_results:
        best_seed = qc.get("final_seed")
        hits = [r for r in refit_results if int(r.get("seed")) == int(best_seed)]
        if hits:
            final_fo = np.asarray(hits[0]["fo"], dtype=float)
    
    if final_fo is None:
        fo_cols = [c for c in subj.columns if c.lower().startswith("fo_s")]
        if fo_cols:
            final_fo = subj[fo_cols].mean(axis=0).to_numpy(dtype=float)
        else:
            raise RuntimeError("Could not recover final FO vector from refit_results.json or subject_metrics.tsv")
    
    dominant_state_idx = int(np.argmax(final_fo))
    reference_state_idx = (REFERENCE_STATE - 1) if REFERENCE_STATE is not None else dominant_state_idx
    
    print("Final FO vector:", np.round(final_fo, 6))
    print("Dominant state :", f"S{dominant_state_idx+1}")
    print("Reference state:", f"S{reference_state_idx+1}")
    
    # =========================
    # Atlas / parcel label handling for Schaefer2018 200/7
    # =========================
    
    SCHAEFER_NETWORK_MAP = {
        "Vis": "Vis",
        "SomMot": "SomMot",
        "DorsAttn": "DorsAttn",
        "SalVentAttn": "SalVentAttn",
        "Limbic": "Limbic",
        "Cont": "Cont",
        "Default": "Default",
    }
    
    # Some Schaefer distributions use abbreviations like SalVentAttn, others may appear as SalVent or VentAttn.
    NETWORK_ALIASES = {
        "Vis": "Vis",
        "Visual": "Vis",
        "SomMot": "SomMot",
        "Somatomotor": "SomMot",
        "DorsAttn": "DorsAttn",
        "DorsalAttn": "DorsAttn",
        "SalVentAttn": "SalVentAttn",
        "SalVent": "SalVentAttn",
        "VentAttn": "SalVentAttn",
        "Limbic": "Limbic",
        "Cont": "Cont",
        "Control": "Cont",
        "Default": "Default",
        "DMN": "Default",
    }
    
    def auto_find_label_file(result_root):
        candidates = []
        for base in [result_root, result_root.parent, result_root.parent.parent]:
            if base and Path(base).exists():
                for pat in [
                    "*label*.tsv", "*labels*.tsv", "*parcel*.tsv", "*lut*.tsv",
                    "*label*.csv", "*labels*.csv", "*parcel*.csv", "*lut*.csv",
                    "*label*.txt", "*labels*.txt", "*parcel*.txt", "*lut*.txt",
                ]:
                    candidates.extend(Path(base).glob(pat))
                for pat in [
                    "**/*label*.tsv", "**/*labels*.tsv", "**/*parcel*.tsv", "**/*lut*.tsv",
                    "**/*label*.csv", "**/*labels*.csv", "**/*parcel*.csv", "**/*lut*.csv",
                ]:
                    hits = list(Path(base).glob(pat))
                    candidates.extend(hits[:20])
        seen, uniq = set(), []
        for c in candidates:
            s = str(c)
            if s not in seen:
                seen.add(s)
                uniq.append(c)
        return uniq
    
    def read_brainstorm_label_txt(path):
        rows = []
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^(\d+)\s+(.+)$", line)
            if not m:
                continue
            rows.append({"index": int(m.group(1)), "label": m.group(2).strip()})
        if not rows:
            raise ValueError(f"Could not parse Brainstorm label txt: {path}")
        return pd.DataFrame(rows)
    
    def standardize_label_table(df, n_expected):
        out = df.copy()
        out.columns = [str(c).strip().lower() for c in out.columns]
        rename_map = {}
        for c in out.columns:
            if c in {"label", "labels", "parcel", "parcel_name", "name", "region"}:
                rename_map[c] = "label"
            elif c in {"network", "net", "rsn"}:
                rename_map[c] = "network"
            elif c in {"hemi", "hemisphere"}:
                rename_map[c] = "hemi"
            elif c in {"index", "idx", "node", "parcel_id", "label_id", "id"}:
                rename_map[c] = "index"
        out = out.rename(columns=rename_map)
    
        if "label" not in out.columns:
            if out.shape[1] == 1:
                out = out.rename(columns={out.columns[0]: "label"})
            else:
                raise ValueError("Could not find a label column in the label file.")
    
        if "index" in out.columns:
            out = out.sort_values("index").reset_index(drop=True)
        else:
            out["index"] = np.arange(1, len(out) + 1)
    
        # Some TSVs may include a background row 0; keep positive labels and then trim.
        out = out[out["index"] > 0].copy().reset_index(drop=True)
        if len(out) < n_expected:
            raise ValueError(f"Label file has only {len(out)} rows after filtering; expected at least {n_expected}.")
        out = out.iloc[:n_expected].copy().reset_index(drop=True)
        return out
    
    def infer_network_from_schaefer_label(label):
        s = str(label).strip()
        tokens = [t for t in re.split(r"[_\-\s]+", s) if t]
        for tok in tokens:
            if tok in NETWORK_ALIASES:
                return NETWORK_ALIASES[tok]
        joined = "_".join(tokens)
        for k, v in NETWORK_ALIASES.items():
            if k in joined:
                return v
        return "Unknown"
    
    def infer_hemi_from_schaefer_label(label):
        s = str(label).strip()
        tokens = [t for t in re.split(r"[_\-\s]+", s) if t]
        if "LH" in tokens or "Left" in tokens:
            return "L"
        if "RH" in tokens or "Right" in tokens:
            return "R"
        sl = s.lower()
        if "_lh_" in sl or sl.startswith("lh_"):
            return "L"
        if "_rh_" in sl or sl.startswith("rh_"):
            return "R"
        return "?"
    
    def compact_schaefer_label(label):
        s = str(label).strip()
        s = re.sub(r"^7Networks_", "", s)
        s = re.sub(r"^17Networks_", "", s)
        return s
    
    def make_fallback_labels(n):
        labels = [f"parcel_{i+1:03d}" for i in range(n)]
        return pd.DataFrame({
            "index": np.arange(1, n + 1),
            "label": labels,
            "label_short": labels,
            "network": ["Unknown"] * n,
            "hemi": ["?"] * n,
        })
    
    def resolve_label_file():
        explicit = resolve_existing_path(PARCEL_LABELS_FILE)
        if explicit is not None:
            return explicit
        preferred = resolve_existing_path(DEFAULT_SCHAEFER_TSV, DEFAULT_BRAINSTORM_TXT)
        if preferred is not None:
            return preferred
        hits = auto_find_label_file(RESULT_ROOT)
        return hits[0] if hits else None
    
    def load_parcel_labels(n_expected):
        label_path = resolve_label_file()
        if label_path is None:
            print("No Schaefer label file found automatically. Using fallback parcel labels.")
            return make_fallback_labels(n_expected), None
    
        label_path = Path(label_path)
        print("Using parcel label file:", label_path)
        if label_path.suffix.lower() == ".txt":
            df = read_brainstorm_label_txt(label_path)
        elif label_path.suffix.lower() == ".csv":
            df = pd.read_csv(label_path)
        else:
            df = pd.read_csv(label_path, sep=None, engine="python")
    
        out = standardize_label_table(df, n_expected)
        if "network" not in out.columns:
            out["network"] = out["label"].map(infer_network_from_schaefer_label)
        if "hemi" not in out.columns:
            out["hemi"] = out["label"].map(infer_hemi_from_schaefer_label)
        out["label_short"] = out["label"].map(compact_schaefer_label)
        return out[["index", "label", "label_short", "network", "hemi"]].reset_index(drop=True), label_path
    
    def network_sort_list(values):
        values = list(pd.unique(pd.Series(values).astype(str)))
        ordered = [n for n in NETWORK_ORDER if n in values]
        extras = sorted([n for n in values if n not in ordered])
        return ordered + extras
    
    parcel_info, label_path_used = load_parcel_labels(P)
    print(parcel_info.head(12))
    print(parcel_info["network"].value_counts(dropna=False))
    
    network_boundaries = []
    network_tick_pos = []
    network_tick_lab = []
    current = None
    start = 0
    for i, net in enumerate(parcel_info["network"].tolist()):
        if current is None:
            current = net
            start = i
        elif net != current:
            network_boundaries.append(i - 0.5)
            network_tick_pos.append((start + i - 1) / 2)
            network_tick_lab.append(current)
            current = net
            start = i
    network_tick_pos.append((start + len(parcel_info) - 1) / 2)
    network_tick_lab.append(current)
    
    print("Network order in parcel table:", network_tick_lab)
    
    # =========================
    # Sanity check: saved signature <-> reconstructed matrices
    # =========================
    
    max_abs_err = np.max(np.abs(np.stack([square_to_ut(corr_mats[k]) for k in range(K)]) - sig_ut))
    print("Max abs reconstruction error from UT vector -> matrix -> UT:", float(max_abs_err))
    assert max_abs_err < 1e-6 + 1e-7, "Unexpected reconstruction error."
    
    # =========================
    # State-wise parcel correlation heatmaps (with Schaefer network boundaries)
    # =========================
    
    def savefig(fig, name):
        # Avoid mixing tight_layout() with figures created using constrained_layout=True,
        # especially after colorbars have been added.
        try:
            use_constrained = bool(fig.get_constrained_layout())
        except Exception:
            use_constrained = False
    
        if not use_constrained:
            try:
                fig.tight_layout()
            except Exception as e:
                print(f"[savefig] tight_layout skipped: {e}")
    
        fig.savefig(FIG_DIR / name, bbox_inches="tight")
        plt.show()
    
    mask = ~np.eye(P, dtype=bool)
    absmax = float(np.nanmax(np.abs(corr_mats[:, mask])))
    norm = TwoSlopeNorm(vmin=-absmax, vcenter=0.0, vmax=absmax)
    
    fig, axes = plt.subplots(1, K, figsize=(5.8*K, 5.6), constrained_layout=True)
    if K == 1:
        axes = [axes]
    for k, ax in enumerate(axes):
        im = ax.imshow(corr_mats[k], cmap="coolwarm", norm=norm, interpolation="nearest")
        title = f"S{k+1}"
        if k == dominant_state_idx:
            title += " (dominant)"
        if k == reference_state_idx:
            title += " (reference)"
        ax.set_title(title)
        ax.set_xlabel("Schaefer-200 parcel index")
        ax.set_ylabel("Schaefer-200 parcel index")
        for b in network_boundaries:
            ax.axhline(b, color="k", lw=0.6, alpha=0.4)
            ax.axvline(b, color="k", lw=0.6, alpha=0.4)
        if len(network_tick_pos) <= 10:
            ax.set_xticks(network_tick_pos)
            ax.set_xticklabels(network_tick_lab, rotation=45, ha="right")
            ax.set_yticks(network_tick_pos)
            ax.set_yticklabels(network_tick_lab)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, fraction=0.03, pad=0.02)
    cbar.set_label("Correlation (r)")
    savefig(fig, "state_parcel_corr_heatmaps_schaefer.png")
    
    # =========================
    # Parcel-level difference maps versus the reference state
    # =========================
    
    other_states = [k for k in range(K) if k != reference_state_idx]
    if other_states:
        diffs = [corr_mats[k] - corr_mats[reference_state_idx] for k in other_states]
        absmax = float(np.nanmax(np.abs(np.stack(diffs))))
        norm = TwoSlopeNorm(vmin=-absmax, vcenter=0.0, vmax=absmax)
    
        fig, axes = plt.subplots(1, len(other_states), figsize=(6.0*len(other_states), 5.6), constrained_layout=True)
        if len(other_states) == 1:
            axes = [axes]
        for ax, k in zip(axes, other_states):
            im = ax.imshow(corr_mats[k] - corr_mats[reference_state_idx], cmap="coolwarm", norm=norm, interpolation="nearest")
            ax.set_title(f"S{k+1} - S{reference_state_idx+1}")
            ax.set_xlabel("Schaefer-200 parcel index")
            ax.set_ylabel("Schaefer-200 parcel index")
            for b in network_boundaries:
                ax.axhline(b, color="k", lw=0.6, alpha=0.4)
                ax.axvline(b, color="k", lw=0.6, alpha=0.4)
            if len(network_tick_pos) <= 10:
                ax.set_xticks(network_tick_pos)
                ax.set_xticklabels(network_tick_lab, rotation=45, ha="right")
                ax.set_yticks(network_tick_pos)
                ax.set_yticklabels(network_tick_lab)
        cbar = fig.colorbar(im, ax=axes, shrink=0.85, fraction=0.03, pad=0.02)
        cbar.set_label("Δ correlation")
        savefig(fig, "state_parcel_corr_differences_vs_reference_schaefer.png")
    
    # =========================
    # Network-level block summaries (Schaefer-7 ordering)
    # =========================
    
    def block_mean(M, labels_df, use_abs=False, network_order=None):
        vals = np.abs(M) if use_abs else M
        nets_raw = labels_df["network"].astype(str).tolist()
        uniq = network_sort_list(nets_raw) if network_order is None else [n for n in network_order if n in set(nets_raw)] + [n for n in sorted(set(nets_raw)) if n not in (network_order or [])]
        B = np.full((len(uniq), len(uniq)), np.nan, dtype=float)
    
        for i, ni in enumerate(uniq):
            ii = np.where(labels_df["network"].values == ni)[0]
            for j, nj in enumerate(uniq):
                jj = np.where(labels_df["network"].values == nj)[0]
                sub = vals[np.ix_(ii, jj)]
                if i == j:
                    mask = ~np.eye(len(ii), dtype=bool)
                    use = sub[mask] if len(ii) > 1 else np.array([np.nan])
                else:
                    use = sub.reshape(-1)
                B[i, j] = np.nanmean(use)
        return uniq, B
    
    network_blocks = {}
    for k in range(K):
        nets, B = block_mean(corr_mats[k], parcel_info, use_abs=USE_ABS_FOR_BLOCK_MEANS, network_order=NETWORK_ORDER)
        network_blocks[f"S{k+1}"] = pd.DataFrame(B, index=nets, columns=nets)
    
    print(network_blocks["S1"])
    
    # =========================
    # Plot state-wise network block heatmaps
    # =========================
    
    blocks = [network_blocks[f"S{k+1}"].to_numpy() for k in range(K)]
    absmax = float(np.nanmax(np.abs(np.stack(blocks))))
    norm = TwoSlopeNorm(vmin=-absmax, vcenter=0.0, vmax=absmax)
    
    fig, axes = plt.subplots(1, K, figsize=(5.8*K, 5.0), constrained_layout=True)
    if K == 1:
        axes = [axes]
    
    labels_n = network_blocks["S1"].index.tolist()
    for k, ax in enumerate(axes):
        mat = network_blocks[f"S{k+1}"].to_numpy()
        im = ax.imshow(mat, cmap="coolwarm", norm=norm, interpolation="nearest")
        title = f"S{k+1}"
        if k == dominant_state_idx:
            title += " (dominant)"
        ax.set_title(title)
        ax.set_xticks(range(len(labels_n)))
        ax.set_yticks(range(len(labels_n)))
        ax.set_xticklabels(labels_n, rotation=45, ha="right")
        ax.set_yticklabels(labels_n)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, fraction=0.03, pad=0.02)
    cbar.set_label("|r|" if USE_ABS_FOR_BLOCK_MEANS else "mean r")
    savefig(fig, "state_network_block_heatmaps_schaefer.png")
    
    # =========================
    # Plot network block differences versus the reference state
    # =========================
    
    other_states = [k for k in range(K) if k != reference_state_idx]
    if other_states:
        diffs = [network_blocks[f"S{k+1}"].to_numpy() - network_blocks[f"S{reference_state_idx+1}"].to_numpy()
                 for k in other_states]
        absmax = float(np.nanmax(np.abs(np.stack(diffs))))
        norm = TwoSlopeNorm(vmin=-absmax, vcenter=0.0, vmax=absmax)
    
        fig, axes = plt.subplots(1, len(other_states), figsize=(6.0*len(other_states), 5.0), constrained_layout=True)
        if len(other_states) == 1:
            axes = [axes]
    
        for ax, k in zip(axes, other_states):
            diff = network_blocks[f"S{k+1}"].to_numpy() - network_blocks[f"S{reference_state_idx+1}"].to_numpy()
            im = ax.imshow(diff, cmap="coolwarm", norm=norm, interpolation="nearest")
            ax.set_title(f"S{k+1} - S{reference_state_idx+1}")
            ax.set_xticks(range(len(labels_n)))
            ax.set_yticks(range(len(labels_n)))
            ax.set_xticklabels(labels_n, rotation=45, ha="right")
            ax.set_yticklabels(labels_n)
        cbar = fig.colorbar(im, ax=axes, shrink=0.85, fraction=0.03, pad=0.02)
        cbar.set_label("Δ " + ("|r|" if USE_ABS_FOR_BLOCK_MEANS else "mean r"))
        savefig(fig, "state_network_block_differences_vs_reference_schaefer.png")
    
    # =========================
    # Rank the biggest network-pair contrasts versus the reference state
    # =========================
    
    def network_contrast_table(state_idx, ref_idx):
        A_ = network_blocks[f"S{state_idx+1}"]
        R_ = network_blocks[f"S{ref_idx+1}"]
        D = A_ - R_
        rows = []
        for i in A_.index:
            for j in A_.columns:
                rows.append({
                    "state": f"S{state_idx+1}",
                    "reference": f"S{ref_idx+1}",
                    "pair": f"{i} -- {j}",
                    "delta": float(D.loc[i, j]),
                    "state_value": float(A_.loc[i, j]),
                    "reference_value": float(R_.loc[i, j]),
                })
        out = pd.DataFrame(rows)
        out["abs_delta"] = out["delta"].abs()
        return out.sort_values(["abs_delta", "pair"], ascending=[False, True]).reset_index(drop=True)
    
    network_contrasts = {}
    for k in range(K):
        if k == reference_state_idx:
            continue
        tbl = network_contrast_table(k, reference_state_idx)
        network_contrasts[f"S{k+1}_vs_S{reference_state_idx+1}"] = tbl
        print("\nTop network contrasts for", f"S{k+1} vs S{reference_state_idx+1}")
        print(tbl.head(15))
    
    for name, tbl in network_contrasts.items():
        tbl.to_csv(FIG_DIR / f"{name}_top_network_contrasts.tsv", sep="\t", index=False)
    
    # =========================
    # Nodal summaries: mean connectivity strength per parcel
    # =========================
    
    def offdiag_row_mean(M):
        M = np.asarray(M, float)
        n = M.shape[0]
        out = (M.sum(axis=1) - np.diag(M)) / np.maximum(n - 1, 1)
        return out
    
    nodal = parcel_info.copy()
    for k in range(K):
        nodal[f"S{k+1}_mean_r"] = offdiag_row_mean(corr_mats[k])
    
    for k in range(K):
        print("\nTop parcels for", f"S{k+1}")
        print(
            nodal[["index", "label_short", "network", "hemi", f"S{k+1}_mean_r"]]
            .sort_values(f"S{k+1}_mean_r", ascending=False)
            .head(15)
        )
    
    nodal.to_csv(FIG_DIR / "nodal_mean_connectivity_by_state.tsv", sep="\t", index=False)
    
    # =========================
    # Parcel-wise contrasts versus the reference state
    # =========================
    
    parcel_contrast_tables = {}
    for k in range(K):
        if k == reference_state_idx:
            continue
        tbl = parcel_info.copy()
        tbl["delta_mean_r"] = offdiag_row_mean(corr_mats[k] - corr_mats[reference_state_idx])
        tbl["abs_delta_mean_r"] = tbl["delta_mean_r"].abs()
        tbl = tbl.sort_values(["abs_delta_mean_r", "label"], ascending=[False, True]).reset_index(drop=True)
        parcel_contrast_tables[f"S{k+1}_vs_S{reference_state_idx+1}"] = tbl
        print("\nTop parcel contrasts for", f"S{k+1} vs S{reference_state_idx+1}")
        print(tbl[["index", "label_short", "network", "hemi", "delta_mean_r", "abs_delta_mean_r"]].head(20))
        tbl.to_csv(FIG_DIR / f"S{k+1}_vs_S{reference_state_idx+1}_top_parcel_contrasts.tsv", sep="\t", index=False)
    
    # =========================
    # Optional: covariance backprojection from covs_pca.npy using preproc_params.npz
    # =========================
    
    def ensure_cov_3d(covs):
        covs = np.asarray(covs)
        if covs.ndim == 2:
            covs = covs[None, ...]
        return covs
    
    def backproject_cov_bold(covs_pca, Vb, nbpc):
        covs_pca = ensure_cov_3d(covs_pca)
        out = []
        for k in range(covs_pca.shape[0]):
            Cbb = covs_pca[k, :nbpc, :nbpc]
            out.append((Vb @ Cbb @ Vb.T).astype(np.float32))
        return np.stack(out, axis=0)
    
    cov_bold = None
    if optional["preproc_params"] and Path(optional["preproc_params"]).exists():
        pp = np.load(optional["preproc_params"])
        Vb = pp["Vb"]
        nbpc = Vb.shape[1]
        cov_bold = backproject_cov_bold(covs_pca, Vb, nbpc)
        print("Backprojected BOLD covariance shape:", cov_bold.shape)
    
        cov_corr = np.stack([corr_from_cov(C) for C in cov_bold], axis=0)
        err = np.max(np.abs(cov_corr - corr_mats))
        print("Max abs error between backprojected corr and saved corr:", float(err))
    else:
        print("preproc_params.npz not found. Skipping covariance backprojection.")
    
    # =========================
    # Summary tables for manuscript writing
    # =========================
    
    summary_rows = []
    for k in range(K):
        state = f"S{k+1}"
        row = {
            "state": state,
            "is_dominant": int(k == dominant_state_idx),
            "is_reference": int(k == reference_state_idx),
            "final_FO": float(final_fo[k]) if k < len(final_fo) else np.nan,
            "self_transition_Akk": float(A[k, k]),
            "expected_dwell_TR": float(1.0 / max(1e-12, 1.0 - A[k, k])),
        }
        nb = network_blocks[state].copy()
        pairs = []
        for i in nb.index:
            for j in nb.columns:
                pairs.append((i, j, float(nb.loc[i, j])))
        pairs = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)
        for rank, (i, j, v) in enumerate(pairs[:5], start=1):
            row[f"top_block_{rank}"] = f"{i} -- {j}"
            row[f"top_block_{rank}_value"] = v
        summary_rows.append(row)
    
    state_summary = pd.DataFrame(summary_rows)
    print(state_summary)
    state_summary.to_csv(FIG_DIR / "state_summary_table.tsv", sep="\t", index=False)
    
    # =========================
    # Plain-language interpretation scaffold
    # =========================
    
    def top_pair_text(tbl, n=5):
        bits = []
        for _, r in tbl.head(n).iterrows():
            bits.append(f"{r['pair']} ({r['delta']:+.3f})")
        return "; ".join(bits)
    
    print("Suggested interpretation scaffold:\n")
    for k in range(K):
        state = f"S{k+1}"
        if k == reference_state_idx:
            print(f"{state}: reference state. Final FO={final_fo[k]:.3f}, A_kk={A[k,k]:.3f}, expected dwell={1/(1-A[k,k]):.2f} TR.")
        else:
            tbl = network_contrasts[f"S{k+1}_vs_S{reference_state_idx+1}"]
            print(
                f"{state} versus S{reference_state_idx+1}: "
                f"Final FO={final_fo[k]:.3f}; A_kk={A[k,k]:.3f}; "
                f"largest network shifts include {top_pair_text(tbl, n=5)}."
            )
    
    # =========================
    # Save a compact session manifest
    # =========================
    
    manifest = {
        "RESULT_ROOT": str(RESULT_ROOT),
        "FINAL_DIR": str(FINAL_DIR),
        "FIG_DIR": str(FIG_DIR),
        "TEMPLATEFLOW_HOME": str(TEMPLATEFLOW_HOME),
        "DEFAULT_SCHAEFER_TSV": str(DEFAULT_SCHAEFER_TSV),
        "DEFAULT_BRAINSTORM_TXT": str(DEFAULT_BRAINSTORM_TXT),
        "label_file_used": str(label_path_used) if label_path_used is not None else None,
        "PREPROC_PARAMS_FILE": str(PREPROC_PARAMS_FILE) if PREPROC_PARAMS_FILE is not None else str(optional["preproc_params"]) if optional["preproc_params"] else None,
        "P": int(P),
        "K": int(K),
        "dominant_state": int(dominant_state_idx + 1),
        "reference_state": int(reference_state_idx + 1),
        "network_order": NETWORK_ORDER,
    }
    (Path(FIG_DIR) / "physiology_notebook_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Saved outputs to:", FIG_DIR)
    
    return {
        "result_root": str(RESULT_ROOT),
        "output_dir": str(FIG_DIR),
        "reference_state": int(reference_state_idx + 1),
        "dominant_state": int(dominant_state_idx + 1),
        "state_summary_table_tsv": str(FIG_DIR / "state_summary_table.tsv"),
    }
