"""Shared Schaefer and cross-modal utilities for the public Stage-6 backends.

These helpers are used by the Stage-6 cross-modal reconstruction and optional
panel-export steps. They keep repeated atlas parsing and covariance
backprojection logic out of the public-facing backend orchestration.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from stage6_backend_common import resolve_existing_path


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
    "DMN": "Default",
}


def ensure_cov_3d(covs: np.ndarray) -> np.ndarray:
    """Ensure covariance arrays are shaped as `(K, D, D)`."""
    covs = np.asarray(covs)
    if covs.ndim == 2:
        covs = covs[None, ...]
    return covs


def parse_schaefer_name(name: str) -> tuple[str | None, str | None]:
    """Extract hemisphere and 7-network label from a Schaefer parcel name."""
    name = str(name)
    hemi = None
    hemi_match = re.search(r"_(LH|RH)_", name)
    if hemi_match:
        hemi = "L" if hemi_match.group(1) == "LH" else "R"

    network = None
    for token in re.split(r"[_\-]", name):
        if token in NETWORK_ALIASES:
            network = NETWORK_ALIASES[token]
            break
    if network is None:
        match = re.search(r"7Networks_(LH|RH)_([^_]+)", name)
        if match:
            network = NETWORK_ALIASES.get(match.group(2), match.group(2))
    return hemi, network


def load_schaefer_table(
    label_file: str | Path | None,
    default_tsv: str | Path | None,
    default_brainstorm_txt: str | Path | None,
) -> pd.DataFrame:
    """Load Schaefer parcel labels from a TSV or Brainstorm-style TXT export."""
    label_path = resolve_existing_path(label_file, default_tsv, default_brainstorm_txt)
    if label_path is None:
        raise FileNotFoundError("Could not find a Schaefer label table (.tsv or .txt).")

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
        ids = pd.to_numeric(raw["atlas_id"], errors="coerce").fillna(
            pd.Series(np.arange(1, len(raw) + 1), index=raw.index)
        ).astype(int).to_numpy()
        names = raw["name"].astype(str).to_numpy()

    out = pd.DataFrame({"atlas_id": ids, "label": names})
    out = out[out["atlas_id"] > 0].copy().iloc[:200].copy()
    parsed = out["label"].apply(parse_schaefer_name)
    out["hemi"] = [p[0] for p in parsed]
    out["network"] = [p[1] for p in parsed]
    out["parcel_idx_1based"] = np.arange(1, len(out) + 1)
    out["parcel_idx_0based"] = np.arange(len(out))
    return out


def reorder_by_network(labels_df: pd.DataFrame, network_order: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Reorder parcels by network, hemisphere, and label for stable plotting."""
    sort_key = pd.Categorical(labels_df["network"], categories=network_order, ordered=True)
    ordered = (
        labels_df.assign(_network_order=sort_key)
        .sort_values(["_network_order", "hemi", "label"])
        .reset_index(drop=True)
    )
    idx = ordered["parcel_idx_0based"].to_numpy()
    return ordered.drop(columns="_network_order"), idx


def aggregate_block_matrix(
    matrix: np.ndarray,
    labels_row: np.ndarray,
    labels_col: np.ndarray | None = None,
    order_row: list[str] | None = None,
    order_col: list[str] | None = None,
) -> pd.DataFrame:
    """Average a parcel-by-parcel matrix into a network block matrix."""
    matrix = np.asarray(matrix, dtype=float)
    labels_row = np.asarray(labels_row)
    labels_col = labels_row if labels_col is None else np.asarray(labels_col)
    order_row = order_row or list(pd.unique(labels_row))
    order_col = order_col or list(pd.unique(labels_col))

    out = np.full((len(order_row), len(order_col)), np.nan, dtype=float)
    for i, row_network in enumerate(order_row):
        idx_r = np.where(labels_row == row_network)[0]
        for j, col_network in enumerate(order_col):
            idx_c = np.where(labels_col == col_network)[0]
            if len(idx_r) == 0 or len(idx_c) == 0:
                continue
            out[i, j] = np.nanmean(matrix[np.ix_(idx_r, idx_c)])
    return pd.DataFrame(out, index=order_row, columns=order_col)


def rank_block_contrasts(
    diff_df: pd.DataFrame,
    top_n: int = 10,
    row_label: str = "row_network",
    col_label: str = "col_network",
) -> pd.DataFrame:
    """Rank the strongest positive or negative network-block contrasts."""
    rows = []
    for row_name in diff_df.index:
        for col_name in diff_df.columns:
            rows.append((row_name, col_name, float(diff_df.loc[row_name, col_name])))
    out = pd.DataFrame(rows, columns=[row_label, col_label, "delta_value"])
    out["abs_delta"] = out["delta_value"].abs()
    return out.sort_values(["abs_delta", "delta_value"], ascending=[False, False]).head(top_n).reset_index(drop=True)


def backproject_modal_blocks(covs_pca: np.ndarray, Vb: np.ndarray, Ve: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backproject PCA-space state covariances into parcel-space BOLD, EEG, and cross-modal blocks."""
    covs_pca = ensure_cov_3d(covs_pca)
    out_bb, out_ee, out_be = [], [], []
    for k in range(covs_pca.shape[0]):
        cbb_p = covs_pca[k, :Vb.shape[1], :Vb.shape[1]]
        cee_p = covs_pca[k, Vb.shape[1]:Vb.shape[1] + Ve.shape[1], Vb.shape[1]:Vb.shape[1] + Ve.shape[1]]
        cbe_p = covs_pca[k, :Vb.shape[1], Vb.shape[1]:Vb.shape[1] + Ve.shape[1]]

        out_bb.append((Vb @ cbb_p @ Vb.T).astype(np.float32))
        out_ee.append((Ve @ cee_p @ Ve.T).astype(np.float32))
        out_be.append((Vb @ cbe_p @ Ve.T).astype(np.float32))

    return np.stack(out_bb), np.stack(out_ee), np.stack(out_be)


def cov_to_crosscorr(Cbb: np.ndarray, Cee: np.ndarray, Cbe: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Convert a cross-covariance block into parcelwise cross-correlations."""
    vb = np.sqrt(np.clip(np.diag(Cbb), eps, None))
    ve = np.sqrt(np.clip(np.diag(Cee), eps, None))
    return Cbe / np.outer(vb, ve)


def zscore_keep(x: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Z-score a 1D timeseries using only the kept time points."""
    x = np.asarray(x, dtype=float)
    keep = np.asarray(keep).astype(bool)
    out = np.full_like(x, np.nan, dtype=float)
    if keep.sum() < 2:
        return out
    mu = np.nanmean(x[keep])
    sd = np.nanstd(x[keep], ddof=0)
    out[keep] = x[keep] - mu if (not np.isfinite(sd) or sd == 0) else (x[keep] - mu) / sd
    return out
