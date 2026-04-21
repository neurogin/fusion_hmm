"""Helper functions for the cleaned Stage-5 HMM model-selection workflow.

This module supports the public Stage-5 notebooks that:
- run the broad LOSO K sweep
- run the manuscript-facing shortlist stability comparison
- rebuild Figure 2 and Table S8 support outputs

Main inputs:
- the canonical Stage-4 `intermediate + nolags + minlen15` segment outputs
- saved Stage-5 screening and shortlist outputs

Main outputs:
- K-sweep tables and plots
- shortlist-stability outputs
- model-selection summary tables and figures

Important note:
- the public notebooks keep the user-facing setup and interpretation visible
- the dense TensorFlow and `osl_dynamics` machinery now lives in normal
  same-directory Python backend modules rather than preserved provenance
  notebooks
- Step 50 focuses on broad screening, Step 51 on the manuscript-facing
  shortlist comparison, and Step 52 on the compact Figure 2 / Table S8 build
- the final choice of `K = 3` remains a documented scientific decision, not
  a single automatic rule
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from stage5_k_sweep_backend import KSweepBackendConfig, run_loso_k_sweep_backend
from stage5_shortlist_backend import ShortlistBackendConfig, run_loso_shortlist_backend


def _maybe_read_table(path: str | Path, sep: str = "\t") -> pd.DataFrame | None:
    """Read a table if it exists, otherwise return None."""
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path, sep=sep)


def _maybe_read_json(path: str | Path) -> dict[str, Any] | None:
    """Read a JSON file if it exists, otherwise return None."""
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_public_k_sweep_step(
    *,
    segments_root: str | Path,
    model_selection_root: str | Path,
    data_variant: str = "intermediate",
    feature_mode: str = "nolags",
    minlen: int = 15,
    manifest_tsv: str | Path | None = None,
    k_grid: list[int] | None = None,
    max_new_pairs_per_run: int = 15,
    gpu_memory_limit_mb: int | None = 4096,
    debug_max_folds: int | None = None,
    debug_k_grid: list[int] | None = None,
    debug_seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Run the public Step-50 screening workflow using the active Python backend."""
    backend_result = run_loso_k_sweep_backend(
        KSweepBackendConfig(
            segments_root=segments_root,
            model_selection_root=model_selection_root,
            data_variant=data_variant,
            feature_mode=feature_mode,
            minlen=minlen,
            manifest_tsv=manifest_tsv,
            k_grid=k_grid,
            max_new_pairs_per_run=max_new_pairs_per_run,
            gpu_memory_limit_mb=gpu_memory_limit_mb,
            debug_max_folds=debug_max_folds,
            debug_k_grid=debug_k_grid,
            debug_seeds=debug_seeds,
        )
    )

    final_root = Path(segments_root) / data_variant
    out_root = Path(model_selection_root) / f"{data_variant}_{feature_mode}_minlen{minlen}"
    manifest_value = backend_result.get("manifest_tsv", "")
    return {
        "backend_module": "stage5_k_sweep_backend.py",
        "backend_status": backend_result.get("status", ""),
        "backend_message": backend_result.get("chunk_message", ""),
        "final_root": str(final_root),
        "out_root": str(out_root),
        "manifest_tsv": str(manifest_value) if manifest_value is not None else "",
        "available_outputs": {
            "cv_results_tsv": str(out_root / "cv_results.tsv"),
            "cv_candidates_long_tsv": str(out_root / "cv_candidates_long.tsv"),
            "summary_byK_selected_tsv": str(out_root / "summary_byK_selected.tsv"),
            "summary_byK_candidates_tsv": str(out_root / "summary_byK_candidates.tsv"),
            "paired_tests_vs_bestK_tsv": str(out_root / "paired_tests_vs_bestK.tsv"),
            "k_selection_recommendation_json": str(out_root / "K_selection_recommendation.json"),
        },
        "summary_byk_selected_df": _maybe_read_table(out_root / "summary_byK_selected.tsv"),
        "summary_byk_candidates_df": _maybe_read_table(out_root / "summary_byK_candidates.tsv"),
        "paired_tests_df": _maybe_read_table(out_root / "paired_tests_vs_bestK.tsv"),
        "recommendation": _maybe_read_json(out_root / "K_selection_recommendation.json"),
    }


def run_public_shortlist_step(
    *,
    segments_root: str | Path,
    shortlist_output_root: str | Path,
    k_list: list[int],
    data_variant: str = "intermediate",
    feature_mode: str = "nolags",
    minlen: int = 15,
    manifest_tsv: str | Path | None = None,
    max_new_folds_per_run: int = 1,
    gpu_memory_limit_mb: int | None = None,
    force_rerun_heldouts: list[str] | None = None,
    debug_subjects: list[str] | None = None,
    debug_seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Run the public Step-51 shortlist workflow using the active Python backend."""
    backend_result = run_loso_shortlist_backend(
        ShortlistBackendConfig(
            segments_root=segments_root,
            shortlist_output_root=shortlist_output_root,
            k_list=k_list,
            data_variant=data_variant,
            feature_mode=feature_mode,
            minlen=minlen,
            manifest_tsv=manifest_tsv,
            max_new_folds_per_run=max_new_folds_per_run,
            gpu_memory_limit_mb=gpu_memory_limit_mb,
            force_rerun_heldouts=force_rerun_heldouts or [],
            debug_subjects=debug_subjects,
            debug_seeds=debug_seeds,
        )
    )

    final_root = Path(segments_root) / data_variant
    out_root = Path(shortlist_output_root) / f"PipelineD_C2_{data_variant}_{feature_mode}_minlen{minlen}"
    manifest_value = backend_result.get("manifest_tsv", "")
    per_k_outputs: dict[int, dict[str, Any]] = {}
    for K in k_list:
        k_dir = out_root / f"K{int(K):02d}"
        per_k_outputs[int(K)] = {
            "k_dir": str(k_dir),
            "state_matching_scores_df": _maybe_read_table(k_dir / "state_matching_scores.tsv"),
            "fold_summaries_table_matched_df": _maybe_read_table(k_dir / "fold_summaries_table_matched.tsv"),
            "invalid_folds_df": _maybe_read_table(k_dir / "invalid_folds.tsv"),
            "reference_fold_txt": str(k_dir / "reference_fold.txt"),
        }

    return {
        "backend_module": "stage5_shortlist_backend.py",
        "backend_status": backend_result.get("status", ""),
        "backend_message": backend_result.get("chunk_message", ""),
        "final_root": str(final_root),
        "out_root": str(out_root),
        "manifest_tsv": str(manifest_value) if manifest_value is not None else "",
        "per_k_outputs": per_k_outputs,
    }


def _require_file(path: str | Path, label: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _load_per_k_tables(stability_output_dir: str | Path, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    stability_output_dir = Path(stability_output_dir)
    k_dir = stability_output_dir / f"K{k:02d}"
    score_path = _require_file(k_dir / "state_matching_scores.tsv", f"K={k} state-matching table")
    fold_path = _require_file(k_dir / "fold_summaries_table_matched.tsv", f"K={k} matched fold summary")
    return pd.read_csv(score_path, sep="\t"), pd.read_csv(fold_path, sep="\t")


def _state_corr_summary(score_df: pd.DataFrame) -> dict[str, float]:
    x = score_df["mean_state_corr"].dropna().astype(float)
    return {
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)),
        "sem": float(x.std(ddof=1) / np.sqrt(len(x))),
        "median": float(x.median()),
        "n": int(len(x)),
    }


def _matched_fo_medians(fold_df: pd.DataFrame) -> pd.Series:
    fo_cols = [c for c in fold_df.columns if c.startswith("FO_s")]
    return pd.Series({c: float(fold_df[c].median()) for c in fo_cols})


def build_figure2_and_table_s8_summary(
    *,
    ksweep_output_dir: str | Path,
    stability_output_dir: str | Path,
    summary_output_dir: str | Path,
    compare_ks: tuple[int, int] = (3, 5),
    decision_table_ks: tuple[int, ...] = (3, 5, 12),
    annotate_ks: tuple[int, ...] = (3, 5, 9, 12),
    figure_name: str = "fusion_hmm_model_selection_summary.png",
    table_name: str = "fusion_hmm_K_selection_compact_table.csv",
    figsize: tuple[float, float] = (12, 10),
    dpi: int = 200,
) -> dict[str, str]:
    """Build the manuscript-facing Figure 2 support plot and compact Table S8 CSV."""
    ksweep_output_dir = Path(ksweep_output_dir)
    stability_output_dir = Path(stability_output_dir)
    summary_output_dir = Path(summary_output_dir)
    summary_output_dir.mkdir(parents=True, exist_ok=True)

    summary_byk_path = _require_file(ksweep_output_dir / "summary_byK_selected.tsv", "K-sweep summary")
    k_selection_path = _require_file(ksweep_output_dir / "K_selection_recommendation.json", "K-selection JSON")
    paired_tests_path = ksweep_output_dir / "paired_tests_vs_bestK.tsv"

    sel = pd.read_csv(summary_byk_path, sep="\t")
    paired = pd.read_csv(paired_tests_path, sep="\t") if paired_tests_path.exists() else None
    krec = json.loads(k_selection_path.read_text(encoding="utf-8"))

    compare_a, compare_b = compare_ks
    a_scores, a_fold = _load_per_k_tables(stability_output_dir, compare_a)
    b_scores, b_fold = _load_per_k_tables(stability_output_dir, compare_b)

    best_k = int(krec["K_best"])
    one_se_threshold = float(krec["oneSE_threshold"])
    one_se_k = int(krec["K_1se"])

    def summarize_k(k: int, fold_df: pd.DataFrame | None = None, score_df: pd.DataFrame | None = None) -> dict[str, float | int | bool]:
        row = sel.loc[sel["K"] == int(k)].iloc[0]
        out: dict[str, float | int | bool] = {
            "K": int(k),
            "mean_test_FE": float(row["fe_test_mean"]),
            "SEM_test_FE": float(row["fe_test_sem"]),
            "feasible_frac": float(row["feasible_frac"]),
            "delta_vs_bestK": float(row["fe_test_mean"] - sel.loc[sel["K"] == best_k, "fe_test_mean"].iloc[0]),
            "within_1SE_rule": bool(row["fe_test_mean"] <= one_se_threshold),
            "sweep_FOmax_median": float(row["fo_max_median"]),
            "sweep_n_active_median": float(row["n_active_median"]),
            "sweep_neff_median": float(row["neff_median"]),
        }
        if (fold_df is not None) and (score_df is not None):
            sc = _state_corr_summary(score_df)
            out.update(
                {
                    "stability_mean_state_corr": sc["mean"],
                    "stability_sem_state_corr": sc["sem"],
                    "stability_median_state_corr": sc["median"],
                    "matched_FOmax_median": float(fold_df["FO_max"].median()),
                    "matched_n_active_median": float(fold_df["n_active"].median()),
                    "matched_neff_median": float(fold_df["neff"].median()),
                }
            )
        return out

    loaded_per_k = {
        compare_a: {"scores": a_scores, "fold": a_fold},
        compare_b: {"scores": b_scores, "fold": b_fold},
    }
    decision_rows = []
    for k in decision_table_ks:
        payload = loaded_per_k.get(int(k))
        if payload is None:
            decision_rows.append(summarize_k(int(k), None, None))
        else:
            decision_rows.append(summarize_k(int(k), payload["fold"], payload["scores"]))
    decision = pd.DataFrame(decision_rows)

    out_table_path = summary_output_dir / table_name
    decision.to_csv(out_table_path, index=False)

    a_sc = _state_corr_summary(a_scores)
    b_sc = _state_corr_summary(b_scores)
    a_fo_medians = _matched_fo_medians(a_fold)
    b_fo_medians = _matched_fo_medians(b_fold)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.0], width_ratios=[1.2, 1.0])

    ax1 = fig.add_subplot(gs[0, :])
    ax1.errorbar(
        sel["K"],
        sel["fe_test_mean"],
        yerr=sel["fe_test_sem"],
        marker="o",
        linewidth=2,
        capsize=4,
    )
    ax1.axhline(one_se_threshold, linestyle="--", linewidth=2, label=f"1-SE threshold ({one_se_threshold:.2f})")
    ax1.axvline(one_se_k, linestyle=":", linewidth=2, label=f"Smallest within 1-SE: K={one_se_k}")
    for k in annotate_ks:
        if int(k) in sel["K"].values:
            y = sel.loc[sel["K"] == int(k), "fe_test_mean"].iloc[0]
            ax1.annotate(f"K={int(k)}", (int(k), y), textcoords="offset points", xytext=(0, 8), ha="center")
    ax1.set_title("A. LOSO-CV K-sweep: mean test free energy ± SEM")
    ax1.set_xlabel("K")
    ax1.set_ylabel("Mean test free energy")
    ax1.legend(frameon=False)
    ax1.grid(False)

    ax2 = fig.add_subplot(gs[1, 0])
    bars_x = np.array([0, 1])
    ax2.bar(bars_x, [a_sc["mean"], b_sc["mean"]], yerr=[a_sc["sem"], b_sc["sem"]], capsize=4)
    ax2.set_xticks(bars_x, [f"K={compare_a}", f"K={compare_b}"])
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel("Mean matched state-signature correlation")
    ax2.set_title(f"B. Stability comparison: K={compare_a} vs K={compare_b}")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(np.arange(len(a_fo_medians)) + 1, a_fo_medians.values, marker="o", linewidth=2, label=f"K={compare_a}")
    ax3.plot(np.arange(len(b_fo_medians)) + 1, b_fo_medians.values, marker="o", linewidth=2, label=f"K={compare_b}")
    ax3.set_xlabel("Matched state index")
    ax3.set_ylabel("Median FO across folds")
    ax3.set_title("C. Matched FO comparison across shortlisted K values")
    ax3.legend(frameon=False)

    fig.suptitle("Fusion HMM model-order selection summary", fontsize=18, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_fig_path = summary_output_dir / figure_name
    fig.savefig(out_fig_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    inputs_used = {
        "summary_byk_tsv": str(summary_byk_path),
        "paired_tests_tsv": str(paired_tests_path) if paired is not None else "",
        "k_selection_json": str(k_selection_path),
        "compare_ks": [int(compare_a), int(compare_b)],
        "compare_a_state_matching": str(stability_output_dir / f"K{compare_a:02d}" / "state_matching_scores.tsv"),
        "compare_a_fold_summary": str(stability_output_dir / f"K{compare_a:02d}" / "fold_summaries_table_matched.tsv"),
        "compare_b_state_matching": str(stability_output_dir / f"K{compare_b:02d}" / "state_matching_scores.tsv"),
        "compare_b_fold_summary": str(stability_output_dir / f"K{compare_b:02d}" / "fold_summaries_table_matched.tsv"),
        "local_minima": krec.get("local_minima", []),
        "shortlist_primary": krec.get("shortlist_primary", []),
        "shortlist_optional": krec.get("shortlist_optional", []),
    }
    (summary_output_dir / "model_selection_summary_inputs.json").write_text(json.dumps(inputs_used, indent=2), encoding="utf-8")

    return {
        "figure_path": str(out_fig_path),
        "table_path": str(out_table_path),
        "inputs_manifest": str(summary_output_dir / "model_selection_summary_inputs.json"),
    }
