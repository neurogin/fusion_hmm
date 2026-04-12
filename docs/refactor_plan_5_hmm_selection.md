# Refactor Plan: Stage 5 HMM Selection

## Scope

This memo covers `notebooks/5_hmm_selection/` only, with two misplaced provenance notebooks in `notebooks/4_alignment/` inspected as context:

- `R01_PipelineC_fusion_hmm_LOSO_K2-12_v3_UNIFIED_TOGGLE_LAGS_GPU_PATCH_Data2Dataset.ipynb`
- `R01_PipelineC_fusion_hmm_LOSO_K2-12_v3_UNIFIED_TOGGLE_LAGS_GPU_PATCH_Data2Dataset-extend.ipynb`

The goal of this pass is inventory and refactor planning only. No files are rewritten here.

This stage corresponds primarily to manuscript model-order selection rather than the final chosen-model fit:

- Main Methods 2.5.1
- Supplementary Methods 1.6-1.7

## Files Inspected

Active Stage-5 files:

- `R01_PipelineC2_Ksweep_LOSO_OOMFIX_steps_XLAoff_chunkresume_v2REBATCH.ipynb`
- `R01_PipelineD_C2_LOSO_stability_shortlist_REWRITE_v3_twostage_configfix.ipynb`
- `R01_PipelineD_C2_LOSO_stability_shortlist.ipynb`
- `R01_PipelineC_LOSO_param_stability_K6-8_intermediate_nolags_minlen15_SEQ10_summaries_ES_LR_GPU4GB_v3DATASETFIX.ipynb`
- `recreate_fusion_hmm_model_selection_summary.ipynb`

Relevant provenance/archive context:

- `notebooks/_archive_raw_original_names/6_loso cv/`
- archived summary CSVs and PNGs for model-selection outputs
- archived markdown/export snapshots of older Pipeline C and D notebooks

## High-Level Assessment

Stage 5 is mostly coherent as a manuscript stage. The active notebooks cluster into three real functions:

1. LOSO K-sweep model-order selection
2. shortlist stability and state-matching checks
3. manuscript-facing summary figure/table reconstruction

There is no strong evidence that this folder is meant to run the final full-data K=3 model. That belongs to the next stage.

The main problems are not stage confusion so much as:

- multiple generations of the shortlist-stability notebook
- one targeted K=6/7/8 stability notebook that looks exploratory or sensitivity-oriented
- hard-coded local paths and GPU/TensorFlow assumptions
- a real manuscript/code tension around which shortlisted K values were emphasized before the final K=3 choice

## File-by-File Assessment

### `R01_PipelineC2_Ksweep_LOSO_OOMFIX_steps_XLAoff_chunkresume_v2REBATCH.ipynb`

- Scientific purpose:
  Main LOSO K-sweep notebook. Runs held-out free-energy evaluation across `K=2:12`, tracks feasibility/collapse across seeds, writes recommendation outputs including 1-SE and local-minima summaries.
- Likely manuscript linkage:
  Main Methods 2.5.1 and Supplementary Methods 1.6.
- Inputs:
  Stage-4 final segment manifest under `FINAL_v3_gnorm_allTR/<variant>/hmm_segments_minlen15_nolags/segments_manifest.tsv` by default, with `FEATURE_MODE` and `MINLEN` toggles still present.
- Outputs:
  `cv_results.tsv`, `cv_candidates_long.tsv`, `fold_meta.tsv`, `summary_byK_selected.tsv`, `summary_byK_candidates.tsv`, `paired_tests_vs_bestK.tsv`, `K_selection_recommendation.json`, and FE/QC plots.
- Classification:
  Core manuscript-relevant logic.
- Overlap:
  Overlaps with the older misplaced Stage-4 `PipelineC` notebooks, but is more mature and more clearly the active path.
- Recommendation:
  Keep and rewrite as the main public Stage-5 entry notebook.

### `R01_PipelineD_C2_LOSO_stability_shortlist_REWRITE_v3_twostage_configfix.ipynb`

- Scientific purpose:
  Shortlist stability notebook. Re-fits shortlisted K values across LOSO folds, performs state matching, and writes fold-to-fold similarity and transition-stability outputs.
- Likely manuscript linkage:
  Main Methods 2.5.1 and Supplementary Methods 1.7.
- Inputs:
  Same Stage-4 final segment manifest family as the K-sweep notebook.
- Outputs:
  `run_meta.json`, `state_matching_scores.tsv`, `sim_matrix_signature.tsv`, `sim_matrix_A.tsv`, `A_mean.tsv`, `A_std.tsv`, `fold_summaries_table_matched.tsv`, seed logs, and QC plots.
- Classification:
  Core manuscript-relevant logic.
- Overlap:
  Strong overlap with the older `R01_PipelineD_C2_LOSO_stability_shortlist.ipynb`.
- Recommendation:
  Keep as the main public shortlist-stability notebook, but Pass 2 must verify the intended active `K_LIST` because the current file has `K_LIST = [5]` while its own comment shows `[3, 5, 9, 12]`.

### `R01_PipelineD_C2_LOSO_stability_shortlist.ipynb`

- Scientific purpose:
  Older shortlist-stability notebook with similar outputs and fewer later fixes.
- Likely manuscript linkage:
  Same general linkage as the rewrite notebook, but likely superseded.
- Inputs:
  Same Stage-4 manifest family.
- Outputs:
  Same core stability products as the rewrite notebook.
- Classification:
  Provenance/superseded notebook.
- Overlap:
  Largely duplicate of the rewrite notebook.
- Recommendation:
  Archive/provenance only. Use only as a donor for intent and historical settings.

### `R01_PipelineC_LOSO_param_stability_K6-8_intermediate_nolags_minlen15_SEQ10_summaries_ES_LR_GPU4GB_v3DATASETFIX.ipynb`

- Scientific purpose:
  Targeted parameter-stability notebook restricted to `K = [6, 7, 8]`.
- Likely manuscript linkage:
  At most supplementary sensitivity/provenance context, not clearly the main manuscript path.
- Inputs:
  Hard-coded canonical Stage-4 no-lag `minlen15` manifest and a run-level QC CSV.
- Outputs:
  Per-fold `covs_pca.npy`, `trans_prob.npy`, `means_pca.npy`, `fo_test.npy`, `fold_summaries.json`, `run_meta.json`, `triu_idx_200.npy`, and `cv_fold_summary.tsv`.
- Classification:
  Exploratory or optional provenance-side validation.
- Overlap:
  Conceptually overlaps with shortlist stability, but uses a narrower K range and diagonal covariances in PCA space, so it should not define the public default path.
- Recommendation:
  Keep archive/provenance only unless Pass 2 finds a specific manuscript-facing role that truly requires public exposure.

### `recreate_fusion_hmm_model_selection_summary.ipynb`

- Scientific purpose:
  Rebuilds the manuscript-style model-selection composite summary using K-sweep outputs plus shortlist-stability outputs.
- Likely manuscript linkage:
  Figure 2 support and Table S8 support.
- Inputs:
  `summary_byK_selected.tsv`, `K_selection_recommendation.json`, and selected shortlist stability outputs for `K=3` and `K=5`.
- Outputs:
  `fusion_hmm_model_selection_summary.png` and `fusion_hmm_K_selection_compact_table.csv`.
- Classification:
  QC/figure/table sidecar, manuscript-facing.
- Overlap:
  No strong overlap with training notebooks, but depends on both K-sweep and shortlist outputs.
- Recommendation:
  Keep as a cleaned public summary-builder notebook.

### Misplaced provenance in `notebooks/4_alignment/`

#### `R01_PipelineC_fusion_hmm_LOSO_K2-12_v3_UNIFIED_TOGGLE_LAGS_GPU_PATCH_Data2Dataset.ipynb`
#### `R01_PipelineC_fusion_hmm_LOSO_K2-12_v3_UNIFIED_TOGGLE_LAGS_GPU_PATCH_Data2Dataset-extend.ipynb`

- Scientific purpose:
  Older LOSO K-sweep notebooks that clearly belong conceptually to Stage 5 rather than Stage 4.
- Relevance:
  Useful provenance for the evolution of the K-sweep path.
- Current issue:
  They default to lagged branches and older path logic, so they should not drive the cleaned public implementation.
- Recommendation:
  Treat as misplaced provenance/context only in this stage plan. Do not expose as public Stage-5 notebooks.

## Merge And Split Recommendations

### Merge candidates

- Merge the public role of `R01_PipelineC2_Ksweep_LOSO_OOMFIX_steps_XLAoff_chunkresume_v2REBATCH.ipynb` into one cleaned public notebook for LOSO K sweep.
- Merge the public role of `R01_PipelineD_C2_LOSO_stability_shortlist_REWRITE_v3_twostage_configfix.ipynb` into one cleaned public notebook for shortlist stability.
- Merge `recreate_fusion_hmm_model_selection_summary.ipynb` into one cleaned figure/table support notebook.

### Files that should stay separate

- Keep K-sweep and shortlist-stability as separate public notebooks. They are distinct manuscript steps with distinct outputs.
- Keep the figure/table summary notebook separate from the training notebooks.

### Files that should likely remain helper-level or provenance-only

- `R01_PipelineD_C2_LOSO_stability_shortlist.ipynb`: provenance only
- `R01_PipelineC_LOSO_param_stability_K6-8_intermediate_nolags_minlen15_SEQ10_summaries_ES_LR_GPU4GB_v3DATASETFIX.ipynb`: likely provenance-only or optional validation only
- the misplaced Stage-4 `PipelineC` notebooks: provenance-only context

## Proposed Cleaned GitHub-Facing File Set

Recommended public-facing Stage-5 set:

- `50_run_loso_k_sweep_model_selection.ipynb`
- `51_run_loso_shortlist_stability_checks.ipynb`
- `52_build_figure2_and_table_s8_model_selection_summary.ipynb`

Optional only if Pass 2 shows a concrete manuscript-facing need:

- `53_optional_validate_targeted_parameter_stability.ipynb`

Current evidence does not justify exposing `53` as part of the default public pipeline.

## Mapping From Current Files To Cleaned Set

- `R01_PipelineC2_Ksweep_LOSO_OOMFIX_steps_XLAoff_chunkresume_v2REBATCH.ipynb`
  -> `50_run_loso_k_sweep_model_selection.ipynb`

- `R01_PipelineD_C2_LOSO_stability_shortlist_REWRITE_v3_twostage_configfix.ipynb`
  -> `51_run_loso_shortlist_stability_checks.ipynb`

- `R01_PipelineD_C2_LOSO_stability_shortlist.ipynb`
  -> archive/provenance only

- `recreate_fusion_hmm_model_selection_summary.ipynb`
  -> `52_build_figure2_and_table_s8_model_selection_summary.ipynb`

- `R01_PipelineC_LOSO_param_stability_K6-8_intermediate_nolags_minlen15_SEQ10_summaries_ES_LR_GPU4GB_v3DATASETFIX.ipynb`
  -> archive/provenance only for now, or optional `53` if later justified

- `notebooks/4_alignment/R01_PipelineC_fusion_hmm_LOSO_K2-12_v3_UNIFIED_TOGGLE_LAGS_GPU_PATCH_Data2Dataset*.ipynb`
  -> misplaced provenance/context for `50`, not public Stage-4 content

## Downstream And Upstream Integration Assumptions To Verify In Pass 2

### Upstream Stage-4 assumptions

Pass 2 should verify that the cleaned Stage-4 outputs match what Stage 5 actually expects:

- canonical dataset branch:
  `FINAL_v3_gnorm_allTR/intermediate/hmm_segments_minlen15_nolags/`
- segment manifest name:
  `segments_manifest.tsv`
- manifest columns seen in active notebooks:
  `run`, `feature_mode`, `lags_tr`, `seg_id`, `start_TR`, `end_TR`, `len_TR`, `start_sec`, `end_sec`, `dur_sec`, `n_features`, `seg_path`
- optional run-level QC CSV used by the targeted K=6/7/8 notebook:
  `qc/per_run_segments_minlen15.csv`

### Canonical dataset assumptions

Pass 2 should verify whether the cleaned public default should be strictly:

- `FEATURE_MODE = "nolags"`
- `MINLEN = 15`

Current evidence says yes. Older notebooks still support lagged branches, but those look like provenance/support paths rather than the frozen paper dataset.

### LOSO fold-definition assumptions

Pass 2 should verify:

- whether folds are defined strictly by held-out subject
- whether the `run` field is parsed deterministically into subject IDs
- whether the final dataset has one run per held-out subject in this stage

### Environment and compute assumptions

Pass 2 should verify which of these must be explicitly documented for the public repo:

- Python environment with `tensorflow` and `osl_dynamics`
- GPU-sensitive settings
- explicit XLA disabling in some notebooks
- GPU memory-cap logic
- chunk/resume behavior
- CPU fallback realism for users without a compatible GPU

### Model-selection output assumptions

Pass 2 should verify which outputs are manuscript-essential and which are only diagnostic:

- `summary_byK_selected.tsv`
- `K_selection_recommendation.json`
- shortlist stability outputs for chosen K values
- compact model-selection table and composite summary figure

## Risks And Ambiguities

### 1. Shortlist ambiguity versus final K=3 paper result

The main K-sweep notebook prints:

- local minima: `[3, 5, 9, 12]`
- recommended shortlist primary: `[5, 9, 12]`
- optional: `[3]`

That does not read naturally alongside the frozen final paper result `K = 3`. This should be treated as a manuscript/code tension to document, not silently harmonize.

### 2. Active shortlist notebook currently uses `K_LIST = [5]`

The rewrite shortlist-stability notebook currently hardcodes `K_LIST = [5]`, while its own inline comment shows `[3, 5, 9, 12]`. The older shortlist notebook uses `[3, 5, 9, 12]`. Pass 2 needs to resolve what the actual intended public-facing shortlist set should be.

### 3. One targeted stability notebook uses different modeling choices

The K=6/7/8 notebook uses diagonal covariances in PCA space and narrower K coverage. That makes it useful provenance, but not a safe public default without stronger manuscript justification.

### 4. Strong environment sensitivity

These notebooks are not manual/hybrid, but they are highly environment-sensitive:

- WSL-style `/mnt/c/...` path assumptions
- TensorFlow and `osl_dynamics`
- GPU memory constraints
- XLA and eager-mode settings
- long-running chunk/resume behavior

### 5. Misplaced provenance from Stage 4

The older `PipelineC` notebooks in `notebooks/4_alignment/` are clearly Stage-5 context. They should be considered provenance only and should not confuse the cleaned Stage-4 public pipeline.

## Bottom-Line Recommendation

Stage 5 should be refactored as a three-notebook public stage:

1. one notebook for LOSO K sweep and K-selection outputs
2. one notebook for shortlisted-K stability and state-matching checks
3. one notebook for the manuscript-facing Figure 2 / Table S8 support outputs

Do not expose the older shortlist notebook or the targeted K=6/7/8 stability notebook as default public entry points unless Pass 2 finds a specific manuscript-facing requirement.

Pass 2 should focus on integration and interpretation, not code recovery:

- verify the exact Stage-4 manifest/QC contract
- verify canonical `nolags + minlen15` usage
- resolve the `K_LIST` ambiguity in the rewrite shortlist notebook
- document the real GPU/TensorFlow/`osl_dynamics` assumptions
- clarify how the manuscript arrives at the final K=3 interpretation from the current selection outputs
