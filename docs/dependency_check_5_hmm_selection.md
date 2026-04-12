# Dependency Check: Stage 5 HMM Selection

## Scope

This memo checks the real dependencies, dataset assumptions, shortlist intent, and environment requirements for `notebooks/5_hmm_selection/` before Stage-5 implementation begins.

The proposed cleaned public-facing Stage-5 set is:

- `50_run_loso_k_sweep_model_selection.ipynb`
- `51_run_loso_shortlist_stability_checks.ipynb`
- `52_build_figure2_and_table_s8_model_selection_summary.ipynb`

This pass also inspects two misplaced provenance notebooks in `notebooks/4_alignment/` because they are historically relevant to Stage 5:

- `R01_PipelineC_fusion_hmm_LOSO_K2-12_v3_UNIFIED_TOGGLE_LAGS_GPU_PATCH_Data2Dataset.ipynb`
- `R01_PipelineC_fusion_hmm_LOSO_K2-12_v3_UNIFIED_TOGGLE_LAGS_GPU_PATCH_Data2Dataset-extend.ipynb`

## Inputs Reviewed

Active Stage-5 notebooks:

- `R01_PipelineC2_Ksweep_LOSO_OOMFIX_steps_XLAoff_chunkresume_v2REBATCH.ipynb`
- `R01_PipelineD_C2_LOSO_stability_shortlist_REWRITE_v3_twostage_configfix.ipynb`
- `R01_PipelineD_C2_LOSO_stability_shortlist.ipynb`
- `R01_PipelineC_LOSO_param_stability_K6-8_intermediate_nolags_minlen15_SEQ10_summaries_ES_LR_GPU4GB_v3DATASETFIX.ipynb`
- `recreate_fusion_hmm_model_selection_summary.ipynb`

Upstream/context files:

- `docs/refactor_plan_5_hmm_selection.md`
- `docs/refactor_plan_4_alignment.md`
- `docs/dependency_check_4_alignment.md`
- `docs/final_dataset_spec.md`
- `docs/reproducibility_notes.md`
- `notebooks/4_alignment/stage4_segment_helpers.py`

## Proposed Cleaned Public-Facing Files Checked

### `50_run_loso_k_sweep_model_selection.ipynb`

Primary source notebook:

- `R01_PipelineC2_Ksweep_LOSO_OOMFIX_steps_XLAoff_chunkresume_v2REBATCH.ipynb`

Status:

- existing logic is sufficient
- no missing helper was identified
- main need is cleaned public wrapping and path cleanup

### `51_run_loso_shortlist_stability_checks.ipynb`

Primary source notebook:

- `R01_PipelineD_C2_LOSO_stability_shortlist_REWRITE_v3_twostage_configfix.ipynb`

Secondary provenance source:

- `R01_PipelineD_C2_LOSO_stability_shortlist.ipynb`

Status:

- existing logic is sufficient
- no missing helper was identified
- main unresolved issue is shortlist interpretation, not code recovery

### `52_build_figure2_and_table_s8_model_selection_summary.ipynb`

Primary source notebook:

- `recreate_fusion_hmm_model_selection_summary.ipynb`

Status:

- figure/table builder logic exists
- no missing code helper was identified
- one wrapper mismatch must be handled in Pass 3:
  the current notebook expects flat files like `state_matching_scores_K03.tsv`, while the active shortlist notebook writes per-K subfolders such as `K03/state_matching_scores.tsv`

## Stage-4 Integration Findings

### Real manifest contract

The active Stage-5 K-sweep and shortlist notebooks both consume `segments_manifest.tsv` from the Stage-4 final segment branch.

From `stage4_segment_helpers.py`, the cleaned Stage-4 segment manifest writes at least these columns:

- `run`
- `feature_mode`
- `lags_tr`
- `seg_id`
- `start_TR`
- `end_TR`
- `len_TR`
- `start_sec`
- `end_sec`
- `dur_sec`
- `n_features`
- `seg_path`

The active Stage-5 notebooks only require a minimal hard dependency on:

- `run`
- `seg_path`

They also make practical use of:

- `seg_id` for deterministic sorting if present
- `start_TR` and `end_TR` for optional contiguity QC in the rewrite shortlist notebook

This means the cleaned Stage-4 outputs match the active Stage-5 manifest contract.

### Segment path handling

Stage 4 writes `seg_path` in the manifest. Stage 5 is robust to either:

- absolute paths
- or paths relative to the manifest directory

Both active Stage-5 core notebooks resolve relative paths against `MANIFEST_TSV.parent`, so there is no recovery blocker here.

### Segment naming

Stage 4 writes segment IDs in the form:

- `sub-XX_ses-YY_task-rest__seg0001`

But Stage 5 does not depend on that exact filename pattern. It uses the manifest’s `seg_path` directly. This is good and should be preserved.

### Dependence on run-level QC CSV

The core public Stage-5 notebooks do not require `qc/per_run_segments_minlen15.csv`.

That CSV appears only in the targeted K=6/7/8 notebook:

- `R01_PipelineC_LOSO_param_stability_K6-8_intermediate_nolags_minlen15_SEQ10_summaries_ES_LR_GPU4GB_v3DATASETFIX.ipynb`

So the run-level QC CSV is not a blocker for the proposed public set.

## Canonical Dataset Branch Findings

The active Stage-5 core notebooks both default to:

- `FEATURE_MODE = "nolags"`
- `MINLEN = 15`

The Stage-4 cleaned helper writes both:

- `hmm_segments_minlen15_nolags`
- optional lagged branches if requested upstream

The frozen dataset docs state that the canonical paper dataset is:

- no-lag
- minimum segment length 15 TR
- 15 runs
- 71 segments
- 3550 retained TRs

So the available evidence supports this conclusion:

- the public Stage-5 default should be strictly `nolags + minlen15`
- lagged branches should remain provenance-only or optional side paths
- the older lagged Stage-5 provenance notebooks should not shape the public default

## Shortlist / K-Selection Findings

### What is directly known

From the active K-sweep notebook:

- local minima are printed as `[3, 5, 9, 12]`
- the JSON recommendation writes:
  - `K_best`
  - `K_1se`
  - `local_minima`
  - `shortlist_primary`
  - `shortlist_optional`
- the current notebook hardcodes:
  - `shortlist_primary = [5, 9, 12]`
  - `shortlist_optional = [K_1se]`

In the observed notebook output, this becomes:

- `K_1se = 3`
- `Local minima: [3, 5, 9, 12]`
- `Recommended shortlist (primary): [5, 9, 12] | optional: [3]`

From the active rewrite shortlist notebook:

- `K_LIST = [5]`
- comment next to it says `#[3, 5, 9, 12]`

From the older shortlist notebook:

- `K_LIST = [3, 5, 9, 12]`

From the manuscript-level frozen dataset docs:

- the final selected model order is `K = 3`

From the summary-figure notebook:

- the main visual comparison is built around `K=3` versus `K=5`
- the compact decision table includes `K=3`, `K=5`, and `K=12`

### Interpretation

This is not a missing dependency problem. It is an interpretation and provenance tension:

- the active K-sweep notebook does not naturally elevate `K=3` as the primary shortlist choice
- the summary notebook clearly treats `K=3` as an important comparison model
- the older shortlist notebook evaluated `[3, 5, 9, 12]`
- the rewrite shortlist notebook currently narrows this to `[5]`

### Bottom line on shortlist intent

The available evidence supports this cautious interpretation:

- `K=3` was clearly retained as a serious manuscript-relevant candidate
- `K=5`, `K=9`, and `K=12` also appear in the shortlist provenance
- the current rewrite notebook’s `K_LIST=[5]` should not be treated as settled public truth without documentation

This ambiguity must be documented explicitly in Pass 3. It should not be silently harmonized.

## LOSO Fold-Definition Findings

Both active core notebooks define folds by subject extracted from the manifest `run` string.

The parsing rule is:

- split `run` on underscores
- return the first token that starts with `sub-`
- otherwise return the first token

Then LOSO is defined as:

- hold out all rows whose parsed `subject` matches the held-out subject

This means:

- fold logic depends on the manifest `run` column containing a BIDS-like `sub-XX` token
- it does not depend on the segment filename itself

### One-run-per-subject assumption

The active code does not strictly require one run per subject. If there were multiple runs sharing the same `sub-XX`, all of them would be held out together.

However, the frozen final dataset spec says:

- 15 runs

and the Stage-4 final dataset is described as leave-one-subject-out. In the frozen paper dataset, that strongly suggests one retained resting-state run per held-out subject.

So the current public dataset assumption appears safe, but it is still somewhat brittle because it depends on:

- `run` naming staying BIDS-like
- no unexpected multiple-run-per-subject expansion being introduced later

## Environment And Compute Assumptions

### Core package assumptions

The active Stage-5 notebooks require at least:

- Python
- `tensorflow`
- `osl_dynamics`
- `numpy`
- `pandas`
- `matplotlib`
- `scipy`

More specifically:

- K-sweep uses `scipy.stats.wilcoxon` and `ttest_rel`
- shortlist stability uses `scipy.optimize.linear_sum_assignment`

### GPU and memory assumptions

The active core notebooks are heavily tuned for GPU-sensitive WSL execution:

- optional GPU memory caps
- `memory_growth` fallback
- XLA disabling
- reduced thread parallelism
- chunk/resume execution to survive long sweeps
- callback disabling and prefetch disabling for stability

### CPU fallback realism

CPU-only execution is supported in code:

- both active core notebooks explicitly print CPU-only when no GPU is visible

But practical CPU-only reproduction is likely slow and possibly painful for the full sweep. So CPU fallback is real in code, but probably not the expected production path.

### Path and OS assumptions

The active notebooks hard-code WSL-style paths such as:

- `/mnt/c/EEGFMRI/...`

This is not a missing dependency, but it is a real public-facing wrapper issue that must be cleaned in Pass 3.

## Missing Or Ambiguous Dependencies

### No true missing helper or recovery item found

I did not find a missing Stage-5 core code dependency that must be recovered before implementation.

### Existing logic that is sufficient

The following are already sufficient as source logic for the cleaned public notebooks:

- `R01_PipelineC2_Ksweep_LOSO_OOMFIX_steps_XLAoff_chunkresume_v2REBATCH.ipynb`
- `R01_PipelineD_C2_LOSO_stability_shortlist_REWRITE_v3_twostage_configfix.ipynb`
- `recreate_fusion_hmm_model_selection_summary.ipynb`

### Interpretation ambiguity that must be documented

- shortlist composition versus final `K=3` interpretation
- rewrite shortlist notebook currently using `K_LIST=[5]`

### Provenance-only notebooks that should not shape the public implementation

- `R01_PipelineD_C2_LOSO_stability_shortlist.ipynb`
- `R01_PipelineC_LOSO_param_stability_K6-8_intermediate_nolags_minlen15_SEQ10_summaries_ES_LR_GPU4GB_v3DATASETFIX.ipynb`
- the misplaced Stage-4 `PipelineC` notebooks

The older misplaced `PipelineC` notebook is especially important not to over-trust, because its manifest auto-find scoring favors paths containing `minlen10`, which is not the canonical final public dataset.

## Recovery Or Wrapper Actions Required Before Pass 3

### Required wrapper action

`52_build_figure2_and_table_s8_model_selection_summary.ipynb` will need a cleaned input path contract.

Reason:

- the current summary notebook expects flat files like:
  - `state_matching_scores_K03.tsv`
  - `fold_summaries_table_matched_K03.tsv`
- the active shortlist notebook writes:
  - `K03/state_matching_scores.tsv`
  - `K03/fold_summaries_table_matched.tsv`

This is a wrapper/integration issue, not a recovery blocker.

### Required documentation action

Pass 3 should make the following explicit:

- Stage 5 public default consumes the canonical Stage-4 no-lag `minlen15` manifest
- lagged branches are optional provenance only
- shortlist ambiguity is preserved and documented
- GPU/TensorFlow/`osl_dynamics` assumptions are real

## Bottom-Line Verdict

Stage 5 is not blocked by missing code recovery.

The active core notebooks match the cleaned Stage-4 outputs well:

- manifest naming matches
- manifest columns match
- segment path handling is already robust
- the core public Stage-5 notebooks do not depend on optional Stage-4 QC CSVs

The main issues before Pass 3 are:

- one real interpretation ambiguity about shortlist composition and the road to final `K=3`
- one wrapper mismatch for the Figure 2 / Table S8 builder
- explicit documentation of the environment assumptions and canonical no-lag `minlen15` branch

So Pass 3 should proceed as cleaned wrappers around the existing logic, not as recovery work.
