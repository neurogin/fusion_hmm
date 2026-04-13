# Dependency Check: Stage 6 HMM Final

## Scope

This memo checks the real dependencies, final-artifact contract, reference-state behavior, and environment assumptions for `notebooks/6_hmm_final/` before Stage-6 implementation begins.

The proposed cleaned public-facing Stage-6 set is:

- `60_fit_final_k3_fusion_hmm.ipynb`
- `61_review_final_k3_fit_qc_and_state_dynamics.ipynb`
- `62_reconstruct_bold_state_networks_and_ranked_contrasts.ipynb`
- `63_reconstruct_crossmodal_state_blocks_and_ranked_contrasts.ipynb`
- `64_build_parcelized_cortical_state_maps.ipynb`
- optional: `65_optional_export_figure4_figure5_panels.ipynb`

This pass focuses on:

- the exact Stage-4 contract consumed by the final-fit notebook,
- whether any real Stage-5 files are needed,
- the true output contract written by the final-fit notebook,
- which downstream notebooks depend on which artifacts,
- and how the manuscript-facing `S2` interpretation is produced in practice.

## Inputs reviewed

Active Stage-6 notebooks:

- `notebooks/6_hmm_final/R01_PipelineE_full_model.ipynb`
- `notebooks/6_hmm_final/PipelineE_K3_results_review_notebook_fixed_paths.ipynb`
- `notebooks/6_hmm_final/PipelineE_K3_state_physiology_notebook_schaefer_adapted_fix1.ipynb`
- `notebooks/6_hmm_final/PipelineE_K3_fusion_input_and_crossmodal_notebook.ipynb`
- `notebooks/6_hmm_final/PipelineE_K3_brainmaps_surface_parcelized_notebook_fix4.ipynb`
- `notebooks/6_hmm_final/PipelineE_K3_Fig4_Fig5_panel_exports_notebook_v3.ipynb`
- `notebooks/6_hmm_final/PipelineE_K3_manuscript_figures_notebook.ipynb`
- `notebooks/6_hmm_final/PipelineE_K3_manuscript_figures_notebook_with_raster.ipynb`

Supporting docs and provenance:

- `docs/refactor_plan_6_hmm_final.md`
- `docs/refactor_plan_5_hmm_selection.md`
- `docs/dependency_check_5_hmm_selection.md`
- `docs/dependency_check_4_alignment.md`
- `docs/final_dataset_spec.md`
- `docs/figure_table_map.md`
- `docs/reproducibility_notes.md`
- `docs/methods_map.md`
- `notebooks/_archive_raw_original_names/7_final model/`

## Proposed cleaned public-facing files checked

### `60_fit_final_k3_fusion_hmm.ipynb`

Primary source notebook:

- `R01_PipelineE_full_model.ipynb`

Status:

- existing logic is sufficient
- no missing helper was identified
- main needs are public wrapper cleanup, path cleanup, and explicit environment notes

### `61_review_final_k3_fit_qc_and_state_dynamics.ipynb`

Primary source notebook:

- `PipelineE_K3_results_review_notebook_fixed_paths.ipynb`

Secondary provenance source:

- selected dynamics and raster sections from `PipelineE_K3_manuscript_figures_notebook_with_raster.ipynb`

Status:

- core review logic exists
- no missing helper was identified
- gamma and Viterbi review support lives mainly in the umbrella provenance notebook, not the cleaner review notebook

### `62_reconstruct_bold_state_networks_and_ranked_contrasts.ipynb`

Primary source notebook:

- `PipelineE_K3_state_physiology_notebook_schaefer_adapted_fix1.ipynb`

Status:

- existing logic is sufficient
- no missing helper was identified
- current notebook requires more files than are conceptually essential, but this is a wrapper issue rather than a recovery blocker

### `63_reconstruct_crossmodal_state_blocks_and_ranked_contrasts.ipynb`

Primary source notebook:

- cross-modal half of `PipelineE_K3_fusion_input_and_crossmodal_notebook.ipynb`

Status:

- existing logic is sufficient
- no missing helper was identified
- this notebook is genuinely mixed and should be split conceptually in Pass 3

### `64_build_parcelized_cortical_state_maps.ipynb`

Primary source notebook:

- `PipelineE_K3_brainmaps_surface_parcelized_notebook_fix4.ipynb`

Status:

- existing logic is sufficient
- no missing helper was identified
- one real interpretation issue remains: the active notebook hardcodes `REFERENCE_STATE = 2`

### Optional `65_optional_export_figure4_figure5_panels.ipynb`

Primary source notebook:

- `PipelineE_K3_Fig4_Fig5_panel_exports_notebook_v3.ipynb`

Status:

- existing logic is sufficient
- no missing helper was identified
- should remain optional because it is panel-export and manual-assembly support, not core computation

## Stage-4 integration findings

### Canonical dataset branch

The full-model notebook explicitly sets:

- `DATA_VARIANT = "intermediate"`
- `FEATURE_MODE = "nolags"`
- `MINLEN = 15`
- `K_FINAL = 3`

This matches the frozen manuscript dataset definition and the cleaned Stage-4 public default.

### Manifest contract actually consumed by the final fit

`R01_PipelineE_full_model.ipynb` auto-finds a manifest under:

- `FINAL_ROOT / "hmm_segments_minlen15_nolags" / "segments_manifest.tsv"`
- fallback: `FINAL_ROOT / "hmm_segments_minlen15" / "segments_manifest.tsv"`
- fallback: highest-scoring `segments_manifest.tsv` anywhere under `FINAL_ROOT`

Required manifest columns are:

- `run`
- `seg_path`

Additional manifest behavior:

- if `seg_id` exists, it is used for deterministic ordering
- `run` is parsed to derive `subject`
- `seg_path` may be absolute or relative to the manifest directory
- all referenced segment files are checked for existence before fitting begins

This means the core Stage-6 fit consumes the cleaned Stage-4 segment manifest directly and does not require extra Stage-4 QC files.

### Other Stage-4 files required by later Stage-6 notebooks

The mixed `PipelineE_K3_fusion_input_and_crossmodal_notebook.ipynb` also pulls in Stage-4 per-run alignment outputs for its figure-style Part A.

It expects:

- `ALIGN_ROOT / sub-XX_ses-YY / bold_pc1.npy`
- `ALIGN_ROOT / sub-XX_ses-YY / eeg_power_tr.npy`
- `ALIGN_ROOT / sub-XX_ses-YY / tr_edges_sec.npy`
- optional `ALIGN_ROOT / sub-XX_ses-YY / eeg_counts_per_tr.npy`
- keep-mask preference:
  - `keep_center_minlen15_lags0.npy`
  - then `keep_center_lags0.npy`
  - then `keep_center_minlen15.npy`

This is a real direct Stage-4 dependency, but only for the alignment-illustration half of that notebook.

## Stage-5 dependency findings

I did not find any active Stage-6 notebook that requires Stage-5 output files such as:

- K-sweep summaries
- shortlist stability tables
- state-matching tables
- Figure 2 / Table S8 outputs

The Stage-6 notebooks appear to depend on Stage 5 only in the scientific sense that:

- the model-selection stage chose `K = 3`

So the Stage-6 code path depends on the Stage-5 decision, not on Stage-5 artifacts.

## Final-model artifact contract

### Root-level outputs written by the full-model notebook

`R01_PipelineE_full_model.ipynb` writes these top-level outputs under `OUT_ROOT`:

- `run_meta.json`
- `preproc_meta.json`
- `preproc_params.npz`
- `seed_candidates.tsv`
- `seed_failures.tsv`
- `candidates_index.json`
- `topM_seeds.json`
- `refit_attempts.tsv`
- `run_metrics.tsv`
- `subject_metrics.tsv`
- `dwell_from_A.tsv`
- `seed_matching_scores.tsv`
- `seed_sim_signature.tsv`
- `seed_sim_A.tsv`
- `seed_A_mean.tsv`
- `seed_A_std.tsv`
- `qc_summary.json`

It also writes a `seeds/seed_###/` tree during screening, containing per-seed artifacts such as:

- `trans_prob.npy`
- `covs_pca.npy`
- optional `means_pca.npy`
- `state_signature_ut_boldcorr.npy`
- `seed_metrics.json`

### `final/` outputs written by the full-model notebook

The notebook writes these authoritative final artifacts under `OUT_ROOT / "final"`:

- `refit_results.json`
- `best_seed.json`
- `trans_prob.npy`
- `covs_pca.npy`
- `state_signature_ut_boldcorr.npy`
- optional `means_pca.npy`

It also writes `final/refit_try_seed###/` subfolders during refit, including:

- `trans_prob.npy`
- `covs_pca.npy`
- optional `means_pca.npy`
- `state_signature_ut_boldcorr.npy`
- `refit_metrics.json`

### Per-run decoding outputs

If enabled, the notebook writes:

- `gamma/<run>/gamma_seg0000.npy`, `gamma_seg0001.npy`, ...
- `viterbi/<run>/viterbi_seg0000.npy`, `viterbi_seg0001.npy`, ...

The saved Gamma files are segment-wise state-probability series.
The saved Viterbi files are segment-wise discrete state paths.

## Downstream notebook dependency findings

### `PipelineE_K3_results_review_notebook_fixed_paths.ipynb`

Active required inputs:

- `qc_summary.json`
- `refit_results.json`
- `subject_metrics.tsv`
- `run_metrics.tsv`
- `dwell_from_A.tsv`
- `trans_prob.npy`
- `state_signature_ut_boldcorr.npy`
- `means_pca.npy`
- `covs_pca.npy`

What it does not appear to require:

- Stage-4 manifest files
- Stage-5 outputs
- Gamma or Viterbi segment files

So the cleaner review notebook can be built directly on the final-fit artifacts alone. If Gamma/Viterbi raster outputs are wanted in the public review notebook, those pieces need to be imported from the umbrella provenance notebook rather than from this smaller review notebook.

### `PipelineE_K3_state_physiology_notebook_schaefer_adapted_fix1.ipynb`

Active required inputs:

- `qc_summary.json`
- `subject_metrics.tsv`
- `run_metrics.tsv`
- `trans_prob.npy`
- `state_signature_ut_boldcorr.npy`
- `covs_pca.npy`

Active optional inputs:

- `refit_results.json`
- `topM_seeds.json`
- `dwell_from_A.tsv`
- `preproc_params.npz`
- Schaefer label TSV or Brainstorm TXT from TemplateFlow

Important detail:

- the active code uses `refit_results.json` first to recover the final FO vector
- fallback is `subject_metrics.tsv`
- `REFERENCE_STATE = None` means “use dominant state from final FO”

This notebook also writes a derived file:

- `state_summary_table.tsv`

That file becomes a later convenience input for other notebooks, but it is not required if `best_seed.json` or `refit_results.json` are available.

### `PipelineE_K3_fusion_input_and_crossmodal_notebook.ipynb`

This notebook is genuinely split across two different dependency families.

Part A requires Stage-4 per-run alignment outputs:

- `bold_pc1.npy`
- `eeg_power_tr.npy`
- `tr_edges_sec.npy`
- optional `eeg_counts_per_tr.npy`
- one of the no-lag keep-mask files listed above

Part B requires final-model outputs:

- `covs_pca.npy`
- `preproc_params.npz`
- `best_seed.json` as the primary reference-state source

Useful fallback inputs for Part B:

- `state_summary_table.tsv`
- `subject_metrics.tsv`
- `run_metrics.tsv`
- `qc_summary.json`

Important detail:

- Part B does not require `state_signature_ut_boldcorr.npy`
- Part B derives cross-modal blocks by backprojecting `covs_pca.npy` with `Vb` and `Ve` from `preproc_params.npz`

This confirms that the notebook should be split conceptually in the public refactor.

### `PipelineE_K3_brainmaps_surface_parcelized_notebook_fix4.ipynb`

Active required inputs:

- `state_signature_ut_boldcorr.npy`
- Schaefer atlas NIfTI from TemplateFlow
- Schaefer atlas TSV from TemplateFlow

Active behavior:

- computes nodal mean connectivity directly from `state_signature_ut_boldcorr.npy`
- does not need `covs_pca.npy`
- does not need `preproc_params.npz`
- does not appear to use `STATE_SUMMARY_FILE` despite the comment saying it is optional for choosing S2 automatically

This means the active notebook hardcodes the reference-state choice rather than deriving it from saved final-fit summaries.

### `PipelineE_K3_Fig4_Fig5_panel_exports_notebook_v3.ipynb`

Active required inputs:

- `state_signature_ut_boldcorr.npy`
- `covs_pca.npy`
- `preproc_params.npz`
- `best_seed.json`
- Schaefer label TSV or Brainstorm TXT

Reference-state behavior:

- dominant state is derived from `best_seed.json` FO vector
- `REFERENCE_STATE_OVERRIDE` can force a different reference state

This notebook does not require `state_summary_table.tsv`.

## Reference-state / state-label findings

### Full-model notebook

The fit notebook itself does not assign the manuscript’s `S2` interpretation directly.

It writes:

- per-run `FO_s01`, `FO_s02`, `FO_s03`
- subject-level weighted `FO_s01`, `FO_s02`, `FO_s03`
- `best_seed.json` with the final FO vector
- `refit_results.json` with per-refit FO vectors

So the fit notebook exposes the occupancy statistics that later notebooks interpret.

### Review notebook

The review notebook defines dominant state as:

- `argmax` across the FO columns for each run or subject

In the saved outputs embedded in that notebook, the reported result is:

- dominant state across subjects: state 2 only
- dominant state across runs: state 2 only

So `S2` is presented there as an empirical result of the saved K=3 fit, not as a hardcoded assumption.

### State physiology notebook

The physiology notebook behaves as follows:

- if `REFERENCE_STATE` is set, use that exact state
- otherwise compute the final FO vector and choose the dominant state by `argmax`

Its active output shows:

- `Final FO vector: [0.089823 0.807597 0.10258]`
- `Dominant state : S2`
- `Reference state: S2`

So in this notebook the `S2` interpretation is data-driven by default, but still configurable.

### Fusion-input / cross-modal notebook

The cross-modal half uses an explicit priority order:

1. `REFERENCE_STATE_OVERRIDE`
2. `best_seed.json` FO vector
3. `state_summary_table.tsv`
4. mean subject FO from `subject_metrics.tsv`
5. mean run FO from `run_metrics.tsv`
6. fallback to `S2`

So in this notebook the public `S2` framing is mostly data-driven when final-fit artifacts are present, but there is a true fallback path that imposes `S2` if nothing better is available.

### Brain-map notebook

This is the strongest place where `S2` is imposed later rather than guaranteed by the outputs.

The active code sets:

- `REFERENCE_STATE = 2`
- `CONTRAST_STATES = [1, 3]`

Even though a comment says `STATE_SUMMARY_FILE` is optional “for choosing S2 automatically,” the current active code does not actually use that file to choose the reference state.

### Bottom line on the manuscript-facing `S2` interpretation

The available evidence supports this summary:

- `S2` is not created by the fit notebook itself
- `S2` emerges from the saved final FO vector in the review and physiology notebooks
- `S2` is recoverable from `best_seed.json` in the panel-export notebook
- `S2` is explicitly hardcoded in the brain-map notebook
- some fallback paths can impose `S2` if final-fit summaries are missing

So Pass 3 should not silently flatten these into one story. The public notebooks should say explicitly whether `S2` is:

- derived from the final FO vector,
- configurable by override,
- or fixed by a manuscript-facing presentation choice

## Environment and plotting assumptions

### Final-fit notebook

Real package and runtime assumptions:

- Python
- `tensorflow`
- `osl_dynamics`
- `numpy`
- `pandas`

Real compute assumptions:

- GPU-sensitive execution
- `DISABLE_XLA_AT_IMPORT = True`
- `DISABLE_PREFETCH = True`
- `DISABLE_CALLBACKS = True`
- `GPU_MEMORY_LIMIT_MB = None` by default, which triggers memory growth
- CPU-only mode exists in code but is likely slower and less practical for the full fit

The saved kernel name is:

- `osl_gpu`

### State physiology notebook

Real assumptions:

- `numpy`
- `pandas`
- `matplotlib`
- Schaefer label TSV from TemplateFlow or Brainstorm TXT

No `nibabel` or `nilearn` is required here.

### Fusion-input / cross-modal notebook

Real assumptions:

- `numpy`
- `pandas`
- `matplotlib`
- Stage-4 per-run alignment outputs for Part A
- Schaefer label TSV or Brainstorm TXT for network labels

### Brain-map notebook

Real assumptions:

- `nibabel`
- `nilearn`
- TemplateFlow Schaefer NIfTI and TSV
- `datasets.fetch_surf_fsaverage`, which may download fsaverage surfaces if they are not already cached

Notably, the proposed public brain-map path does not need:

- `networkx`
- `plotly`

Those appear in the umbrella provenance notebooks, not in the cleaner dedicated brain-map notebook.

### Optional panel-export notebook

Real assumptions:

- `numpy`
- `pandas`
- `matplotlib`
- TemplateFlow Schaefer label TSV or Brainstorm TXT

## Missing or ambiguous dependencies

### No true recovery item found

I did not find a missing Stage-6 helper or missing core file that blocks implementation of the proposed cleaned public set.

### Existing logic that is sufficient

The following active notebooks already contain enough source logic for the cleaned public wrappers:

- `R01_PipelineE_full_model.ipynb`
- `PipelineE_K3_results_review_notebook_fixed_paths.ipynb`
- `PipelineE_K3_state_physiology_notebook_schaefer_adapted_fix1.ipynb`
- `PipelineE_K3_fusion_input_and_crossmodal_notebook.ipynb`
- `PipelineE_K3_brainmaps_surface_parcelized_notebook_fix4.ipynb`
- `PipelineE_K3_Fig4_Fig5_panel_exports_notebook_v3.ipynb`

### Ambiguities that must be documented

1. `S2` is not treated identically across notebooks.
   - some notebooks infer it from final FO
   - one notebook hardcodes it
   - some notebooks keep overrides or fallback defaults

2. `state_summary_table.tsv` is a derived convenience file, not a fundamental final-fit artifact.
   - some later notebooks use it as a fallback
   - the cleaner public refactor should avoid making it look like a required upstream Stage-6 output if `best_seed.json` or `refit_results.json` already provide the needed information

3. The mixed fusion-input notebook pulls an earlier-style alignment illustration into this folder.
   - that part does not belong to the core final-model contract

4. The brain-map notebook comment about “choosing S2 automatically” does not match the active code path.
   - active code fixes `REFERENCE_STATE = 2`

### Provenance-only notebooks that should not shape the public implementation

- `PipelineE_K3_manuscript_figures_notebook.ipynb`
- `PipelineE_K3_manuscript_figures_notebook_with_raster.ipynb`

These notebooks are useful provenance sources, especially for:

- Gamma/Viterbi raster logic
- example Gamma/Viterbi trajectory plotting
- combined figure assembly patterns

But they should not define the public Stage-6 pipeline structure directly.

## Recovery or wrapper actions required before Pass 3

1. Build the cleaned final-fit notebook directly on the Stage-4 manifest contract.
   - keep `intermediate + nolags + minlen15`
   - keep `run` and `seg_path` as the minimal required manifest columns

2. Keep Stage-5 out of the file contract.
   - document that `K = 3` comes from Stage 5
   - do not require Stage-5 artifact files

3. Surface the real final-artifact contract clearly.
   - root-level metadata and QC files
   - `final/` authoritative model outputs
   - per-run `gamma/` and `viterbi/` segment series

4. Split the mixed fusion-input notebook conceptually.
   - Part A should become optional support or move conceptually toward later figure support
   - Part B should become the clean cross-modal reconstruction notebook

5. Document the `S2` behavior explicitly.
   - derived from final FO where that is true
   - configurable where overrides exist
   - fixed by manuscript-facing choice where the active code hardcodes it

6. Keep the brain-map notebook honest.
   - do not pretend the current active code auto-selects the reference state
   - if public cleanup later makes this configurable, that should be described as a wrapper change, not as if it were already the active behavior

7. Keep Gamma/Viterbi raster support separate from the core review notebook unless needed.
   - those sections live mainly in the umbrella provenance notebook

## Bottom-line verdict

Stage 6 is not blocked by missing-code recovery.

It depends directly on Stage 4 for the final retained dataset and does not appear to depend on any real Stage-5 artifact files beyond the scientific choice to carry `K = 3` forward.

The final-model artifact contract is clear and rich:

- top-level provenance, preprocessing, seed-screening, and QC files
- `final/` authoritative model outputs
- per-run `gamma/` and `viterbi/` segment-series outputs

The main thing Pass 3 must preserve carefully is the reference-state story:

- `S2` is often data-driven from the final FO vector
- but it is not uniformly automatic across notebooks
- and one active notebook explicitly hardcodes `S2`

So Pass 3 should proceed as a wrapper and cleanup pass, not as recovery work, while documenting the `S2` interpretation and splitting the mixed fusion-input notebook by scientific role.
