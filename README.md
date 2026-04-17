# fusion_hmm

Code and documentation for the manuscript:

**Fusion hidden Markov modeling reveals a reproducible shared state architecture in simultaneous resting-state EEG-fMRI**


## What this repository is

This repository is the public paper-code companion for the manuscript above.

It is organized as a manuscript-aligned workflow rather than as a generic software package. The main goal is to make the published analysis readable and traceable for an outside scientific reader while preserving the original scientific behavior.

The active public workflow currently runs through:

1. EEG sensor preprocessing and exclusion handling
2. EEG source localization and parcel export
3. BOLD parcel extraction and QC
4. EEG-BOLD timestamp alignment and retained-segment construction
5. LOSO model-order selection
6. final full-data `K = 3` fitting and downstream reconstructions

## Read these first

If you are new to the repo, start here:

- `docs/methods_map.md`
- `docs/manual_steps.md`
- `docs/final_dataset_spec.md`
- `docs/reproducibility_notes.md`
- `docs/figure_table_map.md`

## Current public status

Stages 1 to 6 now have cleaned public-facing workflow files.

The later folders:

- `notebooks/7_summaries/`
- `notebooks/8_figures/`
- `notebooks/9_tables/`

are intentionally empty for now because those summary, figure, and table products are still most cleanly generated inside the upstream stage notebooks.

The main supporting docs already present are:

- `AGENTS.md`
- `docs/methods_map.md`
- `docs/manual_steps.md`
- `docs/repo_scope.md`
- `docs/final_dataset_spec.md`
- `docs/reproducibility_notes.md`
- `docs/figure_table_map.md`
- manuscript and supplement reference files under `docs/_manuscript_reference/`

## Canonical manuscript path

The main public manuscript path is:

- Stage 4 retained dataset branch: `intermediate + nolags + minlen15`
- Stage 5 model-selection story: broad `K = 2..12` screening, then manuscript-facing `K = 3` versus `K = 5` shortlist comparison, final choice `K = 3`
- Stage 6 final model: full-data `K = 3` fit plus downstream `K = 3` review and reconstruction notebooks

## Canonical final dataset

Unless otherwise noted, the repository should reflect the frozen paper dataset:

- no-lag fusion design
- minimum retained segment length = 15 TR
- BOLD TR = 2.1 s
- EEG sampling rate = 250 Hz
- 200 BOLD parcel features
- 200 same-TR EEG parcel-power features
- 400 features per retained TR
- 3550 retained TRs
- 71 retained contiguous segments
- 124.25 usable minutes
- final selected model order = `K = 3`

See also:

- `docs/final_dataset_spec.md`

## Workflow overview

### Stage 1. EEG sensor preprocessing and exclusion handling
Public files in `notebooks/1_eeg_sensor/`:

- `eeg_prune_iclabel_and_export_clean_sets_10.m`
- `11_brainstorm_exclusion_marking_manual.md`
- `export_and_union_merge_brainstorm_exclusions_12.m`
- `eeg_run_qc_and_table_s1_13.m`

### Stage 2. EEG source localization, parcel extraction, and source QC
Public files in `notebooks/2_eeg_source/`:

- `20_prepare_schaefer200_atlas_for_brainstorm.ipynb`
- `21_brainstorm_volume_source_and_atlas_import_manual.md`
- `extract_volgrid_scouts_from_brainstorm_tess_22.m`
- `export_eeg_parcel_pc1_and_gain_normalize_23.m`
- `qc_eeg_source_alignment_table_s2_24.m`
- `25_qc_eeg_parcel_exports_table_s3_and_figures_s2_s4.ipynb`

### Stage 3. BOLD parcel extraction and QC
Public files in `notebooks/3_bold/`:

- `30_map_schaefer200_to_bold_run_grids.ipynb`
- `31_export_bold_parcel_pc1_with_nuisance_regression.ipynb`
- `32_build_table_s4_bold_parcel_atlas_summary.ipynb`
- `33_build_table_s5_and_figure_s5_bold_qc.ipynb`

### Stage 4. EEG-BOLD alignment and retained-segment construction
Public files in `notebooks/4_alignment/`:

- `40_align_eeg_to_bold_trs_and_build_keep_masks.ipynb`
- `41_build_final_no_lag_fusion_observation_segments.ipynb`
- `42_qc_alignment_tables_s6_s7_and_figure1_support.ipynb`

### Stage 5. LOSO model-order selection
Public files in `notebooks/5_hmm_selection/`:

- `50_run_loso_k_sweep_model_selection.ipynb`
- `51_run_loso_shortlist_stability_checks.ipynb`
- `52_build_figure2_and_table_s8_model_selection_summary.ipynb`

### Stage 6. Final full-data K = 3 fit and downstream reconstructions
Public files in `notebooks/6_hmm_final/`:

- `60_fit_final_k3_fusion_hmm.ipynb`
- `61_review_final_k3_fit_qc_and_state_dynamics.ipynb`
- `62_reconstruct_bold_state_networks_and_ranked_contrasts.ipynb`
- `63_reconstruct_crossmodal_state_blocks_and_ranked_contrasts.ipynb`
- `64_build_parcelized_cortical_state_maps.ipynb`
- optional: `65_optional_export_figure4_figure5_panels.ipynb`

## Manual and hybrid steps

This repo is not fully code-only.

Important manual or hybrid steps include:

- Brainstorm EEG exclusion marking
- Brainstorm subject import and anatomy setup
- EEG MNI normalization, BEM generation, and source localization in Brainstorm
- atlas import into Brainstorm as volume scouts
- some screenshot-based or panel-assembly figure work

Those steps are documented in `docs/manual_steps.md`. The public notebooks and scripts do not pretend those steps are fully automated when they are not.

## Active helpers and legacy provenance

Each active stage folder contains:

- public entry files meant for outside readers
- helper files or helper modules used by those public entry files
- preserved older working files kept for provenance

For the active public helper layer:

- Stage 1 and Stage 2 now use descriptive helper names in the cleaned public workflow
- the MATLAB-safe public scripts now place the step number at the end of the filename, for example `eeg_prune_iclabel_and_export_clean_sets_10.m`
- older number-leading `.m` filenames may still appear in these folders as compatibility stubs, but they are no longer the main public entry points
- preserved `r01_` MATLAB implementations remain in place as provenance-compatible low-level code, and the public helper layer now checks those dependencies explicitly instead of assuming they are silently on the MATLAB path
- later Python stages use stage-specific helper modules with plain-language module headers

Historical notebooks and scripts with names such as `r01_*` or `Pipeline*` remain preserved in the workflow folders or in `notebooks/_archive_raw_original_names/`, but they do not define the main public path.

## Repository layout

```text
fusion_hmm/
  README.md
  AGENTS.md

  docs/
    methods_map.md
    manual_steps.md
    figure_table_map.md
    repo_scope.md
    final_dataset_spec.md
    reproducibility_notes.md
    _manuscript_reference/
      FULL_MANUSCRIPT.docx
      SUPPLEMENTAL_MATERIALS.docx

  config/

  notebooks/
    1_eeg_sensor/
      eeg_prune_iclabel_and_export_clean_sets_10.m
      11_brainstorm_exclusion_marking_manual.md
      export_and_union_merge_brainstorm_exclusions_12.m
      eeg_run_qc_and_table_s1_13.m
      helpers/
    2_eeg_source/
      20_prepare_schaefer200_atlas_for_brainstorm.ipynb
      21_brainstorm_volume_source_and_atlas_import_manual.md
      extract_volgrid_scouts_from_brainstorm_tess_22.m
      export_eeg_parcel_pc1_and_gain_normalize_23.m
      qc_eeg_source_alignment_table_s2_24.m
      25_qc_eeg_parcel_exports_table_s3_and_figures_s2_s4.ipynb
    3_bold/
    4_alignment/
    5_hmm_selection/
    6_hmm_final/
    7_summaries/
    8_figures/
    9_tables/
    _archive_raw_original_names/

  scripts/
  src/
  results/
  assets/
```

## Current limitations

This repository is already organized around the final manuscript workflow, but it is still a refactor-phase public release rather than a one-click software package.

In particular:

- some later figure and table products are still generated inside upstream method notebooks
- manual Brainstorm work remains real and is documented rather than hidden
- environment setup still matters for MATLAB, Brainstorm, TensorFlow, `osl_dynamics`, and plotting libraries
- historical provenance notebooks are preserved even when they are no longer the main public entry points

For the most detailed practical caveats, see `docs/reproducibility_notes.md`.
