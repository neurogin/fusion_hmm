# fusion_hmm

Code and documentation for the manuscript:

**Fusion hidden Markov modeling reveals a reproducible shared state architecture in simultaneous resting-state EEG-fMRI**


## Overview

This repository is being prepared as the public code and documentation companion for the manuscript above.

The project develops a whole-brain fusion hidden Markov modeling (HMM) workflow for simultaneous resting-state EEG-fMRI, with emphasis on:

- careful EEG and BOLD preprocessing
- atlas-aligned multimodal feature construction
- timestamp-based EEG-BOLD alignment
- reproducibility-aware model selection
- biological interpretation of the final fusion-state solution

At present, the repository is in an active **refactor phase**. The existing MATLAB and Python notebook workflows are being reorganized into a cleaner, manuscript-aligned, GitHub-facing structure while preserving the original scientific behavior and keeping older working files archived.

## Current status

What is already present in the repository:

- `AGENTS.md`
- `docs/methods_map.md`
- `docs/manual_steps.md`
- `docs/repo_scope.md`
- `docs/final_dataset_spec.md`
- `docs/reproducibility_notes.md`
- `docs/figure_table_map.md`
- manuscript and supplement reference files under `docs/_manuscript_reference/`
- notebook folders organized by major manuscript workflow stage
- archive folder for original notebook/script versions
- cleaned public-facing stage-1 MATLAB entry files under `notebooks/1_eeg_sensor/`
- cleaned public-facing stage-2 EEG source / parcel entry files under `notebooks/2_eeg_source/`

What is not yet fully populated:

- `config/`
- `scripts/`
- `src/`

Also note:

- `notebooks/7_summaries/` is currently empty
- `notebooks/8_figures/` is currently empty
- `notebooks/9_tables/` is currently empty

This is intentional for now. In the current codebase, many summaries, figure-generation steps, and table-generation steps are still embedded within earlier method notebooks rather than separated into dedicated later-stage folders.

## Scientific workflow represented in this repo

The repository is organized around the manuscript workflow:

1. **EEG sensor preprocessing and exclusion handling**
2. **EEG source localization and atlas-aligned parcellation**
3. **BOLD nuisance regression and parcel extraction**
4. **Timestamp-based EEG-BOLD alignment and observation construction**
5. **Fusion HMM model-order selection**
6. **Final full-data K = 3 model fit**
7. **Temporal, BOLD, and cross-modal state summaries**
8. **Figure and table generation**

The final paper workflow centers on the frozen no-lag fusion dataset and the final K = 3 solution.

## Canonical final paper dataset

Unless otherwise noted, the repository should reflect the final paper dataset specification:

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
- final selected model order = **K = 3**

See also:

- `docs/final_dataset_spec.md`

## Current repository structure

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
      10_eeg_prune_iclabel_and_export_clean_sets.m
      11_brainstorm_exclusion_marking_manual.md
      12_export_and_union_merge_brainstorm_exclusions.m
      13_eeg_run_qc_and_table_s1.m
      helpers/
    2_eeg_source/
      20_prepare_schaefer200_atlas_for_brainstorm.ipynb
      21_brainstorm_volume_source_and_atlas_import_manual.md
      22_extract_volgrid_scouts_from_brainstorm_tess.m
      23_export_eeg_parcel_pc1_and_gain_normalize.m
      24_qc_eeg_source_alignment_table_s2.m
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
