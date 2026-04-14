# Methods-to-Code Map

This repository accompanies the manuscript:

**Fusion hidden Markov modeling reveals a reproducible shared state architecture in simultaneous resting-state EEG-fMRI**

This document maps the manuscript methods and supplementary methods to the current repository structure and indicates where manual/hybrid procedures are documented.

## Current status

This repository is in an active refactor phase.

The current goal is to reorganize the existing notebooks into a cleaner public-facing pipeline that follows the manuscript workflow, while preserving scientific behavior and keeping original notebooks archived.

At this stage:

- many analysis steps still exist as `.ipynb` notebooks
- some QC, summary, figure, and table logic is still embedded within earlier method notebooks
- the folders `7_summaries/`, `8_figures/`, and `9_tables/` may remain empty until a later cleanup pass
- original notebook files are preserved separately in `_archive_raw_original_names/`
- manual and hybrid GUI-based steps are documented in `docs/manual_steps.md`
- active public entry files and active helper layers use descriptive names where practical, while older `r01_` and `Pipeline*` names are retained only for provenance or compatibility

## Repository workflow folders

Current notebook folders:

- `notebooks/1_eeg_sensor/`
- `notebooks/2_eeg_source/`
- `notebooks/3_bold/`
- `notebooks/4_alignment/`
- `notebooks/5_hmm_selection/`
- `notebooks/6_hmm_final/`
- `notebooks/7_summaries/`
- `notebooks/8_figures/`
- `notebooks/9_tables/`
- `notebooks/_archive_raw_original_names/`

## Canonical final paper workflow

Unless otherwise noted, refactoring and code organization should follow this final paper workflow:

- no-lag fusion design
- minimum retained segment length = 15 TR
- 200 BOLD parcel features + 200 same-TR EEG parcel-power features
- 400 features per retained TR
- 3550 retained TRs
- 71 retained contiguous segments
- 124.25 usable minutes
- final selected model order = K = 3

---

## 2.1 Dataset and software

**Manuscript sections**
- Main Methods 2.1

**Current repo location**
- `README.md`
- `docs/repo_scope.md`
- `docs/reproducibility_notes.md`
- `config/subjects_runs.csv`

**Notes**
- dataset scope, included runs, and software environment should be documented here

---

## 2.2.1 Sensor-level EEG preprocessing and exclusion marking

**Manuscript sections**
- Main Methods 2.2.1
- Supplementary Methods 1.1
- Supplementary Results 2.1
- Supplementary Table S1

**Current repo location**
- `notebooks/1_eeg_sensor/`

**Current public-facing stage-1 entry files**
- `notebooks/1_eeg_sensor/10_eeg_prune_iclabel_and_export_clean_sets.m`
- `notebooks/1_eeg_sensor/11_brainstorm_exclusion_marking_manual.md`
- `notebooks/1_eeg_sensor/12_export_and_union_merge_brainstorm_exclusions.m`
- `notebooks/1_eeg_sensor/13_eeg_run_qc_and_table_s1.m`

**Stage-1 active helper location**
- `notebooks/1_eeg_sensor/helpers/`
  - `stage1_eeg_sensor_settings.m`
  - `run_iclabel_pruning_and_metadata_export.m`
  - `batch_export_brainstorm_exclusion_events.m`
  - `batch_merge_exclusion_union_masks.m`
  - `summarize_exclusion_union_qc.m`
  - `build_eeg_run_qc_gates_and_manifests.m`

**Preserved legacy implementations**
- the original `r01_*` helper files remain in the same stage folders for provenance and compatibility

**Expected content**
- ICLabel-based component pruning
- export of Brainstorm-facing cleaned EEG
- export of exclusion intervals from Brainstorm
- union-merging of `boundary` and `BAD` intervals
- run-level EEG QC summaries

**Manual/hybrid documentation**
- `docs/manual_steps.md`:
  - Section 1. Manual EEG exclusion marking in Brainstorm
  - Section 2. Brainstorm protocol setup and EEG import

**Notes**
- Brainstorm exclusion marking is manual/hybrid
- exclusions are limited to `boundary` and manually marked `BAD`
- `QRS` is not used as a censoring rule unless it falls inside an already excluded interval
- exported files may also contain `bad_boundary` labels
- original stage-1 drivers and provenance copies are preserved separately and are not the public-facing entry points

---

## 2.2.2 EEG source localization and atlas-aligned volumetric parcellation

**Manuscript sections**
- Main Methods 2.2.2
- Supplementary Methods 1.2
- Supplementary Results 2.2
- Supplementary Table S2
- Supplementary Fig. S1A,B

**Current repo location**
- `notebooks/2_eeg_source/`

**Current public-facing stage-2 entry files**
- `notebooks/2_eeg_source/20_prepare_schaefer200_atlas_for_brainstorm.ipynb`
- `notebooks/2_eeg_source/21_brainstorm_volume_source_and_atlas_import_manual.md`
- `notebooks/2_eeg_source/22_extract_volgrid_scouts_from_brainstorm_tess.m`
- `notebooks/2_eeg_source/24_qc_eeg_source_alignment_table_s2.m`

**Stage-2 helper files used here**
- `notebooks/2_eeg_source/batch_extract_volgrid_scouts_from_brainstorm_tess.m`
- preserved low-level implementation: `notebooks/2_eeg_source/r01_batch_make_volgrid_scouts_from_tess.m`
- preserved one-run implementation: `notebooks/2_eeg_source/r01_make_volgrid_scout_from_tess.m`

**Expected content**
- Brainstorm source workflow support files
- volumetric scout extraction
- atlas/grid compatibility checks
- source-space parcel coverage summaries

**Manual/hybrid documentation**
- `docs/manual_steps.md`:
  - Section 3. EEG volume source localization in Brainstorm
  - Section 4. Atlas preparation for EEG and BOLD parcellation
  - Section 5. Import atlas into Brainstorm as volume scouts
  - Section 6. Extract volume-grid scouts from Brainstorm tess files

**Notes**
- this stage is explicitly hybrid
- Brainstorm handles subject-specific anatomical alignment and volume-grid scout definition
- downstream scripts should not assume constant source-grid size across subjects
- the public Table-S2 support file is written by `24_qc_eeg_source_alignment_table_s2.m`

---

## 2.2.3 EEG parcel extraction and normalization

**Manuscript sections**
- Main Methods 2.2.3
- Supplementary Methods 1.3
- Supplementary Results 2.3
- Supplementary Table S3
- Supplementary Figs. S2-S4

**Current repo location**
- `notebooks/2_eeg_source/`

**Current public-facing stage-2 entry files**
- `notebooks/2_eeg_source/23_export_eeg_parcel_pc1_and_gain_normalize.m`
- `notebooks/2_eeg_source/25_qc_eeg_parcel_exports_table_s3_and_figures_s2_s4.ipynb`

**Stage-2 helper files used here**
- `notebooks/2_eeg_source/batch_export_eeg_parcel_pc_outputs.m`
- `notebooks/2_eeg_source/run_eeg_parcel_export_qc_summaries.m`
- preserved low-level export implementation: `notebooks/2_eeg_source/r01_batch_export_eeg_parcel_pc_v3.m`
- preserved one-run export implementation: `notebooks/2_eeg_source/r01_export_parcel_pc1_one_run_v3.m`
- preserved QC implementations:
  - `notebooks/2_eeg_source/r01_qc_v3_run_timeseries_and_gain_summary.m`
  - `notebooks/2_eeg_source/r01_qc_v3_sign_convention_parcelpc.m`
  - `notebooks/2_eeg_source/r01_qc_v3_pve1_hist_and_lowparcels.m`

**Expected content**
- parcel PC1 extraction
- parcel support thresholding
- deterministic sign fixing
- gain normalization
- PVE summaries and export QC

**Manual/hybrid documentation**
- `docs/manual_steps.md`:
  - Section 7. Hybrid EEG parcel PC extraction workflow
  - Section 8. QC and sanity-check outputs related to manual/hybrid EEG source workflow

**Notes**
- Brainstorm defines parcel membership on the subject-specific volume grid
- MATLAB/Python scripts perform parcel PC extraction, metadata export, and QC summaries
- `23_export_eeg_parcel_pc1_and_gain_normalize.m` preserves the current v3 helper behavior, including PC2 provenance outputs and the restored sample-time sidecar `*_time_sec.npy`
- `25_qc_eeg_parcel_exports_table_s3_and_figures_s2_s4.ipynb` expects the Stage-2 QC sidecars written by `run_eeg_parcel_export_qc_summaries.m`
- `25_qc_eeg_parcel_exports_table_s3_and_figures_s2_s4.ipynb` uses the current v3 CSV outputs and does not port the older MAT-schema-specific exploratory cells wholesale

---

## 2.3 BOLD preprocessing and parcel extraction

**Manuscript sections**
- Main Methods 2.3
- Supplementary Methods 1.4
- Supplementary Results 2.4
- Supplementary Tables S4-S5
- Supplementary Fig. S1C
- Supplementary Fig. S5

**Current repo location**
- `notebooks/3_bold/`

**Public-facing Stage-3 files**
- `notebooks/3_bold/30_map_schaefer200_to_bold_run_grids.ipynb`
- `notebooks/3_bold/31_export_bold_parcel_pc1_with_nuisance_regression.ipynb`
- `notebooks/3_bold/32_build_table_s4_bold_parcel_atlas_summary.ipynb`
- `notebooks/3_bold/33_build_table_s5_and_figure_s5_bold_qc.ipynb`

**Stage-3 helper modules**
- `notebooks/3_bold/stage3_bold_export_helpers.py`
- `notebooks/3_bold/stage3_bold_summary_helpers.py`

**Expected content**
- nuisance design construction
- nuisance regression
- spike regressor handling
- parcel extraction
- BOLD QC summaries

**Manual/hybrid documentation**
- `docs/manual_steps.md`:
  - Section 4. Atlas preparation for EEG and BOLD parcellation
  - Section 9. BOLD atlas alignment note

**Notes**
- BOLD is largely scripted in this repo
- atlas consistency with EEG is handled through shared template-space anatomical correspondence rather than voxel-for-voxel identity
- the standalone atlas-on-BOLD-grid branch is kept for atlas-preservation QC and overlay provenance
- the parcel-export path keeps the exporter-side `res-02` atlas branch authoritative
- Supplementary Table S4 is built by joining atlas QC outputs with exporter metadata
- Supplementary Figure S5 is reconstructed from exporter QC sidecars because one explicit original figure generator was not recovered

---

## 2.4 Timestamp-based alignment and construction of the final fusion observation matrix

**Manuscript sections**
- Main Methods 2.4
- Supplementary Methods 1.5
- Supplementary Tables S6-S7
- Main Results 3.2
- Figure 1

**Current repo location**
- `notebooks/4_alignment/`

**Current public-facing stage-4 entry files**
- `notebooks/4_alignment/40_align_eeg_to_bold_trs_and_build_keep_masks.ipynb`
- `notebooks/4_alignment/41_build_final_no_lag_fusion_observation_segments.ipynb`
- `notebooks/4_alignment/42_qc_alignment_tables_s6_s7_and_figure1_support.ipynb`

**Stage-4 helper files used here**
- `notebooks/4_alignment/stage4_alignment_helpers.py`
- `notebooks/4_alignment/stage4_segment_helpers.py`

**Expected content**
- raw-to-preprocessed EEG timeline reconciliation
- TR-level keep-mask construction
- same-TR no-lag observation matrix construction
- retained-segment export
- final retained dataset manifest

**Manual/hybrid documentation**
- upstream dependency on:
  - `docs/manual_steps.md` Section 1 (Brainstorm exclusion marking)
  - `docs/manual_steps.md` Section 7 (hybrid EEG parcel PC extraction workflow)

**Notes**
- this stage is scripted, but it depends on earlier manual/hybrid EEG exclusions and Brainstorm-derived source outputs
- the cleaned public-facing Stage-4 default is the canonical manuscript dataset:
  - no-lag design
  - minimum retained segment length = 15 TR
- `40_align_eeg_to_bold_trs_and_build_keep_masks.ipynb` preserves the original trigger-based alignment logic:
  - recurring `R128` events in raw and preprocessed EEG event TSVs
  - raw `S1` as the absolute anchor
  - accepted event-label columns `trial_type`, `value`, and `type`
  - accepted time columns `onset`, `start`, `start_sec`, `time`, `latency_sec`, and `latency`
- Stage 4 preserves glob-based run discovery, but the cleaned public notebook now writes an explicit missing-run audit so availability-based exclusions are visible
- lagged alignment outputs remain available for provenance, but `41_build_final_no_lag_fusion_observation_segments.ipynb` presents the no-lag `minlen15` branch as the main public output path

---

## 2.5.1 Fusion HMM model-order selection by LOSO-CV

**Manuscript sections**
- Main Methods 2.5.1
- Supplementary Methods 1.6-1.7
- Supplementary Results 2.6
- Supplementary Table S8
- Figure 2

**Current repo location**
- `notebooks/5_hmm_selection/`

**Current public-facing stage-5 entry files**
- `notebooks/5_hmm_selection/50_run_loso_k_sweep_model_selection.ipynb`
- `notebooks/5_hmm_selection/51_run_loso_shortlist_stability_checks.ipynb`
- `notebooks/5_hmm_selection/52_build_figure2_and_table_s8_model_selection_summary.ipynb`

**Stage-5 helper files used here**
- `notebooks/5_hmm_selection/stage5_hmm_selection_helpers.py`

**Expected content**
- K sweep over candidate model orders
- held-out free energy summaries
- 1-SE rule
- local-minimum shortlist
- cross-fold state matching
- stability metrics

**Notes**
- fully scripted stage
- should consume only the frozen final no-lag 15-TR-minimum dataset
- `50_run_loso_k_sweep_model_selection.ipynb` preserves the broad screening stage over `K = 2..12`
- the screening outputs may still leave `K=12` and lower-K local minima visible
- `51_run_loso_shortlist_stability_checks.ipynb` presents the manuscript-facing shortlist comparison as `K = 3` versus `K = 5`
- higher-K candidates remain part of the screening-stage provenance, but are not the main public comparison carried forward here
- the final manuscript choice remains `K = 3`

---

## 2.5.2 Final full-data K = 3 model fit

**Manuscript sections**
- Main Methods 2.5.2
- Supplementary Methods 1.8
- Supplementary Results 2.7
- Supplementary Table S9

**Current repo location**
- `notebooks/6_hmm_final/`

**Current public-facing stage-6 entry files**
- `notebooks/6_hmm_final/60_fit_final_k3_fusion_hmm.ipynb`
- `notebooks/6_hmm_final/61_review_final_k3_fit_qc_and_state_dynamics.ipynb`
- `notebooks/6_hmm_final/62_reconstruct_bold_state_networks_and_ranked_contrasts.ipynb`
- `notebooks/6_hmm_final/63_reconstruct_crossmodal_state_blocks_and_ranked_contrasts.ipynb`
- `notebooks/6_hmm_final/64_build_parcelized_cortical_state_maps.ipynb`
- optional: `notebooks/6_hmm_final/65_optional_export_figure4_figure5_panels.ipynb`

**Stage-6 helper files used here**
- `notebooks/6_hmm_final/stage6_hmm_final_helpers.py`

**Expected content**
- final retained dataset loading
- run-wise normalization
- PCA reduction
- seed screening and refit
- final model export
- final model QC

**Notes**
- fully scripted stage
- should use the canonical final dataset specification listed above
- `60_fit_final_k3_fusion_hmm.ipynb` consumes the canonical Stage-4 retained dataset directly through `segments_manifest.tsv`
- Stage 5 contributes the scientific decision to carry `K = 3` forward, but it is not a file dependency for the final fit
- the final-model artifact contract includes:
  - root-level provenance, preprocessing, seed-screening, and QC files
  - authoritative model outputs under `final/`
  - per-run decoded outputs under `gamma/` and `viterbi/`

---

## 2.6.1 Temporal summaries of the final-state solution

**Manuscript sections**
- Main Methods 2.6.1
- Supplementary Methods 1.9
- Main Results 3.3
- Figure 3
- Supplementary Fig. S6

**Current repo location**
- currently may be embedded in:
  - `notebooks/6_hmm_final/`
  - or later separated into `notebooks/7_summaries/`

**Current public-facing stage-6 entry file**
- `notebooks/6_hmm_final/61_review_final_k3_fit_qc_and_state_dynamics.ipynb`

**Expected content**
- fractional occupancy
- transition matrix
- dwell times
- gamma activation rasters

**Notes**
- the cleaned public review notebook is built around saved final-fit outputs rather than retraining the HMM
- per-run gamma and Viterbi outputs are part of the Stage-6 artifact contract, but some raster-style figure provenance still lives in preserved umbrella notebooks

---

## 2.6.2 Reconstruction of state-wise BOLD network organization

**Manuscript sections**
- Main Methods 2.6.2
- Supplementary Methods 1.10
- Main Results 3.4
- Figure 4
- Supplementary Table S10

**Current repo location**
- currently may be embedded in:
  - `notebooks/6_hmm_final/`
  - `notebooks/7_summaries/`
  - or `notebooks/8_figures/`

**Current public-facing stage-6 entry file**
- `notebooks/6_hmm_final/62_reconstruct_bold_state_networks_and_ranked_contrasts.ipynb`

**Expected content**
- BOLD covariance backprojection
- parcel-to-network aggregation
- ranked BOLD contrasts relative to S2

**Notes**
- the cleaned public notebook keeps the reference-state logic explicit
- by default it derives the reference state from the saved final FO vector unless the user overrides it

---

## 2.6.3 Reconstruction of descriptive cross-modal BOLD-EEG structure

**Manuscript sections**
- Main Methods 2.6.3
- Supplementary Methods 1.10
- Main Results 3.4
- Figure 5
- Supplementary Table S11

**Current repo location**
- currently may be embedded in:
  - `notebooks/6_hmm_final/`
  - `notebooks/7_summaries/`
  - or `notebooks/8_figures/`

**Current public-facing stage-6 entry file**
- `notebooks/6_hmm_final/63_reconstruct_crossmodal_state_blocks_and_ranked_contrasts.ipynb`

**Expected content**
- cross-modal covariance backprojection
- correlation-like matrix construction
- network-level summaries
- ranked contrasts relative to S2

**Interpretation note**
- these outputs are descriptive, not subject-level inferential statistics

**Notes**
- the cleaned public notebook keeps only the true cross-modal reconstruction half of the mixed provenance notebook
- the earlier alignment-style fusion-input illustration remains provenance/support logic rather than the main Stage-6 public path
- the preserved reference-state behavior usually derives the reference state from `best_seed.json`, but still keeps override and fallback behavior explicit

---

## 2.6.4 Parcelized cortical maps

**Manuscript sections**
- Main Methods 2.6.4
- Supplementary Methods 1.11
- Figure 6

**Current repo location**
- currently may be embedded in:
  - `notebooks/6_hmm_final/`
  - `notebooks/7_summaries/`
  - or `notebooks/8_figures/`

**Current public-facing stage-6 entry file**
- `notebooks/6_hmm_final/64_build_parcelized_cortical_state_maps.ipynb`

**Expected content**
- nodal mean connectivity
- S2 reference map
- S1-S2 and S3-S2 contrast maps
- atlas reference map

**Notes**
- the cleaned public notebook preserves the current active behavior where `REFERENCE_STATE = 2` is imposed explicitly
- this means the manuscript-facing S2 reference map is a presentation choice in this notebook, not a fresh inference from the saved final FO vector

---

## Supplementary figures and tables

**Current repo status**
- these may still be generated inside earlier notebooks rather than from separate dedicated folders

**Current repo folders**
- `notebooks/8_figures/`
- `notebooks/9_tables/`

**Note**
- it is acceptable for these folders to remain empty during early refactoring if figure/table generation is still embedded in the earlier method notebooks

---

## Manual steps documented separately

Detailed Brainstorm and hybrid procedures are documented in:

- `docs/manual_steps.md`

This includes:
1. Brainstorm exclusion marking
2. Brainstorm protocol setup and EEG import
3. EEG volume source localization in Brainstorm
4. Atlas preparation for EEG and BOLD parcellation
5. Import of atlas as Brainstorm volume scouts
6. Extraction of volume-grid scouts from tess files
7. Hybrid EEG parcel PC extraction workflow
8. EEG source-workflow QC and sanity-check outputs
9. BOLD atlas alignment note

---

## Refactor note

This map is a starter version for repo cleanup.

As notebooks are reviewed and consolidated, this file should be updated to include:
- exact notebook filenames
- final renamed public-facing filenames
- keep/merge/split/archive decisions
- figure/table generator locations
- links from each stage to the specific notebooks/scripts that implement it
