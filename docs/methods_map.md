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

**Stage-1 helper location**
- `notebooks/1_eeg_sensor/helpers/`
  - `r01_stage1_params.m`
  - `r01_export_bst_exclusions_Fevents.m`
  - `r01_merge_exclusions_union.m`
  - `r01_qc_excl_union_folder.m`
  - `r01_eeg_runlevel_qc_gates.m`

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
- `notebooks/2_eeg_source/r01_batch_make_volgrid_scouts_from_tess.m`
- `notebooks/2_eeg_source/r01_make_volgrid_scout_from_tess.m`

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
- `notebooks/2_eeg_source/r01_batch_export_eeg_parcel_pc_v3.m`
- `notebooks/2_eeg_source/r01_export_parcel_pc1_one_run_v3.m`
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

---

## 2.5.2 Final full-data K = 3 model fit

**Manuscript sections**
- Main Methods 2.5.2
- Supplementary Methods 1.8
- Supplementary Results 2.7
- Supplementary Table S9

**Current repo location**
- `notebooks/6_hmm_final/`

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

**Expected content**
- fractional occupancy
- transition matrix
- dwell times
- gamma activation rasters

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

**Expected content**
- BOLD covariance backprojection
- parcel-to-network aggregation
- ranked BOLD contrasts relative to S2

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

**Expected content**
- cross-modal covariance backprojection
- correlation-like matrix construction
- network-level summaries
- ranked contrasts relative to S2

**Interpretation note**
- these outputs are descriptive, not subject-level inferential statistics

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

**Expected content**
- nodal mean connectivity
- S2 reference map
- S1-S2 and S3-S2 contrast maps
- atlas reference map

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
