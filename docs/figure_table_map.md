# Figure and Table Map

This document maps manuscript figures and tables to the notebooks, scripts, intermediate outputs, and manual assembly steps that generate them.

It is being created during the repository refactor phase and may initially contain placeholders until the relevant notebooks are reviewed.

## Purpose

This file exists to make figure/table provenance explicit.

As notebooks are reviewed and cleaned, each entry should be updated to show:

- which notebook(s) or script(s) generate the asset
- what upstream inputs are required
- whether the asset is fully scripted, hybrid, or partly manual
- where the final output file is written
- whether the asset is a manuscript figure/table, a supplementary figure/table, or an intermediate QC product

## Status legend

- **Scripted** = generated fully from code
- **Hybrid** = generated from code plus manual setup or manually created inputs
- **Manual assembly** = final figure assembled outside a single executable script/notebook

---

# Main manuscript figures

## Figure 1. Timestamp-based alignment and construction of the final no-lag, 15-TR-minimum EEG-BOLD fusion dataset
**Status:** placeholder  
**Likely source folder(s):**
- `notebooks/4_alignment/`
- possibly `notebooks/8_figures/` later

**Expected components:**
- alignment schematic
- representative aligned run
- retained-TR mask illustration

**Notes:**
- may include manually assembled schematic elements in addition to scripted outputs

---

## Figure 2. LOSO-CV model selection and shortlist stability
**Status:** placeholder  
**Likely source folder(s):**
- `notebooks/5_hmm_selection/`
- possibly `notebooks/8_figures/` later

**Expected components:**
- free-energy curve
- matched state-signature reproducibility
- occupancy summaries
- cross-fold similarity matrices

---

## Figure 3. Final-state dynamics of the full-data K = 3 fusion HMM
**Status:** placeholder  
**Likely source folder(s):**
- `notebooks/6_hmm_final/`
- `notebooks/7_summaries/` later if separated

**Expected components:**
- subject-level FO
- run-level FO
- transition matrix
- kinetic summary diagram

---

## Figure 4. State-wise BOLD network organization
**Status:** placeholder  
**Likely source folder(s):**
- `notebooks/6_hmm_final/`
- `notebooks/7_summaries/`
- `notebooks/8_figures/`

**Expected components:**
- BOLD block matrices
- state contrasts relative to S2
- ranked network contrasts

---

## Figure 5. Descriptive cross-modal BOLD-EEG block structure
**Status:** placeholder  
**Likely source folder(s):**
- `notebooks/6_hmm_final/`
- `notebooks/7_summaries/`
- `notebooks/8_figures/`

**Expected components:**
- cross-modal block matrices
- state contrasts relative to S2
- ranked cross-modal contrasts

---

## Figure 6. Parcelized cortical maps of dominant and contrast BOLD state organization
**Status:** placeholder  
**Likely source folder(s):**
- `notebooks/6_hmm_final/`
- `notebooks/7_summaries/`
- `notebooks/8_figures/`

**Expected components:**
- S2 nodal mean connectivity map
- S1-S2 contrast map
- S3-S2 contrast map
- atlas reference map

---

# Supplementary tables

## Table S1. Run-level summary of EEG retained after preprocessing and exclusion of marked intervals
**Status:** placeholder  
**Likely source folder(s):**
- `notebooks/1_eeg_sensor/`
- possibly `notebooks/9_tables/` later

---

## Table S2. EEG volumetric source-grid atlas alignment and parcel coverage
**Status:** Hybrid
**Current source file(s):**
- `notebooks/2_eeg_source/24_qc_eeg_source_alignment_table_s2.m`

**Upstream requirements:**
- `notebooks/2_eeg_source/22_extract_volgrid_scouts_from_brainstorm_tess.m`
- `notebooks/2_eeg_source/23_export_eeg_parcel_pc1_and_gain_normalize.m`
- Brainstorm manual/hybrid source workflow documented in `docs/manual_steps.md`

**Expected output:**
- `table_s2_eeg_atlas_alignment_summary.csv`

---

## Table S3. EEG parcel-feature export after volumetric source localization and atlas-aligned scout generation
**Status:** Hybrid
**Current source file(s):**
- `notebooks/2_eeg_source/25_qc_eeg_parcel_exports_table_s3_and_figures_s2_s4.ipynb`

**Upstream requirements:**
- `notebooks/2_eeg_source/23_export_eeg_parcel_pc1_and_gain_normalize.m`
- current v3 QC CSV outputs written by:
  - `r01_qc_v3_run_timeseries_and_gain_summary.m`
  - `r01_qc_v3_sign_convention_parcelpc.m`
  - `r01_qc_v3_pve1_hist_and_lowparcels.m`

**Expected output:**
- `table_s3_eeg_parcel_extraction_summary.csv`

---

## Table S4. BOLD parcel extraction and atlas preservation
**Status:** placeholder  
**Likely source folder(s):**
- `notebooks/3_bold/`

---

## Table S5. Motion burden and nuisance-model composition for BOLD preprocessing
**Status:** placeholder  
**Likely source folder(s):**
- `notebooks/3_bold/`

---

## Table S6. Parameters defining the final no-lag, 15-TR-minimum fusion-HMM dataset
**Status:** placeholder  
**Likely source folder(s):**
- `notebooks/4_alignment/`
- `docs/final_dataset_spec.md`

---

## Table S7. Run-level summary of the final no-lag, 15-TR-minimum fusion dataset
**Status:** placeholder  
**Likely source folder(s):**
- `notebooks/4_alignment/`

---

## Table S8. Cross-validated fit and stability metrics used for final fusion-HMM model selection
**Status:** placeholder  
**Likely source folder(s):**
- `notebooks/5_hmm_selection/`

---

## Table S9. Final-model fitting parameters and QC for the full-data K = 3 fusion HMM
**Status:** placeholder  
**Likely source folder(s):**
- `notebooks/6_hmm_final/`

---

## Table S10. Ranked BOLD network contrasts relative to S2
**Status:** placeholder  
**Likely source folder(s):**
- `notebooks/6_hmm_final/`
- `notebooks/7_summaries/`

---

## Table S11. Ranked cross-modal contrasts relative to S2
**Status:** placeholder  
**Likely source folder(s):**
- `notebooks/6_hmm_final/`
- `notebooks/7_summaries/`

---

# Supplementary figures

## Figure S1. Multimodal atlas alignment for EEG source space and BOLD voxel space
**Status:** Hybrid / Manual assembly
**Current source file(s):**
- `notebooks/2_eeg_source/21_brainstorm_volume_source_and_atlas_import_manual.md`
- `notebooks/2_eeg_source/22_extract_volgrid_scouts_from_brainstorm_tess.m`
- `notebooks/2_eeg_source/24_qc_eeg_source_alignment_table_s2.m`
- `notebooks/3_bold/` outputs later in the pipeline

**Notes:**
- final assembly still likely uses Brainstorm screenshots and later BOLD-side outputs

---

## Figure S2. Run-wise median EEG parcel-PC1 scale after gain normalization
**Status:** Hybrid
**Current source file(s):**
- `notebooks/2_eeg_source/25_qc_eeg_parcel_exports_table_s3_and_figures_s2_s4.ipynb`

**Expected output:**
- `fig_s2_gain_pc1_scale_after_gnorm.png`

---

## Figure S3. Pooled distribution of PVE1 across runs and parcels
**Status:** Hybrid
**Current source file(s):**
- `notebooks/2_eeg_source/25_qc_eeg_parcel_exports_table_s3_and_figures_s2_s4.ipynb`

**Expected output:**
- `fig_s3_pve1_histogram_pooled.png`

---

## Figure S4. Run-wise PVE1 quantiles
**Status:** Hybrid
**Current source file(s):**
- `notebooks/2_eeg_source/25_qc_eeg_parcel_exports_table_s3_and_figures_s2_s4.ipynb`

**Expected output:**
- `fig_s4_pve1_run_quantiles.png`

---

## Figure S5. Representative run-level example of BOLD parcel time series after nuisance regression
**Status:** placeholder  
**Likely source folder(s):**
- `notebooks/3_bold/`

---

## Figure S6. Per-run gamma activation raster for the final K = 3 fusion HMM
**Status:** placeholder  
**Likely source folder(s):**
- `notebooks/6_hmm_final/`
- `notebooks/7_summaries/`

---

# Refactor note

As Codex reviews the notebooks, this file should be updated to include for each asset:

- exact notebook/script filename
- exact input files
- exact output files
- whether final assembly is scripted, hybrid, or manual
- whether the asset is already reproducible from the current repo state
