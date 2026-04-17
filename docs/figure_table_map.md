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
**Status:** Hybrid / Manual assembly
**Current source file(s):**
- `notebooks/4_alignment/40_align_eeg_to_bold_trs_and_build_keep_masks.ipynb`
- `notebooks/4_alignment/41_build_final_no_lag_fusion_observation_segments.ipynb`
- `notebooks/4_alignment/42_qc_alignment_tables_s6_s7_and_figure1_support.ipynb`

**Expected components:**
- alignment schematic
- representative aligned run
- retained-TR mask illustration

**Notes:**
- `42_qc_alignment_tables_s6_s7_and_figure1_support.ipynb` writes support plots and a manifest under `manuscript_support/figure1_support/`
- final Figure 1 still includes schematic and layout decisions that remain hybrid/manual rather than one-click scripted

---

## Figure 2. LOSO-CV model selection and shortlist stability
**Status:** active public-facing support notebook available
**Likely source folder(s):**
- `notebooks/5_hmm_selection/`
- possibly `notebooks/8_figures/` later

**Current public-facing source file(s):**
- `notebooks/5_hmm_selection/52_build_figure2_and_table_s8_model_selection_summary.ipynb`

**Expected components:**
- free-energy curve
- matched state-signature reproducibility
- occupancy summaries
- cross-fold similarity matrices

**Notes:**
- the public summary notebook reads broad K-sweep outputs plus shortlist-stability outputs
- the manuscript-facing comparison centers on `K=3` versus `K=5`
- higher-K candidates such as `K=12` remain visible in the screening-stage annotations and compact decision table

---

## Figure 3. Final-state dynamics of the full-data K = 3 fusion HMM
**Status:** active public-facing review notebook available
**Current source file(s):**
- `notebooks/6_hmm_final/61_review_final_k3_fit_qc_and_state_dynamics.ipynb`

**Expected components:**
- subject-level FO
- run-level FO
- transition matrix
- dwell summary
- state-signature similarity review

**Notes:**
- built from the saved final-fit outputs written by `60_fit_final_k3_fusion_hmm.ipynb`
- gamma-raster-style provenance still exists in preserved umbrella notebooks, but those do not define the main public Stage-6 workflow

---

## Figure 4. State-wise BOLD network organization
**Status:** active public-facing reconstruction notebook available
**Current source file(s):**
- `notebooks/6_hmm_final/62_reconstruct_bold_state_networks_and_ranked_contrasts.ipynb`
- optional panel-export support: `notebooks/6_hmm_final/65_optional_export_figure4_figure5_panels.ipynb`

**Expected components:**
- BOLD block matrices
- state contrasts relative to S2
- ranked network contrasts

**Notes:**
- the public reconstruction notebook derives the reference state from the saved final FO vector unless overridden
- the optional panel-export notebook is for manual composite assembly support only

---

## Figure 5. Descriptive cross-modal BOLD-EEG block structure
**Status:** active public-facing reconstruction notebook available
**Current source file(s):**
- `notebooks/6_hmm_final/63_reconstruct_crossmodal_state_blocks_and_ranked_contrasts.ipynb`
- optional panel-export support: `notebooks/6_hmm_final/65_optional_export_figure4_figure5_panels.ipynb`

**Expected components:**
- cross-modal block matrices
- state contrasts relative to S2
- ranked cross-modal contrasts

**Notes:**
- the cleaned public notebook keeps only the true cross-modal reconstruction half of the mixed provenance notebook
- the earlier alignment-style fusion-input illustration is not part of the main public Stage-6 path
- final composite layout remains manual/hybrid even though the panel content is scripted

---

## Figure 6. Parcelized cortical maps of dominant and contrast BOLD state organization
**Status:** active public-facing map notebook available
**Current source file(s):**
- `notebooks/6_hmm_final/64_build_parcelized_cortical_state_maps.ipynb`

**Expected components:**
- S2 nodal mean connectivity map
- S1-S2 contrast map
- S3-S2 contrast map
- atlas reference map

**Notes:**
- the current active map notebook preserves an explicitly imposed `S2` reference state
- nilearn may fetch fsaverage surfaces if they are not already cached locally

---

# Supplementary tables

## Table S1. Run-level summary of EEG retained after preprocessing and exclusion of marked intervals
**Status:** active public-facing build step
**Current source file(s):**
- `notebooks/1_eeg_sensor/eeg_run_qc_and_table_s1_13.m`

**Upstream requirements:**
- `notebooks/1_eeg_sensor/eeg_prune_iclabel_and_export_clean_sets_10.m`
- `notebooks/1_eeg_sensor/export_and_union_merge_brainstorm_exclusions_12.m`
- Brainstorm manual exclusion marking documented in `docs/manual_steps.md`

**Notes:**
- this stage writes run-level QC CSVs and include/exclude manifests that support the manuscript summary table
- any later manuscript-order formatting should remain explicit rather than being treated as a separate active Stage-9 workflow

---

## Table S2. EEG volumetric source-grid atlas alignment and parcel coverage
**Status:** Hybrid
**Current source file(s):**
- `notebooks/2_eeg_source/qc_eeg_source_alignment_table_s2_24.m`

**Upstream requirements:**
- `notebooks/2_eeg_source/extract_volgrid_scouts_from_brainstorm_tess_22.m`
- `notebooks/2_eeg_source/export_eeg_parcel_pc1_and_gain_normalize_23.m`
- Brainstorm manual/hybrid source workflow documented in `docs/manual_steps.md`

**Expected output:**
- `table_s2_eeg_atlas_alignment_summary.csv`

---

## Table S3. EEG parcel-feature export after volumetric source localization and atlas-aligned scout generation
**Status:** Hybrid
**Current source file(s):**
- `notebooks/2_eeg_source/25_qc_eeg_parcel_exports_table_s3_and_figures_s2_s4.ipynb`

**Upstream requirements:**
- `notebooks/2_eeg_source/export_eeg_parcel_pc1_and_gain_normalize_23.m`
- `notebooks/2_eeg_source/run_eeg_parcel_export_qc_summaries.m`

**Notes:**
- the public Stage-2 QC notebook reads the CSV sidecars written by the descriptive Stage-2 QC helper
- the preserved lower-level `r01_qc_v3_*` MATLAB implementations remain available underneath for provenance and compatibility

**Expected output:**
- `table_s3_eeg_parcel_extraction_summary.csv`

---

## Table S4. BOLD parcel extraction and atlas preservation
**Status:** active public-facing build step
**Current source file(s):**
- `notebooks/3_bold/30_map_schaefer200_to_bold_run_grids.ipynb`
- `notebooks/3_bold/31_export_bold_parcel_pc1_with_nuisance_regression.ipynb`
- `notebooks/3_bold/32_build_table_s4_bold_parcel_atlas_summary.ipynb`

**Notes:**
- built by joining exporter `dataset_index.csv` with atlas-preservation `qc_atlas_on_boldgrid_summary.csv`

---

## Table S5. Motion burden and nuisance-model composition for BOLD preprocessing
**Status:** active public-facing build step
**Current source file(s):**
- `notebooks/3_bold/31_export_bold_parcel_pc1_with_nuisance_regression.ipynb`
- `notebooks/3_bold/33_build_table_s5_and_figure_s5_bold_qc.ipynb`

**Notes:**
- numeric fields come from exporter `dataset_index.csv`
- the short `Note` column is kept explicit as provenance annotation rather than treated as a computed metric

---

## Table S6. Parameters defining the final no-lag, 15-TR-minimum fusion-HMM dataset
**Status:** active public-facing build step
**Current source file(s):**
- `notebooks/4_alignment/40_align_eeg_to_bold_trs_and_build_keep_masks.ipynb`
- `notebooks/4_alignment/42_qc_alignment_tables_s6_s7_and_figure1_support.ipynb`

**Upstream requirements:**
- cleaned Stage-2 EEG parcel NPY exports, including `*_PC1_gnorm.npy` and `*_time_sec.npy`
- cleaned Stage-3 BOLD parcel NPY exports
- raw and preprocessed EEG event TSVs containing recurring `R128` triggers

**Expected output:**
- `manuscript_support/table_s6_alignment_parameters.csv`

**Notes:**
- summarizes the parameters preserved in `qc/alignment_parameters_used.json`
- keeps the exposed-versus-active offset-jump-threshold mismatch explicit rather than silently harmonizing it

---

## Table S7. Run-level summary of the final no-lag, 15-TR-minimum fusion dataset
**Status:** active public-facing build step
**Current source file(s):**
- `notebooks/4_alignment/41_build_final_no_lag_fusion_observation_segments.ipynb`
- `notebooks/4_alignment/42_qc_alignment_tables_s6_s7_and_figure1_support.ipynb`

**Expected output:**
- `manuscript_support/table_s7_final_dataset_run_summary.csv`

**Notes:**
- reports per-run retained TRs, usable minutes, and contiguous-segment counts for the canonical no-lag `minlen15` dataset

---

## Table S8. Cross-validated fit and stability metrics used for final fusion-HMM model selection
**Status:** active public-facing support notebook available
**Likely source folder(s):**
- `notebooks/5_hmm_selection/`

**Current public-facing source file(s):**
- `notebooks/5_hmm_selection/52_build_figure2_and_table_s8_model_selection_summary.ipynb`

**Notes:**
- built from `summary_byK_selected.tsv`, `K_selection_recommendation.json`, and the active per-K shortlist outputs under `K03/` and `K05/`

---

## Table S9. Final-model fitting parameters and QC for the full-data K = 3 fusion HMM
**Status:** active public-facing fit and review notebooks available
**Current source file(s):**
- `notebooks/6_hmm_final/60_fit_final_k3_fusion_hmm.ipynb`
- `notebooks/6_hmm_final/61_review_final_k3_fit_qc_and_state_dynamics.ipynb`

**Notes:**
- built from root-level final-fit outputs such as `run_meta.json`, `preproc_meta.json`, `seed_candidates.tsv`, `topM_seeds.json`, `run_metrics.tsv`, `subject_metrics.tsv`, `dwell_from_A.tsv`, and `qc_summary.json`

---

## Table S10. Ranked BOLD network contrasts relative to S2
**Status:** active public-facing reconstruction notebook available
**Current source file(s):**
- `notebooks/6_hmm_final/62_reconstruct_bold_state_networks_and_ranked_contrasts.ipynb`

**Notes:**
- ranked BOLD network contrasts are written relative to the explicit reference state used by the physiology notebook
- by default that reference state is derived from the saved final FO vector unless overridden

---

## Table S11. Ranked cross-modal contrasts relative to S2
**Status:** active public-facing reconstruction notebook available
**Current source file(s):**
- `notebooks/6_hmm_final/63_reconstruct_crossmodal_state_blocks_and_ranked_contrasts.ipynb`

**Notes:**
- ranked cross-modal contrasts are written relative to the reference-state logic preserved from the mixed provenance notebook
- this logic usually derives the reference state from `best_seed.json`, but keeps override and fallback behavior explicit

---

# Supplementary figures

## Figure S1. Multimodal atlas alignment for EEG source space and BOLD voxel space
**Status:** Hybrid / Manual assembly
**Current source file(s):**
- `notebooks/2_eeg_source/21_brainstorm_volume_source_and_atlas_import_manual.md`
- `notebooks/2_eeg_source/extract_volgrid_scouts_from_brainstorm_tess_22.m`
- `notebooks/2_eeg_source/qc_eeg_source_alignment_table_s2_24.m`
- `notebooks/3_bold/30_map_schaefer200_to_bold_run_grids.ipynb`
- `notebooks/3_bold/32_build_table_s4_bold_parcel_atlas_summary.ipynb`

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

## Figure S5. BOLD QC reconstruction from exporter sidecars
**Status:** reconstructed from recovered provenance
**Current source file(s):**
- `notebooks/3_bold/31_export_bold_parcel_pc1_with_nuisance_regression.ipynb`
- `notebooks/3_bold/33_build_table_s5_and_figure_s5_bold_qc.ipynb`

**Notes:**
- built from exporter QC sidecars that are present in the recovered Stage-3 outputs
- the exact original final panel generator was not recovered, so the public repo keeps this reconstruction explicit

---

## Figure S6. Per-run gamma activation raster for the final K = 3 fusion HMM
**Status:** provenance notebook available, not yet a dedicated public Stage-6 entry point
**Current provenance source file(s):**
- `notebooks/6_hmm_final/PipelineE_K3_manuscript_figures_notebook_with_raster.ipynb`

**Notes:**
- gamma and Viterbi segment outputs are part of the Stage-6 final-fit artifact contract
- the cleaner public Stage-6 review notebook currently focuses on saved QC and state-dynamics summaries rather than reproducing the full raster notebook wholesale

---

# Refactor note

As Codex reviews the notebooks, this file should be updated to include for each asset:

- exact notebook/script filename
- exact input files
- exact output files
- whether final assembly is scripted, hybrid, or manual
- whether the asset is already reproducible from the current repo state
