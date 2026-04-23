# Public Workflow Guide

This document gives the shortest complete path through the public repository.

## Run Order

Run the workflow in this order:

1. Stage 1: EEG sensor preprocessing and exclusion handling
2. Stage 2: EEG source localization and parcel export
3. Stage 3: BOLD parcel extraction and QC
4. Stage 4: EEG-BOLD alignment and retained-segment construction
5. Stage 5: LOSO model-order selection
6. Stage 6: final full-data `K = 3` fit and downstream reconstructions

## Stage 1

### Files

- [step10_eeg_prune_iclabel_and_export_clean_sets.m](../notebooks/1_eeg_sensor/step10_eeg_prune_iclabel_and_export_clean_sets.m)
- [step11_brainstorm_exclusion_marking_manual.md](../notebooks/1_eeg_sensor/step11_brainstorm_exclusion_marking_manual.md)
- [step12_export_and_union_merge_brainstorm_exclusions.m](../notebooks/1_eeg_sensor/step12_export_and_union_merge_brainstorm_exclusions.m)
- [step13_eeg_run_qc_and_table_s1.m](../notebooks/1_eeg_sensor/step13_eeg_run_qc_and_table_s1.m)

### What it produces

- Brainstorm-facing cleaned EEG sets
- exported Brainstorm exclusion TSV files
- merged exclusion-union masks
- Stage-1 QC tables and Table S1 support outputs

### Manual boundary

- Step 11 is manual or hybrid

## Stage 2

### Files

- [step20_prepare_schaefer200_atlas_for_brainstorm.ipynb](../notebooks/2_eeg_source/step20_prepare_schaefer200_atlas_for_brainstorm.ipynb)
- [step21_brainstorm_volume_source_and_atlas_import_manual.md](../notebooks/2_eeg_source/step21_brainstorm_volume_source_and_atlas_import_manual.md)
- [step22_extract_volgrid_scouts_from_brainstorm_tess.m](../notebooks/2_eeg_source/step22_extract_volgrid_scouts_from_brainstorm_tess.m)
- [step23_export_eeg_parcel_pc1_and_gain_normalize.m](../notebooks/2_eeg_source/step23_export_eeg_parcel_pc1_and_gain_normalize.m)
- [step24_qc_eeg_source_alignment_table_s2.m](../notebooks/2_eeg_source/step24_qc_eeg_source_alignment_table_s2.m)
- [step25_generate_eeg_parcel_export_qc_sidecars.m](../notebooks/2_eeg_source/step25_generate_eeg_parcel_export_qc_sidecars.m)
- [step26_qc_eeg_parcel_exports_table_s3_and_figures_s2_s4.ipynb](../notebooks/2_eeg_source/step26_qc_eeg_parcel_exports_table_s3_and_figures_s2_s4.ipynb)

### What it produces

- Brainstorm-compatible atlas label text
- scout MAT files
- parcel PC1 exports
- `*_PC1_gnorm.npy`
- `*_time_sec.npy`
- Stage-2 QC sidecars, Table S2 support, and Table S3 support

### Manual boundary

- Step 21 is manual or hybrid

### Important handoff

- Step 25 is the MATLAB QC-sidecar handoff
- Step 26 is the Python summary notebook
- Step 26 expects the parent `parcel_exports/` folder, not the nested `npy/` folder

## Stage 3

### Files

- [step30_map_schaefer200_to_bold_run_grids.ipynb](../notebooks/3_bold/step30_map_schaefer200_to_bold_run_grids.ipynb)
- [step31_export_bold_parcel_pc1_with_nuisance_regression.ipynb](../notebooks/3_bold/step31_export_bold_parcel_pc1_with_nuisance_regression.ipynb)
- [step32_build_table_s4_bold_parcel_atlas_summary.ipynb](../notebooks/3_bold/step32_build_table_s4_bold_parcel_atlas_summary.ipynb)
- [step33_build_table_s5_and_figure_s5_bold_qc.ipynb](../notebooks/3_bold/step33_build_table_s5_and_figure_s5_bold_qc.ipynb)

### What it produces

- atlas-on-grid QC support
- BOLD parcel time-series exports
- Table S4 support
- Table S5 support
- Figure S5 support

## Stage 4

### Files

- [step40_align_eeg_to_bold_trs_and_build_keep_masks.ipynb](../notebooks/4_alignment/step40_align_eeg_to_bold_trs_and_build_keep_masks.ipynb)
- [step41_build_final_no_lag_fusion_observation_segments.ipynb](../notebooks/4_alignment/step41_build_final_no_lag_fusion_observation_segments.ipynb)
- [step42_qc_alignment_tables_s6_s7_and_figure1_support.ipynb](../notebooks/4_alignment/step42_qc_alignment_tables_s6_s7_and_figure1_support.ipynb)

### Canonical branch

- `DATA_VARIANT = "intermediate"`
- `FEATURE_MODE = "nolags"`
- `MINLEN = 15`

### What it produces

- run-level keep masks
- retained segment exports
- Table S6 support
- Table S7 support
- Figure 1 support

### Important input contract

- raw and preprocessed EEG event TSVs
- recurring `R128` triggers
- first raw `S1` event for timing anchoring
- Stage-2 EEG parcel exports and Stage-3 BOLD parcel exports

## Stage 5

### Files

- [step50_run_loso_k_sweep_model_selection.ipynb](../notebooks/5_hmm_selection/step50_run_loso_k_sweep_model_selection.ipynb)
- [step51_run_loso_shortlist_stability_checks.ipynb](../notebooks/5_hmm_selection/step51_run_loso_shortlist_stability_checks.ipynb)
- [step52_build_figure2_and_table_s8_model_selection_summary.ipynb](../notebooks/5_hmm_selection/step52_build_figure2_and_table_s8_model_selection_summary.ipynb)

### Public logic

- Step 50 is the broad screening pass over `K = 2..12`
- Step 51 is the manuscript-facing shortlist comparison centered on `K = 3` and `K = 5`
- Step 52 builds Figure 2 and Table S8 support from the saved outputs

### Runtime note

- Stage 5 is GPU sensitive in normal use because it depends on TensorFlow and `osl_dynamics`

## Stage 6

### Files

- [step60_fit_final_k3_fusion_hmm.ipynb](../notebooks/6_hmm_final/step60_fit_final_k3_fusion_hmm.ipynb)
- [step61_review_final_k3_fit_qc_and_state_dynamics.ipynb](../notebooks/6_hmm_final/step61_review_final_k3_fit_qc_and_state_dynamics.ipynb)
- [step62_reconstruct_bold_state_networks_and_ranked_contrasts.ipynb](../notebooks/6_hmm_final/step62_reconstruct_bold_state_networks_and_ranked_contrasts.ipynb)
- [step63_reconstruct_crossmodal_state_blocks_and_ranked_contrasts.ipynb](../notebooks/6_hmm_final/step63_reconstruct_crossmodal_state_blocks_and_ranked_contrasts.ipynb)
- [step64_build_parcelized_cortical_state_maps.ipynb](../notebooks/6_hmm_final/step64_build_parcelized_cortical_state_maps.ipynb)
- optional: [step65_optional_export_figure4_figure5_panels.ipynb](../notebooks/6_hmm_final/step65_optional_export_figure4_figure5_panels.ipynb)

### Public logic

- Step 60 runs the final full-data `K = 3` fit
- Step 61 reviews the saved fit and state dynamics
- Step 62 reconstructs BOLD-state organization
- Step 63 reconstructs cross-modal state blocks
- Step 64 builds parcelized cortical maps
- Step 65 is optional panel-export support only

### Important note

Stage 6 depends directly on the canonical Stage-4 dataset. Stage 5 matters scientifically because it motivates the choice of `K = 3`, but the final fit does not require Stage-5 result files as an input contract.
