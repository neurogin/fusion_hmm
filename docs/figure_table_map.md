# Figure And Table Map

This document maps manuscript-facing figures and tables to the active public workflow.

It focuses on the public step files that remain in the release tree.

## Status Labels

- `Scripted`: generated directly by the public code path
- `Hybrid`: depends on scripted outputs plus manual or GUI preparation
- `Manual assembly`: the numerical content is scripted, but the final composite layout still requires manual design choices

## Main Figures

### Figure 1. Alignment and retained-dataset support

**Status**

- Hybrid or Manual assembly

**Public source files**

- [step40_align_eeg_to_bold_trs_and_build_keep_masks.ipynb](../notebooks/4_alignment/step40_align_eeg_to_bold_trs_and_build_keep_masks.ipynb)
- [step41_build_final_no_lag_fusion_observation_segments.ipynb](../notebooks/4_alignment/step41_build_final_no_lag_fusion_observation_segments.ipynb)
- [step42_qc_alignment_tables_s6_s7_and_figure1_support.ipynb](../notebooks/4_alignment/step42_qc_alignment_tables_s6_s7_and_figure1_support.ipynb)

**Notes**

- Step 42 writes Figure 1 support products and manuscript-support tables
- the final schematic layout still requires manual assembly choices

### Figure 2. LOSO model selection and shortlist stability

**Status**

- Scripted

**Public source file**

- [step52_build_figure2_and_table_s8_model_selection_summary.ipynb](../notebooks/5_hmm_selection/step52_build_figure2_and_table_s8_model_selection_summary.ipynb)

**Upstream requirements**

- Step 50 broad screening outputs
- Step 51 shortlist outputs

### Figure 3. Final K = 3 state-dynamics review

**Status**

- Scripted

**Public source file**

- [step61_review_final_k3_fit_qc_and_state_dynamics.ipynb](../notebooks/6_hmm_final/step61_review_final_k3_fit_qc_and_state_dynamics.ipynb)

### Figure 4. BOLD-state network organization

**Status**

- Scripted for the panel content
- Manual assembly for the final composite if needed

**Public source files**

- [step62_reconstruct_bold_state_networks_and_ranked_contrasts.ipynb](../notebooks/6_hmm_final/step62_reconstruct_bold_state_networks_and_ranked_contrasts.ipynb)
- optional: [step65_optional_export_figure4_figure5_panels.ipynb](../notebooks/6_hmm_final/step65_optional_export_figure4_figure5_panels.ipynb)

### Figure 5. Cross-modal BOLD-EEG block summaries

**Status**

- Scripted for the panel content
- Manual assembly for the final composite if needed

**Public source files**

- [step63_reconstruct_crossmodal_state_blocks_and_ranked_contrasts.ipynb](../notebooks/6_hmm_final/step63_reconstruct_crossmodal_state_blocks_and_ranked_contrasts.ipynb)
- optional: [step65_optional_export_figure4_figure5_panels.ipynb](../notebooks/6_hmm_final/step65_optional_export_figure4_figure5_panels.ipynb)

### Figure 6. Parcelized cortical maps

**Status**

- Scripted

**Public source file**

- [step64_build_parcelized_cortical_state_maps.ipynb](../notebooks/6_hmm_final/step64_build_parcelized_cortical_state_maps.ipynb)

## Supplementary Tables

### Table S1. EEG run-level preprocessing and exclusion summary

**Status**

- Hybrid

**Public source file**

- [step13_eeg_run_qc_and_table_s1.m](../notebooks/1_eeg_sensor/step13_eeg_run_qc_and_table_s1.m)

### Table S2. EEG source-grid atlas alignment summary

**Status**

- Hybrid

**Public source file**

- [step24_qc_eeg_source_alignment_table_s2.m](../notebooks/2_eeg_source/step24_qc_eeg_source_alignment_table_s2.m)

### Table S3. EEG parcel export summary

**Status**

- Hybrid

**Public source file**

- [step26_qc_eeg_parcel_exports_table_s3_and_figures_s2_s4.ipynb](../notebooks/2_eeg_source/step26_qc_eeg_parcel_exports_table_s3_and_figures_s2_s4.ipynb)

### Table S4. BOLD parcel and atlas summary

**Status**

- Scripted

**Public source file**

- [step32_build_table_s4_bold_parcel_atlas_summary.ipynb](../notebooks/3_bold/step32_build_table_s4_bold_parcel_atlas_summary.ipynb)

### Table S5. BOLD QC summary

**Status**

- Scripted

**Public source file**

- [step33_build_table_s5_and_figure_s5_bold_qc.ipynb](../notebooks/3_bold/step33_build_table_s5_and_figure_s5_bold_qc.ipynb)

### Table S6. Final dataset parameter summary

**Status**

- Scripted

**Public source file**

- [step42_qc_alignment_tables_s6_s7_and_figure1_support.ipynb](../notebooks/4_alignment/step42_qc_alignment_tables_s6_s7_and_figure1_support.ipynb)

### Table S7. Run-level final retained-dataset summary

**Status**

- Scripted

**Public source file**

- [step42_qc_alignment_tables_s6_s7_and_figure1_support.ipynb](../notebooks/4_alignment/step42_qc_alignment_tables_s6_s7_and_figure1_support.ipynb)

### Table S8. Model-selection summary

**Status**

- Scripted

**Public source file**

- [step52_build_figure2_and_table_s8_model_selection_summary.ipynb](../notebooks/5_hmm_selection/step52_build_figure2_and_table_s8_model_selection_summary.ipynb)

## Practical Note

Some manuscript figure layout and panel-composition choices still remain manual by design. The public workflow writes the numerical outputs and panel-support files, then makes those manual boundaries explicit instead of hiding them.
