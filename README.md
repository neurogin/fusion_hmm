# fusion_hmm

Code and documentation for the manuscript:

**Fusion hidden Markov modeling reveals a reproducible shared state architecture in simultaneous resting-state EEG-fMRI**

## What This Repository Is

This repository is a manuscript-aligned reproducibility package for the published workflow. It is organized as a staged analysis sequence rather than as a general-purpose software library.

The public release contains only the active Stage 1 to Stage 6 workflow:

1. EEG sensor preprocessing and exclusion handling
2. EEG source localization and parcel export
3. BOLD parcel extraction and QC
4. EEG-BOLD alignment and retained-segment construction
5. LOSO model-order selection
6. Final full-data `K = 3` fitting and downstream reconstructions

## What Is Included

- active public `stepNN_*` workflow files in `notebooks/1_eeg_sensor/` through `notebooks/6_hmm_final/`
- active helper and backend files used by those public steps
- manual-step markdown files that remain real user steps
- manuscript-facing documentation under `docs/`
- Python environment templates in `environment.yml` and `requirements.txt`

## What Is Not Included

- raw EEG-fMRI data
- large derived outputs
- old development notebooks and internal provenance files
- scratch scripts, smoke-test artifacts, notebook checkpoints, and cache folders

## Read These First

- [docs/setup.md](docs/setup.md)
- [docs/public_workflow.md](docs/public_workflow.md)
- [docs/manual_steps.md](docs/manual_steps.md)
- [docs/methods_map.md](docs/methods_map.md)
- [docs/final_dataset_spec.md](docs/final_dataset_spec.md)
- [docs/reproducibility_notes.md](docs/reproducibility_notes.md)
- [docs/figure_table_map.md](docs/figure_table_map.md)

## Workflow At A Glance

### Stage 1. EEG sensor preprocessing and exclusion handling
- [step10_eeg_prune_iclabel_and_export_clean_sets.m](notebooks/1_eeg_sensor/step10_eeg_prune_iclabel_and_export_clean_sets.m)
- [step11_brainstorm_exclusion_marking_manual.md](notebooks/1_eeg_sensor/step11_brainstorm_exclusion_marking_manual.md)
- [step12_export_and_union_merge_brainstorm_exclusions.m](notebooks/1_eeg_sensor/step12_export_and_union_merge_brainstorm_exclusions.m)
- [step13_eeg_run_qc_and_table_s1.m](notebooks/1_eeg_sensor/step13_eeg_run_qc_and_table_s1.m)

### Stage 2. EEG source localization, parcel extraction, and source QC
- [step20_prepare_schaefer200_atlas_for_brainstorm.ipynb](notebooks/2_eeg_source/step20_prepare_schaefer200_atlas_for_brainstorm.ipynb)
- [step21_brainstorm_volume_source_and_atlas_import_manual.md](notebooks/2_eeg_source/step21_brainstorm_volume_source_and_atlas_import_manual.md)
- [step22_extract_volgrid_scouts_from_brainstorm_tess.m](notebooks/2_eeg_source/step22_extract_volgrid_scouts_from_brainstorm_tess.m)
- [step23_export_eeg_parcel_pc1_and_gain_normalize.m](notebooks/2_eeg_source/step23_export_eeg_parcel_pc1_and_gain_normalize.m)
- [step24_qc_eeg_source_alignment_table_s2.m](notebooks/2_eeg_source/step24_qc_eeg_source_alignment_table_s2.m)
- [step25_generate_eeg_parcel_export_qc_sidecars.m](notebooks/2_eeg_source/step25_generate_eeg_parcel_export_qc_sidecars.m)
- [step26_qc_eeg_parcel_exports_table_s3_and_figures_s2_s4.ipynb](notebooks/2_eeg_source/step26_qc_eeg_parcel_exports_table_s3_and_figures_s2_s4.ipynb)

### Stage 3. BOLD parcel extraction and QC
- [step30_map_schaefer200_to_bold_run_grids.ipynb](notebooks/3_bold/step30_map_schaefer200_to_bold_run_grids.ipynb)
- [step31_export_bold_parcel_pc1_with_nuisance_regression.ipynb](notebooks/3_bold/step31_export_bold_parcel_pc1_with_nuisance_regression.ipynb)
- [step32_build_table_s4_bold_parcel_atlas_summary.ipynb](notebooks/3_bold/step32_build_table_s4_bold_parcel_atlas_summary.ipynb)
- [step33_build_table_s5_and_figure_s5_bold_qc.ipynb](notebooks/3_bold/step33_build_table_s5_and_figure_s5_bold_qc.ipynb)

### Stage 4. EEG-BOLD alignment and retained-segment construction
- [step40_align_eeg_to_bold_trs_and_build_keep_masks.ipynb](notebooks/4_alignment/step40_align_eeg_to_bold_trs_and_build_keep_masks.ipynb)
- [step41_build_final_no_lag_fusion_observation_segments.ipynb](notebooks/4_alignment/step41_build_final_no_lag_fusion_observation_segments.ipynb)
- [step42_qc_alignment_tables_s6_s7_and_figure1_support.ipynb](notebooks/4_alignment/step42_qc_alignment_tables_s6_s7_and_figure1_support.ipynb)

### Stage 5. LOSO model-order selection
- [step50_run_loso_k_sweep_model_selection.ipynb](notebooks/5_hmm_selection/step50_run_loso_k_sweep_model_selection.ipynb)
- [step51_run_loso_shortlist_stability_checks.ipynb](notebooks/5_hmm_selection/step51_run_loso_shortlist_stability_checks.ipynb)
- [step52_build_figure2_and_table_s8_model_selection_summary.ipynb](notebooks/5_hmm_selection/step52_build_figure2_and_table_s8_model_selection_summary.ipynb)

### Stage 6. Final full-data `K = 3` fit and downstream reconstructions
- [step60_fit_final_k3_fusion_hmm.ipynb](notebooks/6_hmm_final/step60_fit_final_k3_fusion_hmm.ipynb)
- [step61_review_final_k3_fit_qc_and_state_dynamics.ipynb](notebooks/6_hmm_final/step61_review_final_k3_fit_qc_and_state_dynamics.ipynb)
- [step62_reconstruct_bold_state_networks_and_ranked_contrasts.ipynb](notebooks/6_hmm_final/step62_reconstruct_bold_state_networks_and_ranked_contrasts.ipynb)
- [step63_reconstruct_crossmodal_state_blocks_and_ranked_contrasts.ipynb](notebooks/6_hmm_final/step63_reconstruct_crossmodal_state_blocks_and_ranked_contrasts.ipynb)
- [step64_build_parcelized_cortical_state_maps.ipynb](notebooks/6_hmm_final/step64_build_parcelized_cortical_state_maps.ipynb)
- optional: [step65_optional_export_figure4_figure5_panels.ipynb](notebooks/6_hmm_final/step65_optional_export_figure4_figure5_panels.ipynb)

## Canonical Manuscript Path

The manuscript-facing default path is:

- Stage 4 canonical dataset branch: `intermediate + nolags + minlen15`
- Stage 5 model-selection narrative: broad `K = 2..12` screening, then a manuscript-facing shortlist comparison centered on `K = 3` and `K = 5`, with final choice `K = 3`
- Stage 6 final model: full-data `K = 3` fit plus the saved-output review and reconstruction steps

See [docs/final_dataset_spec.md](docs/final_dataset_spec.md) for the frozen dataset definition.

## Manual And Hybrid Boundaries

This repository is not fully script-only. Real manual or hybrid steps remain in the public workflow, especially:

- Brainstorm EEG exclusion marking
- Brainstorm source-model setup
- Brainstorm atlas import as volume scouts
- some final figure-panel assembly

Those boundaries are documented explicitly in [docs/manual_steps.md](docs/manual_steps.md).

## Software Summary

The workflow uses a mixed software stack:

- MATLAB for Stage 1 and core parts of Stage 2
- EEGLAB and ICLabel where the Stage-1 and Stage-2 MATLAB steps actually require them
- Brainstorm for the manual or hybrid EEG-source stages
- Python notebooks and helper modules for Stages 2 to 6
- TensorFlow and `osl_dynamics` for Stage 5 and the Stage-6 final fit

Setup details are in [docs/setup.md](docs/setup.md).

## Repository Layout

```text
fusion_hmm/
  README.md
  environment.yml
  requirements.txt
  .gitignore

  docs/
    setup.md
    public_workflow.md
    manual_steps.md
    methods_map.md
    final_dataset_spec.md
    reproducibility_notes.md
    figure_table_map.md
    github_release_checklist.md
    repo_scope.md

  notebooks/
    1_eeg_sensor/
    2_eeg_source/
    3_bold/
    4_alignment/
    5_hmm_selection/
    6_hmm_final/
```

## Before A Public Push

Use [docs/github_release_checklist.md](docs/github_release_checklist.md) for the final packaging checks.

Two release-level items still need manual confirmation outside the code itself:

- the repository license choice
- the author list and citation metadata for `CITATION.cff`
