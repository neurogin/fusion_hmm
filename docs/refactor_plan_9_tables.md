# Refactor Plan: `9_tables`

## Scope

This memo covers the current contents of `notebooks/9_tables/` only, while using the already-cleaned Stage-1 to Stage-6 notebooks, the archive folder, and current repo docs as context.

The goal of this pass is to determine whether Stage 9 currently contains a real table-generation layer, whether table logic is still embedded upstream, and whether this folder should gain cleaned public-facing table builders or remain empty for now.

## Files inspected

Active files in `notebooks/9_tables/`:

- none

Additional context reviewed because Stage 9 is currently empty:

- `docs/figure_table_map.md`
- `docs/methods_map.md`
- `docs/reproducibility_notes.md`
- `docs/refactor_plan_7_summaries.md`
- `docs/refactor_plan_8_figures.md`
- archived table-like outputs under `notebooks/_archive_raw_original_names/`

Archive table/provenance files observed:

- `notebooks/_archive_raw_original_names/2_eeg source atlas alignment/Table_S_EEG_atlas_alignment_summary.csv`
- `notebooks/_archive_raw_original_names/3_bold preprocessing atlas parcel/Table_S_BOLD_motion_nuisance_summary.csv`
- `notebooks/_archive_raw_original_names/3_bold preprocessing atlas parcel/Table_S_BOLD_parcel_atlas_summary.csv`
- `notebooks/_archive_raw_original_names/4_eeg parcel pc extraction/Table_S_EEG_parcel_extraction_summary.csv`
- `notebooks/_archive_raw_original_names/5_multimodal alignment and X/Supplementary_Table_S1_alignment_parameters.csv`
- `notebooks/_archive_raw_original_names/5_multimodal alignment and X/Supplementary_Table_S2_run_level_summary.csv`
- `notebooks/_archive_raw_original_names/5_multimodal alignment and X/Supplementary_Table_S3_segment_manifest.csv`
- `notebooks/_archive_raw_original_names/5_multimodal alignment and X/supplementary_tables_alignment_X.docx`
- `notebooks/_archive_raw_original_names/6_loso cv/fusion_hmm_K_selection_compact_table (1).csv`
- `notebooks/_archive_raw_original_names/6_loso cv/fusion_hmm_K_selection_table_S1_full_labels.csv`

## High-level assessment

`notebooks/9_tables/` is genuinely empty in the current repo snapshot.

That does not look like missing work by itself. It matches the repo policy already documented in:

- `docs/methods_map.md`
- `docs/reproducibility_notes.md`
- `AGENTS.md`

Those files explicitly allow `7_summaries/`, `8_figures/`, and `9_tables/` to remain empty while summary, figure, and table logic is still embedded in earlier stage notebooks.

At present, the real manuscript-table support logic already lives upstream in the cleaned stage notebooks:

- Stage 1 supports Table S1
- Stage 2 supports Tables S2 and S3
- Stage 3 supports Tables S4 and S5
- Stage 4 supports Tables S6 and S7
- Stage 5 supports Table S8
- Stage 6 supports Tables S9, S10, and S11

So Stage 9 is currently a conceptual table layer, not an implemented notebook layer.

My current recommendation is to avoid creating a duplicate Stage-9 wrapper layer right now.

## File-by-file assessment

### Files inside `notebooks/9_tables/`

There are no active files to classify.

### Closely related current table-support notebooks outside the folder

These are the files currently doing the table-stage work in practice:

| Current file | Scientific purpose | Likely manuscript linkage | Classification relative to Stage 9 | Recommendation |
| --- | --- | --- | --- | --- |
| `notebooks/1_eeg_sensor/13_eeg_run_qc_and_table_s1.m` | Builds run-level EEG retention and QC summary after preprocessing and exclusion handling. | Supplementary Table S1. | True table-support logic embedded in Stage 1. | Keep as the active public file for now. Do not duplicate into Stage 9 yet. |
| `notebooks/2_eeg_source/24_qc_eeg_source_alignment_table_s2.m` | Summarizes EEG source-grid atlas alignment and parcel coverage. | Supplementary Table S2. | True table-support logic embedded in Stage 2. | Keep as the active public file for now. |
| `notebooks/2_eeg_source/25_qc_eeg_parcel_exports_table_s3_and_figures_s2_s4.ipynb` | Summarizes EEG parcel export QC and writes Table S3 support. | Supplementary Table S3. | Mixed QC/figure/table notebook embedded in Stage 2. | Keep as the active public file for now. |
| `notebooks/3_bold/32_build_table_s4_bold_parcel_atlas_summary.ipynb` | Builds BOLD atlas-preservation table by joining atlas QC and exporter metadata. | Supplementary Table S4. | True table builder embedded in Stage 3. | Keep as the active public file for now. |
| `notebooks/3_bold/33_build_table_s5_and_figure_s5_bold_qc.ipynb` | Builds BOLD motion and nuisance summary table plus Figure S5 reconstruction. | Supplementary Table S5. | Mixed figure/table notebook embedded in Stage 3. | Keep as the active public file for now. |
| `notebooks/4_alignment/42_qc_alignment_tables_s6_s7_and_figure1_support.ipynb` | Builds alignment-parameter and final retained-dataset run-summary tables. | Supplementary Tables S6 and S7. | Mixed QC/figure/table notebook embedded in Stage 4. | Keep as the active public file for now. |
| `notebooks/5_hmm_selection/52_build_figure2_and_table_s8_model_selection_summary.ipynb` | Builds model-selection summary figure and compact decision table. | Supplementary Table S8. | Mixed figure/table notebook embedded in Stage 5. | Keep as the active public file for now. |
| `notebooks/6_hmm_final/60_fit_final_k3_fusion_hmm.ipynb` | Writes core final-fit provenance and QC tables. | Supplementary Table S9 support. | Core table-support logic embedded in Stage 6 fit notebook. | Keep as the active public file for now. |
| `notebooks/6_hmm_final/61_review_final_k3_fit_qc_and_state_dynamics.ipynb` | Reads final-fit outputs and helps assemble final-model QC summaries. | Supplementary Table S9 support; Figure 3 support. | Mixed review/table notebook embedded in Stage 6. | Keep as the active public file for now. |
| `notebooks/6_hmm_final/62_reconstruct_bold_state_networks_and_ranked_contrasts.ipynb` | Writes ranked BOLD network contrasts relative to the notebook’s explicit reference-state logic. | Supplementary Table S10. | True table-support logic embedded in Stage 6 reconstruction. | Keep as the active public file for now. |
| `notebooks/6_hmm_final/63_reconstruct_crossmodal_state_blocks_and_ranked_contrasts.ipynb` | Writes ranked cross-modal contrasts relative to the preserved reference-state logic. | Supplementary Table S11. | True table-support logic embedded in Stage 6 reconstruction. | Keep as the active public file for now. |

### Archive-only table material

The archive contains several CSV exports and one Word document of supplementary tables, but not a clean standalone Stage-9 notebook layer.

That suggests the historical workflow mostly treated tables as outputs of the scientific stage notebooks, not as a separate table-generation stage.

## Merge and split recommendations

### Merge recommendations

No merge inside `notebooks/9_tables/` is possible because the folder is empty.

At the repo level, no new Stage-9 merge is recommended right now. The current cleaned notebooks already separate table-support logic by scientific stage:

- EEG preprocessing tables in Stage 1
- EEG source and parcel tables in Stage 2
- BOLD tables in Stage 3
- alignment tables in Stage 4
- model-selection table in Stage 5
- final-model and reconstruction tables in Stage 6

Creating a new Stage-9 layer now would mostly duplicate those public notebooks.

### Split recommendations

No split inside `notebooks/9_tables/` is possible because the folder is empty.

Conceptually, the main split that already matters is:

- keep scientific-stage table generation in the stage notebooks where the relevant computations already occur
- keep any later manuscript-only formatting or column-relabeling work explicit if it exists
- avoid pretending that all manuscript tables come from one clean final table notebook when the current pipeline does not work that way

### Files that should likely remain helper-level only

There are no Stage-9 helper files yet.

If Stage 9 is ever populated later, it should probably stay thin and read already-derived table-support CSV or TSV outputs rather than rebuilding heavy scientific logic inside another layer.

## Proposed cleaned GitHub-facing file set

### Recommended now

No new public-facing Stage-9 files should be created in this pass.

Recommended public-facing Stage-9 set for the current repo state:

- none

Recommended Stage-9 behavior for now:

- keep `notebooks/9_tables/` empty
- treat the cleaned Stage-1 to Stage-6 notebooks as the active public table-support layer
- keep any manual manuscript-table formatting boundary explicit rather than hiding it inside a duplicate table stage

### Optional future extraction only if a stricter table layer becomes valuable

If the repo later decides to separate table generation from the scientific stage notebooks more aggressively, the most plausible future Stage-9 public files would be:

1. `90_build_table_s1_eeg_run_summary.m`
2. `91_build_table_s2_eeg_source_alignment.m`
3. `92_build_table_s3_eeg_parcel_export_summary.ipynb`
4. `93_build_table_s4_bold_parcel_atlas_summary.ipynb`
5. `94_build_table_s5_bold_motion_nuisance_summary.ipynb`
6. `95_build_table_s6_alignment_parameters.ipynb`
7. `96_build_table_s7_final_dataset_run_summary.ipynb`
8. `97_build_table_s8_model_selection_summary.ipynb`
9. `98_build_table_s9_final_model_qc.ipynb`
10. `99_build_tables_s10_s11_state_contrast_summaries.ipynb`

I do not currently recommend extracting those files now, because that would mostly wrap outputs that are already clearly tied to their scientific stage notebooks.

## Mapping from current files to cleaned set

### Current recommended state

| Stage-9 cleaned target | Current source |
| --- | --- |
| no Stage-9 public file yet | table logic remains in cleaned Stage-1 to Stage-6 notebooks |

### Optional future extraction mapping

| Possible future Stage-9 file | Current source |
| --- | --- |
| `90_build_table_s1_eeg_run_summary.m` | `notebooks/1_eeg_sensor/13_eeg_run_qc_and_table_s1.m` |
| `91_build_table_s2_eeg_source_alignment.m` | `notebooks/2_eeg_source/24_qc_eeg_source_alignment_table_s2.m` |
| `92_build_table_s3_eeg_parcel_export_summary.ipynb` | `notebooks/2_eeg_source/25_qc_eeg_parcel_exports_table_s3_and_figures_s2_s4.ipynb` |
| `93_build_table_s4_bold_parcel_atlas_summary.ipynb` | `notebooks/3_bold/32_build_table_s4_bold_parcel_atlas_summary.ipynb` |
| `94_build_table_s5_bold_motion_nuisance_summary.ipynb` | `notebooks/3_bold/33_build_table_s5_and_figure_s5_bold_qc.ipynb` |
| `95_build_table_s6_alignment_parameters.ipynb` | `notebooks/4_alignment/42_qc_alignment_tables_s6_s7_and_figure1_support.ipynb` |
| `96_build_table_s7_final_dataset_run_summary.ipynb` | `notebooks/4_alignment/42_qc_alignment_tables_s6_s7_and_figure1_support.ipynb` |
| `97_build_table_s8_model_selection_summary.ipynb` | `notebooks/5_hmm_selection/52_build_figure2_and_table_s8_model_selection_summary.ipynb` |
| `98_build_table_s9_final_model_qc.ipynb` | `notebooks/6_hmm_final/60_fit_final_k3_fusion_hmm.ipynb` and `61_review_final_k3_fit_qc_and_state_dynamics.ipynb` |
| `99_build_tables_s10_s11_state_contrast_summaries.ipynb` | `notebooks/6_hmm_final/62_reconstruct_bold_state_networks_and_ranked_contrasts.ipynb` and `63_reconstruct_crossmodal_state_blocks_and_ranked_contrasts.ipynb` |

## Downstream and upstream integration assumptions to verify in Pass 2

1. Confirm whether any future Stage-9 table layer can read only already-derived table-support CSV or TSV outputs, rather than core scientific artifacts directly.

2. Confirm which tables already have near-final machine-written CSV outputs and which still depend on light manual relabeling, note columns, Word formatting, or manuscript-order adjustments.

3. Check whether Table S8 should remain embedded in `52_build_figure2_and_table_s8_model_selection_summary.ipynb` rather than being split into a pure table notebook.

4. Check whether Table S9 support is really split across `60` and `61`, or whether one of those notebooks already contains the authoritative public export.

5. Check whether Tables S10 and S11 are best left inside the Stage-6 reconstruction notebooks because their meaning depends on the preserved reference-state logic.

6. Verify whether any archive-only CSVs or `supplementary_tables_alignment_X.docx` files reflect:
   - direct outputs of still-executable notebooks
   - or manually assembled manuscript tables that should remain provenance only

7. Confirm whether a future Stage-9 layer would improve clarity, or simply duplicate the cleaned upstream notebooks already listed in `docs/figure_table_map.md`.

## Risks and ambiguities

1. The folder is empty, so the biggest risk is over-correcting.
   - If we force content into Stage 9 now, we may create duplicate public entry points that are less clear than the existing stage notebooks.

2. The table boundary is soft throughout the repo.
   - Several active public notebooks are intentionally mixed figure/table or QC/table notebooks.
   - Forcing pure table notebooks may hide scientific context rather than improve readability.

3. Manual manuscript formatting may still exist for some tables.
   - The archive includes a Word document for alignment supplementary tables, which suggests at least some manuscript-table assembly or formatting was not purely notebook-native.

4. Provenance CSVs can be misleading.
   - A historical CSV export is not the same thing as a clean reproducible public table builder.
   - The public repo should not mistake “a table-shaped file exists in the archive” for “there was a standalone Stage-9 workflow.”

5. A future Stage-9 extraction could overlap heavily with:
   - Stage-5 `52_build_figure2_and_table_s8_model_selection_summary.ipynb`
   - Stage-6 `61`, `62`, and `63`
   - any later figure-only cleanup if Stage 8 is populated later

6. The reference-state logic for Tables S10 and S11 is not uniform.
   - any future standalone table layer would need to preserve the explicit differences already documented upstream rather than silently standardizing them

## Bottom-line recommendation

Stage 9 should remain empty for now.

That is the cleanest and most manuscript-aligned recommendation for the current repo state.

Reason:

- the real table-support logic already exists in the cleaned Stage-1 to Stage-6 notebooks
- `9_tables/` is explicitly allowed to remain empty during this refactor phase
- archive evidence suggests tables were mostly emitted from scientific stage notebooks, not from a separate table stage
- creating a new Stage-9 layer now would mostly duplicate existing public notebooks without improving reproducibility

So the best Stage-9 Pass-1 conclusion is:

- do not force a new Stage-9 implementation yet
- keep using the cleaned upstream notebooks as the active public table-support layer
- use Pass 2 to verify whether a later table-only extraction would truly simplify overlap with upstream stage notebooks and manual manuscript-table assembly, or merely duplicate it
