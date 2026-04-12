# Refactor Plan: `notebooks/4_alignment/`

## Scope

This memo is the Stage 4 Pass 1 inventory and refactor-planning note for:

- `notebooks/4_alignment/`

Stage 4 maps mainly to:

- Main Methods 2.4
- Supplementary Methods 1.5
- Supplementary Tables S6-S7
- Main Results 3.2
- Figure 1

This pass is planning only. No stage files were rewritten, moved, or deleted.

## Files inspected

Active files in `notebooks/4_alignment/`:

1. `R01_PipelineA_align_trmask_lags_v3_gnorm_allTR_INTERMEDIATE.ipynb`
2. `R01_PipelineB_build_X_segments_v3_gnorm_allTR_INTERMEDIATE_TOGGLE_LAGS.ipynb`
3. `R01_PipelineC_fusion_hmm_LOSO_K2-12_v3_UNIFIED_TOGGLE_LAGS_GPU_PATCH_Data2Dataset.ipynb`
4. `R01_PipelineC_fusion_hmm_LOSO_K2-12_v3_UNIFIED_TOGGLE_LAGS_GPU_PATCH_Data2Dataset-extend.ipynb`

Related provenance material reviewed:

- `notebooks/_archive_raw_original_names/5_multimodal alignment and X/R01_PipelineA_align_trmask_lags_v3_gnorm_allTR_INTERMEDIATE.md`
- `notebooks/_archive_raw_original_names/5_multimodal alignment and X/R01_PipelineB_build_X_segments_v3_gnorm_allTR_INTERMEDIATE_TOGGLE_LAGS.md`
- `notebooks/_archive_raw_original_names/5_multimodal alignment and X/Supplementary_Table_S1_alignment_parameters.csv`
- `notebooks/_archive_raw_original_names/5_multimodal alignment and X/Supplementary_Table_S2_run_level_summary.csv`
- `notebooks/_archive_raw_original_names/5_multimodal alignment and X/Supplementary_Table_S3_segment_manifest.csv`
- figure/schematic provenance images in the same archive folder

Supporting docs reviewed:

- `docs/methods_map.md`
- `docs/manual_steps.md`
- `docs/final_dataset_spec.md`
- `docs/reproducibility_notes.md`
- `docs/figure_table_map.md`
- `docs/refactor_plan_2_eeg_source.md`
- `docs/dependency_check_2_eeg_source.md`
- `docs/recovery_check_time_sec_sidecar.md`
- `docs/refactor_plan_3_bold.md`
- `docs/recovery_check_3_bold_provenance.md`

## High-level assessment

Stage 4 currently contains two real alignment notebooks and two later-stage notebooks that are misplaced in this folder.

The real Stage-4 core is:

- `Pipeline A`: raw-to-preprocessed EEG timeline reconciliation, exclusion remapping, TR-level keep masks, and per-run EEG/BOLD alignment products
- `Pipeline B`: final observation-segment construction from Pipeline A outputs

The folder also contains:

- two near-duplicate `Pipeline C` LOSO/HMM notebooks that belong to the next modeling stage, not to alignment

The main refactor need is not to invent new logic. It is to:

- separate true Stage-4 alignment from later HMM-selection code
- make the final manuscript dataset path explicit:
  - no-lag
  - 15-TR minimum
- keep lagged and `minlen10` branches as provenance or optional side outputs rather than the primary public story
- pull helper-like logic out of the large notebooks so the public entry files read more clearly

## File-by-file assessment

### 1. `R01_PipelineA_align_trmask_lags_v3_gnorm_allTR_INTERMEDIATE.ipynb`

**Scientific purpose**

- reconciles raw and preprocessed EEG timelines using recurring trigger events
- maps preprocessed exclusion intervals back onto the raw EEG timeline
- projects usable EEG coverage onto the BOLD TR grid
- applies the final hybrid TR-retention logic plus the EEG sample-completeness gate
- computes same-TR EEG parcel power and lagged EEG parcel-power features
- writes per-run alignment artifacts used by downstream fusion-HMM steps

**Likely manuscript linkage**

- Main Methods 2.4
- Supplementary Methods 1.5
- Figure 1 support
- upstream support for Table S6 and Table S7

**Inputs**

- raw EEG events TSVs
- preprocessed EEG events TSVs
- stage-1 exclusion-union TSVs
- stage-2 EEG parcel exports:
  - `*_PC1_gnorm.npy`
  - `*_time_sec.npy`
- stage-3 BOLD parcel exports:
  - `*_task-rest_parcel_pc1.npy`

**Outputs**

- per-run TR edges, coverage arrays, exclusion summaries, and keep masks
- per-run `bold_pc1.npy`
- per-run `eeg_power_tr.npy`
- per-run `eeg_power_tr_lags.npy`
- per-run `keep_center_*` masks and min-length summaries
- `align_trmask_lags_summary.csv`
- per-run diagnostic plots

**Assessment**

- core manuscript-relevant alignment logic: yes
- helper logic embedded: heavy
- QC/audit material embedded after the main run block: yes
- overlap: strong overlap with `Pipeline B`, which consumes its outputs directly

**Recommendation**

- keep as the main source for cleaned Stage-4 alignment logic
- split embedded utilities into helper-level code
- port only the true pipeline cells into the cleaned public notebook
- keep the appended ad hoc path checks and artifact-existence cells archive-only

**Important notes**

- the notebook header says it writes `bold_tr.npy`, but the code actually writes `bold_pc1.npy`
- it actively requires `*_time_sec.npy`, so Stage-2 sidecar generation is a real dependency, not optional in practice
- it computes both lagged and no-lag center masks from the same per-run outputs; the final paper uses the no-lag branch

### 2. `R01_PipelineB_build_X_segments_v3_gnorm_allTR_INTERMEDIATE_TOGGLE_LAGS.ipynb`

**Scientific purpose**

- loads Pipeline A per-run outputs
- applies keep masks plus a finite-row check
- concatenates BOLD and EEG blocks into per-segment observation matrices
- writes `segments_manifest.tsv` and per-run segment summaries

**Likely manuscript linkage**

- Main Methods 2.4
- Supplementary Methods 1.5
- Table S7 support
- final retained dataset manifest support

**Inputs**

- Pipeline A per-run outputs:
  - `bold_pc1.npy`
  - `eeg_power_tr.npy` or `eeg_power_tr_lags.npy`
  - `keep_center_minlen10_*` or `keep_center_minlen15_*`
  - `tr_edges_sec.npy`

**Outputs**

- one `.npy` observation matrix per retained segment
- `segments_manifest.tsv`
- `per_run_segments_minlen10.csv`
- `per_run_segments_minlen15.csv`
- global QC plots

**Assessment**

- core manuscript-relevant logic: yes
- public entry point candidate: yes
- helper logic embedded: moderate
- exploratory/downstream content mixed in: yes

**Recommendation**

- keep as the main source for cleaned segment-export logic
- make the cleaned public default explicitly:
  - `FEATURE_MODE = "nolags"`
  - `MINLEN = 15`
- keep lagged and `minlen10` builds available only as clearly labeled optional provenance branches
- do not port the preview `osl_dynamics.Data` cell as part of the core public Stage-4 workflow

**Important notes**

- the current defaults center the notebook on `MINLEN_PRIMARY = 10` with `MINLEN_OPTIONAL = 15`, which is not the final manuscript emphasis
- the final paper dataset is no-lag, 15-TR minimum, so the cleaned public wrapper should present that branch first without deleting the preserved optional branches

### 3. `R01_PipelineC_fusion_hmm_LOSO_K2-12_v3_UNIFIED_TOGGLE_LAGS_GPU_PATCH_Data2Dataset.ipynb`

**Scientific purpose**

- loads segment manifests and runs LOSO HMM model-order selection across `K`

**Likely manuscript linkage**

- not Stage 4
- belongs to later HMM-selection work

**Inputs**

- Stage-4 segment manifests and segment arrays
- TensorFlow / `osl-dynamics`

**Outputs**

- LOSO-CV results
- free-energy summaries
- model-order plots

**Assessment**

- misplaced file from the next stage: yes
- core Stage-4 logic: no
- overlap: almost complete overlap with the `-extend` variant

**Recommendation**

- treat as Stage-5/6 contamination in this folder
- do not include in the cleaned Stage-4 public set
- revisit during the HMM-selection planning pass

**Important notes**

- the title still says `K=2..8`, while the code uses `K_GRID = 2:12`
- that is later-stage provenance drift, not something to fix silently during Stage-4 refactoring

### 4. `R01_PipelineC_fusion_hmm_LOSO_K2-12_v3_UNIFIED_TOGGLE_LAGS_GPU_PATCH_Data2Dataset-extend.ipynb`

**Scientific purpose**

- same later-stage LOSO/HMM role as the other `Pipeline C` notebook

**Assessment**

- misplaced from a later stage: yes
- near-duplicate of the other `Pipeline C` notebook: yes
- not part of Stage-4 cleaned public set

**Recommendation**

- keep untouched as provenance for now
- do not let it shape the Stage-4 public refactor
- revisit during the Stage-5 planning pass to decide whether one `Pipeline C` copy is superseded

## Merge and split recommendations

### Merge recommendations

Do not merge `Pipeline A` and `Pipeline B` into one giant notebook.

They are sequential, but they serve distinct manuscript purposes:

- alignment and TR-mask construction
- final observation-segment construction

Keeping them separate will make the public workflow clearer.

### Split recommendations

Split helper-like logic out of the cleaned public notebooks into internal helper files:

- event parsing and run discovery
- raw-to-preprocessed timeline alignment
- exclusion remapping and union canonicalization
- TR-coverage and keep-mask utilities
- EEG power-per-TR and lag-feature builders
- contiguous-segment and manifest-writing utilities

This should improve readability without changing behavior.

### Material that should remain helper-level only

- all low-level interval and mask utilities from `Pipeline A`
- segment and finite-row utilities from `Pipeline B`

### Material that should remain archive/provenance-only

- appended ad hoc QC/path-audit cells in `Pipeline A`
- the `osl_dynamics.Data` preview cell in `Pipeline B`
- archive supplementary tables and schematic images
- both `Pipeline C` notebooks for the purposes of Stage 4

## Proposed cleaned GitHub-facing file set

Recommended public-facing Stage-4 set:

1. `40_align_eeg_to_bold_trs_and_build_keep_masks.ipynb`
   - main scripted alignment notebook
   - reads Stage-1 exclusions, Stage-2 EEG parcel exports, and Stage-3 BOLD parcel exports
   - writes per-run TR masks, same-TR EEG power, optional lagged EEG power, and alignment QC sidecars

2. `41_build_final_no_lag_fusion_observation_segments.ipynb`
   - main scripted final-dataset builder
   - defaults to the canonical manuscript dataset:
     - no-lag
     - 15-TR minimum
   - writes segment arrays, the retained-data manifest, and run-level summaries

3. `42_qc_alignment_tables_s6_s7_and_figure1_support.ipynb`
   - lightweight summary notebook
   - builds manuscript-facing parameter and run-summary tables from outputs of `40` and `41`
   - writes figure-support summaries and representative plots without pretending to automate the whole final figure assembly

Likely internal helper files, not public entry notebooks:

- `stage4_alignment_helpers.py`
- `stage4_segment_helpers.py`

## Mapping from current files to cleaned set

- `R01_PipelineA_align_trmask_lags_v3_gnorm_allTR_INTERMEDIATE.ipynb`
  - primary source for `40_align_eeg_to_bold_trs_and_build_keep_masks.ipynb`
  - embedded utility code should move into Stage-4 helper modules
  - appended audit/QC cells should either feed `42` selectively or stay archive-only

- `R01_PipelineB_build_X_segments_v3_gnorm_allTR_INTERMEDIATE_TOGGLE_LAGS.ipynb`
  - primary source for `41_build_final_no_lag_fusion_observation_segments.ipynb`
  - global summary plots and table-support pieces can feed `42`
  - the `osl_dynamics` preview cell should remain non-core provenance

- `R01_PipelineC_fusion_hmm_LOSO_K2-12_v3_UNIFIED_TOGGLE_LAGS_GPU_PATCH_Data2Dataset.ipynb`
  - not part of the Stage-4 cleaned set
  - carry forward conceptually to the HMM-selection stage

- `R01_PipelineC_fusion_hmm_LOSO_K2-12_v3_UNIFIED_TOGGLE_LAGS_GPU_PATCH_Data2Dataset-extend.ipynb`
  - not part of the Stage-4 cleaned set
  - carry forward conceptually to the HMM-selection stage

- archive markdown/tables/images under `5_multimodal alignment and X`
  - provenance-only references for cleaned documentation, table schema, and figure-support expectations

## Downstream integration assumptions to verify in Pass 2

### Stage-2 EEG-source assumptions

1. Stage 4 expects Stage-2 NPY sidecars, not only MAT files.
   - required in practice:
     - `*_PC1_gnorm.npy`
     - `*_time_sec.npy`

2. The EEG sample-time sidecar is required by current alignment code.
   - the now-documented rule is:
     - `single((0:nTime-1)' / srate)`
   - Pass 2 should verify that the cleaned Stage-2 exporter writes it in the exact location and naming pattern Stage 4 expects.

3. Stage 4 assumes EEG run tags of the form:
   - `sub-XX_ses-YY_desc-ICRej70_clean`
   - Pass 2 should confirm that this remains the cleaned Stage-2 public schema.

### Stage-3 BOLD assumptions

4. Stage 4 expects BOLD parcel PC1 files named like:
   - `sub-XX_ses-YY_task-rest_parcel_pc1.npy`
   - inside the exporter-style `parcel_pc1_v6/npy` branch

5. Stage 4 currently does not consume `dataset_index.csv`.
   - it discovers runs by globbing NPY files and intersecting available inputs
   - Pass 2 should verify whether that preserved behavior is acceptable, or whether important run-level information exists only in `dataset_index.csv`
   - this should be checked, not changed silently

6. Stage 4 strips BOLD filenames down to `sub-XX_ses-YY` for run matching.
   - Pass 2 should confirm that the final dataset truly has one resting-state run per session, so this does not collapse distinct runs

### Current path and folder-layout assumptions

7. The active notebooks assume older local derivative paths such as:
   - `02_derivatives/eeg_source/parcel_pc1/npy`
   - `02_derivatives/bold_parcel/parcel_pc1_v6/npy`
   - `02_derivatives/fusion_prep/align_trmask_lags/FINAL_v3_gnorm_allTR/intermediate`

8. The cleaned Stage-2 and Stage-3 public files already moved toward user-editable root variables.
   - Pass 2 should verify the exact cleaned output locations to avoid breaking Stage-4 integration during implementation

### Final-dataset and downstream-stage assumptions

9. The current notebooks still build broader branches:
   - lagged EEG features
   - `minlen10`
   - `minlen15`

10. The final paper dataset is:
    - no-lag
    - minimum segment length = 15 TR

11. Pass 2 should verify whether later stages consume only the canonical no-lag, `minlen15` branch or whether the lagged branch must remain exposed as an optional public provenance path.

## Risks and ambiguities

- `Pipeline A` mixes core logic with later ad hoc QC cells and path audits.
- `Pipeline B` presents `minlen10` as the primary build even though the final paper uses `minlen15`.
- the Stage-4 folder is contaminated by later HMM-selection notebooks
- `Pipeline A` notebook text says `bold_tr.npy`, but the code writes `bold_pc1.npy`
- current run discovery is availability-driven and may silently drop runs if a Stage-2 or Stage-3 file is missing or renamed
- archive provenance tables are numbered `S1-S3` locally, while the manuscript-facing repo docs now use `S6-S7` for this stage
- final Figure 1 likely remains partly hybrid/manual:
  - archive schematic images exist
  - notebook-generated QC plots exist
  - but a single final figure-assembly script is not evident from this folder

## Bottom-line recommendation

Stage 4 should be cleaned around:

- one main alignment notebook
- one main final no-lag segment-export notebook
- one light QC/table/figure-support notebook

The two `Pipeline C` notebooks should be treated as later-stage material and should not shape the Stage-4 public refactor.

Pass 2 should focus on exact Stage-2 and Stage-3 integration points:

- EEG `*_time_sec.npy` and run-tag schema
- BOLD parcel-export location and filename schema
- whether `dataset_index.csv` contains any run-level information that Stage 4 currently ignores
- whether later modeling stages need any lagged or `minlen10` branch preserved as a public optional path
