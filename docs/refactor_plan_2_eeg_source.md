# Refactor Plan: `notebooks/2_eeg_source/`

## Scope

This memo covers the first-pass inventory and refactor plan for:

- `notebooks/2_eeg_source/`

This folder maps mainly to:

- Main Methods 2.2.2
- Main Methods 2.2.3
- Supplementary Methods 1.2
- Supplementary Methods 1.3
- Supplementary Results 2.2
- Supplementary Results 2.3
- Supplementary Table S2
- Supplementary Table S3
- Supplementary Figures S1A,B and S2-S4

The goal here is planning only. No files were rewritten or moved in this pass.

## Files inspected

I inspected 17 files in `notebooks/2_eeg_source/`:

1. `convert_atlas_to_text.ipynb`
2. `fetch_atlas.ipynb`
3. `r01_batch_export_eeg_parcel_pc_v3.m`
4. `r01_batch_export_parcel_pc_driver_v3.m`
5. `r01_batch_make_volgrid_scouts_from_tess_driver.m`
6. `r01_batch_make_volgrid_scouts_from_tess.m`
7. `r01_eeg_runlevel_qc_gates_driver.m`
8. `r01_eeg_runlevel_qc_gates.m`
9. `r01_export_parcel_pc1_one_run_v3.m`
10. `r01_exporter_qc_driver.m`
11. `r01_figs_eeg_parcel_pc.ipynb`
12. `r01_make_volgrid_scout_from_tess.m`
13. `r01_qc_v3_pve1_hist_and_lowparcels.m`
14. `r01_qc_v3_pve1_per_parcel_summary.m`
15. `r01_qc_v3_run_timeseries_and_gain_summary.m`
16. `r01_qc_v3_sign_convention_parcelpc.m`
17. `r01_readtable.m`

I also checked:

- `docs/manual_steps.md`
- `docs/methods_map.md`
- `docs/figure_table_map.md`
- manuscript and supplement reference text
- archive provenance files under:
  - `notebooks/_archive_raw_original_names/2_eeg source atlas alignment/`
  - `notebooks/_archive_raw_original_names/4_eeg parcel pc extraction/`
  - `notebooks/_archive_raw_original_names/1_eeg preprocessing/`

## High-level assessment

The folder contains five distinct clusters:

1. atlas preparation utilities
2. Brainstorm-derived volume-grid scout extraction
3. batch EEG parcel export and gain normalization
4. parcel-export QC and figure/table sidecars
5. misplaced stage-1 run-QC files

The core final-paper logic is present, but it is fragmented across:

- tiny hard-coded drivers
- reusable MATLAB helpers
- one mixed Python notebook that contains both useful figure logic and clearly older exploratory code

## Scientific settings that should be preserved during refactor

The public refactor should preserve the current stage-2 scientific behavior, including:

- manual Brainstorm source localization and atlas import as explicit hybrid steps
- linear MNI normalization (`maff` / `maff8`, SPM12)
- 3-layer BEM with 2432 vertices per layer
- skull thickness 4 mm
- conductivities 1.0 / 0.0125 / 1.0
- OpenMEEG BEM with adaptive integration
- isotropic 3-mm MRI volume grid
- current-density, unconstrained source reconstruction
- Brainstorm atlas import as `Volume mask or atlas (dilated, MNI space)`
- source-grid-aware scout extraction from Brainstorm tess files
- parcel support threshold `MinVertices = 40`
- deterministic sign convention `maxabs`
- gain-normalization basis `kVertNorm_median`
- current export behavior of saving both raw and gain-normalized parcel outputs

## File-by-file assessment

### 1. `convert_atlas_to_text.ipynb`

- Scientific purpose:
  - converts the Schaefer TemplateFlow atlas `.tsv` label table into the Brainstorm companion `.txt` label format
- Likely manuscript linkage:
  - Supplementary Methods 1.2
  - manual atlas-preparation support for Main Methods 2.2.2
- Inputs:
  - TemplateFlow Schaefer `.tsv`
  - matching Schaefer `.nii.gz`
- Outputs:
  - Brainstorm-readable `.txt` label file with the same basename as the atlas NIfTI
- Final-paper status:
  - core support / hybrid setup
- Overlap:
  - overlaps directly with `fetch_atlas.ipynb`
- Recommendation:
  - merge into one cleaned atlas-preparation notebook or script with `fetch_atlas.ipynb`
- Uncertainties:
  - current file uses hard-coded WSL paths and does not document where the generated `.txt` should live for public users

### 2. `fetch_atlas.ipynb`

- Scientific purpose:
  - fetches the Schaefer-200 / 7-network atlas NIfTI and `.tsv` from TemplateFlow
- Likely manuscript linkage:
  - Supplementary Methods 1.2
  - upstream support for Main Methods 2.2.2 and later BOLD atlas alignment
- Inputs:
  - Python environment with `templateflow`
- Outputs:
  - downloaded atlas files in the TemplateFlow cache
- Final-paper status:
  - core support / hybrid setup
- Overlap:
  - direct precursor to `convert_atlas_to_text.ipynb`
- Recommendation:
  - merge with `convert_atlas_to_text.ipynb`
- Uncertainties:
  - current notebook only prints fetched paths; it does not write a public-facing manifest or a user-oriented note about cache location

### 3. `r01_batch_export_eeg_parcel_pc_v3.m`

- Scientific purpose:
  - batch exporter for run-wise EEG parcel features from volumetric Brainstorm kernels plus subject-specific volume-grid scouts
  - also computes gain-normalized outputs and batch summaries
- Likely manuscript linkage:
  - Main Methods 2.2.3
  - Supplementary Methods 1.3
  - Supplementary Results 2.3
  - Supplementary Table S3
  - likely contributes stage-2 ingredients for Supplementary Table S2 and Figures S2-S4
- Inputs:
  - Brainstorm protocol root
  - `results_MN_EEG_KERNEL_*.mat`
  - cleaned EEG `.set` files from stage 1
  - standardized scout files such as `scout_Schaefer2018_200_7N_dilated_MNI.mat`
  - `writeNPY` on the MATLAB path if `.npy` export is desired
- Outputs:
  - `*_parcelPC_raw.mat`
  - `*_parcelPC_gnorm.mat`
  - `.npy` exports for PC1, PC2, PVE1, PVE2, masks, counts, parcel ids
  - `parcelNames_200.csv`
  - `batch_parcel_gain_summary_v3.csv`
  - `batch_parcel_coverage_summary_v3.csv`
  - `batch_parcel_manifest_v3.csv`
- Final-paper status:
  - core to the final paper
- Overlap:
  - overlaps with the driver and with downstream QC / figure code
- Recommendation:
  - keep the computational logic
  - rewrite as the main public-facing parcel-export entry point, with the current function retained as a helper if needed
- Uncertainties:
  - this file writes both PC1 and PC2 even though the manuscript-facing downstream feature is PC1; that looks intentional for QC/provenance and should not be dropped silently

### 4. `r01_batch_export_parcel_pc_driver_v3.m`

- Scientific purpose:
  - thin hard-coded driver for `r01_batch_export_eeg_parcel_pc_v3.m`
- Likely manuscript linkage:
  - same as the batch exporter, but only as a convenience launcher
- Inputs:
  - machine-specific Windows paths
- Outputs:
  - whatever the batch exporter writes
- Final-paper status:
  - provenance / convenience only
- Overlap:
  - fully overlaps with `r01_batch_export_eeg_parcel_pc_v3.m`
- Recommendation:
  - archive-only
  - replace with a cleaned public entry script later
- Uncertainties:
  - none; this is a standard hard-coded lab driver

### 5. `r01_batch_make_volgrid_scouts_from_tess_driver.m`

- Scientific purpose:
  - thin hard-coded driver for batch extraction of volume-grid scouts from Brainstorm tess files
- Likely manuscript linkage:
  - same stage as Main Methods 2.2.2 / Supplementary Methods 1.2
- Inputs:
  - machine-specific Brainstorm protocol root
- Outputs:
  - scout files via the underlying batch helper
  - `batch_volgrid_scout_build.csv`
- Final-paper status:
  - provenance / convenience only
- Overlap:
  - fully overlaps with `r01_batch_make_volgrid_scouts_from_tess.m`
- Recommendation:
  - archive-only
- Uncertainties:
  - none

### 6. `r01_batch_make_volgrid_scouts_from_tess.m`

- Scientific purpose:
  - finds Brainstorm inverse-kernel files, infers subject/session anatomy folders, reads `GridLoc` size from kernels, and writes scout files extracted from tess files in the correct source-grid index space
- Likely manuscript linkage:
  - Main Methods 2.2.2
  - Supplementary Methods 1.2
  - Supplementary Results 2.2
  - Supplementary Table S2
  - Supplementary Fig. S1A,B
- Inputs:
  - Brainstorm protocol root
  - kernel files under `data/**/results_MN_EEG_KERNEL_*.mat`
  - tess files under `anat/sub-*_ses-*`
- Outputs:
  - per-subject/session standardized scout files
  - summary table returned to caller
- Final-paper status:
  - core to the final paper
- Overlap:
  - overlaps with the driver and with `r01_make_volgrid_scout_from_tess.m`
- Recommendation:
  - keep as a batch helper behind a cleaned public-facing scout-extraction script
- Uncertainties:
  - output summary is useful, but not yet shaped as a manuscript-facing public table

### 7. `r01_eeg_runlevel_qc_gates_driver.m`

- Scientific purpose:
  - stage-1 run-level EEG QC driver, not source-space processing
- Likely manuscript linkage:
  - Main Methods 2.2.1
  - Supplementary Methods 1.1
  - Supplementary Table S1
- Inputs:
  - stage-1 raw / pruned / QC paths through `r01_params()`
- Outputs:
  - stage-1 run-level QC tables and manifests
- Final-paper status:
  - core stage-1 logic, but misplaced in this folder
- Overlap:
  - overlaps with cleaned stage-1 files under `notebooks/1_eeg_sensor/`
- Recommendation:
  - archive or relocate out of stage 2 in a later pass
  - do not include in the cleaned stage-2 public set
- Uncertainties:
  - none; this is clearly a folder-placement error

### 8. `r01_eeg_runlevel_qc_gates.m`

- Scientific purpose:
  - stage-1 sensor-level run QC and include/exclude manifest generation
- Likely manuscript linkage:
  - Main Methods 2.2.1
  - Supplementary Methods 1.1
  - Supplementary Results 2.1
  - Supplementary Table S1
- Inputs:
  - stage-1 raw EEG
  - stage-1 IC-pruned EEG
  - stage-1 exclusion union QC summary
- Outputs:
  - stage-1 run QC CSVs and manifests
- Final-paper status:
  - core stage-1 logic, but misplaced here
- Overlap:
  - overlaps with the cleaned stage-1 helper already placed under `notebooks/1_eeg_sensor/helpers/`
- Recommendation:
  - archive or relocate out of stage 2 in a later pass
  - not part of the stage-2 public set
- Uncertainties:
  - none

### 9. `r01_export_parcel_pc1_one_run_v3.m`

- Scientific purpose:
  - one-run implementation of parcel PC extraction from the Brainstorm imaging kernel and cleaned sensor covariance
  - enforces vertex-support thresholding, deterministic sign fixing, and output diagnostics
- Likely manuscript linkage:
  - Main Methods 2.2.3
  - Supplementary Methods 1.3
  - Supplementary Results 2.3
  - Supplementary Table S3
- Inputs:
  - run tag
  - cleaned stage-1 EEG `.set`
  - one Brainstorm kernel result file
  - one subject/session scout file
  - output directory
  - options structure
- Outputs:
  - `<runTag>_parcelPC_raw.mat`
  - `diagOut` struct with gain / sign / support / PVE summaries
- Final-paper status:
  - core to the final paper
- Overlap:
  - used by the batch exporter; mathematically central to stage 2
- Recommendation:
  - keep as a helper function
  - expose its scientific assumptions clearly in a higher-level public file rather than turning this low-level function into the public entry point
- Uncertainties:
  - none about purpose; the main refactor risk is accidental scientific change if this logic is rewritten too aggressively

### 10. `r01_exporter_qc_driver.m`

- Scientific purpose:
  - thin driver that runs the stage-2 parcel-export QC functions
- Likely manuscript linkage:
  - Supplementary Results 2.3
  - Supplementary Table S3
  - Supplementary Figs. S2-S4
- Inputs:
  - parcel export output directory
- Outputs:
  - whatever the four QC helpers write
- Final-paper status:
  - convenience / provenance only
- Overlap:
  - fully overlaps with the QC helper cluster
- Recommendation:
  - archive-only
  - replace with one cleaned public QC notebook or script
- Uncertainties:
  - none

### 11. `r01_figs_eeg_parcel_pc.ipynb`

- Scientific purpose:
  - mixed Python notebook for parcel-export QC figure and table generation
  - first cell plots stage-2 figure outputs from existing CSVs
  - later cells directly read exporter MAT files and patch parcel names
- Likely manuscript linkage:
  - Supplementary Fig. S2
  - Supplementary Fig. S3
  - Supplementary Fig. S4
  - possibly supplementary table support for Table S2 / Table S3
- Inputs:
  - QC CSVs from the MATLAB QC helpers
  - exporter MAT files
  - TemplateFlow atlas TSV
- Outputs:
  - figure PNGs:
    - `fig_gain_pc1_std_med_per_run_v3.png`
    - `fig_pve1_histogram_pooled_v3.png`
    - `fig_pve1_run_quantiles_v3.png`
    - `fig_lowpve_frequency_top20_v3.png`
    - `fig_sign_qc_passrate_v3.png`
  - extra CSV / figure side products from later cells
- Final-paper status:
  - partly core supplementary figure logic, partly exploratory / obsolete
- Overlap:
  - heavy overlap with the four QC helper functions
  - overlaps with the atlas-prep notebooks because the later cells repull TemplateFlow labels
- Recommendation:
  - split conceptually
  - keep the first-cell figure-generation logic by merging it into a cleaned public stage-2 QC notebook
  - archive the later exploratory MAT-parsing / name-patching cells unless they are deliberately rehabilitated
- Uncertainties:
  - the later cells appear to target older MAT naming / field conventions (`*_parcelPC.mat`, `n_dipoles`, `parcel_names`) rather than the current v3 schema, so they should not be treated as authoritative without validation

### 12. `r01_make_volgrid_scout_from_tess.m`

- Scientific purpose:
  - one-file helper that extracts the relevant Brainstorm volume atlas from a tess file and writes a compact scout file with `Scouts` and `TessNbVertices`
- Likely manuscript linkage:
  - Main Methods 2.2.2
  - Supplementary Methods 1.2
  - Supplementary Results 2.2
  - Supplementary Table S2
- Inputs:
  - tess file
  - output scout filename
  - atlas name substring
  - expected number of source-grid vertices
- Outputs:
  - one standardized scout MAT file
  - diagnostic struct
- Final-paper status:
  - core helper
- Overlap:
  - used by the batch scout builder
- Recommendation:
  - keep as helper
- Uncertainties:
  - none

### 13. `r01_qc_v3_pve1_hist_and_lowparcels.m`

- Scientific purpose:
  - summarizes PVE1 across runs and parcels, writes run quantiles, pooled histogram counts, and low-PVE parcel frequency summaries
- Likely manuscript linkage:
  - Supplementary Results 2.3
  - Supplementary Table S3
  - Supplementary Figs. S3-S4
- Inputs:
  - exporter MAT files (`*_parcelPC_gnorm.mat` preferred, raw fallback)
- Outputs:
  - `batch_pve1_run_quantiles_v3.csv`
  - `batch_pve1_histogram_v3.csv`
  - `batch_pve1_lowparcels_frequency_v3.csv`
  - `batch_pve1_lowparcels_frequency_named_v3.csv`
- Final-paper status:
  - core supplementary QC, with one exploratory side output
- Overlap:
  - overlaps with `r01_figs_eeg_parcel_pc.ipynb`
  - overlaps partially with `r01_qc_v3_pve1_per_parcel_summary.m`
- Recommendation:
  - keep as helper behind a cleaned public QC file
- Uncertainties:
  - the low-parcel-frequency output looks more like optional sensitivity auditing than a direct manuscript asset

### 14. `r01_qc_v3_pve1_per_parcel_summary.m`

- Scientific purpose:
  - computes across-run per-parcel PVE1 summaries and a sensitivity-oriented list of low-PVE parcels
- Likely manuscript linkage:
  - weak / indirect linkage to Supplementary Results 2.3
  - not clearly cited directly in the manuscript wording
- Inputs:
  - exporter MAT files
- Outputs:
  - `batch_pve1_per_parcel_summary_v3.csv`
  - `sensitivity_drop_parcels_by_pve1_v3.csv`
- Final-paper status:
  - exploratory or optional supplementary sensitivity sidecar
- Overlap:
  - overlaps with `r01_qc_v3_pve1_hist_and_lowparcels.m`
  - overlaps with the later exploratory cells in `r01_figs_eeg_parcel_pc.ipynb`
- Recommendation:
  - do not make this a public-facing main file in the first cleaned stage-2 set
  - either keep as an optional helper or leave archive/provenance-only until a direct manuscript use is documented
- Uncertainties:
  - it is useful, but its manuscript necessity is currently unclear

### 15. `r01_qc_v3_run_timeseries_and_gain_summary.m`

- Scientific purpose:
  - run-level QC for parcel time-series scale, NaN burden, gain-normalization consistency, and exporter diagnostics
- Likely manuscript linkage:
  - Supplementary Results 2.3
  - Supplementary Table S3
  - Supplementary Fig. S2
- Inputs:
  - exporter MAT files
- Outputs:
  - `qc_v3/qc_run_timeseries_gain_summary.csv`
- Final-paper status:
  - core supplementary QC
- Overlap:
  - overlaps with `r01_figs_eeg_parcel_pc.ipynb`
  - used by `r01_exporter_qc_driver.m`
- Recommendation:
  - keep as helper behind a cleaned public QC file
- Uncertainties:
  - none

### 16. `r01_qc_v3_sign_convention_parcelpc.m`

- Scientific purpose:
  - validates that saved parcel-PC1 signals are consistent with the exporter’s timecourse-based deterministic sign convention
- Likely manuscript linkage:
  - Supplementary Results 2.3
  - Supplementary Table S3
- Inputs:
  - exporter MAT files
  - saved paths inside `diagOut` back to EEG, kernels, and scouts
- Outputs:
  - `qc_v3_sign/qc_sign_v3_summary.csv`
  - `qc_v3_sign/qc_sign_v3_details.csv`
- Final-paper status:
  - core supplementary QC
- Overlap:
  - overlaps with `r01_figs_eeg_parcel_pc.ipynb`
  - used by `r01_exporter_qc_driver.m`
- Recommendation:
  - keep as helper behind a cleaned public QC file
- Uncertainties:
  - none

### 17. `r01_readtable.m`

- Scientific purpose:
  - scratch-style inspection of include-manifest continuity metrics
- Likely manuscript linkage:
  - stage-1 continuity review, not stage 2
- Inputs:
  - workspace variables `P` and `tag`
  - stage-1 include manifest
- Outputs:
  - none written; console display only
- Final-paper status:
  - exploratory / obsolete and misplaced
- Overlap:
  - overlaps loosely with stage-1 QC review
- Recommendation:
  - archive-only
- Uncertainties:
  - none

## Merge and split recommendations

### Clean merge candidates

- `fetch_atlas.ipynb` + `convert_atlas_to_text.ipynb`
  - these form one coherent atlas-preparation step

- `r01_batch_make_volgrid_scouts_from_tess_driver.m` + `r01_batch_make_volgrid_scouts_from_tess.m`
  - the driver should be absorbed into one public-facing scout-extraction script

- `r01_batch_export_parcel_pc_driver_v3.m` + `r01_batch_export_eeg_parcel_pc_v3.m`
  - same pattern: one public-facing entry file plus helpers

- `r01_exporter_qc_driver.m` + the QC helper cluster
  - these should become one cleaner public QC notebook or script

- the first, figure-oriented cell of `r01_figs_eeg_parcel_pc.ipynb` should merge into the cleaned public QC file

### Files that should remain helper-level rather than public entry points

- `r01_make_volgrid_scout_from_tess.m`
- `r01_export_parcel_pc1_one_run_v3.m`
- `r01_qc_v3_run_timeseries_and_gain_summary.m`
- `r01_qc_v3_sign_convention_parcelpc.m`
- `r01_qc_v3_pve1_hist_and_lowparcels.m`

### Files that should likely stay archive/provenance-only

- `r01_batch_export_parcel_pc_driver_v3.m`
- `r01_batch_make_volgrid_scouts_from_tess_driver.m`
- `r01_exporter_qc_driver.m`
- `r01_readtable.m`
- `r01_eeg_runlevel_qc_gates_driver.m`
- `r01_eeg_runlevel_qc_gates.m`

### File that likely needs conceptual splitting

- `r01_figs_eeg_parcel_pc.ipynb`
  - keep the figure-generation logic
  - archive or rewrite the older MAT-schema-specific table / patch cells

## Proposed cleaned GitHub-facing stage-2 file set

### Public-facing main files

- `20_prepare_schaefer200_atlas_for_brainstorm.ipynb`
  - merge `fetch_atlas.ipynb` and `convert_atlas_to_text.ipynb`
  - purpose: fetch the TemplateFlow atlas and generate the Brainstorm `.txt` label companion

- `21_brainstorm_volume_source_and_atlas_import_manual.md`
  - short stage-local manual/hybrid pointer to `docs/manual_steps.md` Sections 3-6
  - purpose: make the Brainstorm source-localization, MNI, BEM, source, and atlas-import steps explicit

- `22_extract_volgrid_scouts_from_brainstorm_tess.m`
  - public entry point for batch scout extraction after the manual Brainstorm steps are complete
  - should call the scout helpers and write a clean scout-build summary

- `23_export_eeg_parcel_pc1_and_gain_normalize.m`
  - public entry point for run-wise parcel export and gain normalization
  - should call the preserved batch exporter and clearly document inputs, outputs, and preserved options

- `24_qc_eeg_source_alignment_table_s2.m`
  - public source-localization / atlas-alignment QC entry point
  - should summarize scout/grid coverage into a clean manuscript-facing Table-S2 support file
  - should note that representative Figure S1A,B panels remain hybrid because Brainstorm screenshots / external overlays are manual

- `25_qc_eeg_parcel_exports_table_s3_and_figures_s2_s4.ipynb`
  - public parcel-export QC notebook
  - should run or consume the QC helpers, write a clean Table-S3 support summary, and generate the stage-2 QC figures

### Helper functions

- `r01_batch_make_volgrid_scouts_from_tess.m`
- `r01_make_volgrid_scout_from_tess.m`
- `r01_batch_export_eeg_parcel_pc_v3.m`
- `r01_export_parcel_pc1_one_run_v3.m`
- `r01_qc_v3_run_timeseries_and_gain_summary.m`
- `r01_qc_v3_sign_convention_parcelpc.m`
- `r01_qc_v3_pve1_hist_and_lowparcels.m`

### Optional helper or archive-only candidate

- `r01_qc_v3_pve1_per_parcel_summary.m`
  - keep only if the cleaned QC notebook explicitly documents it as an optional sensitivity sidecar
  - otherwise leave it archive-only

### Archive / provenance only

- `r01_batch_export_parcel_pc_driver_v3.m`
- `r01_batch_make_volgrid_scouts_from_tess_driver.m`
- `r01_exporter_qc_driver.m`
- `r01_figs_eeg_parcel_pc.ipynb` in its current mixed form
- `r01_eeg_runlevel_qc_gates_driver.m`
- `r01_eeg_runlevel_qc_gates.m`
- `r01_readtable.m`

## Mapping from current files to cleaned set

- atlas prep:
  - `fetch_atlas.ipynb`
  - `convert_atlas_to_text.ipynb`
  - new file: `20_prepare_schaefer200_atlas_for_brainstorm.ipynb`

- Brainstorm source / atlas manual step:
  - currently documented only in `docs/manual_steps.md`
  - new file: `21_brainstorm_volume_source_and_atlas_import_manual.md`

- scout extraction:
  - `r01_batch_make_volgrid_scouts_from_tess.m`
  - `r01_make_volgrid_scout_from_tess.m`
  - driver archived
  - new public entry: `22_extract_volgrid_scouts_from_brainstorm_tess.m`

- parcel export:
  - `r01_batch_export_eeg_parcel_pc_v3.m`
  - `r01_export_parcel_pc1_one_run_v3.m`
  - driver archived
  - new public entry: `23_export_eeg_parcel_pc1_and_gain_normalize.m`

- source / parcel QC:
  - `r01_qc_v3_run_timeseries_and_gain_summary.m`
  - `r01_qc_v3_sign_convention_parcelpc.m`
  - `r01_qc_v3_pve1_hist_and_lowparcels.m`
  - possibly `r01_qc_v3_pve1_per_parcel_summary.m`
  - figure cell from `r01_figs_eeg_parcel_pc.ipynb`
  - new public entries:
    - `24_qc_eeg_source_alignment_table_s2.m`
    - `25_qc_eeg_parcel_exports_table_s3_and_figures_s2_s4.ipynb`

## Risks and ambiguities

- The Brainstorm source-localization and atlas-import steps are genuinely manual / hybrid. The cleaned stage-2 set should not present them as fully scripted.

- `r01_figs_eeg_parcel_pc.ipynb` mixes at least two generations of logic:
  - a current figure layer that reads QC CSVs
  - older direct HDF5 MAT parsing that appears to assume pre-v3 naming / field conventions
  - this file should not be ported wholesale without pruning

- `Supplementary Table S2` and `Supplementary Table S3` currently appear to be assembled from intermediate summary CSVs rather than written by a single clean public entry point. The refactor should make this explicit instead of pretending that the current folder already has turnkey table writers.

- `r01_qc_v3_pve1_per_parcel_summary.m` is useful, but its direct manuscript role is uncertain. It is safer to treat it as optional unless the table / figure mapping later shows a concrete use.

- `r01_eeg_runlevel_qc_gates.m`, its driver, and `r01_readtable.m` are stage-1 materials that ended up in this folder. They should not shape the stage-2 public design.

- Representative `Figure S1A,B` content likely remains hybrid because it combines Brainstorm screenshots / atlas views with later downstream context. The scripted stage-2 files can support this figure, but probably do not fully generate it.

## Bottom-line recommendation

Stage 2 should be cleaned into:

- one merged atlas-prep notebook
- one manual/hybrid Brainstorm-source markdown pointer
- one public scout-extraction script
- one public parcel-export script
- one source-alignment QC script for Table S2 support
- one parcel-export QC notebook for Table S3 and Figures S2-S4

The mathematical exporter and scout logic should be preserved with minimal rewriting. The biggest clarity win will come from replacing the current hard-coded drivers and mixed QC notebook with a small number of explicit, manuscript-aligned public entry files.
