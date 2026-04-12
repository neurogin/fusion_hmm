# Stage 2 Pass 2: Dependency Check for `notebooks/2_eeg_source/`

## Scope

This memo is the Stage-2 Pass-2 dependency and recovery check for:

- `notebooks/2_eeg_source/`

It starts from:

- `docs/refactor_plan_2_eeg_source.md`

and verifies the proposed cleaned public-facing stage-2 set against the actual current codebase.

This is a memo only. No stage-2 files were rewritten, moved, or implemented in this pass.

## Inputs reviewed

I reviewed:

- `docs/refactor_plan_2_eeg_source.md`
- `docs/methods_map.md`
- `docs/manual_steps.md`
- `docs/reproducibility_notes.md`
- `docs/figure_table_map.md`
- all 17 files currently in `notebooks/2_eeg_source/`
- archive provenance summaries under:
  - `notebooks/_archive_raw_original_names/2_eeg source atlas alignment/`
  - `notebooks/_archive_raw_original_names/4_eeg parcel pc extraction/`
- one downstream alignment notebook to verify whether stage-2 outputs are actually consumed later:
  - `notebooks/4_alignment/R01_PipelineA_align_trmask_lags_v3_gnorm_allTR_INTERMEDIATE.ipynb`

## Proposed cleaned public-facing files checked

The current proposed cleaned stage-2 public set is:

- `20_prepare_schaefer200_atlas_for_brainstorm.ipynb`
- `21_brainstorm_volume_source_and_atlas_import_manual.md`
- `22_extract_volgrid_scouts_from_brainstorm_tess.m`
- `23_export_eeg_parcel_pc1_and_gain_normalize.m`
- `24_qc_eeg_source_alignment_table_s2.m`
- `25_qc_eeg_parcel_exports_table_s3_and_figures_s2_s4.ipynb`

## Dependency status by proposed file

### `20_prepare_schaefer200_atlas_for_brainstorm.ipynb`

Existing support:

- `fetch_atlas.ipynb`
- `convert_atlas_to_text.ipynb`

Current dependency status:

- no missing internal helper
- the logic already exists
- the main issue is hard-coded WSL paths and lack of a public-facing explanation of where the fetched atlas and generated `.txt` should live

External dependencies already supported by code:

- Python
- `templateflow`
- `pandas`

Manual / hybrid notes that must stay explicit:

- `TEMPLATEFLOW_HOME` or equivalent cache location must be readable from the environment used to prepare the atlas
- the generated Brainstorm `.txt` must match the atlas NIfTI basename exactly

Verdict:

- not blocked
- needs a cleaned merged wrapper, not recovery

### `21_brainstorm_volume_source_and_atlas_import_manual.md`

Existing support:

- `docs/manual_steps.md` Sections 3-7
- `docs/reproducibility_notes.md`

Current dependency status:

- no missing helper because this is documentation-only by design
- the manual/hybrid workflow is already documented globally

What still needs to be made explicit in the future stage-local file:

- inputs from stage 1: cleaned EEGLAB `*_clean.set`
- atlas-prep outputs from step 20: Schaefer NIfTI + matching `.txt`
- required Brainstorm outputs for downstream scripts:
  - `results_MN_EEG_KERNEL_*.mat`
  - tess files containing the imported `Volume` atlas
- the known subject-specific note:
  - `sub-13_ses-01` used anatomy from `sub-13_ses-02`

Verdict:

- not blocked
- no recovery needed
- this is mainly a stage-local documentation wrapper around already known manual steps

### `22_extract_volgrid_scouts_from_brainstorm_tess.m`

Existing support:

- `r01_batch_make_volgrid_scouts_from_tess.m`
- `r01_make_volgrid_scout_from_tess.m`

Current dependency status:

- no missing internal helper
- the current scout extraction logic is present and appears sufficient

Hidden assumptions that the cleaned public file should declare explicitly:

- Brainstorm protocol folder structure is assumed:
  - `protocolRoot/data/**/results_MN_EEG_KERNEL_*.mat`
  - `protocolRoot/anat/sub-XX_ses-YY/`
- the atlas has already been imported into the tess file and stored as a `Volume ...` atlas entry
- the code prefers `tess_cortex_pial_low*_fix.mat`, then falls back to `tess_cortex_pial_low*.mat`
- the helper uses the first matching `Volume` atlas if the requested atlas substring is not found
- the standardized scout file is subject/session-level, not run-level

Manual / hybrid prerequisite that must stay explicit:

- Brainstorm atlas import as `Volume mask or atlas (dilated, MNI space)` must already be complete before this script is run

Manuscript-facing output status:

- the helper writes the scout files and can write a batch build summary
- there is still no cleaned public stage-2 entry point that turns this into a manuscript-facing Table-S2 support file by itself

Verdict:

- not blocked
- no helper recovery needed
- needs a cleaned wrapper and clearer documentation of the manual handoff

### `23_export_eeg_parcel_pc1_and_gain_normalize.m`

Existing support:

- `r01_batch_export_eeg_parcel_pc_v3.m`
- `r01_export_parcel_pc1_one_run_v3.m`

Current dependency status:

- no missing internal MATLAB helper for the core export logic
- the main computational path is intact

Helpers that look safe to preserve as-is:

- `r01_batch_export_eeg_parcel_pc_v3.m`
- `r01_export_parcel_pc1_one_run_v3.m`

Important hidden assumptions to declare later:

- EEG clean sets are named like `sub-XX_ses-YY_desc-ICRej70_clean.set`
- the exporter currently infers identity from `sub-` and `ses-` tokens only
- the scout file is expected at:
  - `protocolRoot/anat/sub-XX_ses-YY/scout_Schaefer2018_200_7N_dilated_MNI.mat`
- `ExpectedNScouts = 200`
- `StrictTessMatch = true`
- `MinVertices = 40`
- `SignConvention = maxabs`
- gain normalization is based on global median `kVertNorm_median`

Important external dependencies already supported by code:

- MATLAB
- EEGLAB (`pop_loadset`)
- optional external `writeNPY`

Important downstream note:

- `writeNPY` is optional in the current helper code, but it is not optional for the full pipeline in practice
- the stage-4 alignment notebook explicitly consumes `*_PC1_gnorm.npy`
- `writeNPY` is not vendored anywhere in this repo and must be declared later as an external dependency

True gap found here:

- I could not find any in-repo code that writes the companion `*_time_sec.npy` files expected by the downstream alignment notebook
- stage 4 currently raises an error if the `time_sec.npy` sidecar is missing
- this is not a missing helper inside `notebooks/2_eeg_source/`, but it is a real missing output producer relative to the later pipeline

How serious this gap is:

- it does not block writing a cleaned stage-2 public wrapper around the current exporter
- it does block claiming that stage-2 exports are fully alignment-ready without additional recovery or explicit reconstruction of the time vector

Verdict:

- core exporter logic is not blocked
- one real downstream output gap exists: no producer for `*_time_sec.npy`
- Pass 3 can still build the cleaned public entry file, but the alignment-facing time sidecar should be resolved or at least documented before calling stage 2 fully recovered

### `24_qc_eeg_source_alignment_table_s2.m`

Existing support:

- `r01_batch_make_volgrid_scouts_from_tess.m` summary table
- `r01_batch_export_eeg_parcel_pc_v3.m` coverage summary:
  - `batch_parcel_coverage_summary_v3.csv`
- archive provenance summary:
  - `notebooks/_archive_raw_original_names/2_eeg source atlas alignment/Table_S_EEG_atlas_alignment_summary.csv`

Current dependency status:

- no missing internal helper
- there is also no single current script that already emits a clean manuscript-facing Table-S2 file

What exists today:

- the underlying summary ingredients exist
- the archived Table-S2 provenance CSV gives a strong schema reference

What does not exist today:

- one clean active entry point that reads those ingredients and writes a manuscript-facing stage-2 Table-S2 support file

Verdict:

- not blocked by missing helpers
- requires a new cleaned wrapper / summarizer in Pass 3
- no special recovery beyond using existing summaries and archive provenance as schema guidance

### `25_qc_eeg_parcel_exports_table_s3_and_figures_s2_s4.ipynb`

Existing support:

- `r01_qc_v3_run_timeseries_and_gain_summary.m`
- `r01_qc_v3_sign_convention_parcelpc.m`
- `r01_qc_v3_pve1_hist_and_lowparcels.m`
- optional sidecar:
  - `r01_qc_v3_pve1_per_parcel_summary.m`
- the first figure-producing cell of `r01_figs_eeg_parcel_pc.ipynb`
- archive provenance summary:
  - `notebooks/_archive_raw_original_names/4_eeg parcel pc extraction/Table_S_EEG_parcel_extraction_summary.csv`

Helpers that look safe to preserve as-is:

- `r01_qc_v3_run_timeseries_and_gain_summary.m`
- `r01_qc_v3_sign_convention_parcelpc.m`
- `r01_qc_v3_pve1_hist_and_lowparcels.m`

Optional helper that is not required for the cleaned public set:

- `r01_qc_v3_pve1_per_parcel_summary.m`

Key mixed-logic warning:

- `r01_figs_eeg_parcel_pc.ipynb` should not be ported wholesale
- the first cell is useful and reads current QC CSV outputs
- later cells look older and assume MAT fields / filenames such as:
  - `*_parcelPC.mat`
  - `n_dipoles`
  - `parcel_names`
- those names do not match the current active v3 exporter schema, which uses:
  - `*_parcelPC_raw.mat`
  - `*_parcelPC_gnorm.mat`
  - `n_vertices`
  - `parcelNames`

Additional hidden assumption:

- `r01_qc_v3_sign_convention_parcelpc.m` depends on absolute file paths saved inside `diagOut`
- this QC is robust when run in the same environment immediately after export
- it is less portable if the exported MAT files are copied elsewhere without the original EEG, kernel, and scout files

Verdict:

- not blocked
- no missing internal helper recovery needed
- the safe path is to preserve the MATLAB QC helpers and port only the current figure logic, not the old HDF5 MAT-parsing cells

## Missing or ambiguous dependencies

### True missing dependency that should be treated as recovery work

- No missing internal stage-2 MATLAB helper was found.
- One real downstream output gap was found:
  - `*_time_sec.npy` is expected by the stage-4 alignment notebook
  - no active generator for this sidecar was found anywhere in the repo
  - no archive provenance file in this repo showed where it was created

This is the only item in this pass that looks like actual recovery rather than simple wrapper cleanup.

### Existing helpers already sufficient

- `r01_batch_make_volgrid_scouts_from_tess.m`
- `r01_make_volgrid_scout_from_tess.m`
- `r01_batch_export_eeg_parcel_pc_v3.m`
- `r01_export_parcel_pc1_one_run_v3.m`
- `r01_qc_v3_run_timeseries_and_gain_summary.m`
- `r01_qc_v3_sign_convention_parcelpc.m`
- `r01_qc_v3_pve1_hist_and_lowparcels.m`

### Optional helper or sensitivity sidecar

- `r01_qc_v3_pve1_per_parcel_summary.m`

### Archive / provenance-only material that should not shape the public refactor

- `r01_batch_export_parcel_pc_driver_v3.m`
- `r01_batch_make_volgrid_scouts_from_tess_driver.m`
- `r01_exporter_qc_driver.m`
- the later MAT-parsing cells of `r01_figs_eeg_parcel_pc.ipynb`
- archived summary CSVs used only as schema / provenance references

## Manual/hybrid prerequisites that must stay explicit

The following are genuine manual or hybrid prerequisites and should remain explicit in Pass 3:

- Brainstorm protocol setup and EEG import
- Brainstorm subject anatomy import
- Brainstorm MNI normalization
- Brainstorm BEM generation
- Brainstorm OpenMEEG head model and source reconstruction
- Brainstorm atlas import using `Volume mask or atlas (dilated, MNI space)`
- Brainstorm writing the imported atlas into the tess file before scout extraction
- manual production of the kernel results files:
  - `results_MN_EEG_KERNEL_*.mat`

These are already documented globally in `docs/manual_steps.md`, but the stage-local manual file should make the handoff between stage 1, atlas prep, Brainstorm, scout extraction, and parcel export very explicit.

## Misplaced or stage-contaminating files

The following files are stage-1 contamination inside `notebooks/2_eeg_source/`:

- `r01_eeg_runlevel_qc_gates_driver.m`
- `r01_eeg_runlevel_qc_gates.m`
- `r01_readtable.m`

These should not shape the stage-2 public-facing refactor.

## External/toolbox/environment dependencies to declare later

Dependencies directly supported by the current codebase:

- MATLAB
- EEGLAB
  - `pop_loadset`
- Brainstorm outputs on disk
  - Brainstorm is a manual/hybrid upstream dependency even where active MATLAB files do not call Brainstorm functions directly
- OpenMEEG
  - part of the manual Brainstorm source-model workflow
- SPM12
  - part of the manual Brainstorm MNI-normalization workflow
- Python
- `templateflow`
- `pandas`
- `matplotlib`
- external `writeNPY`
  - not vendored in this repo

Dependencies that should not be treated as core stage-2 public dependencies unless old exploratory logic is intentionally retained:

- `h5py`
- `numpy`

Those currently appear only in the older MAT-parsing cells of `r01_figs_eeg_parcel_pc.ipynb`.

## Outputs that are manuscript-facing in practice but not yet produced by one clean public entry point

- Table S2 support summary
  - active ingredients exist, but there is no clean current public writer
  - provenance reference exists in archive:
    - `Table_S_EEG_atlas_alignment_summary.csv`

- Table S3 support summary
  - active ingredients exist, but there is no clean current public writer
  - provenance reference exists in archive:
    - `Table_S_EEG_parcel_extraction_summary.csv`

- Figure S1A,B support
  - active stage-2 scripts can support the underlying coverage / scout outputs
  - final figure assembly still looks hybrid because it likely uses Brainstorm screenshots and cross-stage context

## Recovery actions required before Pass 3

### Required before calling stage 2 fully recovered

- Resolve the missing provenance for `*_time_sec.npy`
  - either recover the original producer
  - or document a verified reconstruction rule before the cleaned stage-2 export is presented as alignment-ready

### Not true recovery, but important Pass-3 wrapper tasks

- build one merged atlas-prep public notebook from the two current Python notebooks
- build one stage-local manual/hybrid markdown handoff file
- wrap the scout builder in a public entry script
- wrap the parcel exporter in a public entry script
- build a clean Table-S2 summarizer from existing coverage outputs
- build a clean Table-S3 / Figure-S2-S4 notebook from the safe QC helpers and the good figure logic only

## Bottom-line verdict

Stage 2 does **not** appear to be missing its core internal helpers.

The main need before Pass 3 is **not** large-scale recovery. The main need is:

- cleaned public wrappers
- explicit manual/hybrid handoff documentation
- pruning of mixed old/new logic in the QC figure notebook

There is, however, **one real recovery item**:

- no in-repo generator was found for the `*_time_sec.npy` sidecars expected by downstream alignment

So the practical verdict is:

- stage 2 is mostly ready for Pass 3 implementation
- `20`, `21`, `22`, `24`, and `25` are not blocked by missing helpers
- `23` is only partially blocked if it is expected to produce a fully downstream-ready stage-2 output set for alignment
- if Pass 3 proceeds now, the cleaned exporter wrapper should either:
  - include a documented resolution of the `time_sec.npy` gap, or
  - state clearly that this sidecar remains unresolved and will need a follow-up recovery step
