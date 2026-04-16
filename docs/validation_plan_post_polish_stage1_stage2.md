# Post-Polish Validation Plan for Stage 1 and Stage 2

## Scope

This memo is a practical manual validation checklist for the highest-risk repo changes made during the recent polish pass.

It focuses on the active public-facing workflow and helper layer for:

- `notebooks/1_eeg_sensor/`
- `notebooks/2_eeg_source/`

The goal is not to rerun the full project. The goal is to run a small number of targeted smoke tests that would quickly reveal whether the new public/helper layer still produces the same practical outputs and still matches downstream expectations.

This plan assumes you will run the tests on your own machine with your own local data paths and Brainstorm installation.

## Highest-Risk Changes

The highest-risk changes are the ones that rewired the active public-facing scripts to a new helper layer.

### Stage 1 risk points

- The public entry scripts now read settings from `notebooks/1_eeg_sensor/helpers/stage1_eeg_sensor_settings.m`.
- The public entry scripts now call descriptive wrappers instead of directly calling the older `r01_*` helpers.
- Path resolution and helper lookup are now handled one layer earlier than before.
- The compatibility wrapper `r01_stage1_params.m` now sits behind the public settings layer.

### Stage 2 risk points

- The public entry scripts now call wrapper helpers for scout extraction and parcel export.
- The parcel-export QC path is now exposed through `run_eeg_parcel_export_qc_summaries.m`.
- The public QC notebook now expects the new helper path to be the way users produce its prerequisite QC files.
- The verified `*_time_sec.npy` sidecar must still be written correctly, because Stage 4 depends on it directly.

### Lower-risk changes

The Stage 3 to Stage 6 changes were mostly naming and documentation improvements. Those are not the main validation target here. For those later stages, import/path sanity is more important than a full rerun at this point.

## Recommended Validation Order

Run the checks in this order:

1. Stage-1 path and helper sanity only
2. Stage-1 one-run ICLabel pruning smoke test
3. Stage-1 one-run Brainstorm exclusion export and union-merge smoke test
4. Stage-1 one-run QC / Table S1 support smoke test
5. Stage-2 one-run scout extraction smoke test
6. Stage-2 one-run parcel export smoke test
7. Stage-2 one-run QC / Table S2 / Table S3 support smoke test
8. Fast downstream compatibility check against Stage-4 expectations

This order gives you early failure if the rewired helper layer broke path handling, filenames, or schemas.

## Minimum Dataset Subset to Use

Use the smallest subset that still exercises the full public path:

- one subject-session-run with a known-good preprocessed EEGLAB `.set` file
- the matching Brainstorm raw link / event export context for that same run
- the matching Stage-2 Brainstorm source and scout context for that same run

Best choice:

- one run that you know previously made it all the way through Stage 4
- one run that already has an older known-good output bundle you can compare against

If possible, use a run with:

- a non-empty ICLabel rejection set
- at least one Brainstorm BAD or boundary exclusion interval
- a valid Stage-2 kernel and atlas scout file

That avoids a misleading "pass" on a trivial run that does not stress the real logic.

## Stage-1 Smoke Tests

### 1. Path and helper sanity check

#### Script to check

- `notebooks/1_eeg_sensor/10_eeg_prune_iclabel_and_export_clean_sets.m`
- `notebooks/1_eeg_sensor/12_export_and_union_merge_brainstorm_exclusions.m`
- `notebooks/1_eeg_sensor/13_eeg_run_qc_and_table_s1.m`

#### What to do

- Open the scripts and set the editable roots in `stage1_eeg_sensor_settings.m`.
- Confirm that the raw EEGLAB input folder, Brainstorm database root, derivative roots, and QC roots all point where you expect.
- Confirm that MATLAB can resolve the helper folder and the legacy helper folder.

#### Must-check signal

- Each script starts without "function not found" errors.
- The helper wrappers resolve successfully.
- The scripts show the expected folder roots before any real processing starts.

#### Fast-fail sign

- Missing helper errors for any of:
  - `run_iclabel_pruning_and_metadata_export`
  - `batch_export_brainstorm_exclusion_events`
  - `batch_merge_exclusion_union_masks`
  - `summarize_exclusion_union_qc`
  - `build_eeg_run_qc_gates_and_manifests`

### 2. Cleaned EEG export smoke test

#### Script to run

- `notebooks/1_eeg_sensor/10_eeg_prune_iclabel_and_export_clean_sets.m`

#### Smallest practical data subset

- one known-good `.set` file

#### Must-check outputs

In the Stage-1 IC-pruned derivative folder, confirm that the run writes:

- one `withICA` EEGLAB output
- one `clean` EEGLAB output
- the ICLabel pruning summary CSVs

The exact output names may include the standard tag:

- `desc-ICRej70`
- `desc-ICRej70_clean`

#### What to verify

- The cleaned `.set` file exists for the test run.
- The `withICA` `.set` file exists for the test run.
- The filename pattern still includes the expected `desc-ICRej70` tag.
- The pruning summary CSV is updated and includes the test run.
- If you open the pruning CSV, the test run should have sensible kept/rejected IC counts.

#### Nice-to-check

- Open the cleaned `.set` in EEGLAB and confirm it loads normally.
- Open the `withICA` file and confirm it still carries ICA-related information.
- Compare rejected-IC counts for the same run against an older known-good output.

#### Downstream compatibility signal

- The cleaned file name still follows the `sub-XX_ses-YY_desc-ICRej70_clean` convention that later stages expect.

### 3. Brainstorm exclusion export and union-merge smoke test

#### Script to run

- `notebooks/1_eeg_sensor/12_export_and_union_merge_brainstorm_exclusions.m`

#### Smallest practical data subset

- the same single test run, but only if it has matching Brainstorm event content

#### Must-check outputs

Confirm that the following files appear for the test run:

- Brainstorm exclusion export TSV
- exclusion-union TSV

Depending on your existing folder layout, these should live in the Stage-1 masks or exclusion derivatives location.

#### What to verify in the Brainstorm export TSV

- The file exists.
- It is non-empty for a run that truly has BAD or boundary intervals.
- The column names are still what downstream code expects.
- The event labels are still the expected ones:
  - `BAD`
  - `boundary`
  - `bad_boundary`

#### What to verify in the union TSV

- The file exists.
- It includes the merged exclusion intervals for the same run.
- The row count is plausible relative to the raw Brainstorm export.
- The file naming still matches the `*_excl_union.tsv` convention.

#### Nice-to-check

- Compare the number of intervals and total excluded time against an older known-good run.
- Spot-check one or two interval boundaries against the original Brainstorm export.

#### Downstream compatibility signal

- The output file is named exactly as Stage 4 expects:
  - `sub-XX_ses-YY_desc-ICRej70_clean_excl_union.tsv`

### 4. Run-level QC and Table S1 support smoke test

#### Script to run

- `notebooks/1_eeg_sensor/13_eeg_run_qc_and_table_s1.m`

#### Smallest practical data subset

- the same single run is enough for a smoke test
- if easy, a two-run subset is slightly better because it tests include/exclude logic more honestly

#### Must-check outputs

Confirm that the Stage-1 QC area writes the expected run-level support outputs, including:

- exclusion-union QC summary CSV
- run-level EEG QC gates CSV
- any include/exclude manifest or support table written by the public script

#### What to verify

- The QC summary CSVs exist and are readable.
- The tested run is present in the tables.
- The tested run has sensible values for usable fraction and QC flags.
- The tested run is not silently dropped from the manifest.

#### Nice-to-check

- Compare the run's usable fraction and include/exclude decision against an older known-good output.
- Compare total exclusion time and any EMG-proxy field values against the old run.
- Confirm the wording or notes still make the EMG-proxy issue explicit rather than silently harmonized.

#### Downstream compatibility signal

- The tested run still receives the same practical keep/exclude decision as in the known-good run.
- The run-level support tables still have the columns you expect for Supplementary Table S1 support.

## Stage-2 Smoke Tests

### 5. Scout extraction smoke test

#### Script to run

- `notebooks/2_eeg_source/22_extract_volgrid_scouts_from_brainstorm_tess.m`

#### Smallest practical data subset

- one subject-session-run with a valid Brainstorm source kernel and the expected Schaefer 200 scout file

#### Must-check outputs

Confirm that the run writes or refreshes:

- the scout MAT output used downstream
- the build summary CSV, usually `batch_volgrid_scout_build.csv`

#### What to verify

- The output MAT file exists.
- The build summary CSV exists.
- The tested run appears in the summary CSV.
- The summary reports the expected 200-parcel Schaefer atlas path.
- The scout count is 200.

#### Nice-to-check

- Open the MAT file and confirm it contains a `Scouts` structure.
- Compare the scout count, vertex counts, and any grid-size metadata against an older known-good run.

#### Downstream compatibility signal

- The Stage-2 parcel export script can point to the produced scout file without any manual renaming or schema adjustment.

### 6. Parcel export smoke test

#### Script to run

- `notebooks/2_eeg_source/23_export_eeg_parcel_pc1_and_gain_normalize.m`

#### Smallest practical data subset

- the same single test run used for scout extraction

#### Must-check outputs

For the tested run, confirm that the parcel output directory now contains:

- `*_parcelPC_raw.mat`
- `*_parcelPC_gnorm.mat`
- `*_PC1_gnorm.npy`
- `*_time_sec.npy`

Also confirm the batch-level CSVs exist or update:

- parcel gain summary CSV
- parcel coverage summary CSV
- parcel manifest CSV

#### What to verify in the MAT outputs

- The raw MAT file exists.
- The gain-normalized MAT file exists.
- Both load successfully.
- The main PC1 array has 200 parcel columns.
- The raw and gain-normalized files agree on the number of time points.

#### What to verify in the NPY outputs

- `*_PC1_gnorm.npy` exists.
- `*_time_sec.npy` exists.
- The row count of `*_PC1_gnorm.npy` matches the length of `*_time_sec.npy`.
- `*_time_sec.npy` starts at `0`.
- `*_time_sec.npy` is sample-level, not TR-level.
- The last value matches `(nTime - 1) / srate` for the run.

#### Strong recommended exact check for `*_time_sec.npy`

If you can load both the MAT file and the NPY file, verify that:

- `nTime = size(PC1_gnorm, 1)`
- `time_sec = single((0:nTime-1)' / srate)`

This is the critical deterministic rule that Stage 4 expects.

#### Nice-to-check

- Compare the MAT variable names and dimensions against an older known-good export.
- Compare the parcel manifest row for this run against the older known-good export.
- Check whether PC1 means or scales look roughly similar after gain normalization.

#### Downstream compatibility signal

- The output file names still match Stage 4 expectations exactly:
  - `sub-XX_ses-YY_desc-ICRej70_clean_PC1_gnorm.npy`
  - `sub-XX_ses-YY_desc-ICRej70_clean_time_sec.npy`

### 7. Parcel-export QC and Table S2 / Table S3 support smoke test

#### Scripts / notebook to run

- `notebooks/2_eeg_source/run_eeg_parcel_export_qc_summaries.m`
- `notebooks/2_eeg_source/25_qc_eeg_parcel_exports_table_s3_and_figures_s2_s4.ipynb`

#### Smallest practical data subset

- one run is enough for helper-path validation
- a small handful of runs is better if the notebook expects distributions rather than a single row

#### Must-check outputs

First confirm that the MATLAB QC helper writes the expected prerequisite CSVs:

- `qc_v3/qc_run_timeseries_gain_summary.csv`
- `qc_v3_sign/qc_sign_v3_summary.csv`
- `batch_pve1_run_quantiles_v3.csv`
- `batch_pve1_histogram_v3.csv`
- `batch_pve1_lowparcels_frequency_named_v3.csv`

Then run the notebook and confirm it can read those files without path or naming errors.

#### What to verify

- The helper runs without requiring the old `r01_*` command line.
- The notebook loads and completes its summary cells.
- Table-S2 support output is written or refreshed.
- Table-S3 support output is written or refreshed.
- Any figure-support CSVs or images expected from the notebook appear where the notebook says they will.

#### Nice-to-check

- Compare Table S2 and Table S3 support CSV row counts to an older known-good run.
- Compare a few summary values such as run count, parcel coverage rates, or low-parcel frequencies.

#### Downstream compatibility signal

- The notebook runs using the new helper-layer prerequisite path and does not assume the old `r01_*` QC entry points.

## Downstream Compatibility Checks

### Fastest Stage-4 compatibility check

You do not need to rerun full Stage 4 for this pass.

The fastest useful downstream check is to confirm that, for the same run, all three Stage-4 EEG-side inputs exist with the expected naming and compatible shapes:

- `sub-XX_ses-YY_desc-ICRej70_clean_PC1_gnorm.npy`
- `sub-XX_ses-YY_desc-ICRej70_clean_time_sec.npy`
- `sub-XX_ses-YY_desc-ICRej70_clean_excl_union.tsv`

Then verify:

- `size(PC1_gnorm, 1) == length(time_sec)`
- `time_sec(1) == 0`
- the exclusion TSV is non-empty when the run truly had exclusions

If those three files still exist with those names and those basic consistency checks pass, the most likely Stage-2-to-Stage-4 schema breakage has probably not happened.

### Better downstream check, if you want one more step

Open the cleaned Stage-4 notebook that builds alignment masks and check whether it can discover the test run without edits to filename logic.

You do not need to complete full alignment processing. A run-discovery or input-audit cell is enough.

## What to Compare Against Known-Good Outputs

If you have an older known-good run, compare these items first:

### Stage 1 comparisons

- cleaned `.set` filename
- `withICA` filename
- rejected-IC count
- kept-IC count
- Brainstorm exclusion export row count
- union exclusion row count
- total excluded seconds
- usable fraction
- include/exclude decision

### Stage 2 comparisons

- scout count
- parcel count
- MAT output filenames
- NPY output filenames
- `size(PC1_gnorm)`
- `length(time_sec)`
- `time_sec(1)`
- `time_sec(end)`
- batch manifest row for the tested run
- key QC summary row values

### Best comparisons to do exactly

If the old and new runs were produced from the same source inputs, the most useful exact or near-exact comparisons are:

- output filenames
- CSV column names
- per-run row counts
- per-run inclusion/exclusion flags
- per-run parcel matrix dimensions
- `time_sec.npy` reconstruction rule

## Must-Check Outputs

These are the outputs most worth checking because they would reveal a real break in the new public/helper layer.

### Stage 1

- cleaned EEGLAB `.set`
- `withICA` EEGLAB `.set`
- ICLabel pruning summary CSV
- Brainstorm exclusion export TSV
- exclusion-union TSV
- run-level QC / Table S1 support CSVs

### Stage 2

- scout MAT output
- scout build summary CSV
- `*_parcelPC_raw.mat`
- `*_parcelPC_gnorm.mat`
- `*_PC1_gnorm.npy`
- `*_time_sec.npy`
- parcel manifest / gain / coverage CSVs
- QC summary CSVs needed by the public QC notebook

## Nice-to-Check Outputs

These are useful, but lower priority than the list above.

- detailed sign-convention QC summaries
- PVE histogram support files
- figure-support images or PNGs
- any Stage-2 notebook visual outputs
- EEGLAB-level internal field inspection beyond basic loadability

## Fast-Fail Signs to Watch For

Stop early and fix configuration or helper wiring if you see any of these:

- helper function not found errors in Stage 1 or Stage 2
- output files written to unexpected roots because of the new settings layer
- missing `desc-ICRej70_clean` tag in Stage-1 cleaned outputs
- missing Brainstorm export TSV or missing union TSV
- QC tables that do not include the tested run
- missing scout MAT output or scout count not equal to 200
- missing `*_parcelPC_gnorm.mat`
- missing `*_PC1_gnorm.npy`
- missing `*_time_sec.npy`
- `*_time_sec.npy` length not matching `PC1_gnorm.npy`
- `*_time_sec.npy` not starting at `0`
- Stage-2 QC notebook failing because it cannot find the expected QC CSV sidecars

## Files Where Import or Path Sanity Is Enough

These changes were lower risk and do not need a full rerun for this validation pass.

- helper/header documentation improvements in Stages 3 to 6
- repo-level docs and README wording cleanup

For those, a quick import/open sanity check is enough.

## Lowest-Priority Checks

Only do these if the smoke tests above pass and you want extra confidence.

- compare more than one run
- compare sign-convention QC distributions across several runs
- compare PVE histogram summaries across several runs
- run a larger subset through Stage 2 QC notebook outputs
- run a partial Stage-4 alignment audit after confirming the file schema

## Bottom Line

If you only have time for a very small validation pass, do these four things first:

1. Run Stage 1 pruning on one known-good run and confirm the cleaned `.set`, `withICA` `.set`, and pruning CSV appear with the expected `desc-ICRej70` naming.
2. Run Stage 1 Brainstorm exclusion export and union merge for the same run and confirm `*_excl_union.tsv` is produced with the expected naming.
3. Run Stage 2 parcel export for the same run and confirm `*_parcelPC_gnorm.mat`, `*_PC1_gnorm.npy`, and `*_time_sec.npy` are written.
4. Confirm that `length(time_sec)` matches the number of rows in `PC1_gnorm.npy` and that `time_sec(1) == 0`.

Those four checks will catch most practical breakage introduced by the new public/helper layer.
