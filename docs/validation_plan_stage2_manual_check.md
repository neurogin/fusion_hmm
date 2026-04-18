# Stage-2 Manual Validation Plan

## Scope

This memo is a practical manual validation checklist for the active public-facing Stage-2 workflow only.

It covers:

- [extract_volgrid_scouts_from_brainstorm_tess_22.m](/C:/fusion_hmm/notebooks/2_eeg_source/extract_volgrid_scouts_from_brainstorm_tess_22.m)
- [export_eeg_parcel_pc1_and_gain_normalize_23.m](/C:/fusion_hmm/notebooks/2_eeg_source/export_eeg_parcel_pc1_and_gain_normalize_23.m)
- [qc_eeg_source_alignment_table_s2_24.m](/C:/fusion_hmm/notebooks/2_eeg_source/qc_eeg_source_alignment_table_s2_24.m)
- [run_eeg_parcel_export_qc_summaries.m](/C:/fusion_hmm/notebooks/2_eeg_source/run_eeg_parcel_export_qc_summaries.m)
- [25_qc_eeg_parcel_exports_table_s3_and_figures_s2_s4.ipynb](/C:/fusion_hmm/notebooks/2_eeg_source/25_qc_eeg_parcel_exports_table_s3_and_figures_s2_s4.ipynb)
- the active Stage-2 helper layer in [notebooks/2_eeg_source](/C:/fusion_hmm/notebooks/2_eeg_source)

The goal is not to rerun the whole cohort. The goal is to run a small number of smoke tests that would quickly show whether the public Stage-2 layer still writes the expected scout, parcel, NPY, and QC outputs, and whether those outputs still match downstream Stage-4 expectations.

## Recommended Validation Order

Run the checks in this order:

1. Stage-2 path and dependency sanity only
2. Step 22 scout-extraction smoke test
3. Step 23 parcel-export smoke test
4. Step 24 Table-S2 support smoke test
5. Stage-2 QC-helper smoke test
6. Stage-2 notebook 25 smoke test
7. Fast downstream Stage-4 compatibility check

This order gives you fast failure if there is a path problem, a missing helper, or a broken output schema.

## Must-Check Outputs

These are the highest-value outputs to inspect because they would reveal a real Stage-2 contract break:

- standardized scout MAT output under the Brainstorm protocol `anat/sub-XX_ses-YY/` folder
- `04_qc/stage2_eeg_source/tables/batch_volgrid_scout_build.csv`
- `02_derivatives/stage2_eeg_source/parcel_exports/*_parcelPC_raw.mat`
- `02_derivatives/stage2_eeg_source/parcel_exports/*_parcelPC_gnorm.mat`
- `02_derivatives/stage2_eeg_source/parcel_exports/npy/*_PC1_gnorm.npy`
- `02_derivatives/stage2_eeg_source/parcel_exports/npy/*_time_sec.npy`
- `02_derivatives/stage2_eeg_source/parcel_exports/batch_parcel_gain_summary_v3.csv`
- `02_derivatives/stage2_eeg_source/parcel_exports/batch_parcel_coverage_summary_v3.csv`
- `02_derivatives/stage2_eeg_source/parcel_exports/batch_parcel_manifest_v3.csv`
- `04_qc/stage2_eeg_source/tables/table_s2_eeg_atlas_alignment_summary.csv`
- `02_derivatives/stage2_eeg_source/parcel_exports/qc_v3/qc_run_timeseries_gain_summary.csv`
- `02_derivatives/stage2_eeg_source/parcel_exports/qc_v3_sign/qc_sign_v3_summary.csv`
- `02_derivatives/stage2_eeg_source/parcel_exports/batch_pve1_run_quantiles_v3.csv`
- `02_derivatives/stage2_eeg_source/parcel_exports/table_s3_eeg_parcel_extraction_summary.csv`

## Stage-2 Smoke Tests

### 1. Path and dependency sanity

#### What to open

- [extract_volgrid_scouts_from_brainstorm_tess_22.m](/C:/fusion_hmm/notebooks/2_eeg_source/extract_volgrid_scouts_from_brainstorm_tess_22.m)
- [export_eeg_parcel_pc1_and_gain_normalize_23.m](/C:/fusion_hmm/notebooks/2_eeg_source/export_eeg_parcel_pc1_and_gain_normalize_23.m)
- [qc_eeg_source_alignment_table_s2_24.m](/C:/fusion_hmm/notebooks/2_eeg_source/qc_eeg_source_alignment_table_s2_24.m)

#### Smallest practical check

- Fill in `project_root` and `protocol_root`.
- Confirm the Stage-1 clean EEG folder really exists at:
  - `02_derivatives/stage1_eeg_sensor/ic_pruned/clean_sets`
- Confirm the Brainstorm protocol root points to the protocol that contains:
  - `data/.../results_MN_EEG_KERNEL_*.mat`
  - `anat/sub-XX_ses-YY/...`

#### What to verify

- Each script prints the expected roots before doing real work.
- Step 23 finds `pop_loadset` on the MATLAB path.
- If you want NPY outputs, Step 23 also finds `writeNPY`.
- The wrapper helpers resolve without manual path hacks:
  - `batch_extract_volgrid_scouts_from_brainstorm_tess`
  - `batch_export_eeg_parcel_pc_outputs`
  - `run_eeg_parcel_export_qc_summaries`

#### Fast-fail sign

- any “function not found” error for the public wrappers or preserved `r01_*` helpers
- Step 23 warning that `writeNPY` is missing when you expected Stage-4-ready NPY sidecars

### 2. Step 22 scout-extraction smoke test

#### Script to run

- [extract_volgrid_scouts_from_brainstorm_tess_22.m](/C:/fusion_hmm/notebooks/2_eeg_source/extract_volgrid_scouts_from_brainstorm_tess_22.m)

#### Smallest practical subset

Best option:

- a temporary reduced Brainstorm protocol copy containing one known-good subject-session-run

If you do not want to clone a reduced protocol:

- use one known-good subject/session in the real protocol and accept that the batch helper may scan more than one run

Choose a case that already reached later stages before, so you have something known-good to compare against.

#### Output folders and files that should appear

- one scout MAT file under:
  - `protocol_root/anat/sub-XX_ses-YY/scout_Schaefer2018_200_7N_dilated_MNI.mat`
- one summary CSV at:
  - `04_qc/stage2_eeg_source/tables/batch_volgrid_scout_build.csv`

#### What to check

- the scout MAT file exists for the tested subject/session
- the summary CSV exists and includes the tested subject/session
- the summary CSV includes these fields:
  - `sub`
  - `ses`
  - `tessFile`
  - `outScoutFile`
  - `atlasName`
  - `nScouts`
  - `nEmptyScouts`
  - `vertexMin`
  - `vertexMax`
  - `nVertExpected`
- `nScouts` is `200`
- `nVertExpected` is positive
- `vertexMax <= nVertExpected`
- `outScoutFile` points to a file that exists

#### Downstream compatibility signal

- Step 23 can use the produced scout MAT directly, with no manual renaming or copying

### 3. Step 23 parcel-export smoke test

#### Script to run

- [export_eeg_parcel_pc1_and_gain_normalize_23.m](/C:/fusion_hmm/notebooks/2_eeg_source/export_eeg_parcel_pc1_and_gain_normalize_23.m)

#### Smallest practical subset

Best option:

- the same reduced one-run protocol copy from the Step-22 test, plus the matching Stage-1 cleaned `.set` file

If you keep the full protocol root:

- pick one known-good run and focus your checks on that run’s outputs

#### Output folders and files that should appear

Under:

- `02_derivatives/stage2_eeg_source/parcel_exports/`

you should see for the tested run:

- `<runTag>_parcelPC_raw.mat`
- `<runTag>_parcelPC_gnorm.mat`

Under:

- `02_derivatives/stage2_eeg_source/parcel_exports/npy/`

you should see:

- `<runTag>_PC1_gnorm.npy`
- `<runTag>_time_sec.npy`

You will likely also see:

- `<runTag>_PC2_gnorm.npy`
- `<runTag>_PVE1.npy`
- `<runTag>_PVE2.npy`
- `<runTag>_valid_parcel_mask.npy`
- `<runTag>_n_vertices.npy`
- `<runTag>_n_rows.npy`
- `<runTag>_parcel_ids.npy`

Batch-level summary files should also exist:

- `batch_parcel_gain_summary_v3.csv`
- `batch_parcel_coverage_summary_v3.csv`
- `batch_parcel_manifest_v3.csv`
- `parcelNames_200.csv`

#### What to check in the MAT outputs

- both MAT files load without error
- the main parcel matrix is time-by-parcel, with `200` parcel columns
- the raw and gain-normalized MAT files agree on the number of time points
- the run tag follows:
  - `sub-XX_ses-YY_desc-ICRej70_clean`

#### What to check in the CSV summaries

In `batch_parcel_coverage_summary_v3.csv`, confirm the tested run has:

- `n_scouts = 200`
- plausible `n_found`, `n_missing`, and `coverage_frac`
- `minVertices = 40`
- the expected source-grid fields used later by Step 24:
  - `TessNbVertices`
  - `nVert_gridloc`
  - `n_assigned_vertices`
  - `overlap_vertices`

In `batch_parcel_manifest_v3.csv`, confirm the tested run row points to files that really exist.

In `batch_parcel_gain_summary_v3.csv`, confirm the tested run row has plausible non-empty values for fields such as:

- `kVertNorm_median`
- `pc1_std_median`
- `pve1_median`

#### What to check in the NPY outputs

- `<runTag>_PC1_gnorm.npy` exists
- `<runTag>_time_sec.npy` exists
- the first dimension of `PC1_gnorm.npy` matches the length of `time_sec.npy`
- `time_sec.npy` starts at `0`
- `time_sec.npy` is sample-level, not TR-level

Strongly recommended exact check:

- if `nTime = size(PC1_gnorm, 1)` and the exported sampling rate is `srate`, then:
  - `time_sec = single((0:nTime-1)' / srate)`

This exact rule is the most important Stage-2 sidecar contract for downstream alignment.

#### Downstream compatibility signal

For the tested run, the public exporter still writes:

- `sub-XX_ses-YY_desc-ICRej70_clean_PC1_gnorm.npy`
- `sub-XX_ses-YY_desc-ICRej70_clean_time_sec.npy`

with matching sample counts

### 4. Step 24 Table-S2 support smoke test

#### Script to run

- [qc_eeg_source_alignment_table_s2_24.m](/C:/fusion_hmm/notebooks/2_eeg_source/qc_eeg_source_alignment_table_s2_24.m)

#### Smallest practical subset

- the same one-run or one-subject subset used above is enough

#### Output folder and file that should appear

- `04_qc/stage2_eeg_source/tables/table_s2_eeg_atlas_alignment_summary.csv`

#### What to check

- the CSV exists
- the tested run appears in the table
- the table contains these columns:
  - `Subject`
  - `Session`
  - `Run`
  - `Scouts`
  - `Parcels_found`
  - `Parcels_missing`
  - `Coverage_fraction`
  - `MinDipoles`
  - `Parcels_valid_ge_MinDipoles`
  - `Scout_grid_size`
  - `Kernel_grid_size`
  - `Assigned_vertices`
  - `Overlap_vertices`
  - `Overlap_rate`
- `Scouts` is `200` for the tested run
- row count is consistent with the runs present in `batch_parcel_coverage_summary_v3.csv`

#### Nice-to-check

- if `batch_volgrid_scout_build.csv` exists, note whether Step 24 reports any scout-count or grid-size mismatch warning

#### Downstream compatibility signal

- the coverage-summary schema still matches what Step 24 expects, with no missing-column error

### 5. QC-helper smoke test

#### Script to run

- [run_eeg_parcel_export_qc_summaries.m](/C:/fusion_hmm/notebooks/2_eeg_source/run_eeg_parcel_export_qc_summaries.m)

#### Smallest practical subset

- the same parcel-output directory from the Step-23 smoke test

#### Output folders and files that should appear

Under the parcel-output directory:

- `qc_v3/qc_run_timeseries_gain_summary.csv`
- `qc_v3_sign/qc_sign_v3_summary.csv`
- `batch_pve1_run_quantiles_v3.csv`
- `batch_pve1_histogram_v3.csv`
- `batch_pve1_lowparcels_frequency_named_v3.csv`

#### What to check

- the helper runs from the public name, not only from the legacy `r01_*` entry points
- it does not fail to find `*_parcelPC_gnorm.mat` or `*_parcelPC_raw.mat`
- each of the five expected QC sidecars is written

#### Downstream compatibility signal

- Notebook 25 can load these files without any path or naming edits beyond the one top-level `parcel_output_dir`

### 6. Notebook 25 smoke test

#### Notebook to run

- [25_qc_eeg_parcel_exports_table_s3_and_figures_s2_s4.ipynb](/C:/fusion_hmm/notebooks/2_eeg_source/25_qc_eeg_parcel_exports_table_s3_and_figures_s2_s4.ipynb)

#### Smallest practical subset

- one run is enough to confirm helper-path wiring and file discovery
- two or three runs is better if you want the pooled histograms and quantiles to look more realistic

#### Output folders and files that should appear

Inside the parcel-output directory:

- `table_s3_eeg_parcel_extraction_summary.csv`
- `figures_s2_s4/fig_s2_gain_pc1_scale_after_gnorm.png`
- `figures_s2_s4/fig_s3_pve1_histogram_pooled.png`
- `figures_s2_s4/fig_s4_pve1_run_quantiles.png`

Optional extra QC figures may also appear.

#### What to check

- the notebook no longer errors out on missing prerequisite QC CSVs
- `table_s3_eeg_parcel_extraction_summary.csv` is written
- the table contains the expected summary fields:
  - `Run`
  - `Valid parcels`
  - `NaN fraction (PC1)`
  - `Median PC1 scale after gnorm`
  - `Sign-check pass rate`
  - `Median sign-check correlation`
  - `PVE1 q10`
  - `PVE1 q50`
  - `PVE1 q90`
  - `Fraction of parcels with PVE1 < 0.20`
- the three main figure PNGs are written to `figures_s2_s4`

#### Nice-to-check

- spot-check that run counts in Table S3 match the QC CSV inputs
- compare one or two summary values against an older known-good output if available

#### Downstream compatibility signal

- the public QC helper path is sufficient for notebook 25, with no need to call old `r01_*` entry points manually

## Downstream Compatibility Checks

You do not need to rerun all of Stage 4 for this pass.

The fastest useful Stage-4 compatibility check is:

1. pick one tested run
2. confirm these files exist with the exact expected naming:
   - `sub-XX_ses-YY_desc-ICRej70_clean_PC1_gnorm.npy`
   - `sub-XX_ses-YY_desc-ICRej70_clean_time_sec.npy`
3. confirm:
   - `rows(PC1_gnorm.npy) == length(time_sec.npy)`
   - `time_sec.npy(1) == 0`
   - the time step is constant and matches `1 / srate`

If you want one additional check:

- open the cleaned Stage-4 alignment notebook and run only the input-discovery or audit cell that looks for the EEG NPY sidecars

That is enough to reveal a naming or sidecar-schema break without rerunning full alignment.

## What to Compare Against Known-Good Outputs

If you have an older known-good Stage-2 run, compare these items first:

- scout MAT filename and location
- scout count in `batch_volgrid_scout_build.csv`
- parcel MAT filenames
- NPY filenames
- parcel matrix dimensions
- `batch_parcel_manifest_v3.csv` row for the tested run
- `batch_parcel_coverage_summary_v3.csv` row for the tested run
- `batch_parcel_gain_summary_v3.csv` row for the tested run
- `time_sec.npy` length
- `time_sec.npy` first value
- `time_sec.npy` last value
- Table-S2 row count and columns
- Table-S3 row count and columns

The most useful exact comparisons are:

- output filenames
- CSV column names
- row counts
- run tags
- `n_scouts`
- parcel matrix shape
- `time_sec.npy` reconstruction rule

## Likely Usability Issues to Note During Testing

Please note these during manual testing so they can be polished afterward if needed:

- Step 22 writes scout MAT files into the Brainstorm protocol `anat/` tree, not into a Stage-2 derivatives folder. If that feels surprising or easy to miss, it should be documented more prominently later.
- Step 22 and Step 23 are batch-style scripts. A truly minimal smoke test is easiest if you point them to a reduced protocol copy rather than the full cohort.
- Notebook 25 still requires you to set `parcel_output_dir` manually near the top.
- Step 23 warns about missing `writeNPY`, but that warning is easy to overlook even though `*_time_sec.npy` is critical for Stage 4.
- The active helper layer still delegates to preserved low-level `r01_*` implementations underneath. If that causes confusion in practice, it may need a later documentation or promotion pass.

## Fast-Fail Signs

Stop early and fix configuration or helper wiring if you see any of these:

- Step 22 cannot find the Brainstorm protocol root or any `results_MN_EEG_KERNEL_*.mat` files
- Step 22 writes no scout MAT file for the tested subject/session
- `batch_volgrid_scout_build.csv` is missing or shows `nScouts` other than `200`
- Step 23 cannot find the Stage-1 cleaned `.set` file for the tested run
- Step 23 writes MAT files but no `*_PC1_gnorm.npy` when you expected NPY export
- Step 23 writes `*_PC1_gnorm.npy` but no `*_time_sec.npy`
- `*_time_sec.npy` length does not match the first dimension of `*_PC1_gnorm.npy`
- `batch_parcel_coverage_summary_v3.csv` is missing required columns used by Step 24
- Step 24 errors on missing coverage-summary columns
- `run_eeg_parcel_export_qc_summaries.m` fails because it cannot find parcel MAT outputs
- Notebook 25 errors on missing QC CSV sidecars

## Bottom Line

If you only have time for a very small Stage-2 check, do these three things first:

1. Run Step 22 on one known-good subject/session and confirm the standardized scout MAT plus `batch_volgrid_scout_build.csv` are written, with `nScouts = 200`.
2. Run Step 23 for the same case and confirm:
   - `*_parcelPC_gnorm.mat`
   - `*_PC1_gnorm.npy`
   - `*_time_sec.npy`
   are all written.
3. Confirm the exact downstream-critical relation:
   - `rows(PC1_gnorm.npy) == length(time_sec.npy)`
   - `time_sec(1) == 0`

Those checks will catch most practical Stage-2 schema breakage quickly.
