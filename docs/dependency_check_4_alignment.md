# Stage 4 Pass 2: Dependency Check for `notebooks/4_alignment/`

## Scope

This memo is the Stage-4 Pass-2 dependency and integration check for:

- `notebooks/4_alignment/`

It starts from:

- `docs/refactor_plan_4_alignment.md`

and verifies the current Stage-4 notebooks against:

- the cleaned Stage-2 EEG-source outputs
- the cleaned Stage-3 BOLD outputs
- the actual trigger/event assumptions used for EEG-to-fMRI alignment

This is a memo only. No Stage-4 files were rewritten, moved, or implemented in this pass.

## Inputs reviewed

I reviewed:

- `docs/refactor_plan_4_alignment.md`
- `docs/refactor_plan_2_eeg_source.md`
- `docs/dependency_check_2_eeg_source.md`
- `docs/recovery_check_time_sec_sidecar.md`
- `docs/refactor_plan_3_bold.md`
- `docs/recovery_check_3_bold_provenance.md`
- `docs/final_dataset_spec.md`
- `docs/reproducibility_notes.md`
- `docs/figure_table_map.md`
- `notebooks/4_alignment/R01_PipelineA_align_trmask_lags_v3_gnorm_allTR_INTERMEDIATE.ipynb`
- `notebooks/4_alignment/R01_PipelineB_build_X_segments_v3_gnorm_allTR_INTERMEDIATE_TOGGLE_LAGS.ipynb`
- both `Pipeline C` notebooks in `notebooks/4_alignment/`
- `notebooks/2_eeg_source/23_export_eeg_parcel_pc1_and_gain_normalize.m`
- `notebooks/2_eeg_source/r01_batch_export_eeg_parcel_pc_v3.m`
- `notebooks/3_bold/31_export_bold_parcel_pc1_with_nuisance_regression.ipynb`
- `notebooks/3_bold/stage3_bold_export_helpers.py`
- the Stage-3 local recovery example:
  - `_local_recovery_examples/stage3_bold_provenance/parcel_pc1_v6/dataset_index.csv`
- the Stage-2 local recovery example:
  - `_local_recovery_examples/time_sec_sidecar/sub-01_ses-01/`
- archive provenance under:
  - `notebooks/_archive_raw_original_names/5_multimodal alignment and X/`

## Proposed cleaned public-facing files checked

The current proposed cleaned Stage-4 public set is:

- `40_align_eeg_to_bold_trs_and_build_keep_masks.ipynb`
- `41_build_final_no_lag_fusion_observation_segments.ipynb`
- `42_qc_alignment_tables_s6_s7_and_figure1_support.ipynb`

## Stage-2 integration findings

### What Stage 4 actually expects

`Pipeline A` expects Stage-2 EEG parcel exports in a folder matching:

- `.../eeg_source/parcel_pc1/npy`

and looks for files named:

- `sub-*_ses-*_desc-ICRej70_clean_PC1_gnorm.npy`
- `sub-*_ses-*_desc-ICRej70_clean_time_sec.npy`

It reduces each EEG filename to a run key using:

- regex: `sub-\d+_ses-\d+`

The time sidecar is not optional in practice. The code raises:

- `FileNotFoundError` if `*_time_sec.npy` is absent

and checks:

- row count of `PC1_gnorm.npy` exactly matches row count of `time_sec.npy`

### Match to cleaned Stage 2

The cleaned Stage-2 exporter now writes:

- `*_PC1_gnorm.npy`
- `*_time_sec.npy`

using the verified rule:

- `single((0:nTime-1)' / srate)`

and the preserved helper still generates run tags like:

- `sub-01_ses-01_desc-ICRej70_clean`

This means the Stage-2 filename schema and sidecar convention do match what Stage 4 currently expects.

### Remaining mismatch

The remaining Stage-2 issue is location, not naming.

The cleaned Stage-2 public file makes `parcel_output_dir` user-configurable. Stage 4 still hard-codes an older local derivative location:

- `02_derivatives/eeg_source/parcel_pc1/npy`

So:

- file naming matches
- sidecar naming matches
- row/timebase assumptions match
- folder-root assumptions are still old and must be wrapped explicitly in Pass 3

### Hidden upstream dependency beyond Stage 2

Stage 4 also depends on Stage-1 union TSVs named like:

- `sub-*_ses-*_desc-ICRej70_clean_excl_union.tsv`

under:

- `.../02_derivatives/masks/bst_exports`

This naming still matches the cleaned Stage-1 output convention. The remaining issue is again location hard-coding, not missing logic.

## Stage-3 integration findings

### What Stage 4 actually expects

`Pipeline A` expects BOLD parcel outputs in:

- `.../bold_parcel/parcel_pc1_v6/npy`

and looks for files named:

- `sub-*_ses-*_task-rest_parcel_pc1.npy`

It reduces the BOLD filenames to the same run key:

- `sub-\d+_ses-\d+`

### Match to cleaned Stage 3

The cleaned Stage-3 public exporter still defaults to:

- `DERIVATIVES_ROOT / "bold_parcel" / "parcel_pc1_v6"`

and its helper writes:

- `npy/sub-XX_ses-YY_task-rest_parcel_pc1.npy`

The local recovered `dataset_index.csv` confirms this output schema for 15 runs.

So the Stage-3 filename schema and parcel-output folder convention do match the current Stage-4 code path.

### Whether `dataset_index.csv` is required

Current Stage-4 notebooks do not consume:

- `dataset_index.csv`

at all.

Instead, `Pipeline A` discovers runs by globbing the BOLD NPY files and intersecting them with:

- EEG parcel NPY files
- raw event TSVs
- preprocessed event TSVs
- exclusion-union TSVs

I did not find any Stage-4 computation that currently requires a field available only in `dataset_index.csv`.

For the current alignment logic, `dataset_index.csv` is therefore:

- not a missing dependency
- not required to reconstruct TR masks or build segment arrays

But it is still useful as a safer manifest. The current glob-intersection behavior can silently drop runs if any one required input is missing or renamed.

### Is `sub-XX_ses-YY` matching safe?

For this dataset, it appears safe.

Evidence:

- the recovered Stage-3 `dataset_index.csv` has 15 rows
- `runTag` values are of the form `sub-XX_ses-YY_task-rest`
- none of the 15 rows contain `_run-`
- none of the recovered `bold_file` paths contain `_run-`
- the final dataset spec is 15 runs across 12 participants, consistent with one resting-state run per session

So for the manuscript dataset:

- collapsing to `sub-XX_ses-YY` appears safe

But this is not a generic design. It is brittle outside this dataset, and the cleaned public notebooks should document that assumption explicitly.

## EEG event / trigger (`R128`) findings

### Which files are read

`Pipeline A` reads two event-file streams:

- raw EEG events:
  - `RAW_EVENTS_DIR.glob("sub-*_ses-*_task-rest_events.tsv")`
- preprocessed EEG events:
  - `PREPROC_EVENTS_DIR.glob("sub-*_ses-*_task-rest_events.tsv")`

These are expected under older local folders:

- `01_raw/eeg_eeglab/events_raw`
- `01_raw/eeg_eeglab/events_preproc`

### What the code does with triggers

The true alignment helper is:

- `align_raw_preproc(raw_events_tsv, preproc_events_tsv)`

It explicitly extracts:

- `R128` events from both raw and preprocessed event TSVs
- `S1` from the raw event TSV only

It then:

1. matches recurring `R128` inter-trigger intervals between raw and preprocessed time axes
2. builds piecewise offsets from preprocessed time to raw time
3. anchors raw time zero to the first raw `S1` event

So in practice the alignment workflow depends on:

- recurring scanner-trigger events labeled with `R128`
- a raw `S1` anchor event

### Which columns are used

The active robust extractor in the alignment cell searches label information across:

- `trial_type`
- `value`
- `type`

and time information across:

- `onset`
- `start`
- `start_sec`
- `time`
- `latency_sec`
- `latency`

Column-name lookup is case-insensitive via `_find_col`.

### How the trigger code is represented in practice

Direct bundled event TSV examples were not found in the repo or in `_local_recovery_examples`, so the exact concrete label string in the open-data event files could not be verified from a versioned example.

What can be verified from code and comments is:

- the code looks for regex `R128`
- it uses substring matching, not exact equality
- the code comment says this was patched because event codes in the real files often live in `value`, while `type` can contain only generic categories such as `Stimulus` / `Response`
- another code comment says raw `S1` is also stored in `value` in the real files

This means the current logic is robust to practical label forms such as:

- `R128`
- `Stimulus/R128`

as long as the string contains uppercase contiguous `R128`.

### What the trigger matching is and is not robust to

It is robust to:

- `R128` appearing in `trial_type`, `value`, or `type`
- trigger labels embedded inside longer strings, because matching uses `str.contains("R128")`
- multiple possible time-column names

It is not obviously robust to:

- lowercase variants such as `r128`
- split forms such as `R 128`
- missing raw `S1`
- event TSVs that omit all of the expected time-column names

It also requires enough repeated trigger events to pass:

- `len(raw_r128) >= 10`
- `len(pre_r128) >= 10`

### Is this dependency documented clearly enough?

No.

Current repo docs say Stage 4 does raw-to-preprocessed EEG timeline reconciliation, but they do not clearly spell out that the public notebook will require:

- raw EEG event TSVs
- preprocessed EEG event TSVs
- recurring `R128` scanner triggers in both
- a raw `S1` event used as the absolute anchor

This trigger dependency should be explicit in the cleaned public Stage-4 notebook and likely also in `docs/reproducibility_notes.md`.

## Internal helper/dependency status

### Existing logic that is sufficient

There is no missing Stage-4 helper function.

The alignment and segment-building logic already exists inside:

- `Pipeline A`
- `Pipeline B`

and is computationally sufficient for Pass 3.

### What should become helper-level code later

For readability only, not for scientific recovery:

- event-file parsing and column detection
- `R128`-based raw/preprocessed timeline mapping
- exclusion remapping and interval canonicalization
- TR-mask construction
- EEG power-per-TR construction
- segment extraction and manifest writing

### Internal caveat worth preserving

One internal behavior detail should not be silently “cleaned up” later:

- `OFFSET_JUMP_THR` is exposed in the user-input cell, but the active segment-boundary split inside `build_segment_offsets_from_matches` is currently hard-coded at `0.5` seconds

That is not a missing dependency, but it is a provenance-relevant implementation detail that should be preserved or flagged explicitly during refactor rather than harmonized silently.

## Misplaced later-stage material

The two `Pipeline C` notebooks remain later-stage contamination:

- `R01_PipelineC_fusion_hmm_LOSO_K2-12_v3_UNIFIED_TOGGLE_LAGS_GPU_PATCH_Data2Dataset.ipynb`
- `R01_PipelineC_fusion_hmm_LOSO_K2-12_v3_UNIFIED_TOGGLE_LAGS_GPU_PATCH_Data2Dataset-extend.ipynb`

They are not Stage-4 dependencies.

They do, however, clarify one downstream expectation:

- later modeling notebooks look for `hmm_segments_minlen15_*`
- they support both `lags` and `nolags`
- they do not center their main path on `minlen10`

So:

- lagged `minlen15` remains useful as an optional downstream provenance branch
- `minlen10` does not appear to be needed for the later public modeling path

## Missing or ambiguous dependencies

### True missing recovery item

I did not find a missing code helper that blocks Stage 4.

So there is no major Stage-4 recovery task analogous to the earlier `time_sec.npy` recovery.

### Hidden external prerequisite that must be documented

There is, however, one important hidden upstream prerequisite:

- the raw and preprocessed EEG event TSV streams

These are required by current Stage-4 alignment logic, but they are not surfaced as a cleaned earlier-stage public output.

This is best classified as:

- an integration dependency that must be documented and wrapped

not as a missing helper function.

### Ambiguity that remains

No bundled event TSV example was found, so the exact concrete trigger string format in the open-data event files remains only indirectly verified.

What is known:

- the code is built around `R128`
- comments indicate the real files often store event codes in `value`

What remains unverified from bundled inputs:

- whether the concrete label is always `R128`
- or sometimes `Stimulus/R128`

The current regex strategy should cover either case, but the public notebook should say so explicitly.

## Recovery or wrapper actions required before Pass 3

1. Keep the current Stage-2 and Stage-3 filename conventions, because they already match Stage-4 expectations.

2. Replace old hard-coded local derivative locations with explicit user inputs near the top of the cleaned notebook, for example:
   - `RAW_EVENTS_DIR`
   - `PREPROC_EVENTS_DIR`
   - `EXCL_UNION_DIR`
   - `EEG_PARCEL_NPY_DIR`
   - `BOLD_PARCEL_NPY_DIR`
   - `ALIGNMENT_OUTPUT_DIR`

3. Document the trigger dependency explicitly:
   - recurring `R128` events in raw and preprocessed TSVs
   - raw `S1` used as the anchor
   - accepted label columns and time columns

4. Preserve glob-based run discovery, but add an explicit missing-run audit in the public notebook so users can see which runs were excluded because an input stream was absent.

5. Make the cleaned public default:
   - no-lag
   - `minlen15`

6. Keep lagged `minlen15` available only as an optional provenance branch if later-stage model-comparison reproduction is desired.

7. Do not present `minlen10` as the main public output path.

## Bottom-line verdict

No true recovery of missing Stage-4 code is needed before Pass 3.

Stage 4 still matches the cleaned Stage 2 and Stage 3 outputs on the important scientific interface points:

- Stage-2 EEG parcel filename pattern matches
- `*_time_sec.npy` sidecar convention now matches
- Stage-3 BOLD parcel filename pattern matches
- the recovered Stage-3 output schema is consistent with Stage-4 expectations

The main remaining issues are integration and documentation:

- old hard-coded folder locations
- availability-based glob matching
- undocumented dependence on raw/preprocessed EEG event TSVs
- undocumented dependence on `R128` and raw `S1`

For Pass 3, the implementation work should focus on:

- a cleaned alignment notebook that exposes these input roots explicitly
- a cleaned segment-builder notebook whose default path is the canonical no-lag `minlen15` dataset
- a light summary notebook for Tables S6-S7 and Figure-1 support

without changing the existing alignment behavior.
