# Stage 1 Dependency Gap Report

## Scope

This report covers the exact helper-function and file dependencies referenced by the 9 MATLAB files in `notebooks/1_eeg_sensor/`:

- `r01_batch_export_bst_exclusions_Fevents_driver.m`
- `r01_batch_export_bst_exclusions_Fevents.m`
- `r01_batch_merge_exclusions_union_driver.m`
- `r01_batch_merge_exclusions_union.m`
- `r01_eeg_iclabel_prune_and_metadata_driver.m`
- `r01_eeg_iclabel_prune_and_metadata.m`
- `r01_merge_exclusions_union_folder_driver.m`
- `r01_params.m`
- `r01_qc_excl_union_folder_driver.m`

Because stage-1 QC/Table-S1 logic is not fully contained in this folder, this pass also inspected:

- `notebooks/2_eeg_source/r01_eeg_runlevel_qc_gates.m`
- `notebooks/2_eeg_source/r01_eeg_runlevel_qc_gates_driver.m`
- archived provenance under `notebooks/_archive_raw_original_names/1_eeg preprocessing/`
- manuscript-alignment docs under `docs/`

No code was rewritten in this pass.

## Status Legend

- `found and usable`: present in the repo in executable form for this stage
- `found but misplaced`: present in the repo, but only in archive provenance or in the wrong manuscript stage
- `partially recoverable`: missing as an executable repo file, but interface/output behavior can be inferred with moderate confidence
- `missing and not yet reconstructable`: missing and not safely inferable from the repo/manuscript

## Executive Findings

1. One stage-1 helper is truly absent as a repo file: `r01_export_bst_exclusions_Fevents.m`.
2. Two stage-1 helpers exist only as archive provenance copies, not as active `.m` files:
   - `r01_merge_exclusions_union`
   - `r01_qc_excl_union_folder`
3. The run-level QC / Table-S1 implementation exists, but is currently misplaced in stage 2:
   - `notebooks/2_eeg_source/r01_eeg_runlevel_qc_gates.m`
   - its SHA256 matches the archived stage-1 provenance copy exactly
4. No frozen stage-1 intermediate artifacts were found in-repo:
   - no `*_bst_exclusions.tsv`
   - no `*_excl_union.tsv`
   - no `excl_union_qc_summary.csv`
   - no `eeg_run_qc_gates_*.csv`
   - no include/exclude manifests
5. The main implementation blocker is therefore not the merge/QC logic. It is the missing Brainstorm-event exporter and the lack of example exported TSVs to validate a reconstruction against.

## A. Repo-Local Code Dependencies Referenced By `notebooks/1_eeg_sensor/`

| Dependency | Exists in repo? | Where | Called by | Likely inputs | Likely outputs | Inference basis | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `r01_params.m` | Yes | `notebooks/1_eeg_sensor/r01_params.m` | `r01_batch_export_bst_exclusions_Fevents_driver.m`, `r01_eeg_iclabel_prune_and_metadata_driver.m`, `r01_merge_exclusions_union_folder_driver.m`, `r01_qc_excl_union_folder_driver.m` | none | struct `P` with stage-1 paths, ICLabel thresholds, merge/QC warning parameters | direct code inspection | found and usable | Central stage-1 parameter source. Also referenced by the misplaced stage-2 QC driver. |
| `r01_eeg_iclabel_prune_and_metadata.m` | Yes | `notebooks/1_eeg_sensor/r01_eeg_iclabel_prune_and_metadata.m` | `r01_eeg_iclabel_prune_and_metadata_driver.m` | raw EEGLAB `.set` tree, output base dir, QC table dir, ICLabel policy params | `withICA/*.set`, `clean/*.set`, `eeg_ic_prune_summary_<TAG>.csv`, `eeg_ic_prune_lists_<TAG>.csv`, per-run ICLabel CSVs | direct code inspection | found and usable | Self-contained apart from EEGLAB/ICLabel runtime dependencies. |
| `r01_batch_export_bst_exclusions_Fevents.m` | Yes | `notebooks/1_eeg_sensor/r01_batch_export_bst_exclusions_Fevents.m` | `r01_batch_export_bst_exclusions_Fevents_driver.m` | Brainstorm DB root, output dir, optional file filter, raw-link `data_0raw_*.mat` files | per-run `*_bst_exclusions.tsv`, `bst_exclusions_batch_summary.csv` | direct code inspection | found and usable | Batch wrapper is present, but it is not runnable end-to-end without the missing single-run exporter below. |
| `r01_export_bst_exclusions_Fevents.m` | No executable repo file found | not found as `.m` or archive helper; only referenced by callers/comments | `r01_batch_export_bst_exclusions_Fevents.m` and archived `r01_batch_export_bst_exclusions_Fevents.txt` | likely `in_mat` = Brainstorm raw-link MAT; `out_tsv` = per-run exclusion TSV path | likely `*_bst_exclusions.tsv` with at least `label`, `start_sec`, `end_sec`, probably `source`; labels expected to include `BAD`, `boundary`, `bad_boundary` | caller signature, downstream merge helper, manual docs, manuscript wording | partially recoverable | Most important gap. Interface is inferable; exact low-level Brainstorm `F.events` extraction behavior is not validated in-repo. |
| `r01_batch_merge_exclusions_union.m` | Yes | `notebooks/1_eeg_sensor/r01_batch_merge_exclusions_union.m` | `r01_batch_merge_exclusions_union_driver.m` | folder of `*_bst_exclusions.tsv`, merge options | `*_excl_union.tsv` and per-run union QC CSVs | direct code inspection | found and usable | Batch wrapper is present, but depends on the archive-only helper below. |
| `r01_merge_exclusions_union` | Yes, but not as an active stage-1 `.m` file | `notebooks/_archive_raw_original_names/1_eeg preprocessing/r01_merge_exclusions_union.txt` | `r01_batch_merge_exclusions_union.m`, `r01_merge_exclusions_union_folder_driver.m` | `in_tsv`, `out_tsv`, `labels`, `adjacency_tol_sec`, `min_dur_sec`, `write_qc_csv` | `*_excl_union.tsv`, sibling `<base>_qc.csv` | archived helper body | found but misplaced | Fully inspectable. Recovery looks straightforward because the full function body survives in archive form. |
| `r01_qc_excl_union_folder` | Yes, but not as an active stage-1 `.m` file | `notebooks/_archive_raw_original_names/1_eeg preprocessing/r01_qc_excl_union_folder.txt` | `r01_qc_excl_union_folder_driver.m` | `union_dir`, `out_dir`, optional `bst_db_root`, warning thresholds | `excl_union_qc_summary.csv` | archived helper body | found but misplaced | Fully inspectable. This is the folder-level exclusion QC layer that feeds run-level QC. |

## B. Closely Related Stage-1 QC / Table-S1 Dependencies Found Outside `notebooks/1_eeg_sensor/`

| Dependency | Exists in repo? | Where | Called by | Likely inputs | Likely outputs | Inference basis | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `r01_eeg_runlevel_qc_gates.m` | Yes | `notebooks/2_eeg_source/r01_eeg_runlevel_qc_gates.m` and archived `notebooks/_archive_raw_original_names/1_eeg preprocessing/r01_eeg_runlevel_qc_gates.txt` | `notebooks/2_eeg_source/r01_eeg_runlevel_qc_gates_driver.m` | raw EEGLAB dir, stage-1 pruned EEG dir, stage-1 QC tables dir, stage-1 exclusion QC dir | `eeg_run_qc_gates_<TAG>.csv`, `include_manifest.csv`, `include_manifest_<TAG>.csv`, `exclude_manifest.csv`, `exclude_manifest_<TAG>.csv`, `exclude_stems_<TAG>.txt` | direct code inspection plus identical hash of active/archived copies | found but misplaced | This is the manuscript-facing run-level QC/Table-S1 logic for stage 1, but it currently lives in stage 2. |
| `r01_eeg_runlevel_qc_gates_driver.m` | Yes | `notebooks/2_eeg_source/r01_eeg_runlevel_qc_gates_driver.m` | manual execution only | `r01_params()`, then stage-1 QC paths | same outputs as above | direct code inspection | found but misplaced | There is no equivalent active stage-1 driver in `notebooks/1_eeg_sensor/`. |
| dedicated Table-S1 generator script | No standalone file found | not found | none found | unclear | unclear | repo search plus `docs/figure_table_map.md` | partially recoverable | No separate Table-S1 notebook/script was found. Current evidence suggests Supplementary Table S1 is derived directly from `r01_eeg_runlevel_qc_gates.m` outputs rather than from a distinct table-building file. |

## C. Data-File Dependencies Referenced By Stage-1 Code

These are not repo helper functions, but they are exact file-pattern dependencies that determine whether stage 1 can be rerun.

| File dependency | Referenced by | Exists in repo? | Likely role | Recoverability / notes |
| --- | --- | --- | --- | --- |
| raw EEGLAB input tree: `**/*.set` under `root_raw_eeglab` | `r01_eeg_iclabel_prune_and_metadata.m`, `r01_eeg_runlevel_qc_gates.m` | No example data in repo | author-preprocessed sensor-level EEG input | Expected external dataset input; not a code gap. |
| Brainstorm raw-link MATs: `data_0raw_*.mat` under `<bst_db_root>/data/**` | `r01_batch_export_bst_exclusions_Fevents.m`, archived `r01_qc_excl_union_folder.txt` | No example files in repo | source of Brainstorm `BAD` / `boundary` event exports; optional source of run duration | Expected external dataset input; not a code gap, but critical for exporter reconstruction. |
| `*_bst_exclusions.tsv` | `r01_batch_merge_exclusions_union.m`, `r01_merge_exclusions_union_folder_driver.m` | No | exported Brainstorm exclusion intervals | Absence of any example TSVs makes the missing exporter harder to validate. |
| `*_excl_union.tsv` | archived `r01_qc_excl_union_folder.txt`, `r01_eeg_runlevel_qc_gates.m` indirectly via QC summary/union path | No | merged exclusion intervals used for QC and keep-mask construction | Can be regenerated once export + merge steps exist. |
| `excl_union_qc_summary.csv` | `r01_eeg_runlevel_qc_gates.m` | No | folder-level summary of merged exclusions; preferred source for usable fraction and exclusion flags | Can be regenerated once `r01_qc_excl_union_folder` is restored. |
| `eeg_run_qc_gates_<TAG>.csv`, include/exclude manifests | stage-1 manuscript outputs, read downstream by `notebooks/2_eeg_source/r01_readtable.m` | No | run-level QC and retained-run manifest supporting Supplementary Results 2.1 / Table S1 | Can be regenerated once the stage-1 QC gate file is moved back conceptually into stage 1 and the upstream exclusion artifacts exist. |

## D. External Runtime Dependencies (Not Repo Gaps, But Required)

| Runtime dependency | Referenced by | Repo status | Notes |
| --- | --- | --- | --- |
| EEGLAB `pop_loadset`, `eeg_checkset`, `pop_subcomp`, `pop_saveset` | `r01_eeg_iclabel_prune_and_metadata.m`, `r01_eeg_runlevel_qc_gates.m` | external | Explicitly checked in code. |
| ICLabel `iclabel` plugin | `r01_eeg_iclabel_prune_and_metadata.m` | external | Explicitly checked in code. |
| MATLAB `pwelch` | `r01_eeg_runlevel_qc_gates.m` | external/runtime | Used for EMG proxy calculation. |
| Brainstorm raw-link MAT schema (`F`, `F.events`, `F.prop.times`, etc.) | missing exporter, archived exclusion-QC helper | external/runtime data schema | The missing exporter almost certainly depends on this schema. |

## E. What Can Be Inferred Safely

### 1. `r01_merge_exclusions_union`

This helper is fully recoverable from archive provenance. The archived body specifies:

- required columns: `label`, `start_sec`, `end_sec`
- default included labels: `BAD`, `boundary`, `bad_boundary`
- default merge settings:
  - `adjacency_tol_sec = 0.0`
  - `min_dur_sec = 0.0`
- output:
  - `*_excl_union.tsv`
  - sibling QC CSV named `<base>_qc.csv`

This is a placement problem, not a scientific ambiguity problem.

### 2. `r01_qc_excl_union_folder`

This helper is also fully recoverable from archive provenance. The archived body specifies:

- input folder: `*_excl_union.tsv`
- optional Brainstorm DB lookup for run duration
- output:
  - `excl_union_qc_summary.csv`
- warning/flag logic including:
  - `NEG_START`
  - `NEG_END`
  - `SWAPPED_SE`
  - `OVERLAP`
  - `NONPOS_DUR`
  - `TINY_INTERVAL`
  - `HUGE_INTERVAL`
  - `HIGH_EXCL_FRAC`
  - `TOUCHING_WINDOWS`

This is also a placement problem, not a scientific ambiguity problem.

### 3. `r01_eeg_runlevel_qc_gates`

This file is not missing. It is misplaced. Evidence:

- active copy: `notebooks/2_eeg_source/r01_eeg_runlevel_qc_gates.m`
- archived provenance copy: `notebooks/_archive_raw_original_names/1_eeg preprocessing/r01_eeg_runlevel_qc_gates.txt`
- SHA256 of both files is identical

This means the stage-1 QC / Table-S1 logic can be pulled back into stage 1 without reconstructing its behavior from scratch.

### 4. `r01_export_bst_exclusions_Fevents`

This helper is only partially recoverable.

What can be inferred with reasonable confidence:

- function signature is almost certainly:
  - `r01_export_bst_exclusions_Fevents(in_mat, out_tsv)`
- `in_mat` is a Brainstorm raw-link MAT, likely containing `F.events`
- `out_tsv` is a per-run exclusion export
- downstream consumers expect columns at least:
  - `label`
  - `start_sec`
  - `end_sec`
  - probably `source`
- downstream labels expected:
  - `BAD`
  - `boundary`
  - possibly `bad_boundary`

What is not yet validated in-repo:

- exact Brainstorm field access pattern used by the original exporter
- whether events were expanded, normalized, or re-labeled before writing
- whether `bad_boundary` was generated by the exporter or only preserved downstream
- exact TSV column order and any additional provenance columns

Because no frozen `*_bst_exclusions.tsv` examples exist in the repo, a reimplementation would still need careful validation against Brainstorm raw-link files or an original lab-side copy of the exporter.

## F. Additional Provenance Signals

1. Several stage-1 drivers prepend an external lab path:
   - `addpath('C:\EEGFMRI\hmm\R01_rerun\03_code\matlab');`

   This strongly suggests some helpers, especially the missing exporter, may historically have lived outside the current repo snapshot.

2. `docs/manual_steps.md` and the manuscript context agree that stage-1 exclusion export should be limited to:
   - `boundary`
   - manually marked `BAD`

   and that `QRS` is retained rather than censored.

3. `docs/figure_table_map.md` still lists Supplementary Table S1 as a placeholder sourced from stage 1, which is consistent with the current absence of a dedicated table-building notebook.

## G. Bottom-Line Dependency Assessment

### Found and usable

- `r01_params.m`
- `r01_eeg_iclabel_prune_and_metadata.m`
- `r01_batch_export_bst_exclusions_Fevents.m` as a wrapper
- `r01_batch_merge_exclusions_union.m` as a wrapper

### Found but misplaced

- `r01_merge_exclusions_union`
- `r01_qc_excl_union_folder`
- `r01_eeg_runlevel_qc_gates.m`
- `r01_eeg_runlevel_qc_gates_driver.m`

### Partially recoverable

- `r01_export_bst_exclusions_Fevents.m`
- dedicated Table-S1 generator, if one ever existed as a separate file

### Missing and not yet reconstructable

- none at the level of exact symbol names already identified

The main practical blocker remains `r01_export_bst_exclusions_Fevents.m`. Everything downstream of that helper is either present, archive-recoverable, or misplaced but intact.

## H. Implementation Readiness

Stage 1 is **not yet ready for full end-to-end implementation**.

What is ready:

- ICLabel pruning / clean-set export
- recovery of union-merge helper from archive
- recovery of folder-level exclusion QC helper from archive
- reclassification of run-level QC / Table-S1 logic back into stage 1

What is still blocked:

- exact recovery of the Brainstorm event exporter `r01_export_bst_exclusions_Fevents.m`
- validation of its output schema and event-handling behavior, because no example exported TSVs are present in the repo

Practical implication:

- a stage-1 public refactor can proceed structurally
- a scientifically faithful, rerunnable stage-1 implementation still needs either:
  - the original exporter helper, or
  - frozen example `*_bst_exclusions.tsv` outputs, or
  - a validated reconstruction against actual Brainstorm raw-link MAT files
