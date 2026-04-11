# Refactor Plan: `notebooks/1_eeg_sensor/`

## Scope

This is a planning-only first pass for the manuscript stage corresponding to:

- Main Methods 2.2.1
- Supplementary Methods 1.1
- Supplementary Results 2.1
- Supplementary Table S1

No files were moved or rewritten in this pass.

## Context Used

Reviewed before planning:

- `AGENTS.md`
- `docs/methods_map.md`
- `docs/manual_steps.md`
- `docs/final_dataset_spec.md`
- `docs/reproducibility_notes.md`
- `docs/figure_table_map.md`
- `docs/_manuscript_reference/FULL_MANUSCRIPT.docx`
- `docs/_manuscript_reference/SUPPLEMENTAL_MATERIALS.docx`

To resolve missing helper references, I also checked related provenance/context files outside this folder:

- `notebooks/2_eeg_source/r01_eeg_runlevel_qc_gates.m`
- `notebooks/_archive_raw_original_names/1_eeg preprocessing/r01_merge_exclusions_union.txt`
- `notebooks/_archive_raw_original_names/1_eeg preprocessing/r01_qc_excl_union_folder.txt`
- `notebooks/_archive_raw_original_names/1_eeg preprocessing/r01_eeg_runlevel_qc_gates.txt`

These cross-folder checks are contextual only. The assessment below remains focused on `notebooks/1_eeg_sensor/`.

## High-Level Assessment

`notebooks/1_eeg_sensor/` currently contains **8 MATLAB `.m` files and no notebooks**:

- 1 substantive sensor-preprocessing function
- 2 batch wrappers for Brainstorm exclusion export and interval merging
- 1 shared parameter file
- 4 thin driver scripts

The folder does represent the correct manuscript stage, but it is **fragmented and only partially self-contained**:

- the ICLabel pruning step is present and largely complete
- the Brainstorm exclusion export step depends on a missing helper: `r01_export_bst_exclusions_Fevents.m`
- the union merge and union-QC helpers are not present in this folder, although archived provenance copies exist
- the run-level QC gate logic that best matches Supplementary Results 2.1 and Table S1 currently lives in `notebooks/2_eeg_source/`, not here

That means the current folder is best understood as a partially refactored stage scaffold rather than a clean public-facing stage-1 package.

## Manuscript Alignment Summary

Based on the manuscript and supplement, this stage should cover the following sequence:

1. start from author-preprocessed EEGLAB `.set` files
2. prune ICA components using the conservative ICLabel reject-artifacts rule
3. save both a traceable ICA-retaining file and a Brainstorm-facing cleaned file
4. perform manual exclusion marking in Brainstorm using `boundary` and `BAD`
5. export Brainstorm exclusions
6. union-merge overlapping/touching exclusion intervals
7. summarize usable EEG, high-frequency dominance, bad-channel burden, and retained-segment continuity for Supplementary Results 2.1 and Table S1

The current folder captures steps 2, 5, and 6 directly, but steps 4 and 7 are split across documentation, missing helpers, archive provenance, and one stage-1 QC script currently housed in stage 2.

## Per-File Assessment

| File | Scientific purpose | Likely manuscript linkage | Inputs | Outputs | Current role | Overlap | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `r01_params.m` | Centralizes thresholds, paths, protocol root, merge settings, and QC warning settings for the rerun. | Supports all scripted parts of Main 2.2.1 and Supplementary Methods 1.1. Encodes the paper-facing `reject_artifacts` rule at 0.70. | None; returns a MATLAB struct. | `P` struct with paths and thresholds. | Core helper, not a results-producing analysis file by itself. | Used by all driver scripts. Also retains a historical `brain_threshold` branch that does not appear to be the final paper path. | **Keep logic, rewrite presentation.** In the public stage, this should become a repo-local config block or helper, not a standalone public notebook. Historical alternate-policy knobs should be clearly marked as non-paper paths. |
| `r01_eeg_iclabel_prune_and_metadata.m` | Performs ICLabel-based pruning, reconstructs channel EEG after IC removal, writes traceable `withICA` sets plus Brainstorm-facing `clean` sets, and exports IC/run metadata tables. | Main 2.2.1; Supplementary Methods 1.1. This is the clearest scripted implementation of the manuscript’s conservative ICLabel pruning step. | Author-preprocessed EEGLAB `.set` files, EEGLAB on path, ICLabel on path, policy parameters. | `withICA/*.set`, `clean/*.set`, `eeg_ic_prune_summary_<TAG>.csv`, `eeg_ic_prune_lists_<TAG>.csv`, per-run ICLabel tables. | Core final-paper stage logic. | Overlaps only with its thin driver. Also contains an older alternate policy branch (`brain_threshold`) that is not the manuscript-default path. | **Keep, but rewrite as a public-facing stage file.** This should anchor the first cleaned stage-1 notebook/script. The `reject_artifacts` 0.70 policy should be foregrounded; the alternate policy should remain archival or clearly labeled as historical. |
| `r01_eeg_iclabel_prune_and_metadata_driver.m` | Minimal execution wrapper around `r01_eeg_iclabel_prune_and_metadata`. | No separate manuscript content; just runs the step above. | `r01_params.m`, hard-coded `addpath`, MATLAB path setup. | Same outputs as the main pruning function. | Thin convenience wrapper. | Fully subsumed by `r01_eeg_iclabel_prune_and_metadata.m` and any future notebook parameter cell. | **Archive only.** Replace with a notebook top section or a cleaner repo-local runner later. |
| `r01_batch_export_bst_exclusions_Fevents.m` | Scans Brainstorm raw-link MAT files, exports marked exclusion intervals, and writes a batch summary. | Main 2.2.1; Supplementary Methods 1.1. This is the scripted handoff from manual Brainstorm marking to standardized exclusion TSVs. | Brainstorm database root, raw-link `data_0raw_*.mat` files, filter tag, and a missing helper `r01_export_bst_exclusions_Fevents.m`. | `*_bst_exclusions.tsv`, `bst_exclusions_batch_summary.csv`. | Core stage logic in intent, but incomplete in the current repo state. | Overlaps with its driver and sits immediately adjacent to the union-merge step. | **Rewrite and merge with downstream exclusion processing.** This belongs in a cleaned post-Brainstorm exclusion notebook/script, but only after the missing single-run exporter is recovered or reimplemented from provenance. |
| `r01_batch_export_bst_exclusions_Fevents_driver.m` | Runs the batch Brainstorm exclusion export with shared params. | No separate manuscript content. | `r01_params.m`, hard-coded `addpath`. | Same outputs as batch export. | Thin convenience wrapper. | Fully subsumed by the batch export function and future notebook parameter cells. | **Archive only.** |
| `r01_batch_merge_exclusions_union.m` | Batch-merges exported exclusion intervals into non-overlapping union windows. | Main 2.2.1; Supplementary Methods 1.1. Also feeds later alignment and run-level QC. | Folder of `*_bst_exclusions.tsv`, merge parameters, and missing in-folder helper `r01_merge_exclusions_union`. | `*_excl_union.tsv` and per-run QC CSVs. | Core stage logic in intent, but incomplete in the current folder because the single-run helper is absent here. | Strongly overlaps with `r01_merge_exclusions_union_folder_driver.m`, which re-implements the loop directly. | **Merge into the cleaned exclusion-processing notebook/script.** Restore the single-run merge helper from archive provenance and remove the duplicate batch-loop pattern in the public version. |
| `r01_merge_exclusions_union_folder_driver.m` | Alternate batch loop for union-merging exclusion TSVs. | No separate manuscript content. | `r01_params.m`, `*_bst_exclusions.tsv`, hard-coded `addpath`, missing single-run helper. | `*_excl_union.tsv`. | Thin duplicate wrapper. | Duplicates the loop already embodied by `r01_batch_merge_exclusions_union.m`. | **Archive only.** This is duplicate orchestration rather than distinct scientific logic. |
| `r01_qc_excl_union_folder_driver.m` | Runs folder-level QC over merged exclusion windows and writes summary warnings. | Indirectly supports Supplementary Results 2.1 and Table S1, but does not itself produce the final run-level gate table. | `r01_params.m`, merged union TSV folder, hard-coded `addpath`, missing helper `r01_qc_excl_union_folder`. | Expected `excl_union_qc_summary.csv` via the missing helper. | Thin wrapper pointing to absent logic. | Should connect to the true run-level QC/Table S1 logic, which currently lives outside this folder. | **Archive only as a wrapper.** Fold the underlying union QC plus run-level QC/Table-S1 summary logic into a single cleaned QC notebook/script for stage 1. |

## Overlap and Fragmentation Findings

### 1. Driver sprawl

Four of the eight files are drivers. They carry almost no scientific logic and mainly:

- call `r01_params()`
- add a hard-coded MATLAB path
- invoke one function

These should not become four separate public-facing files.

### 2. Duplicate batch-loop logic for exclusion merging

`r01_batch_merge_exclusions_union.m` and `r01_merge_exclusions_union_folder_driver.m` both implement the same batch orchestration idea for union merging. This is a good candidate for consolidation.

### 3. Stage-1 QC logic is split across multiple locations

The manuscript-facing run-level QC step that supports Supplementary Results 2.1 and Table S1 is not contained cleanly inside this folder:

- `r01_qc_excl_union_folder_driver.m` expects a helper that is not present here
- the fuller gate logic appears in `notebooks/2_eeg_source/r01_eeg_runlevel_qc_gates.m`
- archived provenance copies exist under `_archive_raw_original_names/1_eeg preprocessing/`

Scientifically, that QC belongs to stage 1, not stage 2.

### 4. Missing helper dependency blocks a fully traceable stage-1 export path

`r01_batch_export_bst_exclusions_Fevents.m` calls `r01_export_bst_exclusions_Fevents.m`, but that helper was not found in:

- `notebooks/1_eeg_sensor/`
- `notebooks/2_eeg_source/`
- the archive provenance folder that corresponds to stage 1

This is the clearest current blocker to a clean public refactor of the Brainstorm exclusion-export step.

## Cross-Folder Dependencies That Matter for This Stage

These are not in `notebooks/1_eeg_sensor/`, but they materially affect the stage-1 plan.

| File | Why it matters | Planning implication |
| --- | --- | --- |
| `notebooks/2_eeg_source/r01_eeg_runlevel_qc_gates.m` | Contains the run-level QC gates and manifest-writing logic that best matches Supplementary Results 2.1 and Supplementary Table S1. | Stage-1 QC is currently misplaced. The cleaned stage-1 set should absorb this logic conceptually, even if the eventual implementation decides to keep helpers elsewhere. |
| `_archive_raw_original_names/1_eeg preprocessing/r01_merge_exclusions_union.txt` | Preserves the single-run union-merge implementation expected by the stage-1 wrappers. | This helper should be restored or rewritten before public stage-1 cleanup. |
| `_archive_raw_original_names/1_eeg preprocessing/r01_qc_excl_union_folder.txt` | Preserves the single-folder exclusion QC helper expected by the stage-1 wrapper. | This logic belongs in the cleaned stage-1 QC notebook/script. |
| `_archive_raw_original_names/1_eeg preprocessing/r01_eeg_runlevel_qc_gates.txt` | Confirms that the current stage-2 placement of run-level QC came from a stage-1 provenance file. | Supports moving the public narrative for run-level QC back to stage 1 during refactor. |

## Proposed Cleaned GitHub-Facing Stage-1 Set

The current folder should not be cleaned by preserving one public file per current `.m` file. A clearer manuscript-aligned public stage would be:

### `10_eeg_prune_iclabel_and_export_clean_sets`

Purpose:

- run the manuscript-default ICLabel pruning step
- export both traceable `withICA` files and Brainstorm-facing `clean` files
- write pruning metadata tables

Main source material:

- `r01_eeg_iclabel_prune_and_metadata.m`
- `r01_eeg_iclabel_prune_and_metadata_driver.m`
- `r01_params.m`

Status:

- scripted
- core final-paper stage file

### `11_brainstorm_exclusion_marking_manual`

Purpose:

- explicitly document the manual Brainstorm step between pruning and downstream exclusion processing
- point users to `docs/manual_steps.md` rather than pretending this step is fully automated

Main source material:

- `docs/manual_steps.md` sections 1 and 2
- manuscript Main 2.2.1 / Supplementary Methods 1.1 wording

Status:

- manual/hybrid note, not an automated notebook

Recommended form:

- short markdown file or notebook with no executable claims

### `12_export_and_union_merge_brainstorm_exclusions`

Purpose:

- export Brainstorm `BAD` / `boundary` exclusions
- standardize them into per-run union intervals
- write batch and per-run exclusion summaries

Main source material:

- `r01_batch_export_bst_exclusions_Fevents.m`
- `r01_batch_export_bst_exclusions_Fevents_driver.m`
- `r01_batch_merge_exclusions_union.m`
- `r01_merge_exclusions_union_folder_driver.m`
- archived `r01_merge_exclusions_union.txt`
- missing `r01_export_bst_exclusions_Fevents.m` must be recovered or re-authored from provenance

Status:

- scripted/hybrid handoff
- core final-paper stage file

Why merge these pieces:

- they all sit immediately after the manual Brainstorm marking step
- the current files are mostly orchestration wrappers
- combining them improves clarity without collapsing distinct manuscript stages

### `13_eeg_run_qc_and_table_s1`

Purpose:

- compute run-level usable-EEG, spectral, bad-channel, and continuity summaries
- write include/exclude manifests
- produce the manuscript-facing run summary corresponding to Supplementary Results 2.1 and Supplementary Table S1

Main source material:

- `r01_qc_excl_union_folder_driver.m`
- archived `r01_qc_excl_union_folder.txt`
- `notebooks/2_eeg_source/r01_eeg_runlevel_qc_gates.m`
- archived `r01_eeg_runlevel_qc_gates.txt`

Status:

- scripted
- core final-paper stage file

Why keep separate from notebook 12:

- this is the natural place to explain manuscript-facing QC interpretation
- it maps directly to Supplementary Results 2.1 and Table S1
- it should remain distinct from the simpler mechanical export/merge step

## Current-to-Proposed Mapping

| Current file(s) | Proposed destination |
| --- | --- |
| `r01_eeg_iclabel_prune_and_metadata.m` + `r01_eeg_iclabel_prune_and_metadata_driver.m` + config from `r01_params.m` | `10_eeg_prune_iclabel_and_export_clean_sets` |
| manual Brainstorm exclusion marking currently documented only in `docs/manual_steps.md` | `11_brainstorm_exclusion_marking_manual` |
| `r01_batch_export_bst_exclusions_Fevents.m` + `r01_batch_export_bst_exclusions_Fevents_driver.m` + `r01_batch_merge_exclusions_union.m` + `r01_merge_exclusions_union_folder_driver.m` + archived merge helper | `12_export_and_union_merge_brainstorm_exclusions` |
| `r01_qc_excl_union_folder_driver.m` + archived exclusion-QC helper + `notebooks/2_eeg_source/r01_eeg_runlevel_qc_gates.m` | `13_eeg_run_qc_and_table_s1` |
| All four driver scripts as standalone public files | archive/provenance only |

## Keep / Merge / Split / Rewrite / Archive Decisions

### Keep and rewrite

- `r01_eeg_iclabel_prune_and_metadata.m`
- `r01_params.m` as internal config logic, not as a user-facing notebook

### Merge into cleaned public stage files

- `r01_batch_export_bst_exclusions_Fevents.m`
- `r01_batch_merge_exclusions_union.m`
- `r01_qc_excl_union_folder_driver.m` together with its missing/archive helper logic
- stage-1 run-level QC logic currently living in `notebooks/2_eeg_source/r01_eeg_runlevel_qc_gates.m`

### Archive only

- `r01_eeg_iclabel_prune_and_metadata_driver.m`
- `r01_batch_export_bst_exclusions_Fevents_driver.m`
- `r01_merge_exclusions_union_folder_driver.m`
- `r01_qc_excl_union_folder_driver.m` as a standalone wrapper

### No clear split needed

The main pruning function does not need to be split. It already forms one coherent scientific unit.

## Risks, Ambiguities, and Notes to Keep Explicit

### 1. Missing exporter helper

`r01_export_bst_exclusions_Fevents.m` was not found in the repo or stage-1 archive. The public refactor should not pretend the exclusion-export step is fully reproducible until this is resolved.

### 2. Stage bleed into `notebooks/2_eeg_source/`

The run-level QC gate logic that best matches Supplementary Results 2.1 and Table S1 is currently housed in stage 2. That is a manuscript-alignment problem in the repo layout, even if the code itself is still scientifically useful.

### 3. EMG-threshold interpretation needs careful wording

There is a wording/code tension that should be documented rather than silently harmonized:

- the manuscript text says the EMG proxy was descriptive and that run retention depended on the combined QC profile rather than any single metric
- the current run-level gate code uses explicit threshold parameters such as `max_emg_db`

This may be harmless in practice because the exclude manifest was empty, but the cleaned public notebook should explain exactly whether those thresholds were hard gates, descriptive screens, or both.

### 4. Historical alternate policy remains in code

The pruning code still supports a `brain_threshold` branch, but the manuscript-facing workflow uses the conservative `reject_artifacts` policy at 0.70. The public stage should make the paper path unmistakable and avoid foregrounding the alternate branch.

### 5. Hard-coded local paths

Current drivers and params use absolute Windows paths such as:

- `C:\EEGFMRI\hmm\R01_rerun`
- `C:\brainstorm_db\eegfmri_R01_ICRej70`

These should be rewritten later as explicit user-configurable parameters, but without changing thresholds, file-selection logic, or protocol identity.

## Recommended Next Implementation Pass

When stage 1 moves from planning to implementation, the safest order is:

1. keep all originals and provenance copies untouched
2. create the cleaned stage-1 public file set listed above
3. restore or reconstruct the missing exclusion-export and exclusion-QC helpers before claiming a runnable exclusion workflow
4. pull the run-level EEG QC/Table S1 narrative back into stage 1
5. update `docs/methods_map.md` after the cleaned filenames are finalized

## Bottom Line

This folder is close to the right manuscript stage, but it is not yet a clean public-facing stage package.

The clearest refactor path is:

- keep the ICLabel pruning logic as the anchor
- insert an explicit manual Brainstorm note between pruning and downstream scripts
- consolidate exclusion export plus union merge into one public stage file
- consolidate run-level QC plus Table S1 generation into a second public stage file
- archive the thin drivers
- resolve the missing helper and cross-folder stage bleed explicitly rather than silently
