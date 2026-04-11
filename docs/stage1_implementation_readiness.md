# Stage 1 Implementation Readiness

## Scope

This pass reassesses stage 1 only:

- `notebooks/1_eeg_sensor/`
- directly related recovered/provenance files
- manuscript and documentation context for:
  - Main Methods 2.2.1
  - Supplementary Methods 1.1
  - Supplementary Results 2.1
  - Supplementary Table S1

No public-facing stage files were moved or rewritten in this pass.

## Context Reviewed

- `AGENTS.md`
- `docs/methods_map.md`
- `docs/manual_steps.md`
- `docs/final_dataset_spec.md`
- `docs/reproducibility_notes.md`
- `docs/figure_table_map.md`
- `docs/refactor_plan_1_eeg_sensor.md`
- `docs/stage1_dependency_gap_report.md`
- `docs/_manuscript_reference/FULL_MANUSCRIPT.docx`
- `docs/_manuscript_reference/SUPPLEMENTAL_MATERIALS.docx`
- `docs/_manuscript_reference/recovered_stage1_helpers/r01_export_bst_exclusions_Fevents.m`

I also rechecked:

- `notebooks/1_eeg_sensor/r01_batch_export_bst_exclusions_Fevents.m`
- `notebooks/_archive_raw_original_names/1_eeg preprocessing/r01_merge_exclusions_union.txt`
- `notebooks/_archive_raw_original_names/1_eeg preprocessing/r01_qc_excl_union_folder.txt`
- `notebooks/2_eeg_source/r01_eeg_runlevel_qc_gates.m`

## Executive Verdict

The recovered helper `r01_export_bst_exclusions_Fevents.m` **does resolve the main stage-1 code-availability blocker**.

Why:

1. it matches the expected two-argument interface used by the active batch exporter
2. it writes the TSV schema expected by downstream merge logic
3. it exports only the manuscript-relevant exclusion labels:
   - `BAD`
   - `boundary`
   - `bad_boundary`
4. it preserves the manuscript/manual rule that `QRS` is not exported as a censoring label

As a result, stage 1 is now **ready for implementation of the cleaned public-facing stage files**.

That said, stage 1 is **not yet empirically validated end-to-end inside this repo snapshot**, because:

- no example Brainstorm raw-link MATs are present
- no frozen `*_bst_exclusions.tsv` examples are present
- the recovered helper’s handling of point-shaped events still needs to be preserved and later checked against real raw-link files
- the EMG-proxy wording/code tension in run-level QC still needs to be documented explicitly during implementation

So the blocker is no longer “missing code.” The remaining work is now implementation plus later validation/documentation.

## 1. Recovered Helper Assessment

### Helper under review

- `docs/_manuscript_reference/recovered_stage1_helpers/r01_export_bst_exclusions_Fevents.m`

### Interface fit

The recovered helper signature is:

- `r01_export_bst_exclusions_Fevents(rawlink_mat, out_tsv)`

This matches the active batch caller exactly:

- `r01_batch_export_bst_exclusions_Fevents.m`
  calls `r01_export_bst_exclusions_Fevents(in_mat, out_tsv);`

Interface verdict:

- **match**

### Output schema fit

The recovered helper writes a tab-delimited table with columns:

- `label`
- `start_sec`
- `end_sec`
- `source`

This matches downstream expectations:

- `r01_batch_export_bst_exclusions_Fevents.m` reads the TSV and counts:
  - `BAD`
  - `boundary`
  - `bad_boundary`
- archived `r01_merge_exclusions_union.txt` expects TSV columns:
  - `label`
  - `start_sec`
  - `end_sec`
  - `source`
- archived `r01_qc_excl_union_folder.txt` only needs the merged output later, but the merge helper is compatible with this schema

Schema verdict:

- **match**

### Manuscript/manual consistency

The recovered helper exports only:

- `BAD`
- `boundary`
- `bad_boundary`

This is consistent with the documented stage-1 exclusion policy:

- manuscript/manual rule: exclusions are based on inherited `boundary` events plus manually marked `BAD`
- `QRS` is retained and is not used as a censoring rule

The recovered helper is also consistent with a useful supplementary nuance:

- the supplement wording indicates that exclusions were limited to inherited boundary events and manually marked BAD intervals, and that exported files could also contain `bad_boundary` labels

So `bad_boundary` is not a new behavior invented by the recovered file. It is compatible with the supplementary stage-1 description and with downstream code that already expects it.

Scientific-consistency verdict:

- **consistent with the manuscript workflow, with one technical caveat noted below**

## 2. Detailed Compatibility Table

| Check | Result | Assessment |
| --- | --- | --- |
| Batch caller interface | `rawlink_mat, out_tsv` | exact match |
| Batch summary label counts | `BAD`, `boundary`, `bad_boundary` | exact match |
| Merge-helper required columns | `label`, `start_sec`, `end_sec` and expected `source` provenance | exact match |
| QRS handling | not exported | consistent with manuscript/manual policy |
| Empty-output behavior | writes an empty table with the expected column names | good for reproducibility; downstream code should be able to handle empty exports |
| Raw-link load pattern | `load(rawlink_mat, 'F')` then `F = F(1)` | consistent with other stage-1 Brainstorm-facing helpers |
| Event storage assumptions | supports `2xN`, `1xN`, and `Nx2` shapes | robust and plausible for Brainstorm event storage |

## 3. Remaining Uncertainties From The Recovered Helper

### A. Point-event handling needs later data validation

The recovered helper converts `1xN` point events into zero-length intervals:

- `[t; t]`

This is a reasonable defensive choice, but it introduces a specific downstream uncertainty:

- if actual Brainstorm `boundary` events for this dataset are point-shaped rather than interval-shaped
- then zero-length exclusions could survive into the merged union TSVs
- and archived folder-level QC currently flags non-positive interval duration as `NONPOS_DUR`
- and run-level QC treats `NONPOS_DUR` as a severe exclusion failure

Why this does **not** reinstate the old blocker:

- the recovered helper is still structurally compatible with the stage-1 pipeline
- the code is now present and inspectable
- this is a validation issue, not a missing-logic issue

Why this still needs to be stated explicitly:

- without real Brainstorm raw-link files or frozen exported TSVs, I cannot prove whether the actual stage-1 runs store `boundary` as intervals or points

Implementation implication:

- preserve this recovered behavior as-is in the first implementation pass
- do not silently “improve” it yet
- add a note that actual Brainstorm event shapes should be checked when test data are available

### B. The `source` column is additive provenance, not a manuscript issue

The recovered helper writes:

- `source = 'brainstorm'`

This is not specified in the manuscript, but it is already compatible with downstream merge expectations and is scientifically harmless. It should be preserved unless a later validation pass finds a historical reason to change it.

### C. End-to-end run validation is still unavailable in-repo

This repo snapshot still does not contain:

- raw Brainstorm MAT examples
- exported `*_bst_exclusions.tsv` examples
- merged `*_excl_union.tsv` examples

So the recovered helper resolves the code gap, but not the empirical validation gap.

## 4. Reassessed Stage-1 Readiness

## What is now no longer blocked

- ICLabel pruning and clean-set export
- Brainstorm exclusion export code availability
- exclusion TSV schema compatibility with downstream merge logic
- archive recovery of union merge helper
- archive recovery of folder-level exclusion QC helper
- conceptual return of run-level QC / Table-S1 logic to stage 1

## What remains to be handled during implementation

- promote recovered/archive helpers into the cleaned stage-1 structure without overwriting originals
- keep the Brainstorm marking step explicitly manual/hybrid
- preserve the current EMG/QC logic while documenting the manuscript/code wording tension
- preserve the recovered exporter’s event-shape behavior unless later validation justifies a change

## Implementation readiness verdict

Stage 1 is now:

- **ready for implementation of the cleaned public-facing stage files**

Stage 1 is not yet:

- **fully validated as a turnkey rerun from the current repo snapshot**

That distinction matters. The recovered helper removed the main planning/recovery blocker. It did not remove the need for later test-data validation.

## 5. Remaining Non-Blocking Risks To Carry Forward

### 1. EMG-proxy wording/code tension remains

This issue is unchanged by the recovered exporter.

The manuscript says:

- run retention was based on the combined QC profile
- the EMG proxy was descriptive
- it was not a stand-alone exclusion threshold

But `r01_eeg_runlevel_qc_gates.m` still contains a hard gate:

- `max_emg_db = 3.0`

Implementation consequence:

- do not silently delete or relax this gate
- do not silently claim it was never a hard threshold
- carry the current code logic forward and document the discrepancy clearly in the cleaned stage-1 QC file

### 2. Stage-1 QC logic is still physically misplaced

The stage-1 run-level QC / Table-S1 logic remains in:

- `notebooks/2_eeg_source/r01_eeg_runlevel_qc_gates.m`

This is no longer a blocker, but it is still a manuscript-alignment cleanup task.

### 3. Archive-only helpers still need promotion

These helpers are recoverable and inspectable, but still not active stage-1 files:

- `r01_merge_exclusions_union`
- `r01_qc_excl_union_folder`

Again, this is an implementation task, not a blocking uncertainty.

## 6. Refined Cleaned Stage-1 File Plan

Because the existing stage-1 code is MATLAB-based and already scientifically specific, the safest cleaned stage-1 public set should remain:

- MATLAB `.m` stage scripts for scripted steps
- one markdown/manual file for the Brainstorm-only step

This minimizes translation risk and preserves current behavior better than forcing stage 1 into notebooks.

### Recommended cleaned stage-1 set

| Proposed cleaned file | Classification | Purpose | Primary source material |
| --- | --- | --- | --- |
| `10_eeg_prune_iclabel_and_export_clean_sets.m` | public-facing main notebook/script | manuscript-default ICLabel pruning, auditable `withICA` export, Brainstorm-facing `clean` export, ICLabel QC tables | `r01_eeg_iclabel_prune_and_metadata.m`, `r01_params.m` |
| `11_brainstorm_exclusion_marking_manual.md` | manual/hybrid documentation only | documents the required Brainstorm marking step between clean-set export and scripted exclusion processing; should explicitly point back to `docs/manual_steps.md` | `docs/manual_steps.md`, manuscript Methods 2.2.1 / Supplementary Methods 1.1 |
| `12_export_and_union_merge_brainstorm_exclusions.m` | public-facing main notebook/script | scans Brainstorm raw-link MATs, exports `BAD` / `boundary` / `bad_boundary` exclusions, writes per-run exclusion TSVs, merges union intervals, writes batch summaries | `r01_batch_export_bst_exclusions_Fevents.m`, recovered `r01_export_bst_exclusions_Fevents.m`, archived `r01_merge_exclusions_union.txt`, current batch merge wrapper |
| `13_eeg_run_qc_and_table_s1.m` | public-facing main notebook/script | computes run-level exclusion QC plus run-level EEG QC summaries and manifests supporting Supplementary Results 2.1 / Table S1 | archived `r01_qc_excl_union_folder.txt`, `notebooks/2_eeg_source/r01_eeg_runlevel_qc_gates.m` |
| `r01_params.m` or cleaned equivalent config helper | helper function | centralizes stage-1 paths, tag logic, thresholds, and warning parameters | current `r01_params.m` |
| `r01_export_bst_exclusions_Fevents.m` | helper function | single-run Brainstorm exclusion exporter used by the public stage-1 exclusion script | recovered helper in `docs/_manuscript_reference/recovered_stage1_helpers/` |
| `r01_merge_exclusions_union.m` | helper function | single-run union-merge helper for exported exclusions | archived provenance `r01_merge_exclusions_union.txt` |
| `r01_qc_excl_union_folder.m` | helper function | folder-level QC over merged exclusion windows | archived provenance `r01_qc_excl_union_folder.txt` |
| `r01_eeg_runlevel_qc_gates.m` | helper function | run-level EEG QC and manifest writer supporting Supplementary Table S1 | current misplaced copy in `notebooks/2_eeg_source/` plus archived stage-1 provenance |

## 7. What Should Be Archive/Provenance Only

These should remain provenance sources rather than cleaned public-facing entry points:

- `r01_eeg_iclabel_prune_and_metadata_driver.m`
- `r01_batch_export_bst_exclusions_Fevents_driver.m`
- `r01_batch_merge_exclusions_union_driver.m`
- `r01_merge_exclusions_union_folder_driver.m`
- `r01_qc_excl_union_folder_driver.m`
- archive `.txt` helper copies under `notebooks/_archive_raw_original_names/1_eeg preprocessing/`
- recovered helper source under `docs/_manuscript_reference/recovered_stage1_helpers/`

The recovered helper file is important provenance, but its current location under manuscript references is not the final public stage location.

## 8. Bottom Line

The recovered `r01_export_bst_exclusions_Fevents.m` resolves the main stage-1 blocker.

It:

- matches the active caller interface
- writes the expected TSV schema
- respects the documented `boundary` / `BAD` / no-`QRS` censoring policy
- is also compatible with the supplementary note that exported files may contain `bad_boundary`

Remaining issues are now implementation and validation issues, not missing-code blockers.

Final readiness call:

- **Yes, stage 1 is now ready for implementation**

with these explicit caveats:

- preserve the recovered exporter behavior without silent cleanup
- keep Brainstorm marking explicitly manual/hybrid
- document the EMG-proxy wording/code tension rather than harmonizing it silently
- note that raw-link event-shape behavior still needs later validation on real data
