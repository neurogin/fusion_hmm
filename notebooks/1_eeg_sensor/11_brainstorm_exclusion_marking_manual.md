# 11 Brainstorm Exclusion Marking Manual

## What This File Is

This is the manual/hybrid stage that sits between the two scripted stage-1 entry files:

- `10_eeg_prune_iclabel_and_export_clean_sets.m`
- `12_export_and_union_merge_brainstorm_exclusions.m`

It is intentionally a short public-facing guide, not an executable notebook.

## Manuscript Linkage

- Main Methods 2.2.1
- Supplementary Methods 1.1
- Supplementary Results 2.1
- Supplementary Table S1

## What You Do In This Stage

Open the cleaned EEG runs in Brainstorm and mark conservative exclusion intervals that will later be exported and merged by the scripted stage-1 pipeline.

This step is required because the public repository does **not** pretend Brainstorm exclusion marking was fully automated.

## Main Source Of Truth

Follow the detailed instructions in:

- `docs/manual_steps.md`
  - Section 1. Manual EEG exclusion marking in Brainstorm
  - Section 2. Brainstorm protocol setup and EEG import

This file is a short stage-level map to that documentation.

## Inputs

- Cleaned EEGLAB files from step 10:
  - `*_clean.set`
- Brainstorm protocol:
  - `eegfmri_R01_ICRej70`

## Event Names To Use

- `boundary` (lowercase)
- `BAD` (uppercase)

## Key Stage-1 Policy

The exclusion policy is intentionally conservative:

- keep inherited `boundary` events
- add manual `BAD` segments only for genuinely problematic intervals
- do not use `QRS` as a routine censoring marker

`QRS` events are retained unless they fall inside an already excluded interval.

## What The Next Scripted File Will Export

After Brainstorm marking is complete, the next public stage-1 script:

- `12_export_and_union_merge_brainstorm_exclusions.m`

will export and merge these annotations into:

- `*_bst_exclusions.tsv`
- `*_excl_union.tsv`

## Quick Reminder Before Leaving Brainstorm

- confirm `boundary` still exists
- confirm `BAD` only marks genuine problem intervals
- confirm `QRS` was not used as a routine censoring label
- confirm the run is saved before moving to the export step

## Important Refactor Note

This repository keeps Brainstorm exclusion marking explicit as a manual/hybrid dependency on purpose.

The cleaned public stage-1 workflow does not describe this step as if it were fully scripted.
