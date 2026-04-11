# 11 Brainstorm Exclusion Marking Manual

## Status

Manual / hybrid stage. This is not a fully scripted notebook.

## Manuscript Linkage

- Main Methods 2.2.1
- Supplementary Methods 1.1
- Supplementary Results 2.1
- Supplementary Table S1

## Purpose

This stage sits between:

- `10_eeg_prune_iclabel_and_export_clean_sets.m`
- `12_export_and_union_merge_brainstorm_exclusions.m`

Its purpose is to create the conservative Brainstorm exclusion annotations that the later scripted stage exports and merges.

## Source Of Truth

Follow the detailed Brainstorm instructions in:

- `docs/manual_steps.md`
  - Section 1. Manual EEG exclusion marking in Brainstorm
  - Section 2. Brainstorm protocol setup and EEG import

This file is a short stage-level pointer, not a replacement for the full manual document.

## Inputs

- Cleaned EEGLAB files from step 10:
  - `*_clean.set`
- Brainstorm protocol:
  - `eegfmri_R01_ICRej70`

## Required Event Groups

- `boundary` (lowercase)
- `BAD` (uppercase)

## Key Policy

Exclusions are limited to:

- inherited `boundary` events
- manually marked `BAD` segments

`QRS` markers are retained and are not used as routine censoring markers unless they fall inside an already excluded interval.

## Outputs Used By The Next Scripted Stage

These manual Brainstorm annotations are exported later by:

- `12_export_and_union_merge_brainstorm_exclusions.m`

Expected downstream files:

- `*_bst_exclusions.tsv`
- `*_excl_union.tsv`

## Important Refactor Note

This repository intentionally keeps Brainstorm exclusion marking explicit as a manual/hybrid dependency.

The public stage-1 refactor does not pretend this step is fully automated.
