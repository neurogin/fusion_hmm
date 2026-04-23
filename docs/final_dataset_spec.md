# Final Dataset Specification

This document defines the frozen manuscript dataset and modeling defaults used by the public workflow.

Repository steps, comments, and saved outputs should be interpreted against this specification unless a file explicitly says it is reporting a different branch for comparison or provenance.

## Canonical Dataset Identity

The final manuscript dataset uses:

- `DATA_VARIANT = "intermediate"`
- `FEATURE_MODE = "nolags"`
- minimum retained segment length = `15` TR
- final selected model order = `K = 3`

## Source Data Scope

The final retained dataset was built from:

- 15 eyes-open resting-state runs
- 12 participants
- simultaneous resting-state EEG-fMRI data from the open-access dataset used in the manuscript

## Shared Temporal Framework

- BOLD TR = `2.1` s
- EEG sampling rate = `250` Hz

EEG and BOLD are reconciled on the BOLD TR axis using the Stage-4 alignment workflow.

## TR Retention Rules

A TR is retained only if it passes all preserved Stage-4 gates:

1. base EEG coverage rule:
   at least 70% of the TR contains usable EEG
2. hybrid rescue rule:
   at least 50% of the TR is usable EEG, and that usable span forms one contiguous block covering at least 50% of the TR
3. sample-completeness gate:
   at least `max(50, 0.65 x expected EEG samples per TR)` remain after masking

## Final Temporal Design

The final manuscript dataset uses:

- same-TR EEG only
- no lag terms

## Final Segment Export Rule

After TR-level masking, only contiguous retained stretches of at least:

- `15` TR

are exported for downstream modeling.

## Final Observation Definition

Each retained TR contributes one 400-feature fusion observation:

- 200 BOLD parcel PC1 features
- 200 same-TR EEG parcel-power features

The EEG block is computed from gain-normalized parcel-PC1 signals on the same TR.

## PCA Before HMM Fitting

Before HMM fitting, the retained 400-feature observations are reduced to:

- 40 BOLD PCs
- 40 EEG PCs
- 80 modeled dimensions total

## Final Retained Totals

The canonical no-lag, 15-TR-minimum dataset contains:

- 15 runs
- 71 retained contiguous segments
- 3550 retained TRs
- 124.25 usable minutes

## Model-Selection Outcome

The public release preserves the manuscript model-selection story:

- broad LOSO screening over `K = 2..12`
- shortlist comparison centered on `K = 3` and `K = 5`
- final manuscript choice = `K = 3`

## Final Full-Data Model Identity

The final Stage-6 fit uses:

- the canonical Stage-4 retained dataset
- `K = 3`
- modality-specific PCA
- full state covariance matrices
- run-wise normalization within modality
- multi-seed fitting with seed screening and refit

## Short Reference Summary

For quick reference, the public manuscript path is:

- `intermediate + nolags + minlen15`
- 200 BOLD features + 200 EEG features
- 400 features per retained TR
- 3550 retained TRs across 71 retained segments
- final `K = 3`
