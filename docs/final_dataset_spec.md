# Final Dataset Specification

This document defines the **canonical final analysis dataset** used for the manuscript:

**Fusion hidden Markov modeling reveals a dominant backbone state and transient alternatives in simultaneous resting-state EEG-fMRI**

It is intended to serve as the frozen reference for the final paper workflow during repository refactoring and public release preparation.

## Purpose

This file exists to prevent ambiguity during refactoring.

If older notebooks, comments, exports, or intermediate results disagree with the values below, this document should be treated as the reference point for the final paper dataset unless explicitly updated.

## Final analysis dataset identity

The final fusion-HMM analysis used the:

- **no-lag** EEG-BOLD fusion design
- **15-TR minimum retained segment length**
- **final selected model order: K = 3**

This is the canonical dataset and model configuration referred to throughout the manuscript and supplement.

## Source data scope

The final retained dataset was built from:

- **15 eyes-open resting-state runs**
- **12 healthy adult participants**
- simultaneous EEG-fMRI recordings from the open-access dataset used in the manuscript

## Shared temporal framework

- **BOLD TR:** 2.1 s
- **EEG sampling rate:** 250 Hz

EEG and BOLD were aligned on a common TR axis using timestamp-based reconciliation of the raw EEG timeline and the preprocessed EEG timeline before TR-level masking.

## Final TR-retention rules

A BOLD TR was retained only if it passed both the EEG coverage rule and the sample-completeness gate.

### 1. Base EEG coverage rule
A TR was retained directly if at least:

- **70% of the TR duration** contained usable EEG

### 2. Hybrid rescue rule
A partially contaminated TR could still be retained if:

- at least **50% of the TR** was usable EEG, **and**
- the usable EEG formed **one contiguous block**
- spanning at least **50% of the TR**

### 3. Sample-completeness gate
After interval masking, each TR also had to contain at least:

- `max(50, 0.65 × expected EEG samples per TR)`

This excluded sparsely sampled, incomplete, or NaN-prone EEG bins, including tail bins beyond valid EEG coverage.

## Final temporal design

The final manuscript dataset used:

- **same-TR EEG only**
- **no lag terms**

EEG entered the model only as parcel-wise same-TR power values.

## Final segment-export rule

After TR-level masking, only contiguous retained stretches of at least:

- **15 TR**

were exported for downstream modeling.

## Final observation vector definition

Each retained TR contributed one **400-dimensional** fusion observation vector:

- **200 BOLD parcel features**
- **200 EEG parcel-power features**

### BOLD block
- parcel-wise BOLD PC1 values
- feature columns: **0–199**

### EEG block
- parcel-wise same-TR EEG power values
- EEG power defined as mean squared signal within the TR from the gain-normalized parcel-PC1 signal
- feature columns: **200–399**

## Final feature-space definition before model fitting

Before HMM fitting, the retained 400-dimensional observations were reduced by modality-specific PCA:

- **40 BOLD PCs**
- **40 EEG PCs**
- **80 modeled dimensions total**

## Final retained-data totals

The final no-lag, 15-TR-minimum fusion dataset contained:

- **15 runs**
- **71 retained contiguous segments**
- **3550 retained TRs**
- **124.25 usable minutes**

These values are the canonical retained-data totals for the final manuscript dataset.

## Final model-order decision

Model order was selected by:

- leave-one-subject-out cross-validation across **K = 2–12**
- held-out test free energy
- 1-SE model-selection rule
- shortlist stability analysis using matched BOLD state signatures across folds

The final selected model order was:

- **K = 3**

## Final full-data model identity

The final full-data solution used:

- the canonical no-lag, 15-TR-minimum dataset
- **K = 3**
- full state covariance matrices
- run-wise normalization within modality
- modality-specific PCA
- multi-seed fitting and anti-collapse screening

## Interpretation scope

The final K = 3 model was used to generate:

- temporal summaries
- BOLD state reconstructions
- descriptive cross-modal BOLD-EEG reconstructions
- parcelized cortical maps

These downstream summaries are part of the final paper workflow and should be interpreted in light of the final dataset definition above.

## Refactor policy note

During repository cleanup:

- do not silently change any value listed in this file
- do not merge in older alternative dataset variants unless clearly labeled
- do not treat older lagged or differently filtered exports as part of the final paper dataset
- flag any mismatch between older notebooks and this dataset specification

## Short reference summary

For quick reference, the final paper dataset is:

- no-lag
- minimum segment length = 15 TR
- TR = 2.1 s
- EEG sampling rate = 250 Hz
- 200 BOLD + 200 EEG features
- 400 features per retained TR
- 40 + 40 PCA dimensions
- 15 runs
- 71 segments
- 3550 retained TRs
- 124.25 usable minutes
- final model order = **K = 3**