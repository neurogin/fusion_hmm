# 21 Brainstorm Volume Source And Atlas Import Manual

## What This File Is

This is the manual/hybrid stage that sits between the public-facing Stage-2 atlas-preparation notebook and the later scripted scout-extraction and parcel-export files.

It sits between:

- `20_prepare_schaefer200_atlas_for_brainstorm.ipynb`
- `extract_volgrid_scouts_from_brainstorm_tess_22.m`
- `export_eeg_parcel_pc1_and_gain_normalize_23.m`

It is intentionally a short public-facing guide, not an executable notebook.

## Manuscript Linkage

- Main Methods 2.2.2
- Main Methods 2.2.3
- Supplementary Methods 1.2
- Supplementary Methods 1.3
- Supplementary Results 2.2
- Supplementary Results 2.3
- Supplementary Table S2
- Supplementary Table S3
- Supplementary Figures S1A,B and S2-S4

## What You Do In This Stage

Use Brainstorm to complete the subject-specific source-localization and atlas-import steps that the public MATLAB files depend on later.

This step is required because the public repository does **not** pretend that Brainstorm source modeling and atlas import were fully scripted.

## Main Source Of Truth

Follow the detailed instructions in:

- `docs/manual_steps.md`
  - Section 3. EEG volume source localization in Brainstorm
  - Section 4. Atlas preparation for EEG and BOLD parcellation
  - Section 5. Import atlas into Brainstorm as volume scouts
  - Section 6. Extract volume-grid scouts from Brainstorm tess files
  - Section 7. Hybrid EEG parcel PC extraction workflow
  - Section 8. QC and sanity-check outputs related to manual/hybrid EEG source workflow

This file is a stage-level map to those procedures.

## Inputs

- Cleaned EEGLAB runs from Stage 1:
  - `*_clean.set`
- Brainstorm protocol:
  - `eegfmri_R01_ICRej70`
- Atlas files from step 20:
  - `tpl-MNI152NLin2009cAsym_res-01_atlas-Schaefer2018_desc-200Parcels7Networks_dseg.nii.gz`
  - matching `.txt` label file with the same basename

## Brainstorm Steps That Must Stay Manual / Hybrid

For each subject/session:

1. import or confirm subject anatomy
2. run Brainstorm MNI normalization
3. generate the 3-layer BEM surfaces
4. compute the EEG head model with OpenMEEG
5. compute the volumetric EEG source model
6. load the Schaefer atlas with:
   - `Volume mask or atlas (dilated, MNI space)`

These steps are hybrid because Brainstorm performs subject-specific alignment and source-grid atlas assignment inside the GUI and Brainstorm database.

## Outputs The Next Scripted Files Expect

After the Brainstorm work is complete, the later public Stage-2 scripts expect:

- run-level inverse-kernel files such as:
  - `results_MN_EEG_KERNEL_*.mat`
- subject/session anatomy folders such as:
  - `protocolRoot/anat/sub-XX_ses-YY/`
- tess files that now contain the imported `Volume` atlas entry

After that manual work is complete:

- `extract_volgrid_scouts_from_brainstorm_tess_22.m`
  extracts subject/session scout files from the Brainstorm tess files
- `export_eeg_parcel_pc1_and_gain_normalize_23.m`
  uses those scouts plus the kernel files and cleaned EEG to export parcel PCs and QC sidecars

## Known Project Note

The project documentation records one subject-specific exception:

- `sub-13_ses-01` used anatomy from `sub-13_ses-02`

Keep that exception visible. Do not silently normalize it away during refactor or reruns.

## Quick Reminder Before Leaving Brainstorm

- the final source result uses the intended volumetric source model
- the atlas import used the `dilated, MNI` volume-scout option
- the tess file contains the imported `Volume` atlas entry
- the run-level `results_MN_EEG_KERNEL_*.mat` files exist on disk

## What Comes Next

1. run `extract_volgrid_scouts_from_brainstorm_tess_22.m`
2. run `export_eeg_parcel_pc1_and_gain_normalize_23.m`
3. run `qc_eeg_source_alignment_table_s2_24.m`
4. open `25_qc_eeg_parcel_exports_table_s3_and_figures_s2_s4.ipynb`

## Refactor Note

This repository keeps the Brainstorm Stage-2 handoff explicit on purpose.

The cleaned public workflow does not describe this stage as if it were fully scripted.
