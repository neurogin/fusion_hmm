# Manual And Hybrid Steps

This document records the real manual and hybrid parts of the public workflow.

The repository does not pretend these steps are fully scripted. They are part of the reproducible analysis contract and must be followed consistently.

## Manual-Step Summary

The public workflow contains three main manual or hybrid boundaries:

1. Brainstorm EEG exclusion marking in Stage 1
2. Brainstorm source-model setup and atlas import in Stage 2
3. some later manuscript figure-panel assembly, even after the computational outputs have been generated

## Shared Path Concept

Both Stage 1 and Stage 2 use the same public-facing Brainstorm path concept:

- `brainstorm_protocol_root`

This should be the actual Brainstorm protocol folder that directly contains:

- `data/`
- `anat/`

The public MATLAB settings files derive the specific Brainstorm subfolders from that protocol root.

## Stage 1 Manual Step

### Public handoff file

- [step11_brainstorm_exclusion_marking_manual.md](../notebooks/1_eeg_sensor/step11_brainstorm_exclusion_marking_manual.md)

### What remains manual

- opening each imported EEG run in Brainstorm
- inspecting `boundary` events
- adding visually identified `BAD` intervals

### Exclusion policy that must be preserved

- use `boundary` and manual `BAD` intervals as the exclusion basis
- do not use `QRS` as a routine censoring rule
- keep event naming consistent with the public workflow notes

### Downstream scripted handoff

After manual marking, the scripted Stage-1 sequence continues with:

- `step12_export_and_union_merge_brainstorm_exclusions.m`
- `step13_eeg_run_qc_and_table_s1.m`

## Stage 2 Manual And Hybrid Steps

### Public handoff file

- [step21_brainstorm_volume_source_and_atlas_import_manual.md](../notebooks/2_eeg_source/step21_brainstorm_volume_source_and_atlas_import_manual.md)

### What remains hybrid

- creating or opening the Brainstorm protocol
- importing anatomy and cleaned EEG
- running MNI normalization
- building the BEM model
- computing the EEG source model
- importing the Schaefer atlas as Brainstorm volume scouts

### Important practical note

Stage 2 depends on Brainstorm outputs that are subject specific. The public MATLAB steps do not assume a constant source-grid size across subjects.

### Downstream scripted handoff

After the manual or hybrid Brainstorm work is complete, the scripted Stage-2 sequence continues with:

- `step22_extract_volgrid_scouts_from_brainstorm_tess.m`
- `step23_export_eeg_parcel_pc1_and_gain_normalize.m`
- `step24_qc_eeg_source_alignment_table_s2.m`
- `step25_generate_eeg_parcel_export_qc_sidecars.m`
- `step26_qc_eeg_parcel_exports_table_s3_and_figures_s2_s4.ipynb`

## Later Manual Assembly

Some later outputs still require honest manual assembly or layout choices even after the numerical content has been generated.

Examples:

- Figure 1 still includes schematic and layout choices beyond the Stage-4 support notebook
- `step65_optional_export_figure4_figure5_panels.ipynb` exports panel components, but the final composite figure layout remains manual

## What This Means For Reproducibility

Reproducing the public workflow requires both:

- running the scripted steps in the documented order
- following the manual or hybrid Brainstorm procedures consistently

Use this file together with [docs/public_workflow.md](public_workflow.md) and the stage-specific step files.
