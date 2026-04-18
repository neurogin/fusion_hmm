# Manual and Hybrid Steps

This document records the analysis steps in this repository that are **manual** or **hybrid**, meaning they are not performed entirely by the notebooks/scripts alone.

The purpose of this file is to make the Brainstorm- and GUI-dependent parts of the workflow explicit, reproducible, and easy to follow.

## Why this document exists

This repository accompanies the manuscript:

**Fusion hidden Markov modeling reveals a reproducible shared state architecture in simultaneous resting-state EEG-fMRI**

Some parts of the workflow were performed in Brainstorm or other GUI-based tools and then fed into downstream MATLAB/Python scripts. These steps are scientifically important and therefore must be documented in the repository even if they are not executable code.

## Definitions

- **Manual step**: performed directly in a GUI and not automatically reproducible from a notebook/script alone.
- **Hybrid step**: manual software setup or export followed by scripted downstream processing.
- **Scripted step**: performed fully by notebook/script code.

## Current status

This document is being built during the repo refactor phase.  
It reflects the final paper workflow and the current practical procedure used in the project.

## How to use this document with the cleaned workflow

Use this file together with the stage-specific public files:

- Stage 1 manual handoff: `notebooks/1_eeg_sensor/step11_brainstorm_exclusion_marking_manual.md`
- Stage 2 manual handoff: `notebooks/2_eeg_source/step21_brainstorm_volume_source_and_atlas_import_manual.md`

The public MATLAB and notebook entry files point back here when a step remains genuinely manual or hybrid.

---

# 1. Manual EEG exclusion marking in Brainstorm

## Status
Manual / hybrid

## Purpose
To create a conservative, reproducible set of EEG time exclusions that removes:
1. discontinuities already marked as `boundary`
2. additional visually identified artifact periods marked manually as `BAD`

These exclusions are later exported and merged into union exclusion files for downstream EEG QC, feature extraction, and EEG–BOLD alignment.

## Inputs
- Brainstorm protocol: `eegfmri_R01_ICRej70`
- Cleaned EEGLAB files: `*_clean.set`
- Existing event groups may include `boundary`, `QRS`, sync markers, triggers

## Event naming conventions
Naming is case-sensitive.

Required event groups:
- `boundary` (lowercase)
- `BAD` (uppercase)

## Key policy
Exclusions are limited to:
- `boundary`
- manually marked `BAD`

Cardiac events (`QRS`) are **not** used for censoring unless they fall inside an already excluded segment.

This matches the manuscript-facing rule:

> “We excluded time periods corresponding to acquisition discontinuities (boundary events) and additional visually identified artifacts (BAD segments) marked in Brainstorm. Cardiac events (QRS markers) were retained and were not used for censoring unless they occurred within excluded segments.”

## Practical marking guidance

### Boundary handling
For each `boundary` marker:
- inspect a short window before and after it
- if the signal looks stable immediately adjacent, leave it as `boundary` only
- if there is ringing, amplitude shift, baseline disturbance, or other instability extending beyond the boundary, add a `BAD` segment spanning the unstable interval

### BAD marking criteria
Mark a time interval as `BAD` when there is:
- severe attenuation or near-flatlining across multiple channels
- abrupt amplitude jumps or discontinuous baseline shifts not already fully captured by `boundary`
- prolonged movement artifact
- electrode pop
- unstable or clearly non-physiological signal
- post-boundary transient instability

Do **not** mark as `BAD` solely because:
- QRS-related transients appear
- normal oscillatory activity varies naturally

### Style guidance
- prefer one continuous `BAD` segment over many tiny adjacent segments
- mark the whole unstable interval, not just the peak artifact

## Procedure

### Step 1. Open run in Brainstorm
- Open protocol `eegfmri_R01_ICRej70`
- Open the relevant imported EEG run from `*_clean.set`
- Confirm that `boundary` exists
- Create/use `BAD` as needed

### Step 2. Treat boundary events as excluded anchors
- Keep `boundary` events intact
- Inspect signal immediately around each boundary

### Step 3. Add `BAD` segments manually
- Select the bad interval in the EEG viewer
- Add the interval under `BAD` (uppercase)

### Step 4. Save and verify
Before closing the run:
- confirm `boundary` still exists
- confirm `BAD` contains only genuine problematic intervals
- confirm QRS was not used as routine censoring
- confirm excluded time is not excessive unless the run is genuinely poor

## Outputs
Per run, later exported by MATLAB scripts:
- `*_bst_exclusions.tsv`
- `*_excl_union.tsv`

## Downstream scripted handoff
These outputs are consumed by the cleaned Stage-1 public files:

- `notebooks/1_eeg_sensor/step12_export_and_union_merge_brainstorm_exclusions.m`
- `notebooks/1_eeg_sensor/step13_eeg_run_qc_and_table_s1.m`

---

# 2. Brainstorm protocol setup and EEG import

## Status
Manual / hybrid

## Purpose
To create the Brainstorm protocol, import subject anatomy and cleaned EEG, and prepare each run for exclusion marking and source localization.

## Inputs
- Cleaned EEG files: `*_clean.set`
- Subject FreeSurfer/fMRIPrep anatomy folders

## Procedure
For each subject/run:

1. Start Brainstorm
2. Create or open protocol:
   - `eegfmri_R01_ICRej70`
3. Create subject entry
4. Import anatomy folder automatically
5. Import cleaned EEG `.set` file
6. Review electrode registration
7. Project electrodes on surface if needed
8. Save the imported run

## Notes
Brainstorm may generate:
- electrode placement figures
- registration metadata
- subject/run visual diagnostics

These may be useful QC artifacts even if they are not directly included in the manuscript.

---

# 3. EEG volume source localization in Brainstorm

## Status
Manual / hybrid

## Purpose
To generate subject-specific EEG source models in a common MNI-compatible framework for downstream parcel extraction.

## Inputs
- Brainstorm protocol with imported subject anatomy
- Cleaned EEG runs
- Marked exclusions
- Noise covariance computed from the recordings

## Final workflow used
The project’s final workflow used a **volumetric** EEG source model.

Key settings documented in project notes:

- MNI normalization: **linear, maff8 / SPM12 registration**
- BEM surfaces: **3 layers**
- vertices per layer: **2432**
- skull thickness: **4 mm**
- forward model: **OpenMEEG BEM**
- integration: **adaptive integration**
- source space: **MRI volume**, isotropic **3-mm** grid
- conductivities:
  - scalp = 1.0
  - skull = 0.0125
  - brain = 1.0
- inverse/source reconstruction:
  - **current density**
  - **unconstrained** orientations

## Practical Brainstorm sequence
For each subject/session:

1. Load anatomy automatically
2. Run MNI normalization (`maff8`)
3. Generate BEM surfaces
4. Compute noise covariance from recordings
5. Compute head model with OpenMEEG BEM
6. Compute EEG sources using current density, unconstrained orientation

## Important note
Subject grid sizes can differ across subjects.  
Therefore, downstream export code must **not assume a constant dipole count across subjects**.

## Output examples
- source results files such as `results_MN_EEG_KERNEL_*.mat`
- subject-specific volume source grids
- Brainstorm tessellation / atlas-linked files

---

# 4. Atlas preparation for EEG and BOLD parcellation

## Status
Hybrid

## Purpose
To prepare the Schaefer 2018 200-parcel / 7-network atlas in a form usable by both:
- Brainstorm (Windows GUI)
- downstream scripted BOLD and EEG parcellation workflows

## Atlas used
- Schaefer2018
- 200 parcels
- 7 networks
- MNI152NLin2009cAsym space

## Atlas preparation steps

### 4.1 Install and fetch atlas using TemplateFlow
In the appropriate Python environment:

- install `templateflow`
- set `TEMPLATEFLOW_HOME` to a Windows-visible mounted path
- fetch the Schaefer 200/7 atlas
- verify the atlas and `.tsv` label table exist

### 4.2 Create Brainstorm label `.txt`
Brainstorm expects a `.txt` label file with the same basename as the NIfTI atlas.

Each line format:
`<label_integer> <label_name>`

This `.txt` is created from the `.tsv` label table by notebook/script.

## Important note
The atlas file is **subject-agnostic** as a template-space atlas, but it becomes subject-specific in Brainstorm through each subject’s MNI normalization transform and source-grid mapping.

This means:
- one atlas file can be reused across subjects
- parcel coverage can still vary across subjects
- coverage diagnostics remain essential

---

# 5. Import atlas into Brainstorm as volume scouts

## Status
Manual / hybrid

## Purpose
To define parcel membership directly on the subject-specific EEG volume source grid inside Brainstorm.

## Procedure
For each subject/session:

1. Ensure MNI normalization exists
2. Open a source result that displays the volume source grid
3. Go to Scout / Atlas tools
4. Load atlas using:

**“Volume mask or atlas (dilated, MNI space)”**

5. Select the Schaefer atlas NIfTI file
6. Allow Brainstorm to read the matching `.txt` label file

## Why “dilated, MNI” was used
The EEG volume source grid is coarser than the original atlas voxel resolution.

Using **dilated** MNI import:
- reduces parcel dropout caused by grid/voxel mismatch
- improves parcel representation stability
- gives a source-grid-aware scout definition inside Brainstorm

## Important interpretation
This step is central to the project’s **hybrid approach**:
- Brainstorm handles atlas-to-subject alignment and source-grid parcel assignment
- MATLAB/Python handle export, parcel PCA, metadata, and reproducible QC summaries

---

# 6. Extract volume-grid scouts from Brainstorm tess files

## Status
Hybrid

## Purpose
To save standardized subject/session-specific scout files whose indices live in the same index space as the inverse kernel’s volumetric source grid.

## Rationale
Brainstorm writes atlas information into the tessellation file after atlas loading.  
These scouts are then extracted and saved in a standardized form for downstream parcel export.

This avoids the earlier instability caused by direct voxel-space mapping outside Brainstorm.

## Procedure
For each subject/session:

1. Load atlas in Brainstorm as `dilated, MNI`
2. Ensure Brainstorm has written the atlas into the tess file
3. Run the batch scout extraction MATLAB scripts
4. Save standardized scout file, e.g.:

`scout_Schaefer2018_200_7N_dilated_MNI.mat`

## Required contents of standardized scout file
- `Scouts` (200 structs)
- `TessNbVertices` matching the subject/run grid size

## Important note
Different subjects may show different volume-grid sizes (for example ~39k, ~43k, ~50k).  
This is acceptable, but the downstream exporter must use the correct scout file for each subject/run.

## Downstream scripted handoff

These outputs feed the cleaned public Stage-2 files:

- `notebooks/2_eeg_source/step22_extract_volgrid_scouts_from_brainstorm_tess.m`
- `notebooks/2_eeg_source/step23_export_eeg_parcel_pc1_and_gain_normalize.m`
- `notebooks/2_eeg_source/step24_qc_eeg_source_alignment_table_s2.m`
- `notebooks/2_eeg_source/step25_generate_eeg_parcel_export_qc_sidecars.m`
- `notebooks/2_eeg_source/step26_qc_eeg_parcel_exports_table_s3_and_figures_s2_s4.ipynb`

---

# 7. Hybrid EEG parcel PC extraction workflow

## Status
Hybrid

## Purpose
To extract parcel-level EEG source time series and metadata using Brainstorm-defined parcel membership and MATLAB-based parcel export.

## Hybrid logic
Brainstorm defines:
- atlas-to-subject alignment
- source-grid parcel membership

MATLAB scripts perform:
- parcel PC1/PC2 extraction
- metadata export
- `.npy` writing
- coverage and QC summaries

The cleaned public MATLAB entry script for this stage is:

- `notebooks/2_eeg_source/step23_export_eeg_parcel_pc1_and_gain_normalize.m`

It needs EEGLAB on the MATLAB path for batch loading of the cleaned `.set`
files from Stage 1. The optional `writeNPY` dependency is also needed if
you want the downstream-ready `.npy` sidecars, including `*_time_sec.npy`.

## Components of the workflow

### 7.1 Brainstorm manual preparation
- import anatomy
- compute inverse kernels per run
- load atlas as `dilated, MNI`
- save/use standardized scout files

### 7.2 One-run exporter
Inputs:
- kernel results file
- scout atlas file
- output directory

Outputs:
- parcel `.mat`
- `.npy` files
- PVE
- number of dipoles
- valid-parcel metadata
- run-level coverage summaries

### 7.3 Batch exporter
Finds runs, identifies matching scout files, calls the one-run exporter repeatedly, and writes batch summaries.

## Important note
Brainstorm initialization remains advisable even when batch functions mainly operate on file paths, because some lower-level exporter calls still depend on Brainstorm functions.

---

# 8. QC and sanity-check outputs related to manual/hybrid EEG source workflow

## Status
Hybrid

## Purpose
To verify that the manual + hybrid source-parcellation workflow produced stable, interpretable parcel outputs.

## Key checks documented in project notes
- full parcel coverage across runs
- minimal parcel loss under minimum-dipole thresholding
- stable sign convention
- gain normalization behavior
- PVE summaries
- run-level parcel export QC

## Interpretation
These QC steps support the claim that the final atlas alignment and source-space parcelization workflow was stable and suitable for downstream HMM analysis.

---

# 9. BOLD atlas alignment note

## Status
Hybrid / scripted interface note

## Purpose
To document that EEG and BOLD did not require voxel-for-voxel identity, but instead were aligned through a shared anatomical reference framework.

## Key concept
The goal is:
- consistent atlas definition in shared template space
- modality-specific resampling / source-grid representation
- quantified residual mismatch through QC summaries

This is important context for the multimodal alignment described in the manuscript.

## Downstream scripted handoff

This BOLD-side atlas note feeds the cleaned public Stage-3 files:

- `notebooks/3_bold/step30_map_schaefer200_to_bold_run_grids.ipynb`
- `notebooks/3_bold/step31_export_bold_parcel_pc1_with_nuisance_regression.ipynb`

---

# 10. What these manual steps feed into

These manual/hybrid steps feed into later scripted pipeline stages, including:

- EEG exclusion export and union masking
- EEG source parcel PC extraction
- BOLD parcel extraction
- timestamp-based EEG–BOLD alignment
- final no-lag 15-TR-minimum fusion dataset construction
- HMM fitting and descriptive reconstructions

---

# 11. What should be documented later

As the repo refactor continues, this file should be extended with:

- exact filenames of the Brainstorm-derived outputs
- exact notebook/script names that consume those outputs
- screenshots or small GUI references if useful
- notes on any subject-specific exceptions

## Known exception already noted
- `sub-13_ses-01` used anatomy from `sub-13_ses-02`

---

# 12. Refactor policy for Codex and future contributors

When working with this repository:

- do not rewrite manual Brainstorm steps as if they were fully automated
- preserve the distinction between manual, hybrid, and scripted stages
- keep manuscript terminology aligned with the actual workflow
- document manual dependencies explicitly in notebooks and docs
- preserve output provenance wherever possible

