# Setup Guide

This document summarizes the software and path setup needed for the public workflow.

## 1. Software Stack

### MATLAB stages

Stage 1 and the MATLAB parts of Stage 2 require:

- MATLAB
- EEGLAB for the steps that read or write EEGLAB `.set` files
- ICLabel for Stage-1 component pruning
- Brainstorm for the manual or hybrid EEG-source workflow

Stage-2 parcel export is most useful with:

- a working `writeNPY` function on the MATLAB path

because downstream stages expect the exported `*_PC1_gnorm.npy` and `*_time_sec.npy` files.

### Python stages

The Python notebooks and helper modules use a shared scientific stack built around:

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `jupyterlab`

Additional stage-specific packages include:

- `templateflow` for atlas fetching and atlas-path resolution
- `nibabel` and `nilearn` for volumetric and surface-map stages
- `tensorflow` and `osl_dynamics` for Stage 5 and the Stage-6 final fit

## 2. Create The Python Environment

You can start from either:

- [environment.yml](../environment.yml)
- [requirements.txt](../requirements.txt)

Example with conda:

```bash
conda env create -f environment.yml
conda activate fusion_hmm
```

Example with pip in an existing environment:

```bash
pip install -r requirements.txt
```

## 3. TensorFlow And `osl_dynamics`

Stages 5 and 6 are the most environment-sensitive parts of the repo.

They depend on:

- `tensorflow`
- `osl_dynamics`

In practice:

- GPU-backed execution is the intended mode for the full LOSO sweep and final full-data fit
- CPU fallback may work for testing or small checks
- CPU fallback is usually much slower and less practical for the full manuscript workflow

Because TensorFlow installation details depend on your platform and GPU stack, you may need to adjust that part of the environment manually even after using the provided template files.

## 4. Important Path Concepts

### `project_root`

For the MATLAB workflow, `project_root` is the root folder holding the user data and derived outputs, for example:

- `01_raw/`
- `02_derivatives/`
- `04_qc/`

The public Stage-1 and Stage-2 settings files derive their input and output paths from this root.

### `brainstorm_protocol_root`

This is the actual Brainstorm protocol folder. It must directly contain:

- `data/`
- `anat/`

The public MATLAB workflow treats this as the single user-facing Brainstorm path concept.

### `TEMPLATEFLOW_ROOT`

The Stage-2, Stage-3, and Stage-6 atlas-based steps expect access to a readable TemplateFlow cache or equivalent atlas location.

## 5. Public Output Placement

The repository is meant to hold the workflow code and documentation, not the generated data products.

In normal use, keep generated outputs such as:

- derivatives
- QC tables
- model-selection outputs
- final model fits
- panel exports

outside version control.

## 6. Recommended First Read Order

Before running anything, read:

1. [README.md](../README.md)
2. [docs/public_workflow.md](public_workflow.md)
3. [docs/manual_steps.md](manual_steps.md)
4. the stage-specific `stepNN_*` file you plan to run first

## 7. Recommended First Run Order

1. Configure Stage-1 paths in `notebooks/1_eeg_sensor/helpers/stage1_eeg_sensor_settings.m`
2. Run Stage 1
3. Run Stage 2 Step 20
4. Complete the Stage-2 Brainstorm manual workflow in Step 21
5. Continue through Stage 2 to Stage 6 in order
