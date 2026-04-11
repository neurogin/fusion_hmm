# Repository Scope

This repository is being prepared as the public code and documentation companion for the manuscript:

**Fusion hidden Markov modeling reveals a reproducible shared state architecture in simultaneous resting-state EEG-fMRI**

## Current scope

At the current refactor stage, this repository includes:

- repository-level instructions in `AGENTS.md`
- manuscript workflow documentation in `docs/methods_map.md`
- manual and hybrid Brainstorm procedures in `docs/manual_steps.md`
- manuscript and supplement reference copies in `docs/_manuscript_reference/`
- analysis notebooks organized by major manuscript workflow stage in `notebooks/`
- an archive folder of original notebook versions in `notebooks/_archive_raw_original_names/`

## What is currently present in the workflow notebooks

The notebook folders currently cover the main manuscript analysis stages:

- EEG sensor preprocessing and exclusion handling
- EEG source localization and atlas-aligned parcel workflow
- BOLD preprocessing and parcel extraction
- EEG-BOLD timestamp alignment and fusion observation construction
- HMM model-order selection
- final full-data HMM fitting

Some summary, QC, figure, and table logic is still embedded within these earlier method notebooks.

## What is intentionally not yet finalized

The following parts of the intended public repo structure are not yet fully populated:

- `config/`
- `scripts/`
- `src/`
- `docs/figure_table_map.md`
- `docs/final_dataset_spec.md`
- `docs/reproducibility_notes.md`

Also, the folders:

- `notebooks/7_summaries/`
- `notebooks/8_figures/`
- `notebooks/9_tables/`

may currently remain empty because summary, figure, and table generation are still partly embedded in earlier method notebooks.

## What is not the goal of this repo right now

This repository is **not yet** intended to be:

- a fully polished turnkey pipeline
- a complete historical archive of every exploratory notebook
- a dump of all intermediate or abandoned analysis variants
- a repository containing raw dataset files or large private intermediate outputs

## Refactor-stage policy

During this stage, the repository focuses on:

- preserving scientific behavior
- aligning code organization with the manuscript methods flow
- documenting manual versus scripted steps honestly
- preserving original notebook provenance
- preparing a cleaner public-facing structure for the final paper repo

## Expected future additions

As the refactor continues, this repository is expected to gain:

- cleaner public-facing notebook names
- selected figure/table generator notebooks or scripts
- config templates
- reusable code in `src/`
- cleaner runnable entry points in `scripts/`
- additional documentation for figure/table mapping and reproducibility