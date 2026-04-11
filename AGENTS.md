# AGENTS.md

## Project identity

This repository accompanies the manuscript:

**Fusion hidden Markov modeling reveals a reproducible shared state architecture in simultaneous resting-state EEG-fMRI**

The repository is currently in a **refactor and public-release preparation phase**.

The immediate goal is **not** to invent a new pipeline. The goal is to:

1. inspect the existing MATLAB and Python notebook codebase,
2. map code to the actual manuscript workflow,
3. identify overlapping, redundant, messy, or fragmented notebooks,
4. consolidate them into clearer public-facing pipeline notebooks/scripts,
5. preserve scientific behavior and manuscript alignment,
6. add documentation and explanatory notes suitable for a GitHub paper repository.

---

## Primary rule

**Preserve the science.**

Do not silently change:
- thresholds,
- model settings,
- dataset definitions,
- inclusion/exclusion logic,
- alignment rules,
- parcel/network mappings,
- PCA dimensionality,
- state labels/order,
- HMM fitting logic,
- or statistical interpretation

unless explicitly asked.

When uncertain, flag uncertainty instead of guessing.

---

## Read-this-first project context

Before planning or editing, prioritize the following files:

1. `docs/methods_map.md`
2. `docs/manual_steps.md`
3. manuscript reference files in `docs/_manuscript_reference/` (or equivalent reference folder)
4. `docs/final_dataset_spec.md` if present
5. `docs/reproducibility_notes.md` if present

Use these files to align refactoring decisions with the actual manuscript workflow.

---

## Repository state

This repo currently contains:
- documentation files under `docs/`
- configuration files under `config/`
- notebook folders organized by manuscript workflow under `notebooks/`
- empty or not-yet-finalized `scripts/` and `src/` folders for future cleanup
- `results/` and `assets/` for later outputs

Current notebook workflow folders:

- `notebooks/1_eeg_sensor/`
- `notebooks/2_eeg_source/`
- `notebooks/3_bold/`
- `notebooks/4_alignment/`
- `notebooks/5_hmm_selection/`
- `notebooks/6_hmm_final/`
- `notebooks/7_summaries/`
- `notebooks/8_figures/`
- `notebooks/9_tables/`
- `notebooks/_archive_raw_original_names/`

Important:
- `7_summaries/`, `8_figures/`, and `9_tables/` may remain sparse or empty during early refactoring.
- In the current codebase, many QC summaries, figures, and tables are embedded within earlier method notebooks.
- Do **not** force content into `7_summaries/`, `8_figures/`, or `9_tables/` unless doing so clearly improves public readability and reproducibility.

---

## Documentation routing

- `docs/methods_map.md` is the primary map from manuscript methods to repository structure.
- `docs/manual_steps.md` is the primary reference for Brainstorm GUI steps and other manual/hybrid procedures.
- Do not duplicate detailed manual procedures inside `AGENTS.md`; reference those docs instead.
- If notebook organization changes, update `docs/methods_map.md`.
- If a manual or hybrid dependency becomes clearer during refactoring, update `docs/manual_steps.md`.
- If dataset totals, run manifests, or frozen analysis settings are clarified, update `docs/final_dataset_spec.md` or `docs/reproducibility_notes.md` if present.

---

## Canonical manuscript workflow

All refactoring should follow the manuscript methods flow.

### Stage 1: EEG sensor preprocessing and exclusion handling
Corresponds to manuscript:
- Main Methods 2.2.1
- Supplementary Methods 1.1

This stage includes:
- author-preprocessed EEGLAB inputs
- ICLabel-based component rejection
- export of Brainstorm-facing cleaned EEG
- Brainstorm exclusion marking
- merging of BAD/boundary exclusion intervals
- run-level EEG QC summaries

### Stage 2: EEG source localization and atlas-aligned parcellation
Corresponds to manuscript:
- Main Methods 2.2.2
- Supplementary Methods 1.2

This stage includes:
- Brainstorm manual/source workflow
- MNI normalization
- BEM/source model settings
- volume-grid scout extraction
- Schaefer atlas alignment checks

### Stage 3: EEG parcel extraction, sign fixing, and gain normalization
Corresponds to manuscript:
- Main Methods 2.2.3
- Supplementary Methods 1.3

This stage includes:
- parcel PC1 extraction
- parcel support thresholding
- deterministic sign fixing
- gain normalization
- PVE summaries

### Stage 4: BOLD nuisance regression and parcel extraction
Corresponds to manuscript:
- Main Methods 2.3
- Supplementary Methods 1.4

This stage includes:
- confound design building
- nuisance regression
- selected transient regressor handling
- parcel PC1 extraction
- BOLD QC summaries

### Stage 5: Timestamp-based alignment and final observation matrix
Corresponds to manuscript:
- Main Methods 2.4
- Supplementary Methods 1.5

This stage includes:
- raw-to-preprocessed EEG timeline reconciliation
- projection of EEG usable intervals to the BOLD TR axis
- EEG-informed keep-mask construction
- no-lag observation matrix construction
- retained-segment export

### Stage 6: LOSO model-order selection and shortlist stability
Corresponds to manuscript:
- Main Methods 2.5.1
- Supplementary Methods 1.6–1.7

This stage includes:
- K sweep
- held-out free energy summaries
- 1-SE rule
- local minima shortlist
- cross-fold state matching
- stability metrics

### Stage 7: Final full-data K=3 model fit
Corresponds to manuscript:
- Main Methods 2.5.2
- Supplementary Methods 1.8

This stage includes:
- final retained dataset
- PCA reduction
- seed screening/refit
- final selected model
- final fit QC

### Stage 8: Temporal summaries and biological reconstructions
Corresponds to manuscript:
- Main Methods 2.6.1–2.6.4
- Supplementary Methods 1.9–1.11

This stage includes:
- FO, transitions, dwell times
- gamma rasters
- BOLD block reconstructions
- cross-modal block reconstructions
- nodal/parcel maps
- figure/table generation

---

## Canonical final dataset specification

Unless explicitly told otherwise, treat the following as the frozen paper dataset definition:

- no-lag design
- minimum retained segment length = 15 TR
- BOLD TR = 2.1 s
- EEG sampling rate = 250 Hz
- 200 BOLD parcel features
- 200 same-TR EEG parcel-power features
- 400 features per retained TR
- 3550 retained TRs
- 71 retained contiguous segments
- 124.25 usable minutes
- final selected model order = K = 3

If any code, comment, notebook, or output appears inconsistent with these values, do not “fix” it silently. Flag it clearly.

---

## Refactor objective

The main refactor task is to transform a messy collection of analysis notebooks into a cleaner GitHub-ready public pipeline.

This usually means:
- identifying notebooks that overlap heavily,
- determining whether multiple notebooks should become one cleaner notebook,
- determining whether one notebook should be split into separate clearer stages,
- rewriting filenames to be descriptive,
- adding plain-language explanatory markdown,
- separating manual versus scripted steps,
- preserving original notebooks in archive form.

---

## Archive policy

Original notebooks and scripts should be preserved.

Rules:
- never overwrite original notebooks in `notebooks/_archive_raw_original_names/`
- treat archived originals as provenance copies
- write cleaned/refactored versions into the main workflow folders
- preserve traceability from old names to new names where possible

If a notebook is superseded, note:
- original filename
- new filename
- reason for refactor

---

## Manual versus scripted steps

This repository contains both scripted and manual/hybrid analysis stages.

Important:
- Do not rewrite manual Brainstorm work as if it were fully scripted unless that is actually true.
- Where a notebook depends on manual Brainstorm setup or exported files, state that explicitly.
- Prefer honesty over false automation.

Manual or hybrid steps likely include:
- Brainstorm exclusion marking
- Brainstorm subject import and anatomy setup
- MNI normalization
- BEM generation
- source localization
- atlas loading and scout handling
- screenshot-based figure assembly for some supplementary assets

Whenever relevant, document these dependencies in markdown cells or docs.

---

## How to evaluate notebooks during refactor

When reviewing a notebook or script, determine:

1. What is its scientific purpose?
2. Which manuscript method section does it support?
3. What are its inputs?
4. What are its outputs?
5. Is it core to the final paper, supplementary, exploratory, or obsolete?
6. Does it overlap with another notebook?
7. Should it be:
   - kept as-is,
   - merged with another notebook,
   - split into clearer notebooks,
   - rewritten as a cleaner public-facing notebook,
   - archived only?

Do not merge notebooks merely because they are related.
Merge only when:
- they form one coherent manuscript stage,
- their inputs/outputs align naturally,
- and merging improves clarity without hiding important distinctions.

---

## Preferred refactor style

When rewriting code for the public repo:

- prefer clarity over compactness
- keep descriptive filenames
- use markdown cells liberally in notebooks
- explain what the notebook does, why it exists, and what files it writes
- add an “Inputs” section near the top
- add an “Outputs” section near the top
- add notes on manuscript linkage where helpful
- do not bury key parameter definitions deep in the notebook
- keep path handling explicit and centralized where possible

Good public-facing notebook structure:
1. Title / purpose
2. Manuscript linkage
3. Inputs and expected files
4. User-configurable parameters
5. Main analysis steps
6. Outputs written
7. QC or validation checks
8. Notes / limitations

---

## Filenames and naming conventions

Use filenames that are:
- descriptive
- short enough to read easily
- aligned to the manuscript stage
- stable across reruns

Prefer names like:
- `10_eeg_prune_components.ipynb`
- `12_eeg_merge_exclusions.ipynb`
- `42_bold_extract_parcels.ipynb`
- `53_build_observation_matrix.ipynb`

Avoid vague names like:
- `final_final2.ipynb`
- `updated_working_version.ipynb`
- `test_clean.ipynb`

If renaming, preserve mapping from old name to new name in documentation or comments.

---

## Documentation policy

Whenever a refactor produces or changes a major notebook/script, update documentation as needed.

Especially relevant files:
- `docs/methods_map.md`
- `docs/manual_steps.md`
- `docs/figure_table_map.md`
- `docs/final_dataset_spec.md`
- `docs/reproducibility_notes.md`

If a notebook writes a manuscript figure/table input, document:
- what it reads
- what it writes
- which figure/table it supports
- whether the output is intermediate or final

---

## Empty-folder policy

Do not assume that empty folders indicate missing work.

In this repo:
- `notebooks/7_summaries/`
- `notebooks/8_figures/`
- `notebooks/9_tables/`

may be empty because summary, figure, and table logic is currently embedded within earlier method notebooks.

During refactor:
- it is acceptable to keep these folders empty
- it is acceptable to move content into them later
- only move content there if it improves clarity for the public repo

---

## What not to do

Do not:
- silently alter scientific defaults
- standardize everything into one giant notebook
- pretend manual steps are automated
- delete archived originals
- collapse distinct manuscript stages into one file without strong reason
- move notebooks purely for cosmetic reasons if it makes the scientific flow harder to follow
- introduce machine-specific absolute paths
- remove comments/notes that preserve provenance

---

## What success looks like

A successful refactor yields:

- a notebook/script layout that mirrors the manuscript methods flow
- descriptive filenames
- cleaner public-facing notebooks with embedded explanations
- preserved scientific behavior
- preserved archive/original provenance
- explicit documentation of manual steps
- clear linkage from methods to code
- a codebase that a reader can navigate without needing local lab knowledge

---

## Preferred task workflow for Codex

For each workflow folder, proceed in passes:

### Pass 1: inventory only
- summarize each notebook/script
- identify overlap
- recommend keep/merge/split/archive
- do not edit yet

### Pass 2: refactor plan
- propose cleaned public-facing notebook set for that folder
- propose filenames
- propose where manual notes are needed

### Pass 3: implementation
- create cleaned notebooks/scripts
- preserve original files
- add markdown explanations
- avoid scientific changes

### Pass 4: documentation sync
- update methods map and any relevant docs

---

## Folder-by-folder priority order

Default refactor priority:

1. `notebooks/1_eeg_sensor/`
2. `notebooks/2_eeg_source/`
3. `notebooks/3_bold/`
4. `notebooks/4_alignment/`
5. `notebooks/5_hmm_selection/`
6. `notebooks/6_hmm_final/`

Only after these are stable should summaries/figures/tables be separated further if needed.

---

## Final instruction

This is a manuscript-driven refactor.

The code should be reorganized to better express the published workflow, not to reinvent the workflow.
When in doubt, follow the manuscript logic, preserve the current science, and make the repo easier for an outside reader to understand.