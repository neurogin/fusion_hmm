# Reproducibility Notes

This document records practical notes, caveats, and workflow constraints relevant to reproducing the analysis in this repository.

It accompanies the manuscript:

**Fusion hidden Markov modeling reveals a reproducible shared state architecture in simultaneous resting-state EEG-fMRI**

## Purpose

This file is meant to capture details that may not fit cleanly into:

- `README.md`
- `docs/methods_map.md`
- `docs/manual_steps.md`
- `docs/final_dataset_spec.md`

It is especially important during the current refactor stage, when parts of the workflow are still being reorganized.

---

## 1. Current repository status

This repository is in an active refactor phase.

At present:

- many analysis steps are still notebook-based
- some QC, summary, figure, and table logic remain embedded in earlier method notebooks
- `config/`, `scripts/`, and `src/` are not yet fully populated
- `notebooks/7_summaries/`, `8_figures/`, and `9_tables/` may remain empty while summary and asset-generation logic is still embedded elsewhere
- original notebooks are preserved in `notebooks/_archive_raw_original_names/`

Because of this, the current repo should be read as a **manuscript-aligned working reorganization** rather than a fully polished turnkey release.

---

## 2. Manual and hybrid workflow dependence

This project is **not fully code-only**.

Important parts of the workflow depend on Brainstorm GUI procedures and other hybrid steps, including:

- manual EEG exclusion marking (`boundary` + manual `BAD`)
- Brainstorm subject import and anatomy setup
- MNI normalization in Brainstorm
- BEM generation
- EEG source localization
- atlas loading as Brainstorm volume scouts
- extraction of source-grid-aware scouts from Brainstorm-generated files
- some screenshot-based or manually assembled figure components

These manual/hybrid steps are documented in:

- `docs/manual_steps.md`

Reproducibility therefore depends not only on running code, but also on following the documented manual procedures consistently.

---

## 3. Manuscript-driven workflow is the source of truth

The intended code organization follows the manuscript methods flow, not necessarily the original historical order in which notebooks were created.

During refactoring:

- the manuscript and supplement should guide consolidation decisions
- older exploratory notebooks should not automatically be treated as part of the final paper workflow
- if notebook content conflicts with the manuscript, the discrepancy should be flagged rather than silently resolved

Reference materials currently used during refactoring:

- `docs/_manuscript_reference/FULL_MANUSCRIPT.docx`
- `docs/_manuscript_reference/SUPPLEMENTAL_MATERIALS.docx`

---

## 4. Canonical final dataset definition

The frozen paper dataset is defined in:

- `docs/final_dataset_spec.md`

For quick reference, the final paper dataset is:

- no-lag design
- minimum retained segment length = 15 TR
- BOLD TR = 2.1 s
- EEG sampling rate = 250 Hz
- 200 BOLD parcel features
- 200 same-TR EEG parcel-power features
- 400 features per retained TR
- 15 runs
- 71 retained contiguous segments
- 3550 retained TRs
- 124.25 usable minutes
- final selected model order = K = 3

If older notebooks or exports disagree with these values, they should be treated as pre-refactor or alternate-history materials unless explicitly documented otherwise.

---

## 5. Brainstorm-specific reproducibility considerations

### 5.1 Source-grid size can vary across subjects
Subject-specific Brainstorm volume source grids are not assumed to have identical sizes across subjects.

This means:
- exporters must not assume a constant dipole count across runs
- subject/run-specific scout files must match the corresponding source grid
- source-grid compatibility checks are essential

### 5.2 Atlas coverage may still vary even with a common atlas
The Schaefer atlas is template-space and reusable across subjects, but representation on the source grid can vary because of:

- subject-specific MNI normalization
- source-grid sampling differences
- interaction between atlas boundaries and the source grid

This is one reason the hybrid Brainstorm → MATLAB workflow and its coverage diagnostics are important.

### 5.3 “Dilated, MNI” atlas import was intentional
The Brainstorm atlas import mode:

- **Volume mask or atlas (dilated, MNI space)**

was used intentionally to reduce parcel dropout caused by coarse source-grid sampling relative to atlas voxel resolution.

This should not be treated as an arbitrary GUI choice.

---

## 6. EEG exclusion and masking notes

The EEG exclusion policy is intentionally conservative and limited to:

- `boundary`
- manually marked `BAD`

`QRS` markers are retained and are not used for censoring unless they fall inside an already excluded segment.

This policy matters later because exclusion intervals affect:
- usable EEG duration
- TR-level keep-mask construction
- retained segment lengths
- downstream fusion dataset support

Small differences in exclusion handling can materially affect the retained-data totals and the stability of later modeling.

---

## 7. Alignment and segmentation notes

The alignment workflow depends on:
- reconciling raw and preprocessed EEG timelines
- projecting usable EEG intervals onto the BOLD TR grid
- applying both EEG-coverage and sample-completeness criteria
- exporting only contiguous retained stretches of at least 15 TR

Important implication:

When segments are later combined into model inputs, gaps should not be treated as continuous time unless sequence boundaries are explicitly preserved.

During refactoring, any code that concatenates retained segments should be checked carefully to ensure state transitions are not accidentally allowed across true temporal gaps.

---

## 8. Mixed language / mixed environment workflow

This project spans multiple tools and environments, including:

- MATLAB
- Python / Jupyter notebooks
- Brainstorm
- EEGLAB
- WSL / Linux-side tooling
- Windows-visible file paths for some shared resources

This means reproducibility depends on both:
- code logic
- environment conventions

Path handling is especially likely to require cleanup during refactoring.

Common issues to watch for:
- hard-coded Windows paths
- hard-coded WSL-mounted paths
- assumptions about Brainstorm database locations
- assumptions about TemplateFlow cache locations
- assumptions about output folder names from earlier reruns

---

## 9. Notebook refactor caveat

Many current notebooks were developed iteratively and may include:

- historical parameter choices
- temporary path settings
- duplicated helper logic
- partially embedded QC code
- combined method + figure + summary logic in one notebook

During cleanup, notebooks may be:
- kept
- merged
- split
- renamed
- archived

Refactoring should prioritize:
- preserving scientific behavior
- improving clarity
- preserving provenance
- aligning to the manuscript methods flow

Original notebooks should remain preserved in:
- `notebooks/_archive_raw_original_names/`

---

## 10. Figure and table generation caveat

At the current stage, some figures and tables may still be generated inside upstream method notebooks rather than in dedicated `7_summaries/`, `8_figures/`, or `9_tables/` folders.

That is acceptable for now.

However, this means reproducibility of a figure or table may currently depend on:
- earlier notebooks being run in sequence
- intermediate outputs being present
- manually assembled assets, screenshots, or export steps

As the repo matures, this should become more explicit in:
- `docs/figure_table_map.md`

---

## 11. Subject-specific note already known

Known exception noted in project documentation:

- `sub-13_ses-01` used anatomy from `sub-13_ses-02`

This should be preserved in documentation and not silently normalized away during refactoring.

---

## 12. What to do when something disagrees

If a notebook, comment, export, or figure appears inconsistent with the current repo documentation:

1. check `docs/final_dataset_spec.md`
2. check `docs/methods_map.md`
3. check `docs/manual_steps.md`
4. check the manuscript reference files
5. flag the discrepancy explicitly

Do not silently “fix” the mismatch unless the intended correction is certain and documented.

---

## 13. Refactor-phase reproducibility goal

The immediate goal of this repo is not yet perfect one-click execution.

The current reproducibility goal is:

- a manuscript-aligned repository structure
- clear documentation of manual, hybrid, and scripted steps
- preserved scientific provenance
- a cleaner path toward a public paper repo
- reduced ambiguity about what belongs to the final paper workflow

A later stage may add:
- cleaner config templates
- reusable functions in `src/`
- runnable entry points in `scripts/`
- clearer figure/table generators
- stronger environment setup documentation