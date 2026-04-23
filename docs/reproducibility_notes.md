# Reproducibility Notes

This document records the practical constraints and caveats that matter most when reproducing the public workflow.

## 1. The Public Workflow Is Manuscript Driven

The repository is organized around the published workflow, not around the historical order in which the analysis code was first developed.

That means:

- the active `stepNN_*` files are the intended public path
- later summary products remain embedded in Stages 4 to 6 where that keeps the workflow clearest
- manual or hybrid steps stay explicit instead of being presented as fully scripted

## 2. The Canonical Dataset Branch Is Fixed

The public manuscript path uses:

- `DATA_VARIANT = "intermediate"`
- `FEATURE_MODE = "nolags"`
- `MINLEN = 15`
- final selected model order = `K = 3`

If you run a different branch for comparison, label it clearly. Do not silently mix it with the manuscript-facing outputs.

## 3. Manual And Hybrid Steps Still Matter

Reproducibility depends on more than running code.

The workflow also depends on:

- Brainstorm exclusion marking in Stage 1
- Brainstorm source-model setup and atlas import in Stage 2
- some later figure-panel layout choices

Those steps are documented in [docs/manual_steps.md](manual_steps.md).

## 4. Stage-2 Output Contract Matters Downstream

Stage 4 expects the cleaned Stage-2 EEG parcel exports to include:

- `*_PC1_gnorm.npy`
- `*_time_sec.npy`

The `*_time_sec.npy` sidecar is part of the active public contract. It should not be silently dropped or replaced by a different time-base convention.

Stage-2 Step 26 expects:

- `parcel_output_dir` to point to the parent `parcel_exports/` folder

and not to the nested `npy/` folder.

## 5. Stage-3 Uses Two Atlas Branches For Different Jobs

Stage 3 preserves an explicit distinction between:

- the standalone atlas-on-grid branch in Step 30 for atlas-preservation QC and overlay support
- the authoritative exporter-side parcel branch in Step 31 for BOLD parcel time-series export

Do not silently swap one for the other.

## 6. Stage-4 Trigger And Alignment Assumptions Are Real Inputs

Stage 4 depends on:

- raw EEG event TSVs
- preprocessed EEG event TSVs
- recurring `R128` triggers
- the first raw `S1` event as the anchor between raw and preprocessed timing

The alignment helpers search across common event-label and time-column names, but the trigger dependency itself is still part of the true input contract.

Also note:

- run discovery is availability based
- the public notebooks now surface that behavior with an explicit run-input audit
- the exposed `OFFSET_JUMP_THR` setting is left visible even though the effective split behavior remains tied to the preserved helper logic

## 7. Stage-5 Model Selection Is A Two-Step Story

The public release keeps the model-selection narrative explicit:

- Step 50 performs broad LOSO screening over `K = 2..12`
- Step 51 performs the manuscript-facing shortlist comparison centered on `K = 3` and `K = 5`
- Step 52 builds Figure 2 and Table S8 support from those saved outputs

The final choice of `K = 3` should not be described as the result of one fully automatic rule alone.

## 8. Stage-5 And Stage-6 Runtime Requirements Are Heavy

Stages 5 and 6 depend on:

- TensorFlow
- `osl_dynamics`

In practice:

- GPU-backed execution is the intended mode for the main runs
- CPU fallback is possible in principle
- CPU fallback is often much slower and less practical for the full LOSO sweep and final full-data fit

Chunk or resume settings are part of the public notebooks because longer runs may need multiple passes.

## 9. Stage-6 Reference-State Behavior Is Intentionally Not Harmonized

The public release keeps the preserved reference-state behavior explicit:

- the review and physiology path can infer a dominant or reference state from saved FO outputs
- the cross-modal path can recover a reference state from `best_seed.json` with override or fallback behavior
- the cortical-map path preserves an imposed `REFERENCE_STATE = 2`

Those differences are part of the documented workflow. They are not silently normalized into one new rule.

## 10. Generated Outputs Should Stay Out Of Git

This repository is the workflow package, not the derived-data store.

In normal use:

- raw inputs
- derivatives
- QC outputs
- model fits
- panel exports

should be written to user-configured output locations and kept out of version control.

## 11. Public Release Scope

This public release intentionally excludes:

- historical development notebooks
- internal refactor-planning documents
- scratch files and smoke-test artifacts

The remaining tree is meant to be the active public workflow only.
