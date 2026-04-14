# Refactor Plan: `8_figures`

## Scope

This memo covers the current contents of `notebooks/8_figures/` only, while using the cleaned Stage-4 to Stage-6 notebooks and archive provenance as context.

The goal of this pass is to determine whether Stage 8 currently contains a real figure-generation layer, whether figure logic is still embedded upstream, and whether this folder should gain cleaned public-facing figure notebooks or remain empty for now.

## Files inspected

Active files in `notebooks/8_figures/`:

- none

Additional context reviewed because Stage 8 is currently empty:

- `notebooks/4_alignment/`
  - `40_align_eeg_to_bold_trs_and_build_keep_masks.ipynb`
  - `41_build_final_no_lag_fusion_observation_segments.ipynb`
  - `42_qc_alignment_tables_s6_s7_and_figure1_support.ipynb`
- `notebooks/5_hmm_selection/`
  - `52_build_figure2_and_table_s8_model_selection_summary.ipynb`
- `notebooks/6_hmm_final/`
  - `61_review_final_k3_fit_qc_and_state_dynamics.ipynb`
  - `62_reconstruct_bold_state_networks_and_ranked_contrasts.ipynb`
  - `63_reconstruct_crossmodal_state_blocks_and_ranked_contrasts.ipynb`
  - `64_build_parcelized_cortical_state_maps.ipynb`
  - `65_optional_export_figure4_figure5_panels.ipynb`
  - Stage-6 umbrella provenance notebooks
- `notebooks/_archive_raw_original_names/7_final model/`
  - archived PNG figure assets
  - zipped provenance notebooks
  - manuscript-snippet docs
- `notebooks/7_summaries/`
- `notebooks/9_tables/`
- `docs/refactor_plan_7_summaries.md`
- `docs/refactor_plan_6_hmm_final.md`
- `docs/dependency_check_6_hmm_final.md`
- `docs/figure_table_map.md`
- `docs/methods_map.md`
- `docs/reproducibility_notes.md`

## High-level assessment

`notebooks/8_figures/` is genuinely empty in the current repo snapshot.

That does not look like missing work by itself. It matches the current repo policy that:

- `7_summaries/`, `8_figures/`, and `9_tables/` may remain empty
- while summary, figure, and table logic is still embedded in earlier cleaned stage notebooks

At present, the real figure-stage logic is already distributed across earlier stages:

- Figure 1 support lives in Stage 4
- Figure 2 support lives in Stage 5
- Figures 3-6 support lives in Stage 6
- optional Figure 4 / Figure 5 panel-export support also lives in Stage 6

So Stage 8 is currently a conceptual figure layer, not an implemented notebook layer.

My current recommendation is to avoid creating a duplicate Stage-8 wrapper layer right now.

## File-by-file assessment

### Files inside `notebooks/8_figures/`

There are no active files to classify.

### Closely related current figure notebooks outside the folder

These are the files currently doing the figure-stage work in practice:

| Current file | Scientific purpose | Likely manuscript linkage | Classification relative to Stage 8 | Recommendation |
| --- | --- | --- | --- | --- |
| `notebooks/4_alignment/42_qc_alignment_tables_s6_s7_and_figure1_support.ipynb` | Writes support plots and a manifest for the alignment figure. | Figure 1 support. | True figure-support logic currently embedded in Stage 4. | Keep as the active public file for now. Do not duplicate into Stage 8 yet. |
| `notebooks/5_hmm_selection/52_build_figure2_and_table_s8_model_selection_summary.ipynb` | Writes the model-selection figure summary outputs. | Figure 2 support. | True figure-support logic currently embedded in Stage 5. | Keep as the active public file for now. Do not duplicate into Stage 8 yet. |
| `notebooks/6_hmm_final/61_review_final_k3_fit_qc_and_state_dynamics.ipynb` | Builds Figure-3-style final-state dynamics outputs from saved final-fit artifacts. | Figure 3 support; Figure S6-adjacent provenance context. | True figure-support logic currently embedded in Stage 6. | Keep as the active public file for now. Candidate future Stage-8 migration only if a strict figure layer later becomes valuable. |
| `notebooks/6_hmm_final/62_reconstruct_bold_state_networks_and_ranked_contrasts.ipynb` | Builds BOLD-state summary figures and ranked contrasts. | Figure 4 support. | True figure-support logic currently embedded in Stage 6. | Keep as the active public file for now. Strong candidate if a future Stage-8 layer is ever populated. |
| `notebooks/6_hmm_final/63_reconstruct_crossmodal_state_blocks_and_ranked_contrasts.ipynb` | Builds cross-modal summary figures and ranked contrasts. | Figure 5 support. | True figure-support logic currently embedded in Stage 6. | Keep as the active public file for now. Strong candidate if a future Stage-8 layer is ever populated. |
| `notebooks/6_hmm_final/64_build_parcelized_cortical_state_maps.ipynb` | Builds parcelized cortical map figures. | Figure 6 support. | Figure-stage logic already exposed cleanly in Stage 6. | Keep as the active public file for now. No duplicate Stage-8 wrapper needed yet. |
| `notebooks/6_hmm_final/65_optional_export_figure4_figure5_panels.ipynb` | Exports separate Figure 4 and Figure 5 panels for later manual assembly. | Manual panel-assembly support. | True figure-sidecar, not core computational logic. | Keep optional and separate. If Stage 8 is ever populated, this is the strongest true Stage-8 candidate. |
| `notebooks/6_hmm_final/PipelineE_K3_manuscript_figures_notebook.ipynb` | Umbrella manuscript figure notebook mixing multiple figures and stages. | Figures 3-6 provenance. | Provenance-heavy composite, not a clean public figure notebook. | Archive/provenance-only. |
| `notebooks/6_hmm_final/PipelineE_K3_manuscript_figures_notebook_with_raster.ipynb` | Expanded umbrella figure notebook with gamma raster and mixed figure logic. | Figures 3-6 and Figure S6 provenance. | Provenance-heavy composite, not a clean public figure notebook. | Archive/provenance-only. |
| `notebooks/_archive_raw_original_names/7_final model/fig1.png`, `fig3.png`, `kinetics.png`, `run_level_fo.png`, `subj_level_fo.png`, `transition_matrix_A.png` | Archived rendered figure assets. | Likely manuscript-figure or exploratory panel exports. | Output provenance only, not executable logic. | Archive/provenance-only. Do not treat these as active Stage-8 sources. |

## Merge and split recommendations

### Merge recommendations

No merge inside `notebooks/8_figures/` is possible because the folder is empty.

At the repo level, no new figure-layer merge is recommended right now. The current cleaned notebooks already separate the main figure-producing logic by scientific stage:

- alignment figure support in Stage 4
- model-selection figure support in Stage 5
- final-model figure support in Stage 6

Creating a new Stage-8 layer now would mostly duplicate those public notebooks.

### Split recommendations

No split inside `notebooks/8_figures/` is possible because the folder is empty.

Conceptually, the main figure split that already matters is:

- keep scientific reconstruction notebooks in Stages 4 to 6
- keep optional panel-export support separate
- keep manual final composite assembly explicit rather than pretending the final manuscript figures are one-click outputs

The one Stage-6 file that is already closest to a pure figure-stage notebook is:

- `65_optional_export_figure4_figure5_panels.ipynb`

If Stage 8 is ever populated later, that is the cleanest starting point.

### Files that should remain helper-level only

There are no Stage-8 helper files yet.

If Stage 8 is populated later, it should probably stay thin and read already-derived upstream outputs rather than rebuilding heavy scientific logic inside another layer.

## Proposed cleaned GitHub-facing file set

### Recommended now

No new public-facing Stage-8 files should be created in this pass.

Recommended public-facing Stage-8 set for the current repo state:

- none

Recommended Stage-8 behavior for now:

- keep `notebooks/8_figures/` empty
- treat the cleaned Stage-4, Stage-5, and Stage-6 notebooks as the active public figure-support layer
- keep final manual panel composition explicit where it is still genuinely manual

### Optional future extraction only if a stricter figure layer becomes valuable

If the repo later decides to separate figure-generation from the stage notebooks more aggressively, the most plausible future Stage-8 public files would be:

1. `80_build_figure1_alignment_support.ipynb`
2. `81_build_figure2_model_selection_summary.ipynb`
3. `82_build_figure3_state_dynamics_panels.ipynb`
4. `83_build_figure4_bold_state_panels.ipynb`
5. `84_build_figure5_crossmodal_panels.ipynb`
6. `85_build_figure6_cortical_maps.ipynb`
7. optional: `86_optional_build_figure_s6_gamma_raster.ipynb`

Even in that future case, I would still keep manual composite assembly explicit for:

- Figure 1
- Figures 4 and 5 final multi-panel layouts
- any future dedicated Figure S6 raster export

## Mapping from current files to cleaned set

### Current recommended state

| Stage-8 cleaned target | Current source |
| --- | --- |
| no Stage-8 public file yet | figure logic remains in cleaned Stage-4, Stage-5, and Stage-6 notebooks |

### Optional future extraction mapping

| Possible future Stage-8 file | Current source |
| --- | --- |
| `80_build_figure1_alignment_support.ipynb` | `notebooks/4_alignment/42_qc_alignment_tables_s6_s7_and_figure1_support.ipynb` |
| `81_build_figure2_model_selection_summary.ipynb` | `notebooks/5_hmm_selection/52_build_figure2_and_table_s8_model_selection_summary.ipynb` |
| `82_build_figure3_state_dynamics_panels.ipynb` | `notebooks/6_hmm_final/61_review_final_k3_fit_qc_and_state_dynamics.ipynb` plus raster provenance only if later justified |
| `83_build_figure4_bold_state_panels.ipynb` | `notebooks/6_hmm_final/62_reconstruct_bold_state_networks_and_ranked_contrasts.ipynb` plus optional `65_optional_export_figure4_figure5_panels.ipynb` |
| `84_build_figure5_crossmodal_panels.ipynb` | `notebooks/6_hmm_final/63_reconstruct_crossmodal_state_blocks_and_ranked_contrasts.ipynb` plus optional `65_optional_export_figure4_figure5_panels.ipynb` |
| `85_build_figure6_cortical_maps.ipynb` | `notebooks/6_hmm_final/64_build_parcelized_cortical_state_maps.ipynb` |
| `86_optional_build_figure_s6_gamma_raster.ipynb` | `notebooks/6_hmm_final/PipelineE_K3_manuscript_figures_notebook_with_raster.ipynb` provenance only |

Files that should remain outside a future Stage-8 public set:

- `notebooks/6_hmm_final/60_fit_final_k3_fusion_hmm.ipynb`
- Stage-6 umbrella provenance notebooks as umbrella notebooks
- archived PNG assets in `notebooks/_archive_raw_original_names/7_final model/`

## Downstream and upstream integration assumptions to verify in Pass 2

1. Confirm whether a practical future Stage-8 figure layer can read only already-derived figure-support outputs, or whether it still needs core upstream artifacts directly.

2. Confirm the true dependency split for each figure:
   - Figure 1 from Stage-4 support outputs
   - Figure 2 from Stage-5 summary outputs
   - Figures 3-6 from Stage-6 saved final-model artifacts and reconstruction outputs

3. Confirm whether Figure 4 and Figure 5 final panel layouts truly need `65_optional_export_figure4_figure5_panels.ipynb`, or whether `62` and `63` already write sufficient figure-ready outputs on their own.

4. Confirm whether any future Stage-8 notebook would depend on derived Stage-6 convenience files such as:
   - `state_summary_table.tsv`
   rather than on core artifacts like:
   - `best_seed.json`
   - `state_signature_ut_boldcorr.npy`
   - `covs_pca.npy`

5. Confirm the manual-assembly boundary explicitly:
   - Figure 1 still includes hybrid/manual schematic and layout work
   - Figures 4 and 5 still have optional panel-export support but manual final assembly
   - Figure S6 currently survives only as provenance-heavy raster logic, not a clean public entry point

6. Verify plot-library assumptions for any future Stage-8 extraction:
   - likely `matplotlib`, `numpy`, `pandas`
   - `nibabel` and `nilearn` for cortical maps
   - possibly `networkx` or `plotly` only for specific provenance branches, not for the main public path unless confirmed

7. Confirm whether archived PNGs in `notebooks/_archive_raw_original_names/7_final model/` are:
   - direct outputs of still-executable notebooks
   - or hand-touched/manually assembled assets that should remain provenance only

## Risks and ambiguities

1. The folder is empty, so the biggest risk is over-correcting.
   - If we force content into Stage 8 now, we may create duplicate figure wrappers around already-cleaned stage notebooks.

2. The boundary between scientific reconstruction and figure generation is soft.
   - `62`, `63`, and `64` are already both summary notebooks and figure-support notebooks.

3. The boundary between Stage 8 and manual assembly is also soft.
   - Figure 1 is still hybrid/manual.
   - Figure 4 and Figure 5 final composites still rely on optional panel export and manual layout choices.

4. The umbrella provenance notebooks in Stage 6 are tempting but misleading figure sources.
   - They mix multiple figures, multiple stages, and optional presentation logic.
   - They should not define a future Stage-8 public layer directly.

5. Archived PNGs and panel exports may look authoritative even when they are only static provenance outputs.
   - The public repo should not mistake “an old rendered image exists” for “there is a clean executable figure notebook.”

6. A future Stage-8 layer could easily overlap with Stage 9 tables or with the already-cleaned Stage-6 notebooks if the boundary is not chosen carefully.

## Bottom-line recommendation

Stage 8 should remain empty for now.

That is the cleanest and most manuscript-aligned recommendation for the current repo state.

Reason:

- the real figure-support logic already exists in the cleaned Stage-4, Stage-5, and Stage-6 notebooks
- final composite assembly is still partly manual for several key figures
- `8_figures/` is explicitly allowed to remain empty during this refactor phase
- creating a new Stage-8 layer now would mostly duplicate existing public notebooks without improving reproducibility

So the best Stage-8 Pass-1 conclusion is:

- do not force a new Stage-8 implementation yet
- keep using the cleaned upstream notebooks as the active public figure-support layer
- use Pass 2 to verify whether a later figure-only extraction would truly simplify the Stage-6 overlap and manual-assembly boundary, or merely duplicate it
