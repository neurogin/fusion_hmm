# Refactor Plan: `7_summaries`

## Scope

This memo covers the current contents of `notebooks/7_summaries/` only, while using the already-cleaned Stage-6 notebooks and related docs as provenance context.

The goal of this pass is to determine whether Stage 7 currently contains a real summaries layer, whether summary logic is still embedded upstream, and whether this folder should gain cleaned public-facing notebooks or remain empty for now.

## Files inspected

Active files in `notebooks/7_summaries/`:

- none

Additional context reviewed because Stage 7 is currently empty:

- `notebooks/6_hmm_final/`
  - `60_fit_final_k3_fusion_hmm.ipynb`
  - `61_review_final_k3_fit_qc_and_state_dynamics.ipynb`
  - `62_reconstruct_bold_state_networks_and_ranked_contrasts.ipynb`
  - `63_reconstruct_crossmodal_state_blocks_and_ranked_contrasts.ipynb`
  - `64_build_parcelized_cortical_state_maps.ipynb`
  - `65_optional_export_figure4_figure5_panels.ipynb`
  - Stage-6 provenance umbrella notebooks
- `notebooks/_archive_raw_original_names/7_final model/`
- `notebooks/8_figures/`
- `notebooks/9_tables/`
- `docs/refactor_plan_6_hmm_final.md`
- `docs/dependency_check_6_hmm_final.md`
- `docs/methods_map.md`
- `docs/figure_table_map.md`
- `docs/reproducibility_notes.md`

## High-level assessment

`notebooks/7_summaries/` is genuinely empty in the current repo snapshot.

That does not look like missing work by itself. It matches the repo-wide policy already documented in:

- `docs/methods_map.md`
- `docs/reproducibility_notes.md`
- `AGENTS.md`

Those files explicitly allow `7_summaries/`, `8_figures/`, and `9_tables/` to remain sparse or empty while summary, figure, and table logic is still embedded in earlier stage notebooks.

At present, the real Stage-7-like summary logic is already carried by the cleaned Stage-6 public notebooks:

- `61_review_final_k3_fit_qc_and_state_dynamics.ipynb`
- `62_reconstruct_bold_state_networks_and_ranked_contrasts.ipynb`
- `63_reconstruct_crossmodal_state_blocks_and_ranked_contrasts.ipynb`

So the current question is not “which Stage-7 files should be cleaned,” but rather:

- whether those existing Stage-6 summary notebooks should later be re-homed into Stage 7,
- or whether keeping them in Stage 6 is currently clearer and avoids duplicate entry points.

My current recommendation is to avoid creating a duplicate Stage-7 layer right now.

## File-by-file assessment

### Files inside `notebooks/7_summaries/`

There are no active files to classify.

### Closely related current summary notebooks outside the folder

These are the files currently doing the summary-stage work in practice:

| Current file | Scientific purpose | Likely manuscript linkage | Classification relative to Stage 7 | Recommendation |
| --- | --- | --- | --- | --- |
| `notebooks/6_hmm_final/61_review_final_k3_fit_qc_and_state_dynamics.ipynb` | Reviews saved final-fit outputs, including FO, transitions, dwell, and final-fit QC summaries. | Main Methods 2.6.1; Figure 3; Figure S6 support; Table S9 support. | True summary-stage logic currently embedded in Stage 6. | Keep as the active public file for now. Candidate future Stage-7 migration only if the repo later wants a stricter “fit vs summaries” boundary. |
| `notebooks/6_hmm_final/62_reconstruct_bold_state_networks_and_ranked_contrasts.ipynb` | Builds BOLD state-network summaries and ranked contrasts from saved final-model artifacts. | Main Methods 2.6.2; Figure 4; Table S10. | True summary-stage logic currently embedded in Stage 6. | Keep as the active public file for now. Strong candidate if a future Stage-7 folder is populated. |
| `notebooks/6_hmm_final/63_reconstruct_crossmodal_state_blocks_and_ranked_contrasts.ipynb` | Builds descriptive cross-modal summaries and ranked contrasts from saved final-model artifacts. | Main Methods 2.6.3; Figure 5; Table S11. | True summary-stage logic currently embedded in Stage 6. | Keep as the active public file for now. Strong candidate if a future Stage-7 folder is populated. |
| `notebooks/6_hmm_final/64_build_parcelized_cortical_state_maps.ipynb` | Builds parcelized cortical surface maps. | Main Methods 2.6.4; Figure 6. | More map/figure-oriented than summary-oriented. | Do not make this the core Stage-7 entry point. It fits Stage 6 or Stage 8 better than Stage 7. |
| `notebooks/6_hmm_final/65_optional_export_figure4_figure5_panels.ipynb` | Writes separate figure panels for later manual assembly. | Figure 4 and Figure 5 manual assembly support. | Figure-sidecar, not summary logic. | Keep out of Stage 7. Optional only. |
| `notebooks/6_hmm_final/PipelineE_K3_manuscript_figures_notebook.ipynb` | Umbrella manuscript composite notebook mixing review, summary, and figure logic. | Figures 3-6 support. | Provenance-heavy composite, not a clean summary notebook. | Archive/provenance-only. |
| `notebooks/6_hmm_final/PipelineE_K3_manuscript_figures_notebook_with_raster.ipynb` | Expanded umbrella composite notebook with gamma raster and mixed figure logic. | Figures 3-6 and Figure S6 support. | Provenance-heavy composite, not a clean summary notebook. | Archive/provenance-only. |

## Merge and split recommendations

### Merge recommendations

No merge inside `notebooks/7_summaries/` is possible because the folder is empty.

At the repo level, no new merge is recommended right now. The cleaned Stage-6 notebooks already provide a readable separation between:

- final fit (`60`)
- review/dynamics (`61`)
- BOLD summaries (`62`)
- cross-modal summaries (`63`)
- cortical maps (`64`)
- optional panel export (`65`)

Creating another Stage-7 notebook layer now would mostly duplicate those public entry points.

### Split recommendations

No split inside `notebooks/7_summaries/` is possible because the folder is empty.

Conceptually, the main split that already matters is:

- keep final full-data model fitting in Stage 6
- treat the downstream review and reconstruction notebooks as the summary layer, even though they still live physically in `notebooks/6_hmm_final/`

### Files that should remain helper-level only

There are no Stage-7 helper files yet.

If Stage 7 is populated later, it should probably use the existing Stage-6 saved outputs directly rather than introducing a new heavy helper layer immediately.

## Proposed cleaned GitHub-facing file set

### Recommended now

No new public-facing Stage-7 files should be created in this pass.

Recommended public-facing Stage-7 set for the current repo state:

- none

Recommended Stage-7 behavior for now:

- keep `notebooks/7_summaries/` empty
- treat the cleaned Stage-6 summary notebooks as the active public summary layer
- avoid duplicate wrappers that would just point to the same saved final-model artifacts

### Optional future extraction only if a stricter stage boundary becomes valuable

If the repo later decides to separate “final fit” from “summaries” more aggressively, the most plausible future Stage-7 public files would be:

1. `70_summarize_final_state_dynamics.ipynb`
2. `71_summarize_bold_state_networks_and_ranked_contrasts.ipynb`
3. `72_summarize_crossmodal_state_blocks_and_ranked_contrasts.ipynb`

I do **not** currently recommend extracting:

- `64_build_parcelized_cortical_state_maps.ipynb` into Stage 7
- `65_optional_export_figure4_figure5_panels.ipynb` into Stage 7

Those are more naturally map/figure-facing than summary-facing.

## Mapping from current files to cleaned set

### Current recommended state

| Stage-7 cleaned target | Current source |
| --- | --- |
| no Stage-7 public file yet | summary logic remains in `notebooks/6_hmm_final/61` to `64` |

### Optional future extraction mapping

| Possible future Stage-7 file | Current source |
| --- | --- |
| `70_summarize_final_state_dynamics.ipynb` | `notebooks/6_hmm_final/61_review_final_k3_fit_qc_and_state_dynamics.ipynb` |
| `71_summarize_bold_state_networks_and_ranked_contrasts.ipynb` | `notebooks/6_hmm_final/62_reconstruct_bold_state_networks_and_ranked_contrasts.ipynb` |
| `72_summarize_crossmodal_state_blocks_and_ranked_contrasts.ipynb` | `notebooks/6_hmm_final/63_reconstruct_crossmodal_state_blocks_and_ranked_contrasts.ipynb` |

Files that should remain outside a future Stage-7 public set:

- `notebooks/6_hmm_final/60_fit_final_k3_fusion_hmm.ipynb`
- `notebooks/6_hmm_final/64_build_parcelized_cortical_state_maps.ipynb`
- `notebooks/6_hmm_final/65_optional_export_figure4_figure5_panels.ipynb`
- Stage-6 umbrella provenance notebooks

## Downstream and upstream integration assumptions to verify in Pass 2

1. Confirm whether the practical Stage-7 summary layer can read only core Stage-6 final-model artifacts:
   - `qc_summary.json`
   - `run_metrics.tsv`
   - `subject_metrics.tsv`
   - `dwell_from_A.tsv`
   - `final/best_seed.json`
   - `final/trans_prob.npy`
   - `final/covs_pca.npy`
   - `final/means_pca.npy`
   - `final/state_signature_ut_boldcorr.npy`

2. Confirm which derived Stage-6 outputs are convenience files rather than true Stage-7 dependencies:
   - especially `state_summary_table.tsv`

3. Check whether `61`, `62`, and `63` depend only on core final-model artifacts or whether they also depend on each other’s derived outputs.

4. Confirm whether Figure 3 / Figure S6 support should remain embedded in `61`, or whether a later Stage-7 summary notebook would need its own raster-specific sidecar.

5. Confirm whether Stage 7 should stay summary-only, while:
   - `64` remains a map/figure notebook
   - `65` remains optional panel-export support

6. Verify environment assumptions for any future Stage-7 extraction:
   - likely `numpy`, `pandas`, `matplotlib`
   - possibly `nibabel` and `nilearn` only for map-oriented outputs, not for core summaries
   - likely no direct `tensorflow` / `osl_dynamics` dependency if the summary layer only reads saved artifacts rather than refitting models

7. Confirm whether future Stage-8 or Stage-9 cleanup would duplicate any proposed Stage-7 notebook.
   - In particular, avoid creating a Stage-7 summary notebook and then another Stage-8 figure notebook that both rebuild the same panels from the same derived TSVs.

## Risks and ambiguities

1. The folder is empty, so the biggest risk is over-correcting.
   - If we force content into Stage 7 now, we may create duplicate public entry points that are less clear than the existing Stage-6 notebooks.

2. The stage boundary between Stage 6 and Stage 7 is conceptually real but currently soft in the repo.
   - Stage 6 already contains both the final fit and the main downstream summary/reconstruction notebooks.

3. The boundary between Stage 7 and Stage 8 is also soft.
   - `64` and `65` are more figure/map-facing than summary-facing.
   - Forcing them into Stage 7 would blur the folder’s purpose.

4. The `S2` interpretation remains non-uniform across downstream notebooks.
   - any future Stage-7 re-homing must keep explicit whether `S2` is:
     - inferred from final FO,
     - derived from `best_seed.json`,
     - or imposed directly

5. The umbrella provenance notebooks in Stage 6 remain tempting but misleading sources for a Stage-7 public layer.
   - They mix summaries, figures, and optional presentation logic.
   - They should not define the public Stage-7 plan.

## Bottom-line recommendation

Stage 7 should remain empty for now.

That is the cleanest and most manuscript-aligned recommendation for the current repo state.

Reason:

- the real summary-stage logic already exists in the cleaned Stage-6 notebooks
- `7_summaries/`, `8_figures/`, and `9_tables/` are explicitly allowed to remain empty during this refactor phase
- creating a new Stage-7 public layer now would mostly duplicate existing public notebooks without improving reproducibility

So the best Stage-7 Pass-1 conclusion is:

- do not force a new Stage-7 implementation yet
- keep using the cleaned Stage-6 summary notebooks as the active public summary layer
- use Pass 2 to verify whether a later Stage-7 extraction would truly simplify the Stage-6 / Stage-8 / Stage-9 boundary, or merely duplicate it
