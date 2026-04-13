# Refactor Plan: `6_hmm_final`

## Scope

This memo covers the current contents of `notebooks/6_hmm_final/` only.

The goal of this pass is to identify:

- the true final chosen-model fit notebook for the manuscript,
- the downstream K=3 review and reconstruction notebooks that currently live in the same folder,
- overlaps between them,
- which notebooks should become cleaned public-facing entry points,
- and which notebooks should remain archive/provenance-only.

This folder sits downstream of:

- cleaned Stage 4 alignment outputs, especially the canonical `intermediate + nolags + minlen15` retained-segment dataset,
- and the Stage 5 model-selection conclusion that the final manuscript choice is `K = 3`.

## Files inspected

Active files in `notebooks/6_hmm_final/`:

1. `R01_PipelineE_full_model.ipynb`
2. `PipelineE_K3_results_review_notebook_fixed_paths.ipynb`
3. `PipelineE_K3_state_physiology_notebook_schaefer_adapted_fix1.ipynb`
4. `PipelineE_K3_brainmaps_surface_parcelized_notebook_fix4.ipynb`
5. `PipelineE_K3_fusion_input_and_crossmodal_notebook.ipynb`
6. `PipelineE_K3_Fig4_Fig5_panel_exports_notebook_v3.ipynb`
7. `PipelineE_K3_manuscript_figures_notebook.ipynb`
8. `PipelineE_K3_manuscript_figures_notebook_with_raster.ipynb`

Additional provenance context reviewed:

- `notebooks/_archive_raw_original_names/7_final model/`
- `docs/methods_map.md`
- `docs/final_dataset_spec.md`
- `docs/reproducibility_notes.md`
- `docs/figure_table_map.md`

## High-level assessment

This folder contains two different kinds of material:

1. The true final full-data K=3 fusion-HMM fit workflow.
2. A large amount of downstream K=3 review, reconstruction, and figure-oriented logic that really behaves like later temporal-summary / biological-reconstruction / figure-support work.

The core final-fit notebook is clear:

- `R01_PipelineE_full_model.ipynb` is the authoritative full-data fit notebook.

Everything else reads the saved PipelineE outputs and turns them into:

- fit QC,
- state-dynamics summaries,
- BOLD state reconstructions,
- cross-modal descriptive reconstructions,
- parcelized surface maps,
- or manuscript panel exports.

So this folder is scientifically important, but it is internally mixed. It already combines what the repo-wide manuscript map treats as:

- final chosen-model fit,
- temporal summaries,
- biological reconstructions,
- and figure/panel assembly sidecars.

## File-by-file assessment

| Current file | Scientific purpose | Likely manuscript linkage | Classification | Recommendation |
| --- | --- | --- | --- | --- |
| `R01_PipelineE_full_model.ipynb` | Final full-data HMM fit on the retained fusion dataset. Loads `segments_manifest.tsv`, performs global preprocessing, run-wise normalization, PCA, multi-seed screening, Top-M refit, final seed selection, final artifact export, per-run Gamma/Viterbi decoding, run/subject metrics, and QC summary export. | Main Methods 2.5.2; Supplementary Methods 1.8; Supplementary Results 2.7; Table S9. | Core manuscript-relevant final-model logic. | Keep, but rewrite as the main cleaned public Stage-6 entry notebook. Strong helper-splitting candidate later because many reusable functions are embedded here. |
| `PipelineE_K3_results_review_notebook_fixed_paths.ipynb` | Post-fit review of an already-trained final model. Reads `qc_summary.json`, `run_metrics.tsv`, `subject_metrics.tsv`, `dwell_from_A.tsv`, `topM_seeds.json`, and final artifacts. Produces QC and state-dynamics review figures. Does not retrain. | Final-model QC; likely Figure 3 support and Table S9 support. | Core manuscript-relevant QC/review sidecar. | Keep, but merge conceptually with the state-dynamics portions of the larger manuscript-figure notebook into one cleaned public QC/dynamics notebook. |
| `PipelineE_K3_state_physiology_notebook_schaefer_adapted_fix1.ipynb` | Reconstructs state-wise parcel and network BOLD summaries from `state_signature_ut_boldcorr.npy` and optionally from `covs_pca.npy` plus `preproc_params.npz`. Writes state summary table, network block summaries, ranked network contrasts, ranked parcel contrasts, and a physiology manifest. | Main Methods 2.6.2; Supplementary Methods 1.9-1.11; Figures 4-5 support; Tables S10-S11 support. | Core manuscript-relevant downstream summary logic. | Keep, but rewrite as a cleaned public BOLD-state-summary notebook. Good candidate to stay separate from the core fit notebook. |
| `PipelineE_K3_brainmaps_surface_parcelized_notebook_fix4.ipynb` | Builds parcelized cortical surface maps for the chosen K=3 solution using the Schaefer atlas, including S2 nodal mean connectivity, S1-S2 and S3-S2 contrasts, and parcel tables. | Figure 6; Supplementary biological reconstruction support. | Core manuscript-relevant later-stage visualization logic. | Keep as a separate cleaned public notebook, but do not port the final exploratory “Option B” block wholesale without labeling it as a design branch. |
| `PipelineE_K3_fusion_input_and_crossmodal_notebook.ipynb` | Mixed notebook. Part A recreates a representative fusion-input construction plot from Stage-4 alignment outputs. Part B backprojects cross-modal BOLD-EEG blocks from final-model covariance/PCA outputs and writes ranked cross-modal contrasts. | Part A overlaps Figure 1 / methods illustration; Part B supports Figure 5 and cross-modal descriptive results. | Mixed: partly manuscript-relevant, partly stage-contaminating. | Split candidate. Keep the cross-modal reconstruction half for a cleaned public notebook. Treat the representative fusion-input figure half as provenance-only or later support logic rather than the main Stage-6 public path. |
| `PipelineE_K3_Fig4_Fig5_panel_exports_notebook_v3.ipynb` | Re-exports Figure 4 and Figure 5 panels separately for manual assembly. Reads final-model outputs and atlas labels and writes panel-specific PNGs plus a panel manifest. | Manual figure assembly sidecar for Figures 4-5. | QC/figure sidecar, not core computational logic. | Optional only. If exposed at all, keep as a clearly optional panel-export notebook. Otherwise archive-only. |
| `PipelineE_K3_manuscript_figures_notebook.ipynb` | Umbrella K=3 manuscript-figure notebook combining state dynamics, network physiology, parcel heatmaps, ranked contrasts, and transition visualizations. | Figures 3-6 support, but mixed together. | Overlapping umbrella notebook; older provenance. | Archive/provenance-only. Do not make this the public entry point. |
| `PipelineE_K3_manuscript_figures_notebook_with_raster.ipynb` | Expanded umbrella figure notebook that adds per-run Gamma raster and example Gamma/Viterbi plot on top of the broader manuscript-figure bundle. | Figures 3-6 and Figure S6 support, but still mixed together. | Overlapping umbrella notebook; later provenance branch. | Archive/provenance-only. Use only as a source of sections to split into cleaner public notebooks. |

## Merge and split recommendations

### Merge candidates

1. Merge `PipelineE_K3_results_review_notebook_fixed_paths.ipynb` with the state-dynamics subset of `PipelineE_K3_manuscript_figures_notebook_with_raster.ipynb`.
   - Reason: both cover final-fit QC, FO, dwell, transitions, and Gamma/Viterbi-oriented dynamics review.
   - Clean target: one public notebook for final-fit QC and state dynamics.

2. Merge the cross-modal reconstruction half of `PipelineE_K3_fusion_input_and_crossmodal_notebook.ipynb` with the cross-modal portions that are currently duplicated in the umbrella manuscript-figure notebook(s).
   - Reason: there should be one public notebook for descriptive cross-modal state blocks and ranked contrasts.

### Split candidates

1. Split `PipelineE_K3_fusion_input_and_crossmodal_notebook.ipynb`.
   - Part A is a representative fusion-input construction illustration using Stage-4 alignment outputs.
   - Part B is true final-model cross-modal reconstruction logic.
   - These are different scientific roles and should not stay bundled publicly.

2. Split the current umbrella figure notebooks conceptually into:
   - final-fit QC/dynamics,
   - BOLD state summaries,
   - cross-modal summaries,
   - parcelized surface maps,
   - optional panel-export sidecars.

3. Treat the final “Option B” branch inside `PipelineE_K3_brainmaps_surface_parcelized_notebook_fix4.ipynb` as design/provenance rather than default scientific logic.

### Files that should likely remain helper-level only later

There are no standalone helper modules in this folder yet, but helper-like code is heavily embedded in:

- `R01_PipelineE_full_model.ipynb`
- `PipelineE_K3_state_physiology_notebook_schaefer_adapted_fix1.ipynb`
- `PipelineE_K3_fusion_input_and_crossmodal_notebook.ipynb`
- `PipelineE_K3_brainmaps_surface_parcelized_notebook_fix4.ipynb`

In Pass 3, cleaned public notebooks will likely benefit from one or two small helper modules for:

- manifest and path resolution,
- final-artifact loading,
- state-signature and covariance backprojection,
- atlas label parsing,
- and shared plotting/output helpers.

## Proposed cleaned GitHub-facing file set

Recommended public-facing Stage-6 set:

1. `60_fit_final_k3_fusion_hmm.ipynb`
   - authoritative full-data K=3 fit notebook
   - canonical public default should remain `intermediate + nolags + minlen15`

2. `61_review_final_k3_fit_qc_and_state_dynamics.ipynb`
   - post-fit QC
   - seed recap
   - FO, dwell, transitions
   - Gamma/Viterbi review outputs
   - Table S9 and Figure 3 / Figure S6 support

3. `62_reconstruct_bold_state_networks_and_ranked_contrasts.ipynb`
   - BOLD parcel/network state summaries
   - ranked network and parcel contrasts
   - Table S10 support and Figure 4 support

4. `63_reconstruct_crossmodal_state_blocks_and_ranked_contrasts.ipynb`
   - descriptive BOLD-EEG cross-block reconstruction
   - ranked cross-modal contrasts
   - Figure 5 and Table S11 support

5. `64_build_parcelized_cortical_state_maps.ipynb`
   - parcelized cortical surface maps
   - Figure 6 support

Optional only if clearly useful for public readers:

6. `65_optional_export_figure4_figure5_panels.ipynb`
   - panel-level exports for manual figure assembly
   - not core science

Not recommended as public entry notebooks:

- `PipelineE_K3_manuscript_figures_notebook.ipynb`
- `PipelineE_K3_manuscript_figures_notebook_with_raster.ipynb`

These should be treated as umbrella provenance notebooks rather than public workflow definitions.

## Mapping from current files to cleaned set

| Cleaned target | Main current source(s) |
| --- | --- |
| `60_fit_final_k3_fusion_hmm.ipynb` | `R01_PipelineE_full_model.ipynb` |
| `61_review_final_k3_fit_qc_and_state_dynamics.ipynb` | `PipelineE_K3_results_review_notebook_fixed_paths.ipynb` plus the dynamics/raster portions of `PipelineE_K3_manuscript_figures_notebook_with_raster.ipynb` |
| `62_reconstruct_bold_state_networks_and_ranked_contrasts.ipynb` | `PipelineE_K3_state_physiology_notebook_schaefer_adapted_fix1.ipynb` plus any needed BOLD-only summary fragments from the umbrella manuscript-figure notebook(s) |
| `63_reconstruct_crossmodal_state_blocks_and_ranked_contrasts.ipynb` | cross-modal half of `PipelineE_K3_fusion_input_and_crossmodal_notebook.ipynb` plus any needed cross-modal fragments from the umbrella manuscript-figure notebook(s) |
| `64_build_parcelized_cortical_state_maps.ipynb` | `PipelineE_K3_brainmaps_surface_parcelized_notebook_fix4.ipynb` plus any needed surface-map fragments from the umbrella manuscript-figure notebook(s) |
| `65_optional_export_figure4_figure5_panels.ipynb` | `PipelineE_K3_Fig4_Fig5_panel_exports_notebook_v3.ipynb` |

Archive/provenance only:

- `PipelineE_K3_manuscript_figures_notebook.ipynb`
- `PipelineE_K3_manuscript_figures_notebook_with_raster.ipynb`
- the Stage-6 provenance files in `notebooks/_archive_raw_original_names/7_final model/`

## Downstream and upstream integration assumptions to verify in Pass 2

### Upstream assumptions

1. `R01_PipelineE_full_model.ipynb` consumes Stage-4 outputs directly, not Stage-5 outputs.
   - It auto-finds `segments_manifest.tsv` under the final Stage-4 root.
   - It expects at least `run` and `seg_path` columns.

2. The active public default appears to be:
   - `DATA_VARIANT = "intermediate"`
   - `FEATURE_MODE = "nolags"`
   - `MINLEN = 15`

3. The final-fit notebook uses the selected `K = 3`, but does not appear to require any Stage-5 summary file beyond that scientific choice.

### Downstream assumptions

Likely final-model artifact contract for later notebooks:

- root-level outputs:
  - `run_meta.json`
  - `preproc_meta.json`
  - `preproc_params.npz`
  - `qc_summary.json`
  - `run_metrics.tsv`
  - `subject_metrics.tsv`
  - `dwell_from_A.tsv`
  - `topM_seeds.json`
  - seed-screening and seed-stability sidecars
- `final/` outputs:
  - `best_seed.json`
  - `refit_results.json`
  - `refit_metrics.json`
  - `trans_prob.npy`
  - `means_pca.npy`
  - `covs_pca.npy`
  - `state_signature_ut_boldcorr.npy`
- per-run decoding folders:
  - `gamma/<run>/gamma_segXXXX.npy`
  - `viterbi/<run>/viterbi_segXXXX.npy`

Pass 2 should verify which of these are truly required by the later Stage-6 notebooks and by future Stage-7/8 cleanup.

### Environment assumptions

The final-fit notebook is clearly compute-sensitive and environment-sensitive:

- `tensorflow`
- `osl_dynamics`
- GPU-aware execution
- XLA disabling
- memory-growth or GPU memory cap handling
- WSL-style path assumptions

Later downstream notebooks also add:

- `nibabel`
- `nilearn`
- optional `networkx`
- optional `plotly`

The surface notebook may also trigger Nilearn fsaverage fetching if the surface files are not already cached locally.

## Risks and ambiguities

1. This folder mixes final-model fitting with later-stage summary/figure work.
   - If refactored too aggressively into one notebook, the public repo will blur the distinction between fitting the model and interpreting it.

2. `PipelineE_K3_fusion_input_and_crossmodal_notebook.ipynb` is scientifically mixed.
   - Its cross-modal reconstruction half belongs downstream of the final model.
   - Its representative fusion-input illustration half overlaps earlier alignment-stage storytelling.

3. The umbrella manuscript-figure notebooks are highly overlapping.
   - `PipelineE_K3_manuscript_figures_notebook_with_raster.ipynb` appears to supersede `PipelineE_K3_manuscript_figures_notebook.ipynb`.
   - Neither should define the cleaned public workflow directly.

4. Reference-state handling must not be silently rewritten.
   - Several notebooks infer the dominant/reference state from the final FO vector when not explicitly set.
   - The manuscript-facing interpretation centers on S2 as the dominant/reference state.
   - This needs explicit Pass-2 checking rather than silent cleanup.

5. The surface notebook contains a visible design branch.
   - The “Option B” output block is a plotting/provenance variation, not obviously the canonical scientific default.

6. Figure assembly remains partly hybrid/manual.
   - `PipelineE_K3_Fig4_Fig5_panel_exports_notebook_v3.ipynb` is explicitly for separate panel export and later manual assembly.
   - The umbrella figure notebooks are manuscript-oriented composites, not clean computational stages.

## Bottom-line recommendation

Stage 6 should be treated as:

- one core final-model fit notebook,
- one cleaned post-fit QC/dynamics notebook,
- and a small set of downstream K=3 reconstruction notebooks that stay separate by scientific role.

The main public refactor should **not** be built around the umbrella manuscript-figure notebooks. Those are better treated as provenance notebooks that show how multiple later results were once combined in one place.

For Pass 3, the safest public structure is:

- keep the final full-data fit separate,
- keep BOLD-only and cross-modal reconstructions separate,
- keep parcelized surface maps separate,
- and keep panel-export notebooks optional or archive-only.
