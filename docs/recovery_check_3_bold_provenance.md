## Evidence reviewed

Active Stage 3 files:

- `notebooks/3_bold/R01_BOLD_Atlas_to_BoldGrid.ipynb`
- `notebooks/3_bold/R01_BOLD_ParcelPC1_Export.ipynb`
- `notebooks/3_bold/R01_BOLD_ParcelPC1_QC.ipynb`

Archive and provenance material:

- `notebooks/_archive_raw_original_names/3_bold preprocessing atlas parcel/R01_BOLD_Atlas_to_BoldGrid.md`
- `notebooks/_archive_raw_original_names/3_bold preprocessing atlas parcel/R01_BOLD_ParcelPC1_Export.md`
- `notebooks/_archive_raw_original_names/3_bold preprocessing atlas parcel/R01_BOLD_ParcelPC1_QC.md`
- `notebooks/_archive_raw_original_names/3_bold preprocessing atlas parcel/BOLD exporter and nuisance regression.txt`
- `notebooks/_archive_raw_original_names/3_bold preprocessing atlas parcel/BOLD inputs and space.docx`
- `notebooks/_archive_raw_original_names/3_bold preprocessing atlas parcel/Table_S_BOLD_parcel_atlas_summary.csv`
- `notebooks/_archive_raw_original_names/3_bold preprocessing atlas parcel/Table_S_BOLD_motion_nuisance_summary.csv`
- `notebooks/_archive_raw_original_names/3_bold preprocessing atlas parcel/Atlas mappings.png`

Local real-output example bundle:

- `C:\fusion_hmm\_local_recovery_examples\stage3_bold_provenance\atlas_on_boldgrid\`
- `C:\fusion_hmm\_local_recovery_examples\stage3_bold_provenance\parcel_pc1_v6\`

Manuscript references reviewed:

- `docs/refactor_plan_3_bold.md`
- `docs/_manuscript_reference/FULL_MANUSCRIPT.docx`
- `docs/_manuscript_reference/SUPPLEMENTAL_MATERIALS.docx`

## Example output structure observed

The local example bundle shows two separate Stage-3 output branches:

1. `atlas_on_boldgrid/`
   - `sub-01_ses-01_task-rest_space-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-200Parcels7Networks_dseg_onbold.nii.gz`
   - `sub-01_ses-01_task-rest_space-MNI152NLin2009cAsym_atlas_overlay.png`

2. `parcel_pc1_v6/`
   - `atlas_source/`
   - `atlas_on_grid/`
   - `npy/`
   - `qc/`
   - `qc_sign_v6/`
   - `dataset_index.csv`
   - `motion_qc_flags.csv`
   - `qc_motion_to_pc.csv`

Important observations:

- `dataset_index.csv` contains 15 runs, so it is a frozen full-run summary, not just a one-run sample.
- The local example includes only one concrete run's `atlas_on_grid` and `npy` files, so it is a schema/provenance sample rather than a complete rerun directory.
- The sample `parcel_pc1_v6` bundle does not include a `mat/` directory even though `dataset_index.csv` has an `out_mat` column. That means the local example bundle is partial.

## Atlas resolution findings

### Direct evidence

`R01_BOLD_Atlas_to_BoldGrid.ipynb` and its archive markdown copy use a res-01 atlas input:

- `tpl-MNI152NLin2009cAsym_res-01_atlas-Schaefer2018_desc-200Parcels7Networks_dseg.nii.gz`

`R01_BOLD_ParcelPC1_Export.ipynb` and its archive markdown copy use:

- `ATLAS_RESOLUTION = 2   # res-02 (2mm). Set to 1 if you prefer res-01.`

The local real-output `dataset_index.csv` records:

- `atlas_source_nii = .../parcel_pc1_v6/atlas_source/tpl-MNI152NLin2009cAsym_res-02_atlas-Schaefer2018_desc-200Parcels7Networks_dseg.nii.gz`

The local real-output folder also contains that frozen res-02 atlas file under:

- `parcel_pc1_v6/atlas_source/tpl-MNI152NLin2009cAsym_res-02_atlas-Schaefer2018_desc-200Parcels7Networks_dseg.nii.gz`

### Conclusion

For the actual BOLD parcel export path, the manuscript-aligned final atlas appears to be the res-02 Schaefer atlas frozen inside `parcel_pc1_v6/atlas_source/`.

This is stronger than a notebook comment. It is recorded in the frozen output schema that downstream files actually use.

### Important nuance

The separate atlas-mapping notebook is still a real provenance branch, but it appears to be a different branch that used res-01. The sample run's standalone atlas-on-BOLD-grid NIfTI and the export-path atlas-on-grid NIfTI are not the same file:

- different compressed hashes
- different decompressed sizes

So this is a real branch difference, not just a filename change.

## Atlas-on-grid producer findings

### Direct evidence

The atlas notebook writes:

- `*_atlas-Schaefer2018_desc-200Parcels7Networks_dseg_onbold.nii.gz`
- `*_atlas_overlay.png`
- `qc_atlas_on_boldgrid_summary.csv`

The exporter writes:

- `parcel_pc1_v6/atlas_on_grid/<runTag>_atlas_on_grid.nii.gz`

and records that path in `dataset_index.csv` under the `atlas_on_grid` column.

The reproducibility notebook loads:

- `atlas_on_grid = r["atlas_on_grid"]`

from `dataset_index.csv`, so its downstream dependency is the export-path atlas file, not the standalone `atlas_on_boldgrid/` output.

### Conclusion

The atlas-on-grid logic is genuinely duplicated.

For the parcel-export pipeline and downstream QC, the authoritative producer is:

- `R01_BOLD_ParcelPC1_Export.ipynb`

because:

- it freezes the atlas source actually used for export,
- it writes the `atlas_on_grid` files inside `parcel_pc1_v6/`,
- and all downstream Stage-3 QC that depends on atlas-on-grid files points back to those exporter outputs through `dataset_index.csv`.

The separate atlas notebook should be treated as authoritative for:

- atlas-preservation QC,
- overlay PNG generation,
- and likely the BOLD-side support material for Supplementary Figure S1C.

It should not be treated as the authoritative input producer for the parcel export path.

## Table S4 provenance

### Direct evidence

The archived `Table_S_BOLD_parcel_atlas_summary.csv` has columns:

- `Run`
- `Parcels`
- `All-NaN parcels (%)`
- `Mean-fallback parcels (n)`
- `Median voxels/parcel`
- `Atlas labels expected`
- `Atlas labels present`
- `Atlas labels missing`

The exporter writes parcel-level QC fields into `dataset_index.csv`:

- `n_parcels`
- `qc_pct_all_nan_parcels`
- `qc_n_mean_fallback`
- `median_voxels_per_parcel`

These match the archived Table S4 values exactly across all 15 runs after simple percentage conversion for `qc_pct_all_nan_parcels`.

The atlas notebook writes `qc_atlas_on_boldgrid_summary.csv` with:

- `n_labels_expected`
- `n_labels_present`
- `n_labels_missing`

Those are the remaining columns in Table S4.

### Conclusion

Manuscript-facing Table S4 is not produced by one current notebook in one step.

It is a post-assembled join of:

1. exporter `dataset_index.csv`
2. atlas notebook `qc_atlas_on_boldgrid_summary.csv`

No active code was found that writes `Table_S_BOLD_parcel_atlas_summary.csv` directly. The archived final table exists, and its values clearly come from those two upstream summaries.

## Table S5 provenance

### Direct evidence

The archived `Table_S_BOLD_motion_nuisance_summary.csv` has columns:

- `Run`
- `Volumes`
- `FD mean (mm)`
- `FD p95 (mm)`
- `FD max (mm)`
- `FD spikes (%)`
- `Motion-outlier TRs kept (%)`
- `Total regressors`
- `Note`

Every numeric column except `Note` matches `dataset_index.csv` exactly across all 15 runs after simple rounding and percentage conversion:

- `Volumes` -> `n_volumes`
- `FD mean (mm)` -> `qc_fd_mean`
- `FD p95 (mm)` -> `qc_fd_p95`
- `FD max (mm)` -> `qc_fd_max`
- `FD spikes (%)` -> `pct_fd_spikes * 100`
- `Motion-outlier TRs kept (%)` -> `pct_motion_out_any_kept * 100`
- `Total regressors` -> `n_total_regressors`

The separate `motion_qc_flags.csv` helps identify higher-motion runs but does not fully explain the final `Note` column. The `Note` values appear manually curated.

### Conclusion

Table S5 provenance is much cleaner than Table S4:

- the table is derived directly from exporter `dataset_index.csv`
- the only non-programmatic piece appears to be the hand-written `Note` column

No active code was found that writes `Table_S_BOLD_motion_nuisance_summary.csv` directly under that final manuscript filename.

## Figure S5 provenance

### What the manuscript says

The manuscript and supplement tie Figure S5 to:

- low residual FD-to-parcel-PC coupling after nuisance regression
- artifact-structure summaries showing that large parcel blowups were no longer present after the final nuisance model updates

### Active code and outputs found

The exporter notebook writes three relevant QC products:

1. `qc_motion_to_pc.csv`
   - run-level summary of median, p95, and max absolute FD-to-PC correlations

2. `qc/qc_parcel_blowups.csv`
   - run-level artifact-structure summary

3. `qc/qc_parcel_blowups_<runTag>.png`
   - run-level overlay of parcel blowup fraction and FD

The exporter also writes:

4. `qc/<runTag>_qc_pc1_fd.png`
   - first five parcel PC1 traces for a run with FD underneath

### What was not found

- no active notebook cell that assembles a finished `Figure S5`
- no archived final `Figure S5` image in the Stage-3 provenance folder
- no plotting code that turns `qc_motion_to_pc.csv` into a figure panel

### Conclusion

Figure S5 provenance is only partially scripted in the current repo snapshot.

The active code clearly produces the ingredients for the figure, especially:

- `qc_motion_to_pc.csv`
- `qc_parcel_blowups.csv`
- `qc_parcel_blowups_<runTag>.png`

It may also have used a representative `*_qc_pc1_fd.png`, but that is less clearly supported by the manuscript wording.

So the current evidence supports this interpretation:

- **known:** the figure depended on exporter QC sidecars, not on the reproducibility notebook
- **inferred:** the final figure was likely assembled later from those QC sidecars rather than written by one existing notebook
- **unresolved:** the exact final panel composition is not recoverable from the current Stage-3 code alone

## Reproducibility QC status

### Direct evidence

`R01_BOLD_ParcelPC1_QC.ipynb` writes:

- `qc_sign_v6/qc_sign_details.csv`
- `qc_sign_v6/qc_sign_summary.csv`
- `qc_sign_v6/fig_corr_hist.png`
- `qc_sign_v6/fig_passrate_per_run.png`

The supplement text explicitly mentions reproducibility checks and their successful outcome.

### Important contradiction

The notebook header says:

- it uses the exact stored `fd_spike_trs` from `dataset_index.csv`

But the local real-output `dataset_index.csv` has no `fd_spike_trs` column at all.

The QC notebook code also rebuilds the nuisance design from:

- `FD_THRESHOLD`
- `FD_CAT`
- `EXPAND_POST`
- filtered `motion_outlier*`

rather than loading a frozen spike list.

### Conclusion

The reproducibility/sign-check notebook is real and scientifically useful, but it is not manuscript-essential for any named Stage-3 table or figure based on the current evidence.

It is best classified as:

- optional provenance-side validation

not as a required public-facing Stage-3 entry point.

The notebook comment about `fd_spike_trs` is contradicted by both the active code and the example outputs and should be preserved as a discrepancy, not silently corrected during planning.

## Remaining ambiguities

1. The exact published panel composition of Figure S5 is still unresolved.
   The current repo contains the ingredients, but not a single explicit finished-figure generator.

2. The BOLD side of Supplementary Figure S1C may have come from the standalone atlas notebook branch that used res-01.
   If exact visual provenance matters, this needs confirmation before replacing it with a regenerated exporter-path atlas visualization.

3. The local example bundle is partial.
   It confirms the real output schema well, but it does not include every file type mentioned in `dataset_index.csv`.

## Recommended Stage-3 cleaned public-facing set after recovery

The earlier Stage-3 cleaned set still makes sense, but the roles should now be sharper:

1. `30_map_schaefer200_to_bold_run_grids.ipynb`
   - keep as atlas-preservation QC and overlay notebook
   - supports atlas-label checks and likely Figure S1C support
   - do not present it as the authoritative atlas producer for the parcel export path

2. `31_export_bold_parcel_pc1_with_nuisance_regression.ipynb`
   - make this the authoritative Stage-3 export path
   - preserve exporter-side atlas-on-grid writing
   - preserve res-02 atlas freezing

3. `32_build_table_s4_bold_parcel_atlas_summary.ipynb`
   - explicitly join exporter `dataset_index.csv` with atlas notebook `qc_atlas_on_boldgrid_summary.csv`

4. `33_build_table_s5_and_figure_s5_bold_qc.ipynb`
   - build Table S5 directly from `dataset_index.csv`
   - build a clean Figure S5 from the exporter QC sidecars
   - keep any manual figure-panel choices explicit if they cannot be inferred automatically

Optional only:

5. `34_validate_bold_parcel_pc1_reproducibility.ipynb`
   - public optional validation notebook if desired
   - otherwise archive/provenance-only

## Exact questions that still need human confirmation

1. For Supplementary Figure S5, which outputs were actually used in the published figure?
   The strongest candidates are:
   - `qc_motion_to_pc.csv`
   - `qc_parcel_blowups.csv`
   - `qc_parcel_blowups_<runTag>.png`
   - possibly a representative `*_qc_pc1_fd.png`

2. For Supplementary Figure S1C, should the public repo preserve the standalone atlas-mapping visual branch as published provenance, even though the final parcel export path appears to use the res-02 exporter atlas branch?

3. Do you want the reproducibility/sign-check notebook exposed publicly as an optional validation step, or kept archive/provenance-only?
