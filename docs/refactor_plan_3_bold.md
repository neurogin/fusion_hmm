## Scope

This memo is the Stage 3 Pass 1 inventory and refactor-planning note for `notebooks/3_bold/`.

Stage 3 corresponds mainly to:

- Main Methods 2.3
- Supplementary Methods 1.4
- Supplementary Results 2.4
- Supplementary Table S4
- Supplementary Table S5
- Supplementary Figure S1C
- Supplementary Figure S5

This pass does not rewrite, move, or delete any files. It inventories the current stage, separates core manuscript logic from mixed-in QC and exploratory material, and proposes a cleaned public-facing file set for a later implementation pass.

## Files inspected

Files in `notebooks/3_bold/`:

- `R01_BOLD_Atlas_to_BoldGrid.ipynb`
- `R01_BOLD_ParcelPC1_Export.ipynb`
- `R01_BOLD_ParcelPC1_QC.ipynb`

Provenance and manuscript-context files also reviewed:

- `docs/methods_map.md`
- `docs/manual_steps.md`
- `docs/repo_scope.md`
- `docs/final_dataset_spec.md`
- `docs/reproducibility_notes.md`
- `docs/figure_table_map.md`
- `docs/_manuscript_reference/FULL_MANUSCRIPT.docx`
- `docs/_manuscript_reference/SUPPLEMENTAL_MATERIALS.docx`
- provenance copies under `notebooks/_archive_raw_original_names/3_bold preprocessing atlas parcel/`

## High-level assessment

Stage 3 currently contains three notebooks that break into three main clusters:

1. Atlas-to-run-grid mapping:
   `R01_BOLD_Atlas_to_BoldGrid.ipynb`

2. Core BOLD nuisance-regression and parcel-PC1 export:
   the main exporter portion of `R01_BOLD_ParcelPC1_Export.ipynb`

3. Motion, parcel, and reproducibility QC:
   later QC cells in `R01_BOLD_ParcelPC1_Export.ipynb` plus `R01_BOLD_ParcelPC1_QC.ipynb`

The core science for the paper is present, but it is mixed with sidecar diagnostics and ad hoc audit cells. The main refactor need is not scientific recovery at this stage. The main need is to separate:

- the core manuscript-facing export path,
- the atlas-grid preparation path,
- the table/figure support QC path,
- and the exploratory/provenance-only material that should not be ported wholesale.

There are no obvious files from other manuscript stages sitting in `notebooks/3_bold/`. The bigger issue is that several notebooks mix valid final logic with extra checks that should remain archive-only or be ported selectively.

## File-by-file assessment

### `R01_BOLD_Atlas_to_BoldGrid.ipynb`

**Scientific purpose**

Maps the Schaefer 200-parcel atlas onto the exact voxel grid used by each run's BOLD outputs so that parcel extraction is done on a run-specific, voxel-index-consistent atlas image.

**Likely manuscript linkage**

- Main Methods 2.3
- Supplementary Methods 1.4
- Supplementary Results 2.4
- Supplementary Table S4
- likely support for the BOLD side of Supplementary Figure S1C

**Inputs**

- canonical Schaefer atlas NIfTI and labels
- per-run BOLD reference images
- optional brain masks
- optional raw BIDS JSON sidecars for task/TR metadata

**Outputs**

- per-run atlas-on-BOLD-grid NIfTI files
- optional overlay PNGs
- `qc_atlas_on_boldgrid_summary.csv`

**Assessment**

- core manuscript-relevant logic: yes
- QC sidecar content: yes
- exploratory/audit content mixed in: yes
- helper logic embedded: yes

**Keep / merge / split / archive recommendation**

- keep the core atlas-mapping logic
- split off the later audit-only cells from the public-facing version
- keep the ad hoc directory-diagnostic cells archive/provenance-only

**Notes**

This notebook appears to be the cleanest current source for run-grid atlas preparation, but later cells include exploratory checks and at least one brittle path diagnostic that should not shape the cleaned public notebook.

### `R01_BOLD_ParcelPC1_Export.ipynb`

**Scientific purpose**

Runs voxelwise nuisance regression on preprocessed BOLD data, extracts Schaefer parcel PC1 time series, and writes parcel outputs plus run-level metadata and QC summaries.

**Likely manuscript linkage**

- Main Methods 2.3
- Supplementary Methods 1.4
- Supplementary Results 2.4
- Supplementary Table S4
- Supplementary Table S5
- possibly Supplementary Figure S5 support

**Inputs**

- fMRIPrep BOLD NIfTI files
- confounds TSV/JSON sidecars
- brain masks
- Schaefer atlas files
- optional boldref and atlas-on-grid outputs

**Outputs**

- exported parcel PC1 files
- MAT sidecars
- NPY sidecars
- run-level `dataset_index.csv`
- optional atlas-on-grid NIfTIs
- optional QC figures
- additional QC CSVs in later cells

**Assessment**

- core manuscript-relevant logic: yes
- wrapper/driver role: effectively yes, but with major embedded helper logic
- helper logic embedded: substantial
- QC sidecars mixed in: substantial
- exploratory/ad hoc inspections mixed in: yes

**Keep / merge / split / archive recommendation**

- keep the main exporter logic
- split the QC and artifact-inspection sidecars out of the public-facing exporter
- keep the exporter-focused public file separate from later table/figure support QC
- do not port the whole notebook wholesale

**Notes**

This notebook currently serves too many roles at once:

- the actual manuscript exporter,
- atlas resampling fallback,
- dataset index writer,
- motion/nuisance summary generator,
- parcel blowup diagnostic notebook,
- and run-specific troubleshooting notebook.

The cleaned public-facing stage should keep the core exporter path intact but separate the sidecar diagnostics.

### `R01_BOLD_ParcelPC1_QC.ipynb`

**Scientific purpose**

Recomputes parcel PC1 signals for sampled parcels and compares them against saved exports to check sign convention and export reproducibility.

**Likely manuscript linkage**

- supplementary QC support
- possible indirect support for Supplementary Results 2.4
- not clearly mapped to a named manuscript table or figure

**Inputs**

- `dataset_index.csv`
- exported parcel outputs
- original BOLD files
- confounds files
- atlas-on-grid files

**Outputs**

- `qc_sign_details.csv`
- `qc_sign_summary.csv`
- correlation histogram figure
- pass-rate-per-run figure

**Assessment**

- core manuscript logic: no
- supplementary QC sidecar: yes
- helper logic embedded: yes
- likely public-facing only if a strong manuscript-facing role is confirmed

**Keep / merge / split / archive recommendation**

- keep as provenance/QC source material
- selectively merge only the manuscript-relevant parts if later needed
- otherwise treat this notebook as archive/provenance-only rather than a required public entry file

**Notes**

The notebook's opening description says it uses the exact stored FD spike TR list from `dataset_index.csv`, but the implemented design-matrix reconstruction appears to reapply the nuisance rules rather than simply consuming a frozen spike list. That is a real code/documentation tension and should be checked in Pass 2 rather than harmonized silently.

## Merge and split recommendations

### Clear split candidates

1. Split `R01_BOLD_ParcelPC1_Export.ipynb` into:
   - one public-facing exporter entry file
   - one public-facing QC/table/figure support file
   - archive-only ad hoc inspection cells that should not be ported directly

2. Split `R01_BOLD_Atlas_to_BoldGrid.ipynb` into:
   - a public-facing atlas mapping notebook
   - archive-only audit cells and directory diagnostics

### Material that should remain helper-level only

These are currently embedded inside notebooks and should remain helper-level or internal to a cleaned public entry file rather than becoming top-level public notebooks on their own:

- BOLD file and sidecar discovery helpers
- confounds design-matrix construction logic
- parcel PC1 extraction utilities
- atlas resampling utility blocks
- parcel-level recomputation utilities used only for export verification

### Merge candidates

The cleanest merge for public release is to combine the manuscript-facing QC outputs into focused files rather than preserving every original QC notebook boundary:

- atlas-preservation and parcel-coverage summaries can be merged into one cleaned Table S4 support notebook
- motion/nuisance summaries and clearly manuscript-relevant BOLD QC figures can be merged into one cleaned Table S5 / Figure S5 support notebook

The reproducibility/sign-check notebook should only be merged if its outputs are clearly needed for the public manuscript workflow. Otherwise it should stay provenance-only.

## Proposed cleaned GitHub-facing file set

Recommended public-facing Stage 3 set:

1. `30_map_schaefer200_to_bold_run_grids.ipynb`
   - public-facing atlas-on-run-grid preparation
   - writes run-grid atlas files and basic atlas alignment QC

2. `31_export_bold_parcel_pc1_with_nuisance_regression.ipynb`
   - public-facing BOLD exporter
   - preserves nuisance model, parcel extraction, parcel-PC1 sign convention, and output schema

3. `32_qc_bold_atlas_preservation_table_s4.ipynb`
   - manuscript-facing summary notebook for atlas-label preservation and parcel coverage
   - should consume outputs from `30` and `31`
   - supports Supplementary Table S4 and likely the BOLD side of Figure S1C

4. `33_qc_bold_motion_nuisance_table_s5_and_figure_s5.ipynb`
   - manuscript-facing motion/nuisance summary and BOLD QC figure support
   - should consume exporter outputs and motion/QC sidecars from the core export path

Optional, only if later confirmed to be worth exposing:

5. `34_qc_bold_parcel_pc1_reproducibility.ipynb`
   - sampled recomputation/sign-check notebook
   - likely supplementary validation only
   - probably not needed as a first-line public entry point

## Mapping from current files to cleaned set

### `30_map_schaefer200_to_bold_run_grids.ipynb`

Primary source:

- `R01_BOLD_Atlas_to_BoldGrid.ipynb`

Port into cleaned file:

- the main atlas-on-grid resampling path
- the atlas QC summary output
- any clearly useful overlay export

Leave archive-only:

- later ad hoc counting cells
- brittle path diagnostics
- directory inventory checks that are not part of the scientific pipeline

### `31_export_bold_parcel_pc1_with_nuisance_regression.ipynb`

Primary source:

- core exporter portion of `R01_BOLD_ParcelPC1_Export.ipynb`

Port into cleaned file:

- run discovery and input matching
- nuisance-regression design logic
- parcel extraction and sign-fixing logic
- parcel export outputs
- `dataset_index.csv`

Leave for other cleaned QC files or archive-only:

- motion flag summary sidecars
- parcel blowup diagnostics
- one-off run inspections

### `32_qc_bold_atlas_preservation_table_s4.ipynb`

Primary sources:

- `R01_BOLD_Atlas_to_BoldGrid.ipynb`
- exporter metadata from `R01_BOLD_ParcelPC1_Export.ipynb`
- archived `Table_S_BOLD_parcel_atlas_summary.csv`

Expected role:

- create a clean manuscript-facing summary for atlas-label preservation and parcel coverage
- gather the information currently split across atlas QC outputs and exporter metadata

### `33_qc_bold_motion_nuisance_table_s5_and_figure_s5.ipynb`

Primary sources:

- later QC cells in `R01_BOLD_ParcelPC1_Export.ipynb`
- archived `Table_S_BOLD_motion_nuisance_summary.csv`

Expected role:

- consolidate motion/nuisance summary outputs
- support Table S5
- include only the figure outputs that are clearly manuscript-relevant

### `34_qc_bold_parcel_pc1_reproducibility.ipynb` if needed

Primary source:

- `R01_BOLD_ParcelPC1_QC.ipynb`

Expected role:

- optional validation notebook
- not part of the main public run path unless Pass 2 confirms a strong manuscript-facing need

## Risks and ambiguities

1. **Atlas resolution mismatch needs checking.**
   `R01_BOLD_Atlas_to_BoldGrid.ipynb` appears to use a res-01 Schaefer atlas input, while `R01_BOLD_ParcelPC1_Export.ipynb` is configured around TemplateFlow atlas resolution 2. This should be treated as an explicit Pass 2 dependency and manuscript-alignment check, not silently harmonized during implementation.

2. **Atlas-on-grid logic is duplicated across notebooks.**
   The exporter notebook can resample atlas data itself and optionally save atlas-on-grid outputs, while the separate atlas notebook already performs run-grid mapping. Pass 2 should determine which path should be authoritative in the cleaned public release.

3. **Figure S5 mapping is not fully clear from the current notebooks.**
   The current Stage 3 notebooks clearly support motion and nuisance summaries, but the exact current generator for the manuscript-facing Figure S5 is not obvious from the present file set. Some existing QC plots may be sidecar diagnostics rather than the actual manuscript-facing figure.

4. **`R01_BOLD_ParcelPC1_QC.ipynb` contains a wording-versus-code tension.**
   The notebook claims to use frozen FD spike TRs from `dataset_index.csv`, but the implemented code appears to rebuild nuisance logic from rules and confounds instead. This should be documented and checked before any cleaned public QC notebook is implemented.

5. **The BOLD-file selection policy must remain explicit.**
   The exporter currently prefers `desc-preproc_bold.nii.gz` and leaves AROMA variants as commented alternatives. This is a scientifically meaningful choice and should not be silently changed during refactor.

6. **Some figure assembly may remain hybrid/manual.**
   Supplementary Figure S1C appears to depend on atlas-alignment visuals and cross-stage context. The public repo should be honest if final panel assembly is partly manual or staged across earlier notebooks.

## Bottom-line recommendation

Stage 3 is in good shape for a cleaned public refactor, but the current notebooks are too mixed to expose directly.

The best public-facing structure is:

- one atlas-grid preparation notebook,
- one core BOLD export notebook,
- one atlas/parcel summary notebook for Table S4,
- one motion/nuisance QC notebook for Table S5 and Figure S5,
- with the current recomputation/sign-check notebook retained as optional validation or archive/provenance-only unless Pass 2 shows it is manuscript-essential.

Pass 2 should focus on dependency and manuscript-alignment checks rather than broad recovery. The main items to verify next are:

- the atlas resolution mismatch,
- the authoritative atlas-on-grid producer,
- the exact manuscript-facing source for Figure S5,
- and whether the parcel reproducibility QC notebook is truly part of the public workflow or just provenance-side validation.
