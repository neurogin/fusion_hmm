# Recovery Check: `*_time_sec.npy` Sidecar

## Scope

This memo checks the missing stage-2 `*_time_sec.npy` sidecar using the local example bundle:

- `fusion_hmm/_local_recovery_examples/time_sec_sidecar/sub-01_ses-01/`

Goal:

- determine exactly what `*_time_sec.npy` contains
- determine whether it can be reconstructed deterministically from current stage-2 inputs/outputs
- recommend where the logic should live if restored

## Evidence reviewed

I inspected the example files:

- `sub-01_ses-01_desc-ICRej70_clean_PC1_gnorm.npy`
- `sub-01_ses-01_desc-ICRej70_clean_time_sec.npy`
- `sub-01_ses-01_desc-ICRej70_clean_parcelPC_raw.mat`
- `sub-01_ses-01_desc-ICRej70_clean_parcelPC_gnorm.mat`
- `sub-01_ses-01_desc-ICRej70_clean.set`

I also checked:

- active stage-2 exporter code in `notebooks/2_eeg_source/`
- the stage-4 alignment notebook that consumes the sidecar:
  - `notebooks/4_alignment/R01_PipelineA_align_trmask_lags_v3_gnorm_allTR_INTERMEDIATE.ipynb`
- archive references for any older producer logic

## What `time_sec.npy` appears to contain

From the example:

- `PC1_gnorm.npy` has shape:
  - `[150380, 200]`
- `time_sec.npy` has shape:
  - `[150380, 1]`
- `parcelPC_gnorm.mat` contains:
  - `PC1` with size `[150380, 200]`
  - `diagOut.nTime = 150380`
  - `diagOut.srate = 250`
- `parcelPC_raw.mat` contains:
  - `PC1` with size `[150380, 200]`
  - `diagOut.nTime = 150380`
  - `diagOut.srate = 250`
- the clean EEGLAB file contains:
  - `EEG.pnts = 150380`
  - `EEG.srate = 250`
  - `EEG.xmin = 0`

### Direct comparison findings

- `time_sec.npy` length exactly matches the first dimension of `PC1_gnorm.npy`
- `time_sec.npy` starts at `0`
- `time_sec.npy` is sample-level, not TR-level
- `time_sec.npy` reflects the exported parcel sample axis at the EEG sampling rate
- the nominal sampling step is `1 / 250 = 0.004 s`
- because the array is stored as `single`, adjacent differences show small float32 rounding variation

Example values:

- first 10:
  - `0, 0.00400000019, 0.00800000038, ...`
- last value:
  - about `601.515991210938`

### Exact reconstruction check

The example `time_sec.npy` matches **exactly**:

- `single((0:nTime-1)' / srate)`

and also exactly:

- `single((0:EEG.pnts-1)' / EEG.srate)`

It does **not** match exactly:

- `single(EEG.times(:) / 1000)`

In the example, the max absolute difference versus `single(EEG.times/1000)` is:

- `0.00018310546875`

So the sidecar is not just the saved EEGLAB `EEG.times` vector written out to `.npy`.

## Definition inferred from the example

`*_time_sec.npy` appears to be:

- a column vector
- stored as `single`
- sample-index based
- starting at zero
- defined as:

```matlab
time_sec = single((0:nTime-1)' / srate);
```

where:

- `nTime` is the exported parcel time-series length
- `srate` is the clean EEG sampling rate

## Relationship to the other example files

### Relationship to `*_PC1_gnorm.npy`

- same number of rows
- same sample axis
- `PC1_gnorm.npy` matches `parcelPC_gnorm.mat -> PC1` exactly

This strongly suggests that `time_sec.npy` was intended as the sample-time companion for the exported gain-normalized parcel PC matrix.

### Relationship to `*_parcelPC_gnorm.mat` and `*_parcelPC_raw.mat`

- both MAT files contain enough metadata to reconstruct the sidecar deterministically:
  - `diagOut.nTime`
  - `diagOut.srate`
- the sidecar does not depend on gain normalization itself
- raw and gnorm runs share the same sample axis

### Relationship to `*_clean.set`

- the sidecar is also reconstructable from the clean EEGLAB file:
  - `EEG.pnts`
  - `EEG.srate`
- in the example, `EEG.xmin = 0`, which is consistent with the zero-start sidecar
- the exact sidecar still aligns better with index/srate than with `EEG.times`

## Repo and archive search result

I found:

- consumer code in stage 4 that expects `*_time_sec.npy`
- no active producer code in the repo
- no archive file in this repo that clearly writes the sidecar

The stage-4 alignment notebook currently does:

- load `*_PC1_gnorm.npy`
- require `*_time_sec.npy`
- raise `FileNotFoundError` if the time sidecar is absent

So the current repo contains a downstream consumer but no visible producer.

## Whether recovery is possible

Yes. Recovery appears possible and deterministic.

### Why it is reliable

The example shows an exact match to:

```matlab
single((0:nTime-1)' / srate)
```

This formula is fully determined from current stage-2 data that already exist during export:

- the clean EEG length
- the clean EEG sampling rate
- or equivalently `diagOut.nTime` and `diagOut.srate`

This is not a guess from manuscript wording. It is an exact reconstruction rule verified against the example bundle.

## Recommended implementation location

Best location:

- inside a helper used by `23_export_eeg_parcel_pc1_and_gain_normalize.m`

More specifically:

- restore it in the stage-2 exporter helper path, not downstream in stage 4

### Why it belongs logically in stage 2

`time_sec.npy` is:

- the sample axis for the exported parcel PC arrays
- tied directly to the stage-2 EEG parcel export
- written alongside `*_PC1_gnorm.npy`

That makes it a stage-2 export artifact, not a stage-4 derived product.

Stage 4 should consume this sidecar, not define it.

### Best concrete place

The most natural place is the NPY-writing block of:

- `r01_batch_export_eeg_parcel_pc_v3.m`

Reason:

- that helper already writes:
  - `*_PC1_gnorm.npy`
  - `*_PC2_gnorm.npy`
  - `*_PVE1.npy`
  - masks and metadata sidecars
- it already has access, directly or indirectly, to:
  - `nTime`
  - `srate`

An alternative would be adding the sidecar in `r01_export_parcel_pc1_one_run_v3.m`, but that file currently writes the MAT artifact, not the `.npy` sidecars. For traceability and output grouping, the batch exporter’s NPY block is the cleaner fit.

## Remaining ambiguity

The main remaining ambiguity is not the formula itself. The formula looks settled.

The remaining question is policy:

- whether the restored sidecar should be generated from:
  - `diagOut.nTime` and `diagOut.srate`
  - or by re-reading the clean `.set`

For behavior preservation, the simplest and safest rule is:

- use the exact exported time axis length
- use the exact exported sampling rate
- construct:
  - `single((0:nTime-1)' / srate)`

This avoids depending on EEGLAB’s saved `EEG.times` floating representation, which the example does **not** match exactly.

## Bottom line

The example strongly indicates that `*_time_sec.npy` is:

- a sample-level time vector
- aligned to the rows of `*_PC1_gnorm.npy`
- starting at zero
- stored as `single`
- defined exactly by:

```matlab
single((0:nTime-1)' / srate)
```

Recovery is possible and deterministic from current stage-2 data.

It belongs logically in **stage 2**, not stage 4, and should be restored in the helper path behind:

- `23_export_eeg_parcel_pc1_and_gain_normalize.m`

preferably in:

- `r01_batch_export_eeg_parcel_pc_v3.m`

next to the existing `.npy` export logic.
