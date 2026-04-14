function batch = batch_export_eeg_parcel_pc_outputs(protocolRoot, outDir, varargin)
%BATCH_EXPORT_EEG_PARCEL_PC_OUTPUTS Public Stage-2 wrapper for parcel export.
%
% What this helper does:
%   Runs the preserved Stage-2 batch exporter that creates run-wise parcel
%   PC outputs, gain-normalized MAT files, NPY sidecars, and batch summary
%   tables from Brainstorm kernels and standardized scout files.
%
% When it is used:
%   Called by `23_export_eeg_parcel_pc1_and_gain_normalize.m`.
%
% Key inputs:
%   - Brainstorm protocol root
%   - parcel-output directory
%   - the same name-value options accepted by the preserved v3 exporter
%
% Key outputs:
%   Returns the same batch struct as the preserved helper and writes the
%   same parcel MAT, NPY, and CSV outputs as before.
%
% Important note:
%   This wrapper keeps the restored `*_time_sec.npy` sidecar logic and all
%   manuscript-preserved exporter settings unchanged by delegating to
%   `r01_batch_export_eeg_parcel_pc_v3.m`.

batch = r01_batch_export_eeg_parcel_pc_v3(protocolRoot, outDir, varargin{:});

end
