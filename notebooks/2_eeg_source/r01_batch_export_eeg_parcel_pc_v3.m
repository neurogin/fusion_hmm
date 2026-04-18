function batch = r01_batch_export_eeg_parcel_pc_v3(protocolRoot, outDir, varargin)
%R01_BATCH_EXPORT_EEG_PARCEL_PC_V3 Legacy compatibility wrapper.
%
% The active descriptive Stage-2 batch parcel exporter now lives in:
%   notebooks/2_eeg_source/helpers/batch_export_eeg_parcel_pc_outputs.m
%
% This preserved file remains only so older provenance code can still call
% the batch exporter without changing the scientific output contract.

this_dir = fileparts(mfilename('fullpath'));
helper_dir = fullfile(this_dir, 'helpers');
if exist(helper_dir, 'dir') == 7
    addpath(helper_dir);
end

batch = batch_export_eeg_parcel_pc_outputs(protocolRoot, outDir, varargin{:});

end
