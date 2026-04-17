function r01_eeg_iclabel_prune_and_metadata(root_raw_eeglab, out_base_dir, qc_table_dir, varargin)
%R01_EEG_ICLABEL_PRUNE_AND_METADATA Legacy compatibility wrapper.
%
% The active Stage-1 scientific implementation now lives in:
%   notebooks/1_eeg_sensor/helpers/prune_iclabel_components_and_export_metadata.m
%
% This preserved file remains only so older provenance callers still run.

this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
helper_dir = fullfile(this_dir, 'helpers');
if ~isempty(helper_dir)
    addpath(helper_dir);
end

prune_iclabel_components_and_export_metadata(root_raw_eeglab, out_base_dir, qc_table_dir, varargin{:});

end
