function run_iclabel_pruning_and_metadata_export(root_raw_eeglab, out_base_dir, qc_table_dir, varargin)
%RUN_ICLABEL_PRUNING_AND_METADATA_EXPORT Public wrapper for Stage-1 pruning.
%
% What this helper does:
%   Runs the promoted Stage-1 ICLabel pruning helper that writes the
%   "withICA", "clean", and QC-table outputs used in the public workflow.
%
% When it is used:
%   Called by `eeg_prune_iclabel_and_export_clean_sets_10.m`.
%
% Key inputs:
%   - raw EEGLAB directory
%   - output base directory
%   - QC table directory
%   - the same name-value options accepted by the preserved legacy helper
%
% Key outputs:
%   Delegates to the promoted helper and writes the same files as before.
%
% Important note:
%   The scientific logic now lives in
%   `prune_iclabel_components_and_export_metadata.m`. The older
%   `r01_eeg_iclabel_prune_and_metadata.m` file remains only as a
%   compatibility wrapper.

this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
stage1_dir = fileparts(this_dir);

if ~isempty(stage1_dir)
    addpath(stage1_dir);
end
if ~isempty(this_dir)
    addpath(this_dir);
end

assert_dependency_exists(fullfile(this_dir, 'prune_iclabel_components_and_export_metadata.m'), ...
    ['Missing Stage-1 pruning helper:' newline ...
     '  notebooks/1_eeg_sensor/helpers/prune_iclabel_components_and_export_metadata.m']);

prune_iclabel_components_and_export_metadata(root_raw_eeglab, out_base_dir, qc_table_dir, varargin{:});

end

function assert_dependency_exists(path_to_file, message_text)
if exist(path_to_file, 'file') ~= 2
    error('%s', message_text);
end
end
