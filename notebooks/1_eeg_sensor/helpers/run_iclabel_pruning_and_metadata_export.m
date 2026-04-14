function run_iclabel_pruning_and_metadata_export(root_raw_eeglab, out_base_dir, qc_table_dir, varargin)
%RUN_ICLABEL_PRUNING_AND_METADATA_EXPORT Public wrapper for Stage-1 pruning.
%
% What this helper does:
%   Runs the preserved ICLabel pruning and metadata export implementation
%   that writes the "withICA", "clean", and QC-table outputs used in the
%   Stage-1 public workflow.
%
% When it is used:
%   Called by `10_eeg_prune_iclabel_and_export_clean_sets.m`.
%
% Key inputs:
%   - raw EEGLAB directory
%   - output base directory
%   - QC table directory
%   - the same name-value options accepted by the preserved legacy helper
%
% Key outputs:
%   Delegates to the preserved exporter and writes the same files as before.
%
% Important note:
%   This file is the descriptive public-facing wrapper. The underlying
%   scientific implementation still lives in
%   `r01_eeg_iclabel_prune_and_metadata.m`.

stage1_dir = fileparts(fileparts(mfilename('fullpath')));
if ~isempty(stage1_dir)
    addpath(stage1_dir);
end

r01_eeg_iclabel_prune_and_metadata(root_raw_eeglab, out_base_dir, qc_table_dir, varargin{:});

end
