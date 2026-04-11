% 10_eeg_prune_iclabel_and_export_clean_sets
%
% Public-facing stage-1 entry point for manuscript-default EEG sensor
% pruning and Brainstorm-facing clean-set export.
%
% Manuscript linkage:
%   - Main Methods 2.2.1
%   - Supplementary Methods 1.1
%
% Inputs:
%   - Author-preprocessed EEGLAB .set files under P.paths.raw_eeglab_dir
%   - EEGLAB on the MATLAB path
%   - ICLabel installed in the EEGLAB plugins path
%
% Outputs:
%   - <ic_pruned_dir>\withICA\*_withICA.set
%   - <ic_pruned_dir>\clean\*_clean.set
%   - <qc_tables_dir>\eeg_ic_prune_summary_<TAG>.csv
%   - <qc_tables_dir>\eeg_ic_prune_lists_<TAG>.csv
%   - <qc_tables_dir>\*_iclabel_table_<TAG>.csv
%
% Notes:
%   - This script preserves the historical pruning implementation in
%     r01_eeg_iclabel_prune_and_metadata.m.
%   - The manuscript-default path is the conservative ICLabel
%     "reject_artifacts" policy at 0.70.
%   - The legacy "brain_threshold" branch remains preserved in provenance,
%     but it is not the paper path.

stage1_dir = fileparts(mfilename('fullpath'));
if isempty(stage1_dir)
    error('Could not resolve the stage-1 script location. Run this file from disk.');
end

helper_dir = fullfile(stage1_dir, 'helpers');
addpath(stage1_dir);
addpath(helper_dir);

P = r01_stage1_params();

% -------------------------------------------------------------------------
% User-configurable parameters
% -------------------------------------------------------------------------
overwrite_outputs = false;
save_iclabel_tables = true;
save_withICA = true;
save_clean = true;
clean_remove_iclabel = true;
file_pattern = '*.set';

assert_configured_input_dir(P.paths.raw_eeglab_dir, 'P.paths.raw_eeglab_dir');

fprintf('\nStage 1 / Step 10: ICLabel pruning and clean-set export\n');
fprintf('  Raw EEGLAB input: %s\n', char(P.paths.raw_eeglab_dir));
fprintf('  Output directory: %s\n', char(P.paths.ic_pruned_dir));
fprintf('  QC tables dir:    %s\n', char(P.paths.qc_tables_dir));
fprintf('  IC policy:        %s\n', char(P.ic_policy));
fprintf('  Reject threshold: %.2f\n', P.ic_reject_threshold);
fprintf('  Reject classes:   %s\n\n', strjoin(cellstr(P.ic_reject_classes), ', '));

r01_eeg_iclabel_prune_and_metadata( ...
    char(P.paths.raw_eeglab_dir), ...
    char(P.paths.ic_pruned_dir), ...
    char(P.paths.qc_tables_dir), ...
    'ic_policy', char(P.ic_policy), ...
    'reject_threshold', P.ic_reject_threshold, ...
    'reject_classes', P.ic_reject_classes, ...
    'overwrite', overwrite_outputs, ...
    'save_iclabel_tables', save_iclabel_tables, ...
    'save_withICA', save_withICA, ...
    'save_clean', save_clean, ...
    'clean_remove_iclabel', clean_remove_iclabel, ...
    'file_pattern', file_pattern);

fprintf('\nStage 1 / Step 10 complete.\n');
fprintf('Next manual step: review 11_brainstorm_exclusion_marking_manual.md before Brainstorm marking.\n');

function assert_configured_input_dir(path_value, label)
path_char = char(path_value);
if contains(path_char, '<SET_')
    error('%s is still using a placeholder path. Edit helpers/r01_stage1_params.m first.', label);
end
if ~exist(path_char, 'dir')
    error('%s does not exist: %s', label, path_char);
end
end
