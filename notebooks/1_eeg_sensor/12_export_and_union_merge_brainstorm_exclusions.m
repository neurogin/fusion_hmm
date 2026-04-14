% 12_export_and_union_merge_brainstorm_exclusions
%
% What this file does:
%   Convert manual Brainstorm exclusion markings into standardized TSV files,
%   then merge overlapping or touching exclusions into one union mask per run.
%
% Manuscript linkage:
%   - Main Methods 2.2.1
%   - Supplementary Methods 1.1
%
% Manual dependency:
%   Brainstorm marking must already be complete. This file does not create
%   BAD or boundary annotations; it only exports and merges them.
%
% Preserved implementation note:
%   This public entry file keeps the recovered exporter behavior unchanged,
%   including its current point-event handling. The low-level export and
%   merge logic still lives in the preserved helper functions.

% -------------------------------------------------------------------------
% Step 0. Locate this stage folder and add the stage-1 helper path
% -------------------------------------------------------------------------
stage1_dir = fileparts(mfilename('fullpath'));
if isempty(stage1_dir)
    error('Could not resolve the stage-1 script location. Run this file from disk.');
end

helper_dir = fullfile(stage1_dir, 'helpers');
config_file = fullfile(helper_dir, 'stage1_eeg_sensor_settings.m');

addpath(stage1_dir);
addpath(helper_dir);

% -------------------------------------------------------------------------
% Step 1. Load the stage-1 configuration
%
% Edit path placeholders in helpers/stage1_eeg_sensor_settings.m before running.
% -------------------------------------------------------------------------
P = stage1_eeg_sensor_settings();

brainstorm_db_root = char(P.paths.brainstorm_db_root);
bst_export_dir = char(P.paths.bst_export_dir);
file_filter = char(P.file_filter);
merge_tolerance_sec = P.mask.merge.adjacency_tol_sec;
min_union_duration_sec = P.mask.merge.min_dur_sec;

% -------------------------------------------------------------------------
% Step 2. User-editable run options
% -------------------------------------------------------------------------
overwrite_existing_outputs = true;
recursive_bst_scan = true;

% -------------------------------------------------------------------------
% Step 3. Validate the required Brainstorm folder
% -------------------------------------------------------------------------
assert_configured_input_dir(brainstorm_db_root, 'P.paths.brainstorm_db_root', config_file);

% -------------------------------------------------------------------------
% Step 4. Print a short run summary for the user
% -------------------------------------------------------------------------
fprintf('\nStage 1 / Step 12: Export and union-merge Brainstorm exclusions\n');
fprintf('  Brainstorm DB root: %s\n', brainstorm_db_root);
fprintf('  Export directory:   %s\n', bst_export_dir);
fprintf('  File filter:        %s\n', file_filter);
fprintf('  Labels kept:        BAD, boundary, bad_boundary\n');
fprintf('  Merge tolerance:    %.3f sec\n', merge_tolerance_sec);
fprintf('  Min union duration: %.3f sec\n\n', min_union_duration_sec);

% -------------------------------------------------------------------------
% Step 5. Export Brainstorm BAD / boundary labels to per-run TSV files
%
% The preserved exporter helper is intentionally used as-is here.
% -------------------------------------------------------------------------
batch_export_brainstorm_exclusion_events( ...
    brainstorm_db_root, ...
    bst_export_dir, ...
    'overwrite', overwrite_existing_outputs, ...
    'file_filter', file_filter, ...
    'recursive', recursive_bst_scan);

% -------------------------------------------------------------------------
% Step 6. Merge exported exclusions into one union interval list per run
% -------------------------------------------------------------------------
batch_merge_exclusion_union_masks( ...
    bst_export_dir, ...
    'adjacency_tol_sec', merge_tolerance_sec, ...
    'min_dur_sec', min_union_duration_sec, ...
    'labels', ["BAD","boundary","bad_boundary"], ...
    'overwrite', overwrite_existing_outputs);

% -------------------------------------------------------------------------
% Step 7. Point the user to the next stage
% -------------------------------------------------------------------------
fprintf('\nStage 1 / Step 12 complete.\n');
fprintf('Next scripted step: run 13_eeg_run_qc_and_table_s1.m.\n');

function assert_configured_input_dir(path_value, label, config_file)
path_char = char(path_value);
if contains(path_char, '<SET_')
    error('%s is still using a placeholder path. Edit %s first.', label, config_file);
end
if ~exist(path_char, 'dir')
    error('%s does not exist: %s', label, path_char);
end
end
