% 10_eeg_prune_iclabel_and_export_clean_sets
%
% What this file does:
%   Run the manuscript-default ICLabel pruning step on author-preprocessed
%   EEGLAB datasets, then write:
%     1. auditable "withICA" outputs that keep ICA metadata
%     2. Brainstorm-facing "clean" outputs with ICA fields cleared
%     3. run-level ICLabel QC tables
%
% Manuscript linkage:
%   - Main Methods 2.2.1
%   - Supplementary Methods 1.1
%
% Manual dependency:
%   The outputs of this file feed the later manual Brainstorm marking step
%   documented in 11_brainstorm_exclusion_marking_manual.md.
%
% Preserved implementation note:
%   This file is a readable public entry point. It now calls a descriptive
%   public wrapper helper, while the preserved low-level implementation
%   remains available underneath for provenance compatibility.

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
% The manuscript-default ICLabel policy is stored there as:
%   - ic_policy = "reject_artifacts"
%   - reject_threshold = 0.70
% -------------------------------------------------------------------------
P = stage1_eeg_sensor_settings();

raw_eeglab_dir = char(P.paths.raw_eeglab_dir);
ic_pruned_dir = char(P.paths.ic_pruned_dir);
qc_tables_dir = char(P.paths.qc_tables_dir);
ic_policy = char(P.ic_policy);
reject_threshold = P.ic_reject_threshold;
reject_classes = P.ic_reject_classes;

% -------------------------------------------------------------------------
% Step 2. User-editable run options
%
% These settings only control how this public entry point runs the
% preserved helper function. They do not change the underlying helper logic.
% -------------------------------------------------------------------------
overwrite_outputs = false;
save_iclabel_tables = true;
save_withICA = true;
save_clean = true;
clean_remove_iclabel = true;
file_pattern = '*.set';

% -------------------------------------------------------------------------
% Step 3. Validate the required input folder
% -------------------------------------------------------------------------
assert_configured_input_dir(raw_eeglab_dir, 'P.paths.raw_eeglab_dir', config_file);

% -------------------------------------------------------------------------
% Step 4. Print a short run summary for the user
% -------------------------------------------------------------------------
fprintf('\nStage 1 / Step 10: ICLabel pruning and clean-set export\n');
fprintf('  Raw EEGLAB input: %s\n', raw_eeglab_dir);
fprintf('  Output directory: %s\n', ic_pruned_dir);
fprintf('  QC tables dir:    %s\n', qc_tables_dir);
fprintf('  IC policy:        %s\n', ic_policy);
fprintf('  Reject threshold: %.2f\n', reject_threshold);
fprintf('  Reject classes:   %s\n\n', strjoin(cellstr(reject_classes), ', '));

% -------------------------------------------------------------------------
% Step 5. Run the preserved pruning/export implementation
% -------------------------------------------------------------------------
run_iclabel_pruning_and_metadata_export( ...
    raw_eeglab_dir, ...
    ic_pruned_dir, ...
    qc_tables_dir, ...
    'ic_policy', ic_policy, ...
    'reject_threshold', reject_threshold, ...
    'reject_classes', reject_classes, ...
    'overwrite', overwrite_outputs, ...
    'save_iclabel_tables', save_iclabel_tables, ...
    'save_withICA', save_withICA, ...
    'save_clean', save_clean, ...
    'clean_remove_iclabel', clean_remove_iclabel, ...
    'file_pattern', file_pattern);

% -------------------------------------------------------------------------
% Step 6. Point the user to the next stage
% -------------------------------------------------------------------------
fprintf('\nStage 1 / Step 10 complete.\n');
fprintf('Next manual step: review 11_brainstorm_exclusion_marking_manual.md before Brainstorm marking.\n');

function assert_configured_input_dir(path_value, label, config_file)
path_char = char(path_value);
if contains(path_char, '<SET_')
    error('%s is still using a placeholder path. Edit %s first.', label, config_file);
end
if ~exist(path_char, 'dir')
    error('%s does not exist: %s', label, path_char);
end
end
