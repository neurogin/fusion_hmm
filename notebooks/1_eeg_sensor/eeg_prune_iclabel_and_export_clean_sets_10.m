% eeg_prune_iclabel_and_export_clean_sets_10
%
% What this file does:
%   Run the manuscript-default ICLabel pruning step on author-preprocessed
%   EEGLAB datasets, then write:
%     1. auditable "withICA" outputs that keep ICA metadata
%     2. Brainstorm-facing "clean" outputs with ICA fields cleared
%     3. run-level ICLabel QC tables
%
% Before you run this file:
%   First open `helpers/stage1_eeg_sensor_settings.m` and fill in the two
%   placeholder root paths there. Step 10 will not run correctly until that
%   settings file points to your raw EEGLAB folder and Brainstorm protocol.
%
% Manuscript linkage:
%   - Main Methods 2.2.1
%   - Supplementary Methods 1.1
%
% Manual dependency:
%   The outputs of this file feed the later manual Brainstorm marking step
%   documented in brainstorm_exclusion_marking_manual_11.md.
%
% Preserved implementation note:
%   This file is a readable public entry point. It now calls a descriptive
%   public wrapper helper, while the preserved low-level implementation
%   remains available underneath for provenance compatibility.

% -------------------------------------------------------------------------
% Step 0. Locate this stage folder and add the stage-1 helper path
% -------------------------------------------------------------------------
this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
if isempty(this_dir)
    error('Could not resolve the stage-1 script location. Run this file from disk.');
end

helper_dir = fullfile(this_dir, 'helpers');
config_file = fullfile(helper_dir, 'stage1_eeg_sensor_settings.m');

addpath(this_dir);
addpath(helper_dir);

% -------------------------------------------------------------------------
% Step 1. Load the stage-1 configuration
%
% First-run setup:
%   Open the settings file below and fill it out before running this script:
%     helpers/stage1_eeg_sensor_settings.m
%
% The manuscript-default ICLabel policy is stored there as:
%   - ic_policy = "reject_artifacts"
%   - reject_threshold = 0.70
% -------------------------------------------------------------------------
P = stage1_eeg_sensor_settings();

raw_eeglab_dir = char(P.paths.raw_eeglab_dir);
ic_pruned_dir = char(P.paths.ic_pruned_dir);
with_ica_dir = char(P.paths.with_ica_dir);
clean_sets_dir = char(P.paths.clean_sets_dir);
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
% Step 3. Validate the required runtime dependencies and folders
% -------------------------------------------------------------------------
assert_required_function('pop_loadset', ...
    'Stage-1 Step 10 needs EEGLAB on the MATLAB path for batch .set loading and saving.');
assert_required_function('iclabel', ...
    'Stage-1 Step 10 needs the ICLabel plugin on the MATLAB path for batch component pruning.');

assert_configured_input_dir(raw_eeglab_dir, 'P.paths.raw_eeglab_dir', config_file);
ensure_dir(ic_pruned_dir);
ensure_dir(with_ica_dir);
ensure_dir(clean_sets_dir);
ensure_dir(qc_tables_dir);

% -------------------------------------------------------------------------
% Step 4. Print a short run summary for the user
% -------------------------------------------------------------------------
fprintf('\nStage 1 / Step 10: ICLabel pruning and clean-set export\n');
fprintf('  Settings file:        %s\n', config_file);
fprintf('  Raw EEGLAB input: %s\n', raw_eeglab_dir);
fprintf('  Stage-1 output root: %s\n', ic_pruned_dir);
fprintf('  with_ica dir:        %s\n', with_ica_dir);
fprintf('  clean_sets dir:      %s\n', clean_sets_dir);
fprintf('  QC tables dir:       %s\n', qc_tables_dir);
fprintf('  IC policy:           %s\n', ic_policy);
fprintf('  Reject threshold:    %.2f\n', reject_threshold);
fprintf('  Reject classes:      %s\n', strjoin(cellstr(reject_classes), ', '));
fprintf(['  Dependency note: batch execution needs EEGLAB + ICLabel on the MATLAB path.' newline ...
         '  GUI inspection of the resulting .set files is a separate manual step.' newline newline]);

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
    'file_pattern', file_pattern, ...
    'withICA_subdir', 'with_ica', ...
    'clean_subdir', 'clean_sets');

% -------------------------------------------------------------------------
% Step 6. Point the user to the next stage
% -------------------------------------------------------------------------
fprintf('\nStage 1 / Step 10 complete.\n');
fprintf('Next manual step: review brainstorm_exclusion_marking_manual_11.md before Brainstorm marking.\n');

function assert_configured_input_dir(path_value, label, config_file)
path_char = char(path_value);
if contains(path_char, '<SET_')
    error('%s is still using a placeholder path. Edit %s first.', label, config_file);
end
if ~exist(path_char, 'dir')
    error('%s does not exist: %s', label, path_char);
end
end

function assert_required_function(func_name, message_text)
if exist(func_name, 'file') ~= 2
    error('%s', message_text);
end
end

function ensure_dir(path_value)
if ~exist(path_value, 'dir')
    mkdir(path_value);
end
end
