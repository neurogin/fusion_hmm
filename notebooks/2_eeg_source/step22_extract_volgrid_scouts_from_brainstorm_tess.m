% step22_extract_volgrid_scouts_from_brainstorm_tess
%
% What this file does:
%   Read the Brainstorm tess files that already contain the imported
%   Schaefer volume atlas, then write one standardized scout MAT per
%   subject/session in the same vertex-index space as the run-level
%   Brainstorm inverse kernels.
%
% When to run it:
%   Run this after the manual Brainstorm source-localization and atlas-
%   import steps are complete, and before parcel PC export.
%
% Manuscript linkage:
%   - Main Methods 2.2.2
%   - Supplementary Methods 1.2
%   - Supplementary Results 2.2
%   - Supplementary Table S2
%   - Supplementary Fig. S1A,B support
%
% Inputs expected:
%   - Brainstorm protocol root
%   - run-level results_MN_EEG_KERNEL_*.mat files
%   - subject/session tess files that already contain the imported atlas
%
% Outputs written:
%   - one standardized scout MAT per subject/session
%   - one scout-build summary CSV
%
% Manual dependency:
%   Brainstorm source localization and atlas import must already be
%   complete. See step21_brainstorm_volume_source_and_atlas_import_manual.md.

% -------------------------------------------------------------------------
% Step 0. Locate this stage folder and add the local helper path
% -------------------------------------------------------------------------
this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
if isempty(this_dir)
    error('Could not resolve the stage-2 script location. Run this file from disk.');
end

helper_dir = fullfile(this_dir, 'helpers');
addpath(this_dir);
addpath(helper_dir);

% -------------------------------------------------------------------------
% Step 1. User-editable roots and inputs
% -------------------------------------------------------------------------
project_root = '<SET_PROJECT_ROOT>'; % Main project folder that contains 02_derivatives and 04_qc for this repo run
brainstorm_protocol_root = '<SET_BRAINSTORM_PROTOCOL_ROOT>'; % Actual Brainstorm protocol folder that directly contains data\ and anat\

stage2_qc_root = fullfile(project_root, '04_qc', 'stage2_eeg_source', 'tables');

scout_filename = 'scout_Schaefer2018_200_7N_dilated_MNI.mat';
atlas_name_contains = 'atlas-Schaefer2018_desc-200Parcels7Networks';
kernel_pattern = 'results_MN_EEG_KERNEL_*.mat';

summary_csv = fullfile(stage2_qc_root, 'batch_volgrid_scout_build.csv');
overwrite_existing_summary = true;

% -------------------------------------------------------------------------
% Step 2. Validate inputs and create outputs
% -------------------------------------------------------------------------
assert_configured_input_dir(brainstorm_protocol_root, 'brainstorm_protocol_root');
assert_configured_input_dir(project_root, 'project_root');
ensure_parent_dir(summary_csv);

% -------------------------------------------------------------------------
% Step 3. Print a short run summary for the user
% -------------------------------------------------------------------------
fprintf('\nStage 2 / Step 22: Extract Brainstorm volume-grid scouts\n');
fprintf('  Project root:             %s\n', project_root);
fprintf('  Brainstorm protocol root: %s\n', brainstorm_protocol_root);
fprintf('  Scout filename:           %s\n', scout_filename);
fprintf('  Atlas name filter:        %s\n', atlas_name_contains);
fprintf('  Kernel pattern:           %s\n', kernel_pattern);
fprintf('  QC summary CSV:           %s\n\n', summary_csv);

% -------------------------------------------------------------------------
% Step 4. Run the preserved scout-extraction helper logic
% -------------------------------------------------------------------------
T = batch_extract_volgrid_scouts_from_brainstorm_tess( ...
    brainstorm_protocol_root, ...
    scout_filename, ...
    atlas_name_contains, ...
    kernel_pattern);

% -------------------------------------------------------------------------
% Step 5. Write a readable build summary for later QC
% -------------------------------------------------------------------------
if exist(summary_csv, 'file') && ~overwrite_existing_summary
    error('Summary CSV already exists and overwrite_existing_summary=false: %s', summary_csv);
end

writetable(T, summary_csv);

% -------------------------------------------------------------------------
% Step 6. Point the user to the next stage
% -------------------------------------------------------------------------
fprintf('Wrote scout-build summary: %s\n', summary_csv);
fprintf('Next public step: run step23_export_eeg_parcel_pc1_and_gain_normalize.m.\n');

function assert_configured_input_dir(path_value, label)
path_char = char(path_value);
if isempty(strtrim(path_char)) || contains(path_char, '<SET_')
    error('%s is still using a placeholder path. Edit this script before running.', label);
end
if ~exist(path_char, 'dir')
    error('%s does not exist: %s', label, path_char);
end
end

function ensure_parent_dir(file_path)
parent_dir = fileparts(file_path);
if ~isempty(parent_dir) && ~exist(parent_dir, 'dir')
    mkdir(parent_dir);
end
end
