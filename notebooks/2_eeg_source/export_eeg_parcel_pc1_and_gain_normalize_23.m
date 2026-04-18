% export_eeg_parcel_pc1_and_gain_normalize_23
%
% What this file does:
%   Export run-wise EEG parcel PCs from Brainstorm volumetric inverse
%   kernels and Brainstorm-defined volume-grid scouts, then apply the
%   manuscript-preserved gain normalization and write batch summaries.
%
% When to run it:
%   Run this after the manual Brainstorm Stage-2 work is complete and after
%   the standardized scout files have already been written by step 22.
%
% Manuscript linkage:
%   - Main Methods 2.2.3
%   - Supplementary Methods 1.3
%   - Supplementary Results 2.3
%   - Supplementary Table S3 support
%   - Supplementary Figs. S2-S4 support
%
% Inputs expected:
%   - Brainstorm protocol root with kernel files
%   - Stage-1 cleaned EEGLAB .set files
%   - subject/session scout MAT files from step 22
%
% Outputs written:
%   - *_parcelPC_raw.mat
%   - *_parcelPC_gnorm.mat
%   - batch_parcel_gain_summary_v3.csv
%   - batch_parcel_coverage_summary_v3.csv
%   - batch_parcel_manifest_v3.csv
%   - .npy sidecars when WriteNPY=true, including *_time_sec.npy
%
% Manual dependency:
%   Brainstorm source localization and atlas import must already be
%   complete, and the standardized scout files must already exist.
%
% Preserved implementation note:
%   This public entry script now calls a descriptive public wrapper helper,
%   while the preserved v3 exporter implementation remains available
%   underneath for provenance compatibility. It does not change the parcel
%   support threshold, sign convention, gain normalization basis, or QC
%   metadata logic. The restored sample-time sidecar is written as
%   single((0:nTime-1)' / srate) beside the exported *_PC1_gnorm.npy files
%   when WriteNPY=true.

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
project_root = '<SET_PROJECT_ROOT>'; % Main project folder that contains the Stage-1 outputs and will receive Stage-2 derivatives and QC tables
brainstorm_protocol_root = '<SET_BRAINSTORM_PROTOCOL_ROOT>'; % Actual Brainstorm protocol folder that directly contains data\ and anat\

clean_eeg_dir = fullfile(project_root, '02_derivatives', 'stage1_eeg_sensor', 'ic_pruned', 'clean_sets');
stage2_derivatives_root = fullfile(project_root, '02_derivatives', 'stage2_eeg_source');
parcel_output_dir = fullfile(stage2_derivatives_root, 'parcel_exports');
npy_output_dir = fullfile(parcel_output_dir, 'npy');

desc_tag = 'desc-ICRej70';
kernel_pattern = 'results_MN_EEG_KERNEL_*.mat';
scout_filename = 'scout_Schaefer2018_200_7N_dilated_MNI.mat';

expected_n_scouts = 200;
strict_tess_match = true;

min_vertices = 40;
num_pc = 2;
save_pc2 = true;
sign_convention = 'maxabs';
block_size = 0;
gain_norm = true;
gain_cap = [0, Inf];

write_npy = true;
scale_chunk_rows = 20000;

overwrite_existing_outputs = true;
verbose_output = true;
continue_on_fail = false;

% -------------------------------------------------------------------------
% Step 2. Validate dependencies, inputs, and outputs
% -------------------------------------------------------------------------
ensure_eeglab_ready('stage_label', 'Stage-2 Step 23');

assert_configured_input_dir(brainstorm_protocol_root, 'brainstorm_protocol_root');
assert_configured_input_dir(project_root, 'project_root');
assert_configured_input_dir(clean_eeg_dir, 'clean_eeg_dir');
ensure_dir(parcel_output_dir);
if write_npy
    ensure_dir(npy_output_dir);
end

if write_npy && exist('writeNPY', 'file') ~= 2
    warning(['writeNPY is not on the MATLAB path. The preserved helper can still run,' newline ...
        'but downstream alignment-ready .npy sidecars, including *_time_sec.npy,' newline ...
        'will not be created.']);
end

% -------------------------------------------------------------------------
% Step 3. Print a short run summary for the user
% -------------------------------------------------------------------------
fprintf('\nStage 2 / Step 23: Export EEG parcel PC1 and gain-normalized outputs\n');
fprintf('  Project root:             %s\n', project_root);
fprintf('  Brainstorm protocol root: %s\n', brainstorm_protocol_root);
fprintf('  Stage-1 clean EEG dir:    %s\n', clean_eeg_dir);
fprintf('  Stage-2 output root:      %s\n', stage2_derivatives_root);
fprintf('  Parcel output dir:        %s\n', parcel_output_dir);
fprintf('  NPY output dir:           %s\n', npy_output_dir);
fprintf('  Desc tag:                 %s\n', desc_tag);
fprintf('  Scout filename:           %s\n', scout_filename);
fprintf('  Expected scouts:          %d\n', expected_n_scouts);
fprintf('  Min vertices:             %d\n', min_vertices);
fprintf('  Sign convention:          %s\n', sign_convention);
fprintf('  Gain normalization:       %d\n', gain_norm);
fprintf('  Write NPY sidecars:       %d\n', write_npy);
fprintf(['  Dependency note: batch execution needs EEGLAB on the MATLAB path.' newline ...
         '  This script initializes EEGLAB in no-GUI mode for batch use.' newline ...
         '  writeNPY is additionally needed if you want the downstream-ready NPY' newline ...
         '  sidecars, including *_time_sec.npy, from this public script.' newline newline]);

% -------------------------------------------------------------------------
% Step 4. Run the preserved exporter helper logic
% -------------------------------------------------------------------------
batch = batch_export_eeg_parcel_pc_outputs( ...
    brainstorm_protocol_root, ...
    parcel_output_dir, ...
    'EEGCleanDir', clean_eeg_dir, ...
    'KernelPattern', kernel_pattern, ...
    'DescTag', desc_tag, ...
    'ScoutFilename', scout_filename, ...
    'ExpectedNScouts', expected_n_scouts, ...
    'StrictTessMatch', strict_tess_match, ...
    'MinVertices', min_vertices, ...
    'NumPC', num_pc, ...
    'SavePC2', save_pc2, ...
    'SignConvention', sign_convention, ...
    'BlockSize', block_size, ...
    'Overwrite', overwrite_existing_outputs, ...
    'Verbose', verbose_output, ...
    'GainNorm', gain_norm, ...
    'GainCap', gain_cap, ...
    'WriteNPY', write_npy, ...
    'NPYDir', npy_output_dir, ...
    'ScaleChunkRows', scale_chunk_rows, ...
    'ContinueOnFail', continue_on_fail);

% -------------------------------------------------------------------------
% Step 5. Point the user to the next QC steps
% -------------------------------------------------------------------------
fprintf('\nStage 2 / Step 23 complete.\n');
fprintf('  Gain summary:     %s\n', batch.gainCsv);
fprintf('  Coverage summary: %s\n', batch.coverageCsv);
fprintf('  Manifest:         %s\n', batch.manifestCsv);
fprintf(['If WriteNPY succeeded, each run now also has a sample-time sidecar named ' ...
    '*_time_sec.npy beside *_PC1_gnorm.npy.\n']);
fprintf('Next scripted step: run qc_eeg_source_alignment_table_s2_24.m.\n');

function assert_configured_input_dir(path_value, label)
path_char = char(path_value);
if isempty(strtrim(path_char)) || contains(path_char, '<SET_')
    error('%s is still using a placeholder path. Edit this script before running.', label);
end
if ~exist(path_char, 'dir')
    error('%s does not exist: %s', label, path_char);
end
end

function ensure_dir(path_value)
if ~exist(path_value, 'dir')
    mkdir(path_value);
end
end
