% 23_export_eeg_parcel_pc1_and_gain_normalize
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
%   - stage-1 cleaned EEGLAB .set files
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
%   This public entry script wraps the existing v3 exporter helpers. It
%   does not change the parcel support threshold, sign convention, gain
%   normalization basis, or QC metadata logic. The restored sample-time
%   sidecar is written as single((0:nTime-1)' / srate) beside the exported
%   *_PC1_gnorm.npy files when WriteNPY=true.

% -------------------------------------------------------------------------
% Step 0. Locate this stage folder and add it to the MATLAB path
% -------------------------------------------------------------------------
stage2_dir = fileparts(mfilename('fullpath'));
if isempty(stage2_dir)
    error('Could not resolve the stage-2 script location. Run this file from disk.');
end
addpath(stage2_dir);

% -------------------------------------------------------------------------
% Step 1. User-editable inputs
% -------------------------------------------------------------------------
protocol_root = '<SET_BRAINSTORM_PROTOCOL_ROOT>';
clean_eeg_dir = '<SET_STAGE1_CLEAN_EEG_DIR>';
parcel_output_dir = '<SET_STAGE2_PARCEL_OUTPUT_DIR>';

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
npy_output_dir = '';
scale_chunk_rows = 20000;

overwrite_existing_outputs = true;
verbose_output = true;
continue_on_fail = false;

% -------------------------------------------------------------------------
% Step 2. Validate the required input folders
% -------------------------------------------------------------------------
assert_configured_input_dir(protocol_root, 'protocol_root');
assert_configured_input_dir(clean_eeg_dir, 'clean_eeg_dir');
assert_configured_output_dir(parcel_output_dir, 'parcel_output_dir');

% writeNPY remains an external dependency. The exporter helper still runs
% without it, but the downstream alignment-ready .npy outputs will be
% incomplete if it is missing.
if write_npy && exist('writeNPY', 'file') ~= 2
    warning(['writeNPY is not on the MATLAB path. The preserved helper will skip .npy output, ' ...
        'which means downstream alignment-ready sidecars will not be created.']);
end

% -------------------------------------------------------------------------
% Step 3. Print a short run summary for the user
% -------------------------------------------------------------------------
fprintf('\nStage 2 / Step 23: Export EEG parcel PC1 and gain-normalized outputs\n');
fprintf('  Brainstorm protocol root: %s\n', protocol_root);
fprintf('  Stage-1 clean EEG dir:    %s\n', clean_eeg_dir);
fprintf('  Parcel output dir:        %s\n', parcel_output_dir);
fprintf('  Desc tag:                 %s\n', desc_tag);
fprintf('  Scout filename:           %s\n', scout_filename);
fprintf('  Expected scouts:          %d\n', expected_n_scouts);
fprintf('  Min vertices:             %d\n', min_vertices);
fprintf('  Sign convention:          %s\n', sign_convention);
fprintf('  Gain normalization:       %d\n', gain_norm);
fprintf('  Write NPY sidecars:       %d\n\n', write_npy);

% -------------------------------------------------------------------------
% Step 4. Run the preserved exporter helper logic
% -------------------------------------------------------------------------
batch = r01_batch_export_eeg_parcel_pc_v3( ...
    protocol_root, ...
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
fprintf('Next scripted step: run 24_qc_eeg_source_alignment_table_s2.m.\n');

function assert_configured_input_dir(path_value, label)
path_char = char(path_value);
if isempty(strtrim(path_char)) || contains(path_char, '<SET_')
    error('%s is still using a placeholder path. Edit this script before running.', label);
end
if ~exist(path_char, 'dir')
    error('%s does not exist: %s', label, path_char);
end
end

function assert_configured_output_dir(path_value, label)
path_char = char(path_value);
if isempty(strtrim(path_char)) || contains(path_char, '<SET_')
    error('%s is still using a placeholder path. Edit this script before running.', label);
end
if ~exist(path_char, 'dir')
    mkdir(path_char);
end
end
