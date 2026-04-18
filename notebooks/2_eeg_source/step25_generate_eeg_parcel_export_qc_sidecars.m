% step25_generate_eeg_parcel_export_qc_sidecars
%
% What this file does:
%   Run the preserved Stage-2 MATLAB QC-sidecar helper on the parcel-export
%   folder so the public Python notebook can build Table S3 and the
%   Supplementary Figures S2-S4 support outputs.
%
% When to run it:
%   Run this after Stage-2 Step 23 has finished writing the parcel MAT/NPY
%   outputs and after Step 24 if you are following the public step order.
%
% Manuscript linkage:
%   - Main Methods 2.2.3 support
%   - Supplementary Methods 1.3
%   - Supplementary Results 2.3
%   - Supplementary Table S3 support
%   - Supplementary Figures S2-S4 support
%
% Inputs expected:
%   - Stage-2 parcel export folder:
%       02_derivatives/stage2_eeg_source/parcel_exports
%   - parcel MAT outputs and batch summaries already written by Step 23
%
% Outputs written:
%   - qc_v3/qc_run_timeseries_gain_summary.csv
%   - qc_v3_sign/qc_sign_v3_summary.csv
%   - batch_pve1_run_quantiles_v3.csv
%   - batch_pve1_histogram_v3.csv
%   - batch_pve1_lowparcels_frequency_named_v3.csv
%
% Manual dependency:
%   None beyond the upstream manual/hybrid Brainstorm work already required
%   for Steps 22 and 23.

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

stage2_derivatives_root = fullfile(project_root, '02_derivatives', 'stage2_eeg_source');
parcel_output_dir = fullfile(stage2_derivatives_root, 'parcel_exports');

gain_qc_csv = fullfile(parcel_output_dir, 'qc_v3', 'qc_run_timeseries_gain_summary.csv');
sign_qc_csv = fullfile(parcel_output_dir, 'qc_v3_sign', 'qc_sign_v3_summary.csv');
pve_run_csv = fullfile(parcel_output_dir, 'batch_pve1_run_quantiles_v3.csv');
pve_hist_csv = fullfile(parcel_output_dir, 'batch_pve1_histogram_v3.csv');
pve_lowfreq_csv = fullfile(parcel_output_dir, 'batch_pve1_lowparcels_frequency_named_v3.csv');

% -------------------------------------------------------------------------
% Step 2. Validate inputs and helper availability
% -------------------------------------------------------------------------
assert_configured_input_dir(project_root, 'project_root');
assert_configured_input_dir(parcel_output_dir, 'parcel_output_dir');
assert_dependency_exists(fullfile(helper_dir, 'run_eeg_parcel_export_qc_summaries.m'), ...
    ['Missing Stage-2 QC-sidecar helper:' newline ...
     '  notebooks/2_eeg_source/helpers/run_eeg_parcel_export_qc_summaries.m']);

% -------------------------------------------------------------------------
% Step 3. Print a short run summary for the user
% -------------------------------------------------------------------------
fprintf('\nStage 2 / Step 25: Generate EEG parcel-export QC sidecars\n');
fprintf('  Project root:             %s\n', project_root);
fprintf('  Stage-2 output root:      %s\n', stage2_derivatives_root);
fprintf('  Parcel output dir:        %s\n', parcel_output_dir);
fprintf('  Gain QC summary:          %s\n', gain_qc_csv);
fprintf('  Sign QC summary:          %s\n', sign_qc_csv);
fprintf('  PVE run quantiles:        %s\n', pve_run_csv);
fprintf('  PVE histogram:            %s\n', pve_hist_csv);
fprintf('  Low-PVE parcel frequency: %s\n\n', pve_lowfreq_csv);

% -------------------------------------------------------------------------
% Step 4. Run the preserved helper logic
% -------------------------------------------------------------------------
run_eeg_parcel_export_qc_summaries(parcel_output_dir);

% -------------------------------------------------------------------------
% Step 5. Point the user to the next public step
% -------------------------------------------------------------------------
fprintf('\nStage 2 / Step 25 complete.\n');
fprintf('Next public step: open step26_qc_eeg_parcel_exports_table_s3_and_figures_s2_s4.ipynb.\n');

function assert_configured_input_dir(path_value, label)
path_char = char(path_value);
if isempty(strtrim(path_char)) || contains(path_char, '<SET_')
    error('%s is still using a placeholder path. Edit this script before running.', label);
end
if ~exist(path_char, 'dir')
    error('%s does not exist: %s', label, path_char);
end
end

function assert_dependency_exists(path_to_file, message_text)
if exist(path_to_file, 'file') ~= 2
    error('%s', message_text);
end
end
