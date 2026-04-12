% 13_eeg_run_qc_and_table_s1
%
% What this file does:
%   1. summarize merged exclusion windows at the run level
%   2. compute run-level EEG QC metrics and include/exclude manifests
%   3. write the stage-1 outputs that support Supplementary Table S1
%
% Manuscript linkage:
%   - Main Methods 2.2.1
%   - Supplementary Methods 1.1
%   - Supplementary Results 2.1
%   - Supplementary Table S1
%
% Important wording caveat preserved intentionally:
%   The helper logic still includes an explicit max_emg_db threshold even
%   though the manuscript describes the EMG proxy as descriptive rather than
%   a stand-alone exclusion threshold. This file makes that tension visible
%   and does not silently harmonize it.

% -------------------------------------------------------------------------
% Step 0. Locate this stage folder and add the stage-1 helper path
% -------------------------------------------------------------------------
stage1_dir = fileparts(mfilename('fullpath'));
if isempty(stage1_dir)
    error('Could not resolve the stage-1 script location. Run this file from disk.');
end

helper_dir = fullfile(stage1_dir, 'helpers');
config_file = fullfile(helper_dir, 'r01_stage1_params.m');

addpath(stage1_dir);
addpath(helper_dir);

% -------------------------------------------------------------------------
% Step 1. Load the stage-1 configuration
%
% Edit path placeholders in helpers/r01_stage1_params.m before running.
% -------------------------------------------------------------------------
P = r01_stage1_params();

raw_eeglab_dir = char(P.paths.raw_eeglab_dir);
ic_pruned_dir = char(P.paths.ic_pruned_dir);
bst_export_dir = char(P.paths.bst_export_dir);
qc_tables_dir = char(P.paths.qc_tables_dir);
qc_exclusions_dir = char(P.paths.qc_exclusions_dir);
brainstorm_db_root = char(P.paths.brainstorm_db_root);
iclabel_tag = char(P.iclabel_tag);

min_usable_frac = P.qc.run.min_usable_frac;
max_emg_db = P.qc.run.max_emg_db;
max_badchan_abs = P.qc.run.max_badchan_abs;
max_badchan_frac = P.qc.run.max_badchan_frac;
allow_unknown_usable = P.qc.run.allow_unknown_usable;
load_raw_duration = P.qc.run.load_raw_duration;
hf_band = P.qc.run.hf_band;
lf_band = P.qc.run.lf_band;

merge_tolerance_sec = P.mask.merge.adjacency_tol_sec;
max_excl_frac_warn = P.qc.excl.max_excl_frac_warn;
max_interval_sec_warn = P.qc.excl.max_interval_sec_warn;
min_interval_sec_warn = P.qc.excl.min_interval_sec_warn;

% -------------------------------------------------------------------------
% Step 2. User-editable run options
% -------------------------------------------------------------------------
overwrite_existing_outputs = true;
estimate_run_duration_from_brainstorm = true;

% -------------------------------------------------------------------------
% Step 3. Validate the required input folders
% -------------------------------------------------------------------------
assert_configured_input_dir(raw_eeglab_dir, 'P.paths.raw_eeglab_dir', config_file);
assert_configured_input_dir(ic_pruned_dir, 'P.paths.ic_pruned_dir', config_file);
assert_configured_input_dir(bst_export_dir, 'P.paths.bst_export_dir', config_file);

bst_db_root_for_qc = '';
if estimate_run_duration_from_brainstorm
    assert_configured_input_dir(brainstorm_db_root, 'P.paths.brainstorm_db_root', config_file);
    bst_db_root_for_qc = brainstorm_db_root;
end

% -------------------------------------------------------------------------
% Step 4. Print a short run summary for the user
% -------------------------------------------------------------------------
fprintf('\nStage 1 / Step 13: Run-level EEG QC and Table S1 support exports\n');
fprintf('  Union TSV directory:   %s\n', bst_export_dir);
fprintf('  Exclusion QC dir:      %s\n', qc_exclusions_dir);
fprintf('  Run-level QC tables:   %s\n', qc_tables_dir);
fprintf('  Tag:                   %s\n', iclabel_tag);
fprintf('  Min usable fraction:   %.2f\n', min_usable_frac);
fprintf('  Max EMG proxy (dB):    %.1f\n', max_emg_db);
fprintf('  Max bad channels abs:  %d\n', max_badchan_abs);
fprintf('  Max bad channels frac: %.2f\n\n', max_badchan_frac);

% -------------------------------------------------------------------------
% Step 5. Summarize merged exclusion windows and warning flags
%
% This writes excl_union_qc_summary.csv, which the run-level QC helper uses
% as its preferred source for usable-fraction and exclusion-flag inputs.
% -------------------------------------------------------------------------
r01_qc_excl_union_folder( ...
    bst_export_dir, ...
    qc_exclusions_dir, ...
    'bst_db_root', bst_db_root_for_qc, ...
    'adjacency_tol_sec', merge_tolerance_sec, ...
    'max_excl_frac_warn', max_excl_frac_warn, ...
    'max_interval_sec_warn', max_interval_sec_warn, ...
    'min_interval_sec_warn', min_interval_sec_warn);

% -------------------------------------------------------------------------
% Step 6. Compute run-level EEG QC tables and include/exclude manifests
%
% The preserved helper keeps the current QC gate behavior unchanged,
% including the explicit max_emg_db threshold noted above.
% -------------------------------------------------------------------------
r01_eeg_runlevel_qc_gates( ...
    raw_eeglab_dir, ...
    ic_pruned_dir, ...
    qc_tables_dir, ...
    qc_exclusions_dir, ...
    'tag', iclabel_tag, ...
    'min_usable_frac', min_usable_frac, ...
    'max_emg_db', max_emg_db, ...
    'max_badchan_abs', max_badchan_abs, ...
    'max_badchan_frac', max_badchan_frac, ...
    'allow_unknown_usable', allow_unknown_usable, ...
    'overwrite', overwrite_existing_outputs, ...
    'load_raw_duration', load_raw_duration, ...
    'hf_band', hf_band, ...
    'lf_band', lf_band);

% -------------------------------------------------------------------------
% Step 7. Remind the user what to review next
% -------------------------------------------------------------------------
fprintf('\nStage 1 / Step 13 complete.\n');
fprintf('Review the QC CSVs and manifests before using them downstream.\n');

function assert_configured_input_dir(path_value, label, config_file)
path_char = char(path_value);
if contains(path_char, '<SET_')
    error('%s is still using a placeholder path. Edit %s first.', label, config_file);
end
if ~exist(path_char, 'dir')
    error('%s does not exist: %s', label, path_char);
end
end
