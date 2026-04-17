% eeg_run_qc_and_table_s1_13
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
%
% Before you run this file:
%   1. Make sure `helpers/stage1_eeg_sensor_settings.m` is filled out.
%   2. Make sure step 10 and step 12 have already finished successfully.
%   3. This script reads the clean EEG sets and merged exclusion TSV files
%      from those earlier Stage-1 outputs, then writes the QC tables and
%      manifests used for Supplementary Table S1 support.

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
% Fill out the settings file below before running this script:
%   helpers/stage1_eeg_sensor_settings.m
% -------------------------------------------------------------------------
P = stage1_eeg_sensor_settings();

raw_eeglab_dir = char(P.paths.raw_eeglab_dir);
ic_pruned_dir = char(P.paths.ic_pruned_dir);
with_ica_dir = char(P.paths.with_ica_dir);
clean_sets_dir = char(P.paths.clean_sets_dir);
union_mask_dir = char(P.paths.union_mask_dir);
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
% Step 3. Validate the required runtime dependencies and folders
% -------------------------------------------------------------------------
assert_required_function('pop_loadset', ...
    'Stage-1 Step 13 needs EEGLAB on the MATLAB path for batch loading of cleaned .set files.');

assert_configured_input_dir(raw_eeglab_dir, 'P.paths.raw_eeglab_dir', config_file);
assert_configured_input_dir(ic_pruned_dir, 'P.paths.ic_pruned_dir', config_file);
assert_configured_input_dir(union_mask_dir, 'P.paths.union_mask_dir', config_file);
ensure_dir(qc_tables_dir);
ensure_dir(qc_exclusions_dir);

bst_db_root_for_qc = '';
if estimate_run_duration_from_brainstorm
    assert_configured_input_dir(brainstorm_db_root, 'P.paths.brainstorm_db_root', config_file);
    bst_db_root_for_qc = brainstorm_db_root;
end

% -------------------------------------------------------------------------
% Step 4. Print a short run summary for the user
% -------------------------------------------------------------------------
fprintf('\nStage 1 / Step 13: Run-level EEG QC and Table S1 support exports\n');
fprintf('  Settings file:        %s\n', config_file);
fprintf('  Raw EEGLAB input:      %s\n', raw_eeglab_dir);
fprintf('  with_ica dir:          %s\n', with_ica_dir);
fprintf('  clean_sets dir:        %s\n', clean_sets_dir);
fprintf('  Union TSV directory:   %s\n', union_mask_dir);
fprintf('  Exclusion QC dir:      %s\n', qc_exclusions_dir);
fprintf('  Run-level QC tables:   %s\n', qc_tables_dir);
fprintf('  Tag:                   %s\n', iclabel_tag);
fprintf('  Min usable fraction:   %.2f\n', min_usable_frac);
fprintf('  Max EMG proxy (dB):    %.1f\n', max_emg_db);
fprintf('  Max bad channels abs:  %d\n', max_badchan_abs);
fprintf('  Max bad channels frac: %.2f\n', max_badchan_frac);
fprintf(['  Dependency note: batch execution needs EEGLAB on the MATLAB path to' newline ...
         '  read the cleaned Stage-1 .set files. GUI review of those files is separate.' newline ...
         '  This script writes run-level QC tables and include/exclude manifests.' newline newline]);

% -------------------------------------------------------------------------
% Step 5. Summarize merged exclusion windows and warning flags
% -------------------------------------------------------------------------
summarize_exclusion_union_qc( ...
    union_mask_dir, ...
    qc_exclusions_dir, ...
    'bst_db_root', bst_db_root_for_qc, ...
    'adjacency_tol_sec', merge_tolerance_sec, ...
    'max_excl_frac_warn', max_excl_frac_warn, ...
    'max_interval_sec_warn', max_interval_sec_warn, ...
    'min_interval_sec_warn', min_interval_sec_warn);

% -------------------------------------------------------------------------
% Step 6. Compute run-level EEG QC tables and include/exclude manifests
% -------------------------------------------------------------------------
build_eeg_run_qc_gates_and_manifests( ...
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
