% 13_eeg_run_qc_and_table_s1
%
% Public-facing stage-1 entry point for run-level EEG exclusion QC and the
% manuscript-facing run summary that supports Supplementary Results 2.1 and
% Supplementary Table S1.
%
% Manuscript linkage:
%   - Main Methods 2.2.1
%   - Supplementary Methods 1.1
%   - Supplementary Results 2.1
%   - Supplementary Table S1
%
% Inputs:
%   - Cleaned EEG outputs from step 10
%   - Merged exclusion TSVs from step 12
%   - Raw EEGLAB inputs for duration fallback
%
% Outputs:
%   - <qc_exclusions_dir>\excl_union_qc_summary.csv
%   - <qc_tables_dir>\eeg_run_qc_gates_<TAG>.csv
%   - <qc_tables_dir>\include_manifest.csv
%   - <qc_tables_dir>\include_manifest_<TAG>.csv
%   - <qc_tables_dir>\exclude_manifest.csv
%   - <qc_tables_dir>\exclude_manifest_<TAG>.csv
%   - <qc_tables_dir>\exclude_stems_<TAG>.txt
%
% Important note:
%   - This public stage preserves the current run-level QC gate behavior,
%     including the explicit max_emg_db threshold in
%     r01_eeg_runlevel_qc_gates.m.
%   - The manuscript text describes the EMG proxy as descriptive rather
%     than a stand-alone exclusion threshold. That wording/code tension is
%     preserved and made explicit here; it is not silently harmonized.

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
overwrite_existing_outputs = true;
estimate_run_duration_from_brainstorm = true;

assert_configured_input_dir(P.paths.raw_eeglab_dir, 'P.paths.raw_eeglab_dir');
assert_configured_input_dir(P.paths.ic_pruned_dir, 'P.paths.ic_pruned_dir');
assert_configured_input_dir(P.paths.bst_export_dir, 'P.paths.bst_export_dir');

bst_db_root_for_qc = '';
if estimate_run_duration_from_brainstorm
    assert_configured_input_dir(P.paths.brainstorm_db_root, 'P.paths.brainstorm_db_root');
    bst_db_root_for_qc = char(P.paths.brainstorm_db_root);
end

fprintf('\nStage 1 / Step 13: Run-level EEG QC and Table S1 support exports\n');
fprintf('  Union TSV directory:   %s\n', char(P.paths.bst_export_dir));
fprintf('  Exclusion QC dir:      %s\n', char(P.paths.qc_exclusions_dir));
fprintf('  Run-level QC tables:   %s\n', char(P.paths.qc_tables_dir));
fprintf('  Tag:                   %s\n', char(P.iclabel_tag));
fprintf('  Min usable fraction:   %.2f\n', P.qc.run.min_usable_frac);
fprintf('  Max EMG proxy (dB):    %.1f\n', P.qc.run.max_emg_db);
fprintf('  Max bad channels abs:  %d\n', P.qc.run.max_badchan_abs);
fprintf('  Max bad channels frac: %.2f\n\n', P.qc.run.max_badchan_frac);

% Step 13A. Summarize merged exclusion windows and warning flags.
r01_qc_excl_union_folder( ...
    char(P.paths.bst_export_dir), ...
    char(P.paths.qc_exclusions_dir), ...
    'bst_db_root', bst_db_root_for_qc, ...
    'adjacency_tol_sec', P.mask.merge.adjacency_tol_sec, ...
    'max_excl_frac_warn', P.qc.excl.max_excl_frac_warn, ...
    'max_interval_sec_warn', P.qc.excl.max_interval_sec_warn, ...
    'min_interval_sec_warn', P.qc.excl.min_interval_sec_warn);

% Step 13B. Compute run-level gates and include/exclude manifests.
r01_eeg_runlevel_qc_gates( ...
    char(P.paths.raw_eeglab_dir), ...
    char(P.paths.ic_pruned_dir), ...
    char(P.paths.qc_tables_dir), ...
    char(P.paths.qc_exclusions_dir), ...
    'tag', char(P.iclabel_tag), ...
    'min_usable_frac', P.qc.run.min_usable_frac, ...
    'max_emg_db', P.qc.run.max_emg_db, ...
    'max_badchan_abs', P.qc.run.max_badchan_abs, ...
    'max_badchan_frac', P.qc.run.max_badchan_frac, ...
    'allow_unknown_usable', P.qc.run.allow_unknown_usable, ...
    'overwrite', overwrite_existing_outputs, ...
    'load_raw_duration', P.qc.run.load_raw_duration, ...
    'hf_band', P.qc.run.hf_band, ...
    'lf_band', P.qc.run.lf_band);

fprintf('\nStage 1 / Step 13 complete.\n');
fprintf('Review the QC CSVs and manifests before using them downstream.\n');

function assert_configured_input_dir(path_value, label)
path_char = char(path_value);
if contains(path_char, '<SET_')
    error('%s is still using a placeholder path. Edit helpers/r01_stage1_params.m first.', label);
end
if ~exist(path_char, 'dir')
    error('%s does not exist: %s', label, path_char);
end
end
