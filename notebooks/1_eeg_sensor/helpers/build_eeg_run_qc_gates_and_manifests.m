function [QC, MAN, EXC] = build_eeg_run_qc_gates_and_manifests(root_raw_eeglab, eeg_ic_pruned_dir, qc_tables_dir, qc_exclusions_dir, varargin)
%BUILD_EEG_RUN_QC_GATES_AND_MANIFESTS Public wrapper for Stage-1 run QC.
%
% What this helper does:
%   Computes the run-level EEG QC metrics, include/exclude decisions, and
%   manifest files that support the manuscript-facing Stage-1 outputs.
%
% When it is used:
%   Called by `eeg_run_qc_and_table_s1_13.m`.
%
% Key inputs:
%   - raw EEGLAB directory
%   - Stage-1 pruned EEG directory
%   - Stage-1 QC tables directory
%   - Stage-1 exclusion-summary directory
%   - the same name-value options accepted by the preserved legacy helper
%
% Key outputs:
%   Returns the same tables as the legacy helper and writes the same CSV and
%   manifest files as before.
%
% Important note:
%   The current QC behavior, including the explicit EMG-proxy gate that is
%   kept visible for provenance, now lives in the descriptive helper
%   `build_runlevel_qc_gates.m`. The older `r01_*` file remains only as a
%   compatibility wrapper.

this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
assert_dependency_exists(fullfile(this_dir, 'build_runlevel_qc_gates.m'), ...
    ['Missing Stage-1 QC implementation:' newline ...
     '  notebooks/1_eeg_sensor/helpers/build_runlevel_qc_gates.m']);

[QC, MAN, EXC] = build_runlevel_qc_gates( ...
    root_raw_eeglab, ...
    eeg_ic_pruned_dir, ...
    qc_tables_dir, ...
    qc_exclusions_dir, ...
    'clean_subdir', 'clean_sets', ...
    'withICA_subdir', 'with_ica', ...
    varargin{:});

end

function assert_dependency_exists(path_to_file, message_text)
if exist(path_to_file, 'file') ~= 2
    error('%s', message_text);
end
end
