function summarize_exclusion_union_qc(union_dir, out_dir, varargin)
%SUMMARIZE_EXCLUSION_UNION_QC Public wrapper for exclusion QC summaries.
%
% What this helper does:
%   Scans the merged exclusion-union TSV files for a folder, computes run-
%   level exclusion metrics, and writes the QC summary used by Stage-1 EEG
%   run-level gating.
%
% When it is used:
%   Called by `eeg_run_qc_and_table_s1_13.m`.
%
% Key inputs:
%   - folder containing `*_excl_union.tsv`
%   - output directory for the QC summary CSV
%   - the same name-value options accepted by the preserved legacy helper
%
% Key outputs:
%   Writes `excl_union_qc_summary.csv`.
%
% Important note:
%   This wrapper keeps the current exclusion-summary behavior unchanged by
%   delegating to `r01_qc_excl_union_folder.m`.

this_file = mfilename('fullpath');
this_dir = fileparts(this_file);

assert_dependency_exists(fullfile(this_dir, 'r01_qc_excl_union_folder.m'), ...
    ['Missing preserved Stage-1 exclusion-summary helper:' newline ...
     '  notebooks/1_eeg_sensor/helpers/r01_qc_excl_union_folder.m']);

r01_qc_excl_union_folder(union_dir, out_dir, varargin{:});

end

function assert_dependency_exists(path_to_file, message_text)
if exist(path_to_file, 'file') ~= 2
    error('%s', message_text);
end
end
