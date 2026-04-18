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
%   - public alias: `brainstorm_protocol_root`
%
% Key outputs:
%   Writes `excl_union_qc_summary.csv`.
%
% Important note:
%   This wrapper keeps the current exclusion-summary behavior unchanged by
%   delegating to `summarize_exclusion_union_folder_qc.m`.

this_file = mfilename('fullpath');
this_dir = fileparts(this_file);

assert_dependency_exists(fullfile(this_dir, 'summarize_exclusion_union_folder_qc.m'), ...
    ['Missing Stage-1 exclusion-summary helper:' newline ...
     '  notebooks/1_eeg_sensor/helpers/summarize_exclusion_union_folder_qc.m']);

mapped_varargin = map_public_brainstorm_option(varargin);
summarize_exclusion_union_folder_qc(union_dir, out_dir, mapped_varargin{:});

end

function mapped_varargin = map_public_brainstorm_option(varargin_in)
mapped_varargin = varargin_in;

is_name = cellfun(@(x) ischar(x) || isstring(x), mapped_varargin);
name_cells = cellfun(@char, mapped_varargin(is_name), 'UniformOutput', false);
name_positions = find(is_name);

protocol_match = strcmpi(name_cells, 'brainstorm_protocol_root');
if ~any(protocol_match)
    return;
end

protocol_name_pos = name_positions(find(protocol_match, 1, 'last'));
if protocol_name_pos >= numel(mapped_varargin)
    error('Option "brainstorm_protocol_root" must be followed by a folder path.');
end

protocol_value = mapped_varargin{protocol_name_pos + 1};
mapped_varargin(protocol_name_pos:protocol_name_pos + 1) = [];

is_name = cellfun(@(x) ischar(x) || isstring(x), mapped_varargin);
if any(strcmpi(cellfun(@char, mapped_varargin(is_name), 'UniformOutput', false), 'bst_db_root'))
    return;
end

mapped_varargin = [mapped_varargin, {'bst_db_root', protocol_value}];
end

function assert_dependency_exists(path_to_file, message_text)
if exist(path_to_file, 'file') ~= 2
    error('%s', message_text);
end
end
