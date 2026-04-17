function batch_export_brainstorm_exclusion_events(bst_db_root, out_dir, varargin)
%BATCH_EXPORT_BRAINSTORM_EXCLUSION_EVENTS Public wrapper for exclusion export.
%
% What this helper does:
%   Scans Brainstorm raw-link MAT files and exports the manual `BAD`,
%   `boundary`, and `bad_boundary` events to one TSV per run.
%
% When it is used:
%   Called by `export_and_union_merge_brainstorm_exclusions_12.m`.
%
% Key inputs:
%   - Brainstorm database root
%   - output directory for exported TSV files
%   - the same name-value options accepted by the preserved batch exporter
%
% Key outputs:
%   Writes one `*_bst_exclusions.tsv` per run plus a batch summary CSV.
%
% Important note:
%   The recovered export behavior is preserved exactly, including its
%   current point-event handling, through the legacy implementation
%   `r01_batch_export_bst_exclusions_Fevents.m`.

this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
stage1_dir = fileparts(this_dir);

if ~isempty(stage1_dir)
    addpath(stage1_dir);
end
if ~isempty(this_dir)
    addpath(this_dir);
end

assert_dependency_exists(fullfile(stage1_dir, 'r01_batch_export_bst_exclusions_Fevents.m'), ...
    ['Missing preserved Stage-1 batch exporter:' newline ...
     '  notebooks/1_eeg_sensor/r01_batch_export_bst_exclusions_Fevents.m']);
assert_dependency_exists(fullfile(this_dir, 'r01_export_bst_exclusions_Fevents.m'), ...
    ['Missing preserved Stage-1 one-run exporter:' newline ...
     '  notebooks/1_eeg_sensor/helpers/r01_export_bst_exclusions_Fevents.m' newline ...
     'The public helper layer keeps the recovered Brainstorm export behavior' newline ...
     'through this low-level implementation.']);

r01_batch_export_bst_exclusions_Fevents(bst_db_root, out_dir, varargin{:});

end

function assert_dependency_exists(path_to_file, message_text)
if exist(path_to_file, 'file') ~= 2
    error('%s', message_text);
end
end
