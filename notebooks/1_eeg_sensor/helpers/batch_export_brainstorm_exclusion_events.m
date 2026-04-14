function batch_export_brainstorm_exclusion_events(bst_db_root, out_dir, varargin)
%BATCH_EXPORT_BRAINSTORM_EXCLUSION_EVENTS Public wrapper for exclusion export.
%
% What this helper does:
%   Scans Brainstorm raw-link MAT files and exports the manual `BAD`,
%   `boundary`, and `bad_boundary` events to one TSV per run.
%
% When it is used:
%   Called by `12_export_and_union_merge_brainstorm_exclusions.m`.
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

stage1_dir = fileparts(fileparts(mfilename('fullpath')));
if ~isempty(stage1_dir)
    addpath(stage1_dir);
end

r01_batch_export_bst_exclusions_Fevents(bst_db_root, out_dir, varargin{:});

end
