function batch_merge_exclusion_union_masks(in_dir, varargin)
%BATCH_MERGE_EXCLUSION_UNION_MASKS Public wrapper for union-merge masking.
%
% What this helper does:
%   Reads the exported Brainstorm exclusion TSV files for a folder and
%   merges overlapping or touching intervals into one union mask per run.
%
% When it is used:
%   Called by `12_export_and_union_merge_brainstorm_exclusions.m`.
%
% Key inputs:
%   - folder containing `*_bst_exclusions.tsv`
%   - the same name-value options accepted by the preserved batch merger
%
% Key outputs:
%   Writes `*_excl_union.tsv` and per-run QC CSV sidecars.
%
% Important note:
%   The interval-merging logic itself remains in the preserved legacy
%   implementation `r01_batch_merge_exclusions_union.m`.

stage1_dir = fileparts(fileparts(mfilename('fullpath')));
if ~isempty(stage1_dir)
    addpath(stage1_dir);
end

r01_batch_merge_exclusions_union(in_dir, varargin{:});

end
