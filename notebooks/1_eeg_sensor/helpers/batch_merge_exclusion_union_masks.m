function batch_merge_exclusion_union_masks(export_dir, union_dir, varargin)
%BATCH_MERGE_EXCLUSION_UNION_MASKS Public helper for union-mask export.
%
% What this helper does:
%   Reads the exported Brainstorm exclusion TSV files for a folder and
%   writes the merged union intervals into a separate public-facing
%   `union_masks` folder.
%
% When it is used:
%   Called by `step12_export_and_union_merge_brainstorm_exclusions.m`.
%
% Key inputs:
%   - folder containing `*_bst_exclusions.tsv`
%   - output folder for `*_excl_union.tsv`
%   - the same name-value options accepted by the preserved one-run merger
%
% Key outputs:
%   Writes `*_excl_union.tsv` and per-run QC CSV sidecars in `union_dir`.
%
% Important note:
%   The interval-merging logic lives in the descriptive one-run helper
%   `merge_exclusion_union_masks.m`, which is the active public
%   implementation used by the Stage-1 workflow.

this_file = mfilename('fullpath');
this_dir = fileparts(this_file);

assert_dependency_exists(fullfile(this_dir, 'merge_exclusion_union_masks.m'), ...
    ['Missing Stage-1 one-run union-merger:' newline ...
     '  notebooks/1_eeg_sensor/helpers/merge_exclusion_union_masks.m']);

p = inputParser;
p.addParameter('adjacency_tol_sec', 0.0, @(x) isnumeric(x) && isscalar(x) && x>=0);
p.addParameter('min_dur_sec', 0.0, @(x) isnumeric(x) && isscalar(x) && x>=0);
p.addParameter('labels', ["BAD","boundary","bad_boundary"], @(x) isstring(x) || iscellstr(x));
p.addParameter('overwrite', true, @(x) islogical(x) && isscalar(x));
p.parse(varargin{:});

export_dir = char(export_dir);
union_dir = char(union_dir);
ensure_dir(union_dir);

files = dir(fullfile(export_dir, '*_bst_exclusions.tsv'));
fprintf('Found %d *_bst_exclusions.tsv files in %s\n', numel(files), export_dir);

for i = 1:numel(files)
    in_tsv = fullfile(files(i).folder, files(i).name);
    base = regexprep(files(i).name, '_bst_exclusions\.tsv$', '');
    out_tsv = fullfile(union_dir, sprintf('%s_excl_union.tsv', base));

    if exist(out_tsv, 'file') && ~p.Results.overwrite
        fprintf('[%d/%d] SKIP (exists): %s\n', i, numel(files), out_tsv);
        continue;
    end

    fprintf('[%d/%d] Merge: %s\n', i, numel(files), files(i).name);
    merge_exclusion_union_masks(in_tsv, out_tsv, ...
        'labels', p.Results.labels, ...
        'adjacency_tol_sec', p.Results.adjacency_tol_sec, ...
        'min_dur_sec', p.Results.min_dur_sec, ...
        'write_qc_csv', true);
end

fprintf('Batch union-mask export complete.\n');

end

function ensure_dir(path_value)
if ~exist(path_value, 'dir')
    mkdir(path_value);
end
end

function assert_dependency_exists(path_to_file, message_text)
if exist(path_to_file, 'file') ~= 2
    error('%s', message_text);
end
end
