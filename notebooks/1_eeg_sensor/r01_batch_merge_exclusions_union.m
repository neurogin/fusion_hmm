function r01_batch_merge_exclusions_union(in_dir, varargin)
% Batch merge exclusions into union intervals for all *_bst_exclusions.tsv files in a folder.
%
% Writes alongside each input:
%   *_excl_union.tsv
%   *_excl_union_qc.csv

p = inputParser;
p.addParameter('adjacency_tol_sec', 0.0, @(x) isnumeric(x) && isscalar(x) && x>=0);
p.addParameter('min_dur_sec', 0.0, @(x) isnumeric(x) && isscalar(x) && x>=0);
p.addParameter('labels', ["BAD","boundary","bad_boundary"], @(x) isstring(x) || iscellstr(x));
p.addParameter('overwrite', true, @(x) islogical(x) && isscalar(x));
p.parse(varargin{:});

tol = p.Results.adjacency_tol_sec;
min_dur = p.Results.min_dur_sec;
labels = string(p.Results.labels);
overwrite = p.Results.overwrite;

files = dir(fullfile(in_dir, '*_bst_exclusions.tsv'));
fprintf('Found %d *_bst_exclusions.tsv files in %s\n', numel(files), in_dir);

for i = 1:numel(files)
    in_tsv = fullfile(files(i).folder, files(i).name);
    out_tsv = regexprep(in_tsv, '_bst_exclusions\.tsv$', '_excl_union.tsv');

    if exist(out_tsv,'file') && ~overwrite
        fprintf('[%d/%d] SKIP (exists): %s\n', i, numel(files), out_tsv);
        continue;
    end

    fprintf('[%d/%d] Merge: %s\n', i, numel(files), files(i).name);
    r01_merge_exclusions_union(in_tsv, out_tsv, ...
        'labels', labels, ...
        'adjacency_tol_sec', tol, ...
        'min_dur_sec', min_dur, ...
        'write_qc_csv', true);
end

fprintf('Batch union merge complete.\n');
end
