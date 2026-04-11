addpath('C:\EEGFMRI\hmm\R01_rerun\03_code\matlab');
P = r01_params();

in_dir = P.deriv.masks_bst_exports;
files = dir(fullfile(in_dir, '*_bst_exclusions.tsv'));

fprintf('Found %d *_bst_exclusions.tsv files in %s\n', numel(files), in_dir);

for i = 1:numel(files)
    in_tsv  = fullfile(files(i).folder, files(i).name);
    out_tsv = regexprep(in_tsv, '_bst_exclusions\.tsv$', '_excl_union.tsv');

    fprintf('[%d/%d] Union: %s\n', i, numel(files), files(i).name);

    r01_merge_exclusions_union( ...
        in_tsv, out_tsv, ...
        'adjacency_tol_sec', P.mask.merge.adjacency_tol_sec, ...
        'min_dur_sec', P.mask.merge.min_dur_sec);
end
