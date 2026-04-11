function r01_batch_export_bst_exclusions_Fevents(bst_db_root, out_dir, varargin)
% Batch export Brainstorm exclusions (BAD/boundary/bad_boundary) from raw-link MAT files.
% Requires: r01_export_bst_exclusions_Fevents.m on MATLAB path.
%
% Inputs:
%   bst_db_root : Brainstorm DB root folder (contains "data" folder)
%   out_dir     : where to write exported TSVs + summary CSV
%
% Optional name-value:
%   'overwrite'      (default true)  - overwrite existing TSVs
%   'file_filter'    (default '')    - substring filter for input MAT paths (e.g., 'ICBrain70_clean')
%   'recursive'      (default true)  - scan bst_db_root\data\**\data_0raw_*.mat
%
% Outputs:
%   out_dir\bst_exclusions_batch_summary.csv
%   out_dir\<basename>_bst_exclusions.tsv

p = inputParser;
p.addParameter('overwrite', true, @(x) islogical(x) && isscalar(x));
p.addParameter('file_filter', '', @(x) ischar(x) || isstring(x));
p.addParameter('recursive', true, @(x) islogical(x) && isscalar(x));
p.parse(varargin{:});

overwrite  = p.Results.overwrite;
file_filter = char(p.Results.file_filter);
recursive  = p.Results.recursive;

if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

data_root = fullfile(bst_db_root, 'data');
if ~exist(data_root, 'dir')
    error('Could not find Brainstorm data folder: %s', data_root);
end

% Find raw-link files
if recursive
    rawfiles = dir(fullfile(data_root, '**', 'data_0raw_*.mat'));
else
    rawfiles = dir(fullfile(data_root, 'data_0raw_*.mat'));
end

% Optional filter (e.g., only ICBrain70_clean)
if ~isempty(file_filter)
    keep = false(numel(rawfiles),1);
    for i = 1:numel(rawfiles)
        fullp = fullfile(rawfiles(i).folder, rawfiles(i).name);
        keep(i) = contains(fullp, file_filter);
    end
    rawfiles = rawfiles(keep);
end

fprintf('Found %d raw-link MAT files.\n', numel(rawfiles));
if isempty(rawfiles)
    return;
end

summary = {};
n_ok = 0; n_fail = 0; n_skip = 0;

for i = 1:numel(rawfiles)
    in_mat = fullfile(rawfiles(i).folder, rawfiles(i).name);

    % Output name: strip "data_0raw_" prefix if present
    [~, base, ~] = fileparts(in_mat);
    base2 = regexprep(base, '^data_0raw_', '');
    out_tsv = fullfile(out_dir, sprintf('%s_bst_exclusions.tsv', base2));

    if exist(out_tsv,'file') && ~overwrite
        fprintf('[%d/%d] SKIP (exists): %s\n', i, numel(rawfiles), out_tsv);
        n_skip = n_skip + 1;
        summary(end+1,:) = make_summary_row(in_mat, out_tsv, 'skipped', NaN, NaN, NaN, NaN, NaN); %#ok<AGROW>
        continue;
    end

    fprintf('[%d/%d] Export: %s\n', i, numel(rawfiles), base2);

    try
        r01_export_bst_exclusions_Fevents(in_mat, out_tsv);

        % Summarize TSV contents
        T = readtable(out_tsv, 'FileType','text', 'Delimiter','\t');

        nBAD = sum(strcmpi(T.label,'BAD'));
        nBnd = sum(strcmpi(T.label,'boundary'));
        nBB  = sum(strcmpi(T.label,'bad_boundary'));

        % Basic sanity: duration of excluded intervals (sec)
        if height(T) > 0
            dur = T.end_sec - T.start_sec;
            total_excl = sum(dur(~isnan(dur)));
            min_t = min(T.start_sec);
            max_t = max(T.end_sec);
        else
            total_excl = 0;
            min_t = NaN;
            max_t = NaN;
        end

        n_ok = n_ok + 1;
        summary(end+1,:) = make_summary_row(in_mat, out_tsv, 'ok', nBAD, nBnd, nBB, total_excl, min_t, max_t); %#ok<AGROW>

    catch ME
        n_fail = n_fail + 1;
        warning('FAILED export for %s\n  %s', in_mat, ME.message);
        summary(end+1,:) = make_summary_row(in_mat, out_tsv, 'fail', NaN, NaN, NaN, NaN, NaN, NaN, ME.message); %#ok<AGROW>
    end
end

% Write batch summary CSV
S = cell2table(summary, 'VariableNames', { ...
    'rawlink_mat','out_tsv','status', ...
    'n_BAD','n_boundary','n_bad_boundary', ...
    'total_excluded_sec','min_start_sec','max_end_sec','error_message'});

summary_csv = fullfile(out_dir, 'bst_exclusions_batch_summary.csv');
writetable(S, summary_csv);

fprintf('\nBatch export complete.\n');
fprintf('  OK:   %d\n', n_ok);
fprintf('  FAIL: %d\n', n_fail);
fprintf('  SKIP: %d\n', n_skip);
fprintf('Summary: %s\n', summary_csv);

end

% -------- helper to build a summary row (keeps code readable) ----------
function row = make_summary_row(in_mat, out_tsv, status, nBAD, nBnd, nBB, total_excl, min_t, max_t, err)
if nargin < 10; err = ""; end
row = {in_mat, out_tsv, status, nBAD, nBnd, nBB, total_excl, min_t, max_t, char(err)};
end
