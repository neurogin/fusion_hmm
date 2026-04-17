function batch_export_brainstorm_exclusion_events(bst_db_root, out_dir, varargin)
%BATCH_EXPORT_BRAINSTORM_EXCLUSION_EVENTS Public Stage-1 batch exclusion exporter.
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
%   - the same name-value options used in the preserved batch workflow
%
% Key outputs:
%   Writes one `*_bst_exclusions.tsv` per run plus a batch summary CSV.
%
% Important note:
%   The recovered export behavior is preserved exactly, including its
%   current point-event handling. The per-run export logic now lives in the
%   descriptive helper `export_brainstorm_exclusion_events.m`.

this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
stage1_dir = fileparts(this_dir);

if ~isempty(stage1_dir)
    addpath(stage1_dir);
end
if ~isempty(this_dir)
    addpath(this_dir);
end

assert_dependency_exists(fullfile(this_dir, 'export_brainstorm_exclusion_events.m'), ...
    ['Missing Stage-1 one-run exclusion exporter:' newline ...
     '  notebooks/1_eeg_sensor/helpers/export_brainstorm_exclusion_events.m']);

p = inputParser;
p.addParameter('overwrite', true, @(x) islogical(x) && isscalar(x));
p.addParameter('file_filter', '', @(x) ischar(x) || isstring(x));
p.addParameter('recursive', true, @(x) islogical(x) && isscalar(x));
p.parse(varargin{:});

overwrite = p.Results.overwrite;
file_filter = char(p.Results.file_filter);
recursive = p.Results.recursive;

if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

data_root = fullfile(bst_db_root, 'data');
if ~exist(data_root, 'dir')
    error('Could not find Brainstorm data folder: %s', data_root);
end

if recursive
    rawfiles = dir(fullfile(data_root, '**', 'data_0raw_*.mat'));
else
    rawfiles = dir(fullfile(data_root, 'data_0raw_*.mat'));
end

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

    [~, base, ~] = fileparts(in_mat);
    base2 = regexprep(base, '^data_0raw_', '');
    out_tsv = fullfile(out_dir, sprintf('%s_bst_exclusions.tsv', base2));

    if exist(out_tsv,'file') && ~overwrite
        fprintf('[%d/%d] SKIP (exists): %s\n', i, numel(rawfiles), out_tsv);
        n_skip = n_skip + 1;
        summary(end+1,:) = make_summary_row(in_mat, out_tsv, 'skipped', NaN, NaN, NaN, NaN, NaN, NaN); %#ok<AGROW>
        continue;
    end

    fprintf('[%d/%d] Export: %s\n', i, numel(rawfiles), base2);

    try
        export_brainstorm_exclusion_events(in_mat, out_tsv);

        T = readtable(out_tsv, 'FileType','text', 'Delimiter','\t');

        nBAD = sum(strcmpi(T.label,'BAD'));
        nBnd = sum(strcmpi(T.label,'boundary'));
        nBB  = sum(strcmpi(T.label,'bad_boundary'));

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

function assert_dependency_exists(path_to_file, message_text)
if exist(path_to_file, 'file') ~= 2
    error('%s', message_text);
end
end

function row = make_summary_row(in_mat, out_tsv, status, nBAD, nBnd, nBB, total_excl, min_t, max_t, err)
if nargin < 10; err = ""; end
row = {in_mat, out_tsv, status, nBAD, nBnd, nBB, total_excl, min_t, max_t, char(err)};
end
