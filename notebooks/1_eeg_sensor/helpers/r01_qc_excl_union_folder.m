function r01_qc_excl_union_folder(union_dir, out_dir, varargin)
% QC scan of *_excl_union.tsv files.
% Flags common problems and outliers; optionally estimates run duration from Brainstorm rawlink MATs.
%
% Inputs:
%   union_dir : folder containing *_excl_union.tsv files
%   out_dir   : where to write QC CSV
%
% Optional name-value:
%   'bst_db_root'         : Brainstorm DB root (contains 'data'); if provided, we try to load run duration
%   'adjacency_tol_sec'   : tolerance for "touching" (used for gap metrics only; default 0)
%   'max_excl_frac_warn'  : warn if excluded fraction exceeds this (default 0.20)
%   'max_interval_sec_warn': warn if any single interval exceeds this (default 30)
%   'min_interval_sec_warn': warn if any interval is shorter than this (default 0.05)
%
% Outputs:
%   out_dir\excl_union_qc_summary.csv

p = inputParser;
p.addParameter('bst_db_root', '', @(x) ischar(x) || isstring(x));
p.addParameter('adjacency_tol_sec', 0.0, @(x) isnumeric(x) && isscalar(x) && x>=0);
p.addParameter('max_excl_frac_warn', 0.20, @(x) isnumeric(x) && isscalar(x) && x>0 && x<1);
p.addParameter('max_interval_sec_warn', 30.0, @(x) isnumeric(x) && isscalar(x) && x>0);
p.addParameter('min_interval_sec_warn', 0.05, @(x) isnumeric(x) && isscalar(x) && x>=0);
p.parse(varargin{:});

bst_db_root = char(p.Results.bst_db_root);
tol = p.Results.adjacency_tol_sec;
max_frac = p.Results.max_excl_frac_warn;
max_int = p.Results.max_interval_sec_warn;
min_int = p.Results.min_interval_sec_warn;

if ~exist(out_dir,'dir'); mkdir(out_dir); end

files = dir(fullfile(union_dir, '*_excl_union.tsv'));
fprintf('Found %d *_excl_union.tsv files in %s\n', numel(files), union_dir);

rows = {};
for i = 1:numel(files)
    in_tsv = fullfile(files(i).folder, files(i).name);

    T = readtable(in_tsv, 'FileType','text','Delimiter','\t');
    if isempty(T)
        rows(end+1,:) = make_row(in_tsv, NaN, 0, 0, NaN, NaN, NaN, NaN, "EMPTY", ""); %#ok<AGROW>
        continue;
    end

    % Required columns
    if ~all(ismember({'start_sec','end_sec'}, T.Properties.VariableNames))
        rows(end+1,:) = make_row(in_tsv, NaN, height(T), NaN, NaN, NaN, NaN, NaN, "BAD_COLS", "Missing start_sec/end_sec"); %#ok<AGROW>
        continue;
    end

    s = T.start_sec; e = T.end_sec;

    % Drop NaNs
    good = ~(isnan(s) | isnan(e));
    s = s(good); e = e(good);

    % Fix swapped endpoints for QC purposes (flag it)
    swapped = sum(s > e);
    tmp = s(s>e); s(s>e) = e(s>e); e(s>e) = tmp;

    % Sort
    [s, idx] = sort(s); e = e(idx);

    % Compute overlap violations
    overlap_ct = sum(s(2:end) < (e(1:end-1) - 1e-9));

    dur = e - s;
    total_excl = sum(dur);
    min_dur = min(dur); max_dur = max(dur);

    % Gap metrics
    gaps = s(2:end) - e(1:end-1);
    min_gap = NaN;
    if ~isempty(gaps); min_gap = min(gaps); end

    % Optional: estimate run duration from Brainstorm rawlink
    run_dur = NaN;
    if ~isempty(bst_db_root)
        rawlink = guess_rawlink_from_union(bst_db_root, files(i).name);
        if ~isempty(rawlink)
            run_dur = bst_rawlink_duration_sec(rawlink);
        end
    end

    excl_frac = NaN;
    if ~isnan(run_dur) && run_dur > 0
        excl_frac = total_excl / run_dur;
    end

    % Flags
    flags = strings(0,1);
    if any(s < -1e-6); flags(end+1) = "NEG_START"; end %#ok<AGROW>
    if any(e < -1e-6); flags(end+1) = "NEG_END"; end %#ok<AGROW>
    if swapped > 0; flags(end+1) = "SWAPPED_SE"; end %#ok<AGROW>
    if overlap_ct > 0; flags(end+1) = "OVERLAP"; end %#ok<AGROW>
    if any(dur <= 0); flags(end+1) = "NONPOS_DUR"; end %#ok<AGROW>
    if ~isnan(min_dur) && min_dur < min_int; flags(end+1) = "TINY_INTERVAL"; end %#ok<AGROW>
    if ~isnan(max_dur) && max_dur > max_int; flags(end+1) = "HUGE_INTERVAL"; end %#ok<AGROW>
    if ~isnan(excl_frac) && excl_frac > max_frac; flags(end+1) = "HIGH_EXCL_FRAC"; end %#ok<AGROW>
    if ~isnan(min_gap) && min_gap >= 0 && min_gap <= tol; flags(end+1) = "TOUCHING_WINDOWS"; end %#ok<AGROW>

    flag_str = "OK";
    if ~isempty(flags); flag_str = strjoin(flags, "|"); end

    rows(end+1,:) = make_row(in_tsv, run_dur, numel(s), overlap_ct, total_excl, excl_frac, min_dur, max_dur, flag_str, ""); %#ok<AGROW>
end

S = cell2table(rows, 'VariableNames', { ...
    'union_tsv','run_duration_sec','n_intervals','n_overlap_violations', ...
    'total_excluded_sec','excluded_fraction','min_interval_sec','max_interval_sec', ...
    'flags','notes'});

out_csv = fullfile(out_dir, 'excl_union_qc_summary.csv');
writetable(S, out_csv);
fprintf('Wrote QC summary: %s\n', out_csv);

end

% ------- helpers -------
function row = make_row(tsv, run_dur, n_int, n_ov, tot, frac, mind, maxd, flags, notes)
row = {string(tsv), run_dur, n_int, n_ov, tot, frac, mind, maxd, string(flags), string(notes)};
end

function rawlink = guess_rawlink_from_union(bst_db_root, union_filename)
% union filename example: sub-01_ses-01_desc-ICBrain70_clean_excl_union.tsv
base = regexprep(union_filename, '_excl_union\.tsv$', '');
% rawlink mat base is typically data_0raw_<base>.mat
target = ['data_0raw_' base '.mat'];
hits = dir(fullfile(bst_db_root, 'data', '**', target));
if isempty(hits)
    rawlink = '';
else
    rawlink = fullfile(hits(1).folder, hits(1).name);
end
end

function dur = bst_rawlink_duration_sec(rawlink_mat)
dur = NaN;
try
    S = load(rawlink_mat, 'F');
    if ~isfield(S,'F') || isempty(S.F); return; end
    F = S.F; if numel(F) > 1; F = F(1); end

    % Common Brainstorm patterns: try several
    if isfield(F,'prop') && isstruct(F.prop)
        if isfield(F.prop,'times') && numel(F.prop.times) >= 2
            dur = double(F.prop.times(2) - F.prop.times(1));
            return;
        end
        if isfield(F.prop,'sfreq') && isfield(F.prop,'nSamples') && F.prop.sfreq > 0 && F.prop.nSamples > 1
            dur = double((F.prop.nSamples - 1) / F.prop.sfreq);
            return;
        end
    end

    % F.Time can sometimes be a vector of sample times
    if isfield(F,'Time') && ~isempty(F.Time)
        t = F.Time;
        if isnumeric(t)
            if isvector(t)
                dur = double(max(t) - min(t));
                return;
            elseif numel(t) >= 2
                dur = double(t(2) - t(1));
                return;
            end
        end
    end
catch
    dur = NaN;
end
end
