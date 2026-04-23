function [QC, MAN, EXC] = build_runlevel_qc_gates(root_raw_eeglab, eeg_ic_pruned_dir, qc_tables_dir, qc_exclusions_dir, varargin)
%BUILD_RUNLEVEL_QC_GATES Stage-1 run-level EEG QC and manifest builder.
%
% What this helper does:
%   Computes the Stage-1 run-level QC table plus include and exclude
%   manifests from:
%     1. cleaned EEG `.set` files
%     2. exclusion-union QC summaries
%     3. retained-sample spectral and bad-channel checks
%
% When it is used:
%   Called by `build_eeg_run_qc_gates_and_manifests.m` from the public
%   Stage-1 script `step13_eeg_run_qc_and_table_s1.m`.
%
% Important note:
%   The current QC logic is preserved exactly, including the explicit
%   `max_emg_db` gate that remains intentionally visible because of the
%   manuscript-versus-code wording tension around the EMG proxy.
%
% Primary inputs:
%   - Cleaned EEGLAB sets: <eeg_ic_pruned_dir>\<clean_subdir>\*_desc-<TAG>_clean.set
%   - Exclusion union QC summary: <qc_exclusions_dir>\excl_union_qc_summary.csv
%
% Metrics:
%   (1) usable_fraction (preferred): 1 - excluded_fraction from excl_union_qc_summary.csv
%       Fallbacks if missing:
%         - 1 - total_excluded_sec/run_duration_sec_from_bst
%         - 1 - total_excluded_sec/dur_raw_sec (raw .set duration)
%         - dur_clean_sec/dur_raw_sec (last resort)
%   (2) EMG proxy (dB): 10*log10( P(30-80 Hz) / P(8-13 Hz) ) using median channel
%       computed on retained samples when union TSV is available
%   (3) bad-channel count: robust STD outliers + flat channels, computed on retained samples when possible
%
% Writes:
%   <qc_tables_dir>\eeg_run_qc_gates_<TAG>.csv
%   <qc_tables_dir>\include_manifest.csv
%   <qc_tables_dir>\include_manifest_<TAG>.csv
%   <qc_tables_dir>\exclude_manifest.csv
%   <qc_tables_dir>\exclude_manifest_<TAG>.csv
%   <qc_tables_dir>\exclude_stems_<TAG>.txt
%
% Optional name-value:
%   'tag'                 (default 'ICRej70')
%   'min_usable_frac'     (default 0.70)
%   'max_emg_db'          (default 3.0)
%   'max_badchan_abs'     (default 10)
%   'max_badchan_frac'    (default 0.10)
%   'allow_unknown_usable'(default false)
%   'overwrite'           (default true)
%   'load_raw_duration'   (default false)
%   'hf_band'             (default [30 80])
%   'lf_band'             (default [8 13])
%   'clean_subdir'        (default 'clean')
%   'withICA_subdir'      (default 'withICA')

% -------------------------
% Options
% -------------------------
p = inputParser;
p.addParameter('tag', 'ICRej70', @(x) ischar(x) || isstring(x));
p.addParameter('min_usable_frac', 0.70, @(x) isnumeric(x) && isscalar(x) && x>0 && x<=1);
p.addParameter('max_emg_db', 3.0, @(x) isnumeric(x) && isscalar(x));
p.addParameter('max_badchan_abs', 10, @(x) isnumeric(x) && isscalar(x) && x>=0);
p.addParameter('max_badchan_frac', 0.10, @(x) isnumeric(x) && isscalar(x) && x>=0 && x<=1);
p.addParameter('allow_unknown_usable', false, @(x) islogical(x) && isscalar(x));
p.addParameter('overwrite', true, @(x) islogical(x) && isscalar(x));
p.addParameter('load_raw_duration', false, @(x) islogical(x) && isscalar(x));
p.addParameter('hf_band', [30 80], @(x) isnumeric(x) && numel(x)==2);
p.addParameter('lf_band', [8 13], @(x) isnumeric(x) && numel(x)==2);
p.addParameter('clean_subdir', 'clean', @(x) ischar(x) || isstring(x));
p.addParameter('withICA_subdir', 'withICA', @(x) ischar(x) || isstring(x));
p.parse(varargin{:});

opt.tag = char(p.Results.tag);
opt.min_usable_frac      = p.Results.min_usable_frac;
opt.max_emg_db           = p.Results.max_emg_db;
opt.max_badchan_abs      = p.Results.max_badchan_abs;
opt.max_badchan_frac     = p.Results.max_badchan_frac;
opt.allow_unknown_usable = p.Results.allow_unknown_usable;
opt.overwrite_out        = p.Results.overwrite;
opt.load_raw_duration    = p.Results.load_raw_duration;
opt.hf_band              = p.Results.hf_band;
opt.lf_band              = p.Results.lf_band;
opt.clean_subdir         = char(p.Results.clean_subdir);
opt.withICA_subdir       = char(p.Results.withICA_subdir);

root_raw_eeglab   = char(root_raw_eeglab);
eeg_ic_pruned_dir = char(eeg_ic_pruned_dir);
qc_tables_dir     = char(qc_tables_dir);
qc_exclusions_dir = char(qc_exclusions_dir);

% -------------------------
% Paths / checks
% -------------------------
if ~exist(qc_tables_dir,'dir'); mkdir(qc_tables_dir); end

clean_dir   = fullfile(eeg_ic_pruned_dir, opt.clean_subdir);
withICA_dir = fullfile(eeg_ic_pruned_dir, opt.withICA_subdir);

if exist('pop_loadset','file') ~= 2
    error('EEGLAB is not on the MATLAB path (pop_loadset missing). Add EEGLAB, then rerun.');
end

% -------------------------
% Load exclusion QC summary
% -------------------------
excl_csv = fullfile(qc_exclusions_dir, 'excl_union_qc_summary.csv');
EX = table();
if exist(excl_csv,'file')
    EX = readtable(excl_csv);
else
    warning('Missing excl_union_qc_summary.csv at: %s. usable_fraction will rely on fallback.', excl_csv);
end

exMap = containers.Map('KeyType','char','ValueType','double');
if ~isempty(EX) && hasVar(EX,'union_tsv')
    for i = 1:height(EX)
        u = safeChar(EX.union_tsv(i));
        [~, base, ~] = fileparts(u);
        [sub,ses,run] = parse_ids(base);
        stem = make_stem(sub,ses,run);
        if ~isempty(stem) && ~isKey(exMap, stem)
            exMap(stem) = i;
        end
    end
end

% -------------------------
% Index raw .set by stem
% -------------------------
rawFiles = dir(fullfile(root_raw_eeglab, '**', '*.set'));
rawMap = containers.Map('KeyType','char','ValueType','char');
dupRaw = containers.Map('KeyType','char','ValueType','double');

for i = 1:numel(rawFiles)
    fn = rawFiles(i).name;
    [sub,ses,run] = parse_ids(fn);
    stem = make_stem(sub,ses,run);
    if isempty(stem); continue; end
    fp = fullfile(rawFiles(i).folder, rawFiles(i).name);
    if ~isKey(rawMap, stem)
        rawMap(stem) = fp;
        dupRaw(stem) = 1;
    else
        dupRaw(stem) = dupRaw(stem) + 1;
    end
end

% -------------------------
% Find clean files
% -------------------------
pat = sprintf('*_desc-%s_clean.set', opt.tag);
cleanFiles = dir(fullfile(clean_dir, pat));
if isempty(cleanFiles)
    error('No clean files found: %s', fullfile(clean_dir, pat));
end

% -------------------------
% Define table schema ONCE
% -------------------------
varNames = { ...
    'stem','sub','ses','run', ...
    'raw_set','clean_set','withICA_set', ...
    'srate_hz','n_chan', ...
    'dur_raw_sec','dur_clean_sec', ...
    'run_duration_sec_from_bst','total_excluded_sec','excluded_fraction','usable_fraction', ...
    'kept_n_segments','kept_total_sec', ...
    'kept_min_sec','kept_median_sec','kept_max_sec', ...
    'kept_largest_frac','kept_adjacency_frac','effective_transition_pairs', ...
    'excl_flags', ...
    'emg_proxy_db','n_badchan','badchan_cap','n_raw_candidates', ...
    'include','reasons'};

nVars = numel(varNames);
rows = cell(0, nVars);

% -------------------------
% Main loop
% -------------------------
for i = 1:numel(cleanFiles)

    cleanPath = fullfile(cleanFiles(i).folder, cleanFiles(i).name);
    [sub,ses,run] = parse_ids(cleanFiles(i).name);
    stem = make_stem(sub,ses,run);

    rawPath = '';
    if isKey(rawMap, stem); rawPath = rawMap(stem); end

    withICAPath = '';
    base_clean = regexprep(cleanFiles(i).name, '_clean\.set$', '');
    cand_withICA = fullfile(withICA_dir, sprintf('%s_withICA.set', base_clean));
    if exist(cand_withICA,'file'); withICAPath = cand_withICA; end

    % Load CLEAN
    EEGc = pop_loadset('filename', cleanFiles(i).name, 'filepath', cleanFiles(i).folder);
    EEGc = eeg_checkset(EEGc);

    fs = double(EEGc.srate);
    nchan = double(EEGc.nbchan);
    dur_clean_sec = double(EEGc.pnts / EEGc.srate);

    % Exclusion metrics + union TSV
    union_tsv = '';
    run_dur_sec_bst = NaN;
    total_excl_sec = NaN;
    excl_frac = NaN;
    excl_flags = "MISSING_EXCL_SUMMARY";

    if ~isempty(EX) && isKey(exMap, stem)
        rix = exMap(stem);
        union_tsv        = getTblChar(EX, rix, 'union_tsv', '');
        run_dur_sec_bst  = getTblNum(EX, rix, 'run_duration_sec', NaN);
        total_excl_sec   = getTblNum(EX, rix, 'total_excluded_sec', NaN);
        excl_frac        = getTblNum(EX, rix, 'excluded_fraction', NaN);
        excl_flags       = string(getTblChar(EX, rix, 'flags', 'OK'));
    end

    if isnan(excl_frac) && ~isnan(total_excl_sec) && ~isnan(run_dur_sec_bst) && run_dur_sec_bst > 0
        excl_frac = total_excl_sec / run_dur_sec_bst;
    end

    % Load RAW duration only if needed
    dur_raw_sec = NaN;
    need_raw_for_excl = isnan(excl_frac) && ~isnan(total_excl_sec);
    need_raw_for_ratio = isnan(excl_frac) && isnan(total_excl_sec);
    need_raw = opt.load_raw_duration || need_raw_for_excl || need_raw_for_ratio;

    if need_raw && ~isempty(rawPath) && exist(rawPath,'file')
        [rf, rn, re] = fileparts(rawPath);
        EEGr = pop_loadset('filename', [rn re], 'filepath', rf);
        EEGr = eeg_checkset(EEGr);
        dur_raw_sec = double(EEGr.pnts / EEGr.srate);

        if isnan(excl_frac) && ~isnan(total_excl_sec) && dur_raw_sec > 0
            excl_frac = total_excl_sec / dur_raw_sec;
        end
    end

    % Usable fraction (final)
    usable_frac = NaN;
    if ~isnan(excl_frac)
        usable_frac = 1 - excl_frac;
    elseif ~isnan(dur_raw_sec) && dur_raw_sec > 0
        usable_frac = dur_clean_sec / dur_raw_sec;
    end

    % keepMask based on union TSV (true=kept)
    keepMask = true(1, EEGc.pnts);
    keepMask = apply_union_tsv_exclusions(keepMask, union_tsv, fs);

    % Kept fragmentation metrics (+ effective_transition_pairs)
    K = kept_fragmentation_metrics(keepMask, fs);

    % EMG proxy & bad channel count (computed on kept data if possible)
    emg_db = compute_emg_proxy_db(EEGc, keepMask, opt.hf_band, opt.lf_band);
    n_badchan = count_bad_channels(EEGc, keepMask);
    badchan_cap = min(opt.max_badchan_abs, ceil(opt.max_badchan_frac * nchan));

    % Gate decision + reasons
    include = true;
    reasons = strings(0,1);

    severeFlags = ["BAD_COLS","OVERLAP","NONPOS_DUR","NEG_START","NEG_END","NAN_COLS"];
    if ~isempty(excl_flags) && excl_flags ~= "OK" && excl_flags ~= "MISSING_EXCL_SUMMARY"
        if any(contains(excl_flags, severeFlags))
            include = false;
            reasons(end+1) = "excl_union_flags_severe"; 
        else
            reasons(end+1) = "excl_union_flags_warn"; 
        end
    end

    if isnan(usable_frac)
        if ~opt.allow_unknown_usable
            include = false;
            reasons(end+1) = "usable_frac_unknown"; 
        else
            reasons(end+1) = "usable_frac_unknown_warn"; 
        end
    elseif usable_frac < opt.min_usable_frac
        include = false;
        reasons(end+1) = sprintf('usable_frac<%.2f', opt.min_usable_frac); 
    end

    if ~isnan(emg_db) && emg_db > opt.max_emg_db
        include = false;
        reasons(end+1) = sprintf('emg_db>%.1f', opt.max_emg_db); 
    elseif isnan(emg_db)
        reasons(end+1) = "emg_db_nan"; 
    end

    if n_badchan > badchan_cap
        include = false;
        reasons(end+1) = sprintf('badchan>%d', badchan_cap); 
    end

    reasons_str = "";
    if ~isempty(reasons); reasons_str = strjoin(reasons, ';'); end

    n_raw_dups = 0;
    if isKey(dupRaw, stem); n_raw_dups = dupRaw(stem); end

    % Build row PROGRAMMATICALLY in varNames order
    row = cell(1, nVars);
    row = put(row, varNames, 'stem', stem);
    row = put(row, varNames, 'sub', char(sub));
    row = put(row, varNames, 'ses', char(ses));
    row = put(row, varNames, 'run', char(run));
    row = put(row, varNames, 'raw_set', rawPath);
    row = put(row, varNames, 'clean_set', cleanPath);
    row = put(row, varNames, 'withICA_set', withICAPath);

    row = put(row, varNames, 'srate_hz', fs);
    row = put(row, varNames, 'n_chan', nchan);

    row = put(row, varNames, 'dur_raw_sec', dur_raw_sec);
    row = put(row, varNames, 'dur_clean_sec', dur_clean_sec);

    row = put(row, varNames, 'run_duration_sec_from_bst', run_dur_sec_bst);
    row = put(row, varNames, 'total_excluded_sec', total_excl_sec);
    row = put(row, varNames, 'excluded_fraction', excl_frac);
    row = put(row, varNames, 'usable_fraction', usable_frac);

    row = put(row, varNames, 'kept_n_segments', K.kept_n_segments);
    row = put(row, varNames, 'kept_total_sec', K.kept_total_sec);
    row = put(row, varNames, 'kept_min_sec', K.kept_min_sec);
    row = put(row, varNames, 'kept_median_sec', K.kept_median_sec);
    row = put(row, varNames, 'kept_max_sec', K.kept_max_sec);
    row = put(row, varNames, 'kept_largest_frac', K.kept_largest_frac);
    row = put(row, varNames, 'kept_adjacency_frac', K.kept_adjacency_frac);
    row = put(row, varNames, 'effective_transition_pairs', K.effective_transition_pairs);

    row = put(row, varNames, 'excl_flags', char(excl_flags));

    row = put(row, varNames, 'emg_proxy_db', emg_db);
    row = put(row, varNames, 'n_badchan', n_badchan);
    row = put(row, varNames, 'badchan_cap', badchan_cap);
    row = put(row, varNames, 'n_raw_candidates', n_raw_dups);

    row = put(row, varNames, 'include', include);
    row = put(row, varNames, 'reasons', char(reasons_str));

    % Validate and append
    if numel(row) ~= nVars
        error('Internal error: row length != nVars');
    end
    rows(end+1,:) = row; 
end

QC = cell2table(rows, 'VariableNames', varNames);

% Defensive: ensure include is logical
if iscell(QC.include)
    QC.include = cellfun(@(x) logical(x), QC.include);
elseif ~islogical(QC.include)
    QC.include = logical(QC.include);
end


% -------------------------
% Write QC table
% -------------------------
qc_csv = fullfile(qc_tables_dir, sprintf('eeg_run_qc_gates_%s.csv', opt.tag));
writeIfAllowed(QC, qc_csv, opt.overwrite_out);

% -------------------------
% Include manifest
% -------------------------
MAN = QC(QC.include == true, :);
MAN = MAN(:, { ...
    'stem','sub','ses','run', ...
    'clean_set','raw_set', ...
    'excluded_fraction','usable_fraction', ...
    'kept_n_segments','kept_median_sec','kept_max_sec','kept_largest_frac','kept_adjacency_frac','effective_transition_pairs', ...
    'emg_proxy_db','n_badchan','n_chan','srate_hz'});

man_csv_tag = fullfile(qc_tables_dir, sprintf('include_manifest_%s.csv', opt.tag));
man_csv     = fullfile(qc_tables_dir, 'include_manifest.csv');
writeIfAllowed(MAN, man_csv_tag, opt.overwrite_out);
writeIfAllowed(MAN, man_csv, opt.overwrite_out);

% -------------------------
% Exclude manifest + drop list
% -------------------------
EXC = QC(QC.include == false, :);

if ~isempty(EXC)
    EXC = EXC(:, { ...
        'stem','sub','ses','run', ...
        'clean_set','raw_set','withICA_set', ...
        'excluded_fraction','usable_fraction', ...
        'kept_n_segments','kept_median_sec','kept_max_sec','kept_largest_frac','kept_adjacency_frac','effective_transition_pairs', ...
        'emg_proxy_db','n_badchan','n_chan','srate_hz', ...
        'excl_flags','reasons'});

    rs = string(EXC.reasons);
    EXC.fail_excl_union = contains(rs, "excl_union_flags_severe");
    EXC.fail_usable     = contains(rs, "usable_frac<") | contains(rs, "usable_frac_unknown");
    EXC.fail_emg        = contains(rs, "emg_db>");
    EXC.fail_badchan    = contains(rs, "badchan>");
    EXC.n_fail          = double(EXC.fail_excl_union) + double(EXC.fail_usable) + double(EXC.fail_emg) + double(EXC.fail_badchan);
else
    EXC = EXC; % keep empty
end

exc_csv_tag = fullfile(qc_tables_dir, sprintf('exclude_manifest_%s.csv', opt.tag));
exc_csv     = fullfile(qc_tables_dir, 'exclude_manifest.csv');
writeIfAllowed(EXC, exc_csv_tag, opt.overwrite_out);
writeIfAllowed(EXC, exc_csv, opt.overwrite_out);

drop_txt = fullfile(qc_tables_dir, sprintf('exclude_stems_%s.txt', opt.tag));
if ~(exist(drop_txt,'file') && ~opt.overwrite_out)
    if isempty(EXC)
        writelines(string.empty(0,1), drop_txt);
    else
        writelines(string(EXC.stem), drop_txt);
    end
    fprintf('Wrote drop list: %s\n', drop_txt);
else
    warning('Drop list exists and overwrite=false: %s', drop_txt);
end

end

% =========================================================================
% Utility: safe "write if allowed"
% =========================================================================
function writeIfAllowed(T, path, overwriteFlag)
if exist(path,'file') && ~overwriteFlag
    warning('File exists and overwrite=false: %s', path);
else
    writetable(T, path);
    fprintf('Wrote: %s\n', path);
end
end

% =========================================================================
% Utility: table var exists?
% =========================================================================
function tf = hasVar(T, varname)
tf = any(strcmp(T.Properties.VariableNames, varname));
end

% =========================================================================
% Utility: safeChar
% =========================================================================
function s = safeChar(x)
try
    s = char(x);
catch
    s = char(string(x));
end
end

% =========================================================================
% Utility: safe table getters
% =========================================================================
function out = getTblChar(T, rix, varname, defaultVal)
out = defaultVal;
if isempty(T) || ~hasVar(T, varname); return; end
out = safeChar(T.(varname)(rix));
end

function out = getTblNum(T, rix, varname, defaultVal)
out = defaultVal;
if isempty(T) || ~hasVar(T, varname); return; end
try
    out = double(T.(varname)(rix));
catch
    out = defaultVal;
end
end

% =========================================================================
% Utility: index by name, then assign
% =========================================================================
function row = put(row, varNames, name, value)
    ix = find(strcmp(varNames, name), 1);
    if isempty(ix)
        error('Unknown varName: %s', name);
    end
    row{ix} = value;
end


% =========================================================================
% Parse IDs
% =========================================================================
function [sub, ses, run] = parse_ids(fname)
sub = ""; ses = ""; run = "";
tok = regexp(fname, '(sub-[A-Za-z0-9]+)', 'tokens', 'once'); if ~isempty(tok); sub = string(tok{1}); end
tok = regexp(fname, '(ses-[A-Za-z0-9]+)', 'tokens', 'once'); if ~isempty(tok); ses = string(tok{1}); end
tok = regexp(fname, '(run-[A-Za-z0-9]+)', 'tokens', 'once'); if ~isempty(tok); run = string(tok{1}); end
end

function stem = make_stem(sub, ses, run)
parts = strings(0,1);
if strlength(sub)>0; parts(end+1)=sub; end 
if strlength(ses)>0; parts(end+1)=ses; end 
if strlength(run)>0; parts(end+1)=run; end 
if isempty(parts); stem = ''; else; stem = char(strjoin(parts,'_')); end
end

% =========================================================================
% Apply union TSV exclusions to keepMask
% =========================================================================
function keepMask = apply_union_tsv_exclusions(keepMask, union_tsv, fs)
if isempty(union_tsv) || ~exist(union_tsv,'file')
    return;
end
try
    U = readtable(union_tsv, 'FileType','text', 'Delimiter','\t');
    if isempty(U) || height(U) == 0; return; end
    if ~all(ismember({'start_sec','end_sec'}, U.Properties.VariableNames)); return; end

    s = double(U.start_sec);
    e = double(U.end_sec);
    good = ~(isnan(s) | isnan(e));
    s = s(good); e = e(good);

    swap = s > e;
    tmp = s(swap); s(swap)=e(swap); e(swap)=tmp;

    n = numel(keepMask);
    for j = 1:numel(s)
        a = floor(s(j)*fs) + 1;
        b = ceil(e(j)*fs);
        a = max(1, min(a, n));
        b = max(1, min(b, n));
        if b >= a
            keepMask(a:b) = false;
        end
    end
catch
    % Ignore; upstream flags should catch malformed TSV
end
end

% =========================================================================
% Kept fragmentation metrics (includes effective_transition_pairs)
% =========================================================================
function K = kept_fragmentation_metrics(keepMask, fs)
K = struct();
keepMask = keepMask(:)';

K.kept_n_segments = 0;
K.kept_total_sec = NaN;
K.kept_min_sec = NaN;
K.kept_median_sec = NaN;
K.kept_max_sec = NaN;
K.kept_largest_frac = NaN;
K.kept_adjacency_frac = NaN;
K.effective_transition_pairs = NaN;

if isempty(keepMask) || ~any(keepMask)
    K.kept_n_segments = 0;
    K.kept_total_sec = 0;
    K.kept_adjacency_frac = 0;
    K.effective_transition_pairs = 0;
    return;
end

segs = contiguous_segments(keepMask);   % kept segments
K.kept_n_segments = numel(segs);

lens_samp = zeros(K.kept_n_segments, 1);
for i = 1:K.kept_n_segments
    a = segs{i}(1); b = segs{i}(2);
    lens_samp(i) = max(0, b - a + 1);
end
lens_sec = lens_samp / fs;

K.kept_total_sec = sum(lens_sec);
K.kept_min_sec = min(lens_sec);
K.kept_median_sec = median(lens_sec);
K.kept_max_sec = max(lens_sec);

if K.kept_total_sec > 0
    K.kept_largest_frac = K.kept_max_sec / K.kept_total_sec;
end

N_kept = sum(lens_samp);
if N_kept <= 1
    K.kept_adjacency_frac = 0;
    K.effective_transition_pairs = 0;
else
    adj_pairs = sum(max(0, lens_samp - 1)); % Sigma(L-1)
    K.effective_transition_pairs = adj_pairs;
    K.kept_adjacency_frac = adj_pairs / (N_kept - 1);
end
end

function segs = contiguous_segments(mask)
mask = mask(:)';
d = diff([false mask false]);
starts = find(d==1);
ends   = find(d==-1) - 1;
segs = cell(numel(starts),1);
for i = 1:numel(starts)
    segs{i} = [starts(i), ends(i)];
end
end

% =========================================================================
% EMG proxy (correct: no compression before segmentation)
% =========================================================================
function emg_db = compute_emg_proxy_db(EEG, keepMask, hf_band, lf_band)
x = double(EEG.data);
sig = median(x, 1);
sig = sig - mean(sig);
fs  = double(EEG.srate);

if nargin < 2 || isempty(keepMask)
    keepMask = true(1, numel(sig));
else
    keepMask = keepMask(:)';
    if numel(keepMask) ~= numel(sig)
        warning('keepMask length mismatch; ignoring keepMask for EMG proxy.');
        keepMask = true(1, numel(sig));
    end
end

keptN = sum(keepMask);
if keptN < max(512, round(2*fs))
    emg_db = NaN;
    return;
end

segments = contiguous_segments(keepMask);

win   = max(256, min(round(2*fs), numel(sig)));
nover = floor(win/2);
nfft  = max(512, 2^nextpow2(win));

Pxx_sum = [];
W_sum = 0;
f = [];

for k = 1:numel(segments)
    a = max(1, min(segments{k}(1), numel(sig)));
    b = max(1, min(segments{k}(2), numel(sig)));
    if b <= a; continue; end

    seg = sig(a:b);
    if numel(seg) < win; continue; end

    [pxx, f0] = pwelch(seg, win, nover, nfft, fs);

    if isempty(Pxx_sum)
        Pxx_sum = zeros(size(pxx));
        f = f0;
    end
    w = numel(seg);
    Pxx_sum = Pxx_sum + pxx * w;
    W_sum = W_sum + w;
end

if W_sum == 0
    emg_db = NaN;
    return;
end

pxx = Pxx_sum / W_sum;
hf = bandpower_from_psd(pxx, f, hf_band);
lf = bandpower_from_psd(pxx, f, lf_band);

if lf <= 0 || hf <= 0 || isnan(lf) || isnan(hf)
    emg_db = NaN;
else
    emg_db = 10*log10(hf/lf);
end
end

function bp = bandpower_from_psd(pxx, f, band)
idx = f >= band(1) & f <= band(2);
if ~any(idx)
    bp = NaN; return;
end
bp = trapz(f(idx), pxx(idx));
end

% =========================================================================
% Bad channel count (robust STD outliers + flat)
% =========================================================================
function nbad = count_bad_channels(EEG, keepMask)
x = double(EEG.data);

if nargin >= 2 && ~isempty(keepMask)
    keepMask = keepMask(:)';
    if numel(keepMask) == size(x,2) && any(keepMask) && ~all(keepMask)
        x2 = x(:, keepMask);
        if size(x2,2) >= 200
            x = x2;
        end
    end
end

ch_std = std(x, 0, 2);
medv = median(ch_std);
madv = mad(ch_std, 1);
if madv == 0
    z = zeros(size(ch_std));
else
    z = (ch_std - medv) / madv;
end

flat = ch_std < 1e-12;
noisy = z > 5;

nbad = sum(flat | noisy);
end
