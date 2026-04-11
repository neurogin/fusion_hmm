function out_tsv = r01_merge_exclusions_union(in_tsv, out_tsv, varargin)
% r01_merge_exclusions_union
%
% Merge overlapping/adjacent exclusion intervals into a union list.
%
% Inputs:
%   in_tsv  : TSV produced by r01_export_bst_exclusions_Fevents
%             columns: label, start_sec, end_sec, source
%   out_tsv : output TSV path for merged union intervals
%
% Optional name-value:
%   'labels'           : cellstr/strings of labels to include (default: BAD,boundary,bad_boundary)
%   'adjacency_tol_sec': treat intervals within this gap as contiguous (default: 0.0)
%   'min_dur_sec'      : drop merged intervals shorter than this (default: 0.0)
%   'write_qc_csv'     : write QC summary CSV alongside out_tsv (default: true)
%
% Output:
%   out_tsv path (returned)

p = inputParser;
p.addParameter('labels', ["BAD","boundary","bad_boundary"], @(x) isstring(x) || iscellstr(x));
p.addParameter('adjacency_tol_sec', 0.0, @(x) isnumeric(x) && isscalar(x) && x>=0);
p.addParameter('min_dur_sec', 0.0, @(x) isnumeric(x) && isscalar(x) && x>=0);
p.addParameter('write_qc_csv', true, @(x) islogical(x) && isscalar(x));
p.parse(varargin{:});

labels_keep = string(p.Results.labels);
tol = p.Results.adjacency_tol_sec;
min_dur = p.Results.min_dur_sec;
write_qc = p.Results.write_qc_csv;

if ~exist(in_tsv,'file')
    error('Input TSV not found: %s', in_tsv);
end

T = readtable(in_tsv, 'FileType','text', 'Delimiter','\t');

% Basic column validation
req = ["label","start_sec","end_sec"];
for r = 1:numel(req)
    if ~any(strcmpi(T.Properties.VariableNames, req(r)))
        error('Missing required column "%s" in %s', req(r), in_tsv);
    end
end

% Keep desired labels only
lab = string(T.label);
keep = false(height(T),1);
for i = 1:numel(labels_keep)
    keep = keep | strcmpi(lab, labels_keep(i));
end
T = T(keep, :);

if isempty(T)
    % Write empty union file for reproducibility
    U = table(string.empty(0,1), zeros(0,1), zeros(0,1), ...
        'VariableNames', {'label','start_sec','end_sec'});
    ensure_outdir(out_tsv);
    writetable(U, out_tsv, 'FileType','text', 'Delimiter','\t');
    if write_qc
        write_qc_summary(in_tsv, out_tsv, 0, 0, 0, 0, 0);
    end
    fprintf('No intervals to merge -> wrote empty union TSV: %s\n', out_tsv);
    return;
end

% Sanitize intervals: ensure start <= end, drop NaNs
s = T.start_sec;
e = T.end_sec;
good = ~(isnan(s) | isnan(e));
s = s(good); e = e(good);

swap = s > e;
tmp = s(swap); s(swap) = e(swap); e(swap) = tmp;

% Sort by start time
[ss, idx] = sort(s);
ee = e(idx);

% Merge
merged_s = [];
merged_e = [];

cur_s = ss(1);
cur_e = ee(1);

for i = 2:numel(ss)
    if ss(i) <= (cur_e + tol)
        % overlap or within tolerance gap
        cur_e = max(cur_e, ee(i));
    else
        merged_s(end+1,1) = cur_s; %#ok<AGROW>
        merged_e(end+1,1) = cur_e; %#ok<AGROW>
        cur_s = ss(i);
        cur_e = ee(i);
    end
end
merged_s(end+1,1) = cur_s;
merged_e(end+1,1) = cur_e;

% Drop too-short merged intervals if requested
dur = merged_e - merged_s;
keep2 = dur >= min_dur;
merged_s = merged_s(keep2);
merged_e = merged_e(keep2);

% Write union table
U = table(repmat("excl_union", numel(merged_s), 1), merged_s, merged_e, ...
    'VariableNames', {'label','start_sec','end_sec'});

ensure_outdir(out_tsv);
writetable(U, out_tsv, 'FileType','text', 'Delimiter','\t');

% QC summary
n_in = height(readtable(in_tsv,'FileType','text','Delimiter','\t'));
n_used = numel(ss);
n_out = height(U);
total_in = sum(ee - ss);
total_out = sum(merged_e - merged_s);

if write_qc
    write_qc_summary(in_tsv, out_tsv, n_in, n_used, n_out, total_in, total_out);
end

fprintf('Merged %d intervals (used=%d) -> %d union intervals: %s\n', n_in, n_used, n_out, out_tsv);

end

% ------------ helpers ------------
function ensure_outdir(pathstr)
d = fileparts(pathstr);
if ~isempty(d) && ~exist(d,'dir')
    mkdir(d);
end
end

function write_qc_summary(in_tsv, out_tsv, n_in, n_used, n_out, total_in, total_out)
[odir, obase, ~] = fileparts(out_tsv);
qc_csv = fullfile(odir, sprintf('%s_qc.csv', obase));
S = table(string(in_tsv), string(out_tsv), n_in, n_used, n_out, total_in, total_out, ...
    'VariableNames', {'input_tsv','union_tsv','n_rows_input','n_rows_used','n_union','total_dur_input_sec','total_dur_union_sec'});
writetable(S, qc_csv);
end
