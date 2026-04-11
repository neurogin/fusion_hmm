function r01_export_bst_exclusions_Fevents(rawlink_mat, out_tsv)

% r01_export_bst_exclusions_Fevents
%
% Export Brainstorm interval events from F(1).events for labels:
%   BAD, boundary, bad_boundary
%
% Writes TSV with columns:
%   label   start_sec   end_sec   source
%
% Notes:
% - Brainstorm interval events are typically stored in ev(i).times as 2xN:
%       [start_times; end_times]
%   For a single interval, this is 2x1 (NOT a point vector).
% - Point events can be stored as 1xN; we convert them to zero-length intervals (t;t).
% - Some versions may store intervals as Nx2; we transpose to 2xN.

if ~exist(rawlink_mat, 'file')
    error('File not found: %s', rawlink_mat);
end

% Load only variable 'F' (per your inspection output)
S = load(rawlink_mat, 'F');
if ~isfield(S,'F') || isempty(S.F)
    error('Variable F not found or empty in: %s', rawlink_mat);
end

F = S.F;
if numel(F) > 1
    F = F(1);
end

if ~isfield(F,'events') || isempty(F.events)
    error('F(1).events not found or empty in: %s', rawlink_mat);
end

ev = F.events;
want = ["BAD","boundary","bad_boundary"];

rows = {};
for i = 1:numel(ev)
    lab = string(ev(i).label);
    if ~any(strcmpi(lab, want))
        continue;
    end

    t = ev(i).times;
    if isempty(t)
        continue;
    end

    % ----------------------------
    % Normalize 't' to 2 x N intervals
    % ----------------------------
    if size(t,1) == 2
        % Already 2xN interval format (includes 2x1 single-interval case)
        % do nothing

    elseif size(t,1) == 1
        % 1xN point events -> convert to 2xN zero-length intervals
        t = [t; t];

    elseif size(t,2) == 2
        % Nx2 intervals -> transpose to 2xN
        t = t';

    else
        warning('Skipping "%s": unexpected times shape [%d x %d]', lab, size(t,1), size(t,2));
        continue;
    end

    % Final sanity check
    if size(t,1) ~= 2
        warning('Skipping "%s": normalization failed, got shape [%d x %d]', lab, size(t,1), size(t,2));
        continue;
    end

    % Append rows
    for k = 1:size(t,2)
        rows(end+1,:) = {char(lab), t(1,k), t(2,k), 'brainstorm'}; %#ok<AGROW>
    end
end

% Write output
if isempty(rows)
    warning('No matching events (BAD/boundary/bad_boundary) found in %s', rawlink_mat);
    % Still write an empty table for reproducibility
    T = cell2table(cell(0,4), 'VariableNames', {'label','start_sec','end_sec','source'});
else
    T = cell2table(rows, 'VariableNames', {'label','start_sec','end_sec','source'});
end

% Ensure output directory exists
out_dir = fileparts(out_tsv);
if ~isempty(out_dir) && ~exist(out_dir,'dir')
    mkdir(out_dir);
end

writetable(T, out_tsv, 'FileType','text', 'Delimiter','\t');
fprintf('Exported %d intervals -> %s\n', height(T), out_tsv);

end
