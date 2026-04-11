function r01_eeg_iclabel_prune_and_metadata(root_raw_eeglab, out_base_dir, qc_table_dir, varargin)
% R01 EEG: ICLabel-based pruning + metadata export.
%
% Two supported policies:
%   (A) "brain_threshold": keep ICs with Brain >= threshold
%   (B) "reject_artifacts": drop ICs if max(reject_classes) >= reject_threshold
%
% Outputs:
%   <out_base_dir>\withICA\<stem>_desc-<TAG>_withICA.set
%   <out_base_dir>\clean\<stem>_desc-<TAG>_clean.set
%   <qc_table_dir>\eeg_ic_prune_summary_<TAG>.csv
%   <qc_table_dir>\eeg_ic_prune_lists_<TAG>.csv
%   <qc_table_dir>\<stem>_iclabel_table_<TAG>.csv
%
% Notes:
% - Designed to be robust with MATLAB strings by converting to char before EEGLAB I/O.
% - "withICA" retains ICA + ICLabel for traceability.
% - "clean" is Brainstorm-friendly (ICA fields cleared; channel data reconstructed).

% -----------------------
% Parse inputs
% -----------------------
p = inputParser;
p.addParameter('ic_policy', 'brain_threshold', @(x) ischar(x) || isstring(x));
p.addParameter('threshold', 0.70, @(x) isnumeric(x) && isscalar(x) && x>=0 && x<=1); % brain_threshold
p.addParameter('reject_threshold', 0.70, @(x) isnumeric(x) && isscalar(x) && x>=0 && x<=1); % reject_artifacts
p.addParameter('reject_classes', ["Eye","Muscle","Heart","LineNoise","ChannelNoise","Other"], @(x) ischar(x) || isstring(x) || iscellstr(x));
p.addParameter('overwrite', false, @(x) islogical(x) && isscalar(x));
p.addParameter('save_iclabel_tables', true, @(x) islogical(x) && isscalar(x));
p.addParameter('save_withICA', true, @(x) islogical(x) && isscalar(x));
p.addParameter('save_clean', true, @(x) islogical(x) && isscalar(x));
p.addParameter('clean_remove_iclabel', true, @(x) islogical(x) && isscalar(x));
p.addParameter('file_pattern', '*.set', @(x) ischar(x) || isstring(x));
p.parse(varargin{:});

ic_policy = string(p.Results.ic_policy);
thr_brain = p.Results.threshold;
thr_rej   = p.Results.reject_threshold;
rej_classes = string(p.Results.reject_classes);
overwrite = p.Results.overwrite;
save_iclabel_tables = p.Results.save_iclabel_tables;
save_withICA = p.Results.save_withICA;
save_clean = p.Results.save_clean;
clean_remove_iclabel = p.Results.clean_remove_iclabel;
file_pattern = char(p.Results.file_pattern);

% -----------------------
% EEGLAB / ICLabel checks
% -----------------------
if exist('pop_loadset','file') ~= 2
    error('EEGLAB not on path. Add EEGLAB to MATLAB path first.');
end
if exist('iclabel','file') ~= 2
    error('ICLabel plugin not found (iclabel.m). Install ICLabel and add it to EEGLAB plugins.');
end

% -----------------------
% Force char paths (EEGLAB compatibility)
% -----------------------
root_raw_eeglab = char(root_raw_eeglab);
out_base_dir    = char(out_base_dir);
qc_table_dir    = char(qc_table_dir);

% Output subfolders
out_dir_withICA = fullfile(out_base_dir, 'withICA');
out_dir_clean   = fullfile(out_base_dir, 'clean');

ensure_dir(out_base_dir);
ensure_dir(out_dir_withICA);
ensure_dir(out_dir_clean);
ensure_dir(qc_table_dir);

% -----------------------
% Determine tag for filenames
% -----------------------
switch lower(ic_policy)
    case "brain_threshold"
        tag = sprintf('ICBrain%02d', round(100 * thr_brain));
    case "reject_artifacts"
        tag = sprintf('ICRej%02d', round(100 * thr_rej));
    otherwise
        error('Unknown ic_policy: %s (expected "brain_threshold" or "reject_artifacts")', ic_policy);
end

% -----------------------
% Find input .set files
% -----------------------
set_files = dir(fullfile(root_raw_eeglab, '**', file_pattern));
if isempty(set_files)
    error('No .set files found under: %s (pattern=%s)', root_raw_eeglab, file_pattern);
end

% -----------------------
% QC tables containers
% -----------------------
summary_rows = {}; % run-level
list_rows    = {}; % keep/drop lists

% ICLabel class order
class_names = ["Brain","Muscle","Eye","Heart","LineNoise","ChannelNoise","Other"];

% Map reject classes to indices (for reject_artifacts policy)
rej_idx = [];
if lower(ic_policy) == "reject_artifacts"
    rej_idx = map_iclabel_classes_to_idx(rej_classes, class_names);
end

% -----------------------
% Main loop
% -----------------------
for i = 1:numel(set_files)
    in_set = fullfile(set_files(i).folder, set_files(i).name);

    [sub, ses, run] = parse_bids_like_ids(set_files(i).name);
    stem = make_stem(sub, ses, run);

    fprintf('\n[%d/%d] Loading: %s\n', i, numel(set_files), in_set);

    EEG = pop_loadset('filename', set_files(i).name, 'filepath', set_files(i).folder);
    EEG = eeg_checkset(EEG);

    dur_sec = EEG.pnts / EEG.srate;
    nbchan  = EEG.nbchan;
    srate   = EEG.srate;

    bcount_before = count_boundary_events(EEG);

    hasICA = isfield(EEG,'icaweights') && ~isempty(EEG.icaweights) && ...
             isfield(EEG,'icawinv')    && ~isempty(EEG.icawinv)    && ...
             isfield(EEG,'icasphere')  && ~isempty(EEG.icasphere);

    nIC = 0; nKeep = 0; nDrop = 0;
    keepListStr = ""; dropListStr = "";
    out_withICA_path = "";
    out_clean_path   = "";

    if ~hasICA
        warning('No ICA decomposition found for %s. Skipping pruning; exporting metadata only.', in_set);
    else
        % Ensure ICLabel classifications exist
        hasICLabel = isfield(EEG,'etc') && isfield(EEG.etc,'ic_classification') && ...
                     isfield(EEG.etc.ic_classification,'ICLabel') && ...
                     isfield(EEG.etc.ic_classification.ICLabel,'classifications');

        if ~hasICLabel
            fprintf('  Running ICLabel...\n');
            EEG = iclabel(EEG);
            EEG = eeg_checkset(EEG);
        end

        cls = EEG.etc.ic_classification.ICLabel.classifications; % nIC x 7
        if isempty(cls) || size(cls,2) < 7
            warning('ICLabel classifications unexpected for %s. Skipping pruning.', in_set);
            hasICA = false;
        else
            nIC = size(cls,1);

            % Decide keep/drop based on policy
            switch lower(ic_policy)
                case "brain_threshold"
                    brainProb = cls(:,1);
                    keep = find(brainProb >= thr_brain);
                    drop = setdiff(1:nIC, keep);

                case "reject_artifacts"
                    % Drop IC if ANY selected artifact/other class has prob >= thr_rej
                    % Equivalent to: max(cls(:, rej_idx), [], 2) >= thr_rej
                    art_block = cls(:, rej_idx);
                    art_max = max(art_block, [], 2);
                    drop = find(art_max >= thr_rej);
                    keep = setdiff(1:nIC, drop);

                otherwise
                    error('Unknown ic_policy: %s', ic_policy);
            end

            nKeep = numel(keep);
            nDrop = numel(drop);
            keepListStr = strjoin(string(keep), ',');
            dropListStr = strjoin(string(drop), ',');

            % Per-IC ICLabel table
            if save_iclabel_tables
                [artMaxVal, artMaxClass] = compute_artifact_max(cls, class_names, rej_idx);

                keepFlag = false(nIC,1); keepFlag(keep) = true;
                dropFlag = false(nIC,1); dropFlag(drop) = true;

                dropReason = strings(nIC,1);
                switch lower(ic_policy)
                    case "brain_threshold"
                        dropReason(dropFlag) = "Brain<thr";
                    case "reject_artifacts"
                        % For dropped ICs, annotate which class was max (within selected reject set)
                        dropReason(dropFlag) = artMaxClass(dropFlag) + ">=thr";
                end

                icTab = table((1:nIC)', cls(:,1), cls(:,2), cls(:,3), cls(:,4), cls(:,5), cls(:,6), cls(:,7), ...
                    keepFlag, dropFlag, dropReason, artMaxVal, artMaxClass, ...
                    'VariableNames', {'IC','Brain','Muscle','Eye','Heart','LineNoise','ChannelNoise','Other', ...
                                     'Keep','Drop','DropReason','ArtifactMax','ArtifactMaxClass'});

                icTabFile = fullfile(qc_table_dir, sprintf('%s_iclabel_table_%s.csv', stem, tag));
                writetable(icTab, icTabFile);
            end

            % Report
            switch lower(ic_policy)
                case "brain_threshold"
                    fprintf('  Pruning ICs: keep %d / %d (Brain >= %.2f)\n', nKeep, nIC, thr_brain);
                case "reject_artifacts"
                    fprintf('  Pruning ICs: keep %d / %d (Reject if max(%s) >= %.2f)\n', ...
                        nKeep, nIC, strjoin(rej_classes, '|'), thr_rej);
            end

            % Reconstruct EEG by removing dropped components
            if isempty(drop)
                EEG_pruned = EEG; % nothing to drop
            else
                EEG_pruned = pop_subcomp(EEG, drop, 0);
                EEG_pruned = eeg_checkset(EEG_pruned);
            end

            % Attach provenance
            EEG_pruned = attach_r01_provenance(EEG_pruned, in_set, ic_policy, thr_brain, thr_rej, rej_classes, keep, drop, tag);

            % Output base
            base = sprintf('%s_desc-%s', stem, tag);

            % -------- Save WITH-ICA (auditable) --------
            if save_withICA
                out_withICA_name = sprintf('%s_withICA.set', base);
                out_withICA_path = fullfile(out_dir_withICA, out_withICA_name);

                if exist(out_withICA_path,'file') && ~overwrite
                    fprintf('  withICA exists, not overwriting: %s\n', out_withICA_path);
                else
                    EEG_withICA = EEG_pruned;
                    EEG_withICA.setname = out_withICA_name;

                    % EEGLAB compatibility: enforce char scalars
                    out_withICA_name_c = char(out_withICA_name);
                    out_dir_withICA_c  = char(out_dir_withICA);
                    EEG_withICA.filename = out_withICA_name_c;
                    EEG_withICA.filepath = out_dir_withICA_c;

                    EEG_withICA = pop_saveset(EEG_withICA, 'filename', out_withICA_name_c, 'filepath', out_dir_withICA_c);
                    fprintf('  Saved withICA: %s\n', out_withICA_path);

                    b_after = count_boundary_events(EEG_withICA);
                    if b_after ~= bcount_before
                        warning('Boundary count changed after pruning (withICA) (%d -> %d). Inspect %s', bcount_before, b_after, out_withICA_name_c);
                    end
                end
            end

            % -------- Save CLEAN (Brainstorm-friendly) --------
            if save_clean
                out_clean_name = sprintf('%s_clean.set', base);
                out_clean_path = fullfile(out_dir_clean, out_clean_name);

                if exist(out_clean_path,'file') && ~overwrite
                    fprintf('  clean exists, not overwriting: %s\n', out_clean_path);
                else
                    EEG_clean = EEG_pruned;

                    % Clear ICA-related fields so dataset is plain channel EEG
                    EEG_clean = strip_ica_fields(EEG_clean, clean_remove_iclabel);

                    EEG_clean.setname = out_clean_name;
                    EEG_clean = eeg_checkset(EEG_clean);

                    % EEGLAB compatibility: enforce char scalars
                    out_clean_name_c = char(out_clean_name);
                    out_dir_clean_c  = char(out_dir_clean);
                    EEG_clean.filename = out_clean_name_c;
                    EEG_clean.filepath = out_dir_clean_c;

                    EEG_clean = pop_saveset(EEG_clean, 'filename', out_clean_name_c, 'filepath', out_dir_clean_c);
                    fprintf('  Saved clean:  %s\n', out_clean_path);

                    b_after = count_boundary_events(EEG_clean);
                    if b_after ~= bcount_before
                        warning('Boundary count changed after pruning (clean) (%d -> %d). Inspect %s', bcount_before, b_after, out_clean_name_c);
                    end
                end
            end
        end
    end

    % Run-level summary
    summary_rows(end+1,:) = { ...
        char(sub), char(ses), char(run), ...
        in_set, ...
        char(out_withICA_path), char(out_clean_path), ...
        srate, nbchan, dur_sec, ...
        bcount_before, ...
        nIC, nKeep, nDrop, ...
        char(ic_policy), ...
        tag, ...
        thr_brain, thr_rej, ...
        char(strjoin(rej_classes, '|')) ...
        }; %#ok<AGROW>

    list_rows(end+1,:) = { ...
        char(sub), char(ses), char(run), ...
        char(keepListStr), char(dropListStr) ...
        }; %#ok<AGROW>
end

% Write summary tables
summaryTab = cell2table(summary_rows, 'VariableNames', { ...
    'sub','ses','run', ...
    'input_set','derived_withICA_set','derived_clean_set', ...
    'srate_hz','n_channels','duration_sec', ...
    'n_boundary_events', ...
    'n_IC','n_IC_keep','n_IC_drop', ...
    'ic_policy','tag', ...
    'brain_threshold','reject_threshold','reject_classes'});

listTab = cell2table(list_rows, 'VariableNames', { ...
    'sub','ses','run','IC_keep_list','IC_drop_list'});

writetable(summaryTab, fullfile(qc_table_dir, sprintf('eeg_ic_prune_summary_%s.csv', tag)));
writetable(listTab,    fullfile(qc_table_dir, sprintf('eeg_ic_prune_lists_%s.csv',   tag)));

fprintf('\nDone. Wrote QC tables to: %s\n', qc_table_dir);

end

% ======================================================================
% Helpers
% ======================================================================

function ensure_dir(d)
d = char(d);
if ~exist(d, 'dir'); mkdir(d); end
end

function n = count_boundary_events(EEG)
n = 0;
if isfield(EEG,'event') && ~isempty(EEG.event)
    types = {EEG.event.type};
    for k = 1:numel(types)
        try
            if (ischar(types{k}) || isstring(types{k})) && strcmpi(string(types{k}), "boundary")
                n = n + 1;
            end
        catch
        end
    end
end
end

function [sub, ses, run] = parse_bids_like_ids(fname)
sub = ""; ses = ""; run = "";
tok = regexp(fname, '(sub-[A-Za-z0-9]+)', 'tokens', 'once'); if ~isempty(tok); sub = string(tok{1}); end
tok = regexp(fname, '(ses-[A-Za-z0-9]+)', 'tokens', 'once'); if ~isempty(tok); ses = string(tok{1}); end
tok = regexp(fname, '(run-[A-Za-z0-9]+)', 'tokens', 'once'); if ~isempty(tok); run = string(tok{1}); end
end

function stem = make_stem(sub, ses, run)
% Keep compatibility with your prior naming:
%   sub-XX_ses-YY[_run-ZZ]
parts = strings(0,1);
if strlength(sub) > 0; parts(end+1) = sub; end %#ok<AGROW>
if strlength(ses) > 0; parts(end+1) = ses; end %#ok<AGROW>
if strlength(run) > 0; parts(end+1) = run; end %#ok<AGROW>
if isempty(parts)
    error('Could not parse sub/ses from filename; add run token or adjust parse logic.');
end
stem = char(strjoin(parts, "_"));
end

function idx = map_iclabel_classes_to_idx(rej_classes, class_names)
idx = zeros(1,0);
for k = 1:numel(rej_classes)
    hit = find(strcmpi(class_names, rej_classes(k)));
    if isempty(hit)
        error('Unknown ICLabel class in reject_classes: %s', rej_classes(k));
    end
    idx(end+1) = hit; %#ok<AGROW>
end
idx = unique(idx);
end

function [artMaxVal, artMaxClass] = compute_artifact_max(cls, class_names, rej_idx)
% Compute max prob among selected reject classes; if rej_idx empty, do it across non-brain classes.
nIC = size(cls,1);
if isempty(rej_idx)
    rej_idx = 2:7;
end
block = cls(:, rej_idx);
[artMaxVal, j] = max(block, [], 2);
% Map j back to class index
classIdx = rej_idx(j);
artMaxClass = class_names(classIdx);
artMaxClass = reshape(string(artMaxClass), [nIC, 1]);
end

function EEG = attach_r01_provenance(EEG, in_set, ic_policy, thr_brain, thr_rej, rej_classes, keep, drop, tag)
if ~isfield(EEG,'etc') || isempty(EEG.etc)
    EEG.etc = struct();
end
EEG.etc.r01 = struct();
EEG.etc.r01.input_set = char(in_set);
EEG.etc.r01.ic_policy = char(ic_policy);
EEG.etc.r01.tag = char(tag);
EEG.etc.r01.brain_threshold = thr_brain;
EEG.etc.r01.reject_threshold = thr_rej;
EEG.etc.r01.reject_classes = char(strjoin(rej_classes, '|'));
EEG.etc.r01.keep_ic = keep(:)';
EEG.etc.r01.drop_ic = drop(:)';
end

function EEG = strip_ica_fields(EEG, remove_iclabel)
% Remove ICA matrices so dataset is plain channel EEG (data already reconstructed).
fields_to_clear = {'icaweights','icasphere','icawinv','icachansind','icaact'};
for i = 1:numel(fields_to_clear)
    fn = fields_to_clear{i};
    if isfield(EEG, fn); EEG.(fn) = []; end
end

% Remove ICLabel classification if requested
if remove_iclabel
    if isfield(EEG,'etc') && isfield(EEG.etc,'ic_classification')
        EEG.etc = rmfield_if_exists(EEG.etc, 'ic_classification');
    end
end

% Remove rejection vectors that reference components
if isfield(EEG,'reject')
    EEG.reject = rmfield_if_exists(EEG.reject, 'gcompreject');
end
end

function S = rmfield_if_exists(S, fieldname)
if isstruct(S) && isfield(S, fieldname)
    S = rmfield(S, fieldname);
end
end
