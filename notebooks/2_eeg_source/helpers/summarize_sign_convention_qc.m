function QC = summarize_sign_convention_qc(outDir, varargin)
%SUMMARIZE_SIGN_CONVENTION_QC Check exporter-consistent parcel PC1 signs.
%
% What this helper does:
%   Samples valid parcels from each exported run, reconstructs the raw and
%   timecourse-sign-fixed parcel PC1 traces from the kernel and EEG data,
%   and confirms that the stored exporter output matches the preserved sign
%   convention logic.
%
% When it is used:
%   Called by `run_eeg_parcel_export_qc_summaries.m`.
%
% Key inputs:
%   - `outDir`: Stage-2 parcel-output directory
%   - optional QC settings preserved from the original helper
%
% Key outputs:
%   Writes:
%   - `qc_v3_sign/qc_sign_v3_summary.csv`
%   - `qc_v3_sign/qc_sign_v3_details.csv`

ip = inputParser;
ip.addRequired('outDir', @(x) ischar(x) || isstring(x));
ip.addParameter('QCSubdir', "qc_v3_sign", @(x) ischar(x) || isstring(x));
ip.addParameter('PreferGNORM', true, @(x) islogical(x) && isscalar(x));
ip.addParameter('ParcelsPerRun', 25, @(x) isnumeric(x) && isscalar(x) && x >= 1);
ip.addParameter('RandomSeed', 1, @(x) isnumeric(x) && isscalar(x));
ip.addParameter('CorrThr', 0.99, @(x) isnumeric(x) && isscalar(x));
ip.addParameter('Verbose', true, @(x) islogical(x) && isscalar(x));
ip.addParameter('Overwrite', true, @(x) islogical(x) && isscalar(x));
ip.parse(outDir, varargin{:});
opt = ip.Results;

outDir = char(outDir);
qcDir = fullfile(outDir, char(opt.QCSubdir));
if ~exist(qcDir, 'dir')
    mkdir(qcDir);
end

outSummary = fullfile(qcDir, 'qc_sign_v3_summary.csv');
outDetail = fullfile(qcDir, 'qc_sign_v3_details.csv');

if ~opt.Overwrite && exist(outSummary, 'file') && exist(outDetail, 'file')
    QC.summary = readtable(outSummary);
    QC.details = readtable(outDetail);
    QC.qcDir = qcDir;
    return;
end

gnormFiles = dir(fullfile(outDir, '*_parcelPC_gnorm.mat'));
rawFiles = dir(fullfile(outDir, '*_parcelPC_raw.mat'));
if opt.PreferGNORM && ~isempty(gnormFiles)
    matFiles = gnormFiles;
    mode = "gnorm";
elseif ~isempty(rawFiles)
    matFiles = rawFiles;
    mode = "raw";
else
    error('No *_parcelPC_gnorm.mat or *_parcelPC_raw.mat found in: %s', outDir);
end

rng(opt.RandomSeed);

detailRows = {};
detailVar = { ...
    'runTag','mode','parcel_idx','parcel_name','valid', ...
    'n_vertices','n_rows', ...
    'corr_tcfix','corr_raw','corr_neg_raw','pass', ...
    'tcfix_flipped','tcfix_ref_t','tcfix_ref_val_pre' ...
};

summaryRows = {};
summaryVar = { ...
    'runTag','mode','n_valid_total','n_tested','n_pass','pass_rate', ...
    'corr_tcfix_median','corr_tcfix_min','corr_tcfix_p10', ...
    'frac_raw_negcorr_gt_thr' ...
};

for kf = 1:numel(matFiles)
    matPath = fullfile(matFiles(kf).folder, matFiles(kf).name);
    M = matfile(matPath);

    d = M.diagOut;
    runTag = string(d.runTag);

    if opt.Verbose
        fprintf('\n[%d/%d] QC sign v3: %s (mode=%s)\n', kf, numel(matFiles), runTag, mode);
    end

    if ~isfield(d, 'resFile') || ~exist(d.resFile, 'file')
        warning('Missing resFile for %s. Skipping.', runTag);
        continue;
    end
    if ~isfield(d, 'scoutMat') || ~exist(d.scoutMat, 'file')
        warning('Missing scoutMat for %s. Skipping.', runTag);
        continue;
    end
    if ~isfield(d, 'eegSetFile') || ~exist(d.eegSetFile, 'file')
        warning('Missing eegSetFile for %s. Skipping.', runTag);
        continue;
    end

    signConvention = "maxabs";
    if isfield(d, 'SignConvention') && ~isempty(d.SignConvention)
        signConvention = lower(string(d.SignConvention));
    end

    kernelData = load(d.resFile, 'ImagingKernel', 'GridLoc');
    K = double(kernelData.ImagingKernel);
    [nRowsKernel, nChanK] = size(K);
    nVert = size(kernelData.GridLoc, 1);

    if mod(nRowsKernel, nVert) ~= 0
        warning('Kernel rows not divisible by nVert for %s. Skipping.', runTag);
        continue;
    end
    rows_per_vertex = nRowsKernel / nVert;

    atlasData = load(d.scoutMat, 'Scouts');
    Scouts = atlasData.Scouts;
    nParc = numel(Scouts);

    validMask = logical(M.valid_parcel_mask);
    validMask = validMask(:);
    parcelNames = M.parcelNames;
    parcelNames = parcelNames(:);
    nVertices = double(M.n_vertices);
    nVertices = nVertices(:);
    nRows = double(M.n_rows);
    nRows = nRows(:);

    if numel(validMask) ~= nParc || numel(parcelNames) ~= nParc
        warning('Mask/names mismatch for %s. Skipping.', runTag);
        continue;
    end

    validIdx = find(validMask);
    if isempty(validIdx)
        warning('No valid parcels for %s. Skipping.', runTag);
        continue;
    end

    nPick = min(opt.ParcelsPerRun, numel(validIdx));
    pick = validIdx(randperm(numel(validIdx), nPick));

    [eegPath, eegName, eegExt] = fileparts(char(d.eegSetFile));
    EEG = pop_loadset('filename', [eegName eegExt], 'filepath', eegPath);
    if size(EEG.data, 1) ~= nChanK
        warning('Channel mismatch for %s. Skipping.', runTag);
        continue;
    end
    F = double(EEG.data);
    F = F - mean(F, 2);
    nTime = size(F, 2);

    Sxx = (F * F.') / max(1, (nTime - 1));

    corrVals = [];
    nTested = 0;
    nPass = 0;
    nRawNegHigh = 0;

    for ii = 1:numel(pick)
        p = pick(ii);

        vertices = double(Scouts(p).Vertices(:));
        vertices = vertices(isfinite(vertices) & vertices >= 1 & vertices <= nVert);
        vertices = unique(vertices);
        if numel(vertices) < 5
            continue;
        end

        dipIdx = zeros(numel(vertices) * rows_per_vertex, 1);
        kk = 1;
        for j = 1:numel(vertices)
            base = (vertices(j) - 1) * rows_per_vertex;
            for o = 1:rows_per_vertex
                dipIdx(kk) = base + o;
                kk = kk + 1;
            end
        end
        dipIdx = dipIdx(dipIdx >= 1 & dipIdx <= nRowsKernel);
        if isempty(dipIdx)
            continue;
        end

        Kp = K(dipIdx, :);

        Cp = Kp * Sxx * Kp.';
        Cp = 0.5 * (Cp + Cp.');
        trCp = trace(Cp);
        if ~(isfinite(trCp) && trCp > 0)
            continue;
        end

        try
            [V1, ~] = eigs(Cp, 1, 'largestreal');
            w1 = real(V1(:, 1));
        catch
            [Vfull, Dfull] = eig(Cp);
            lam = real(diag(Dfull));
            [~, ord] = sort(lam, 'descend');
            w1 = real(Vfull(:, ord(1)));
        end

        pc_raw = ((w1.' * Kp) * F).';
        [pc_tcfix, tcflip, tref, vpre] = tc_sign_fix(pc_raw, signConvention);
        pc_stored = double(M.PC1(:, p));

        r_tc = fast_corr(pc_stored, pc_tcfix);
        r_raw = fast_corr(pc_stored, pc_raw);
        r_neg_raw = fast_corr(pc_stored, -pc_raw);

        pass = (r_tc >= opt.CorrThr);

        nTested = nTested + 1;
        nPass = nPass + double(pass);
        corrVals(end+1, 1) = r_tc; %#ok<AGROW>

        if isfinite(r_neg_raw) && r_neg_raw >= opt.CorrThr
            nRawNegHigh = nRawNegHigh + 1;
        end

        detailRows(end+1, :) = { ...
            char(runTag), char(mode), double(p), string(parcelNames{p}), true, ...
            nVertices(p), nRows(p), ...
            double(r_tc), double(r_raw), double(r_neg_raw), logical(pass), ...
            double(tcflip), double(tref), double(vpre) ...
        }; %#ok<AGROW>
    end

    if nTested == 0
        warning('No parcels tested for %s. Skipping summary row.', runTag);
        continue;
    end

    finiteCorr = corrVals(isfinite(corrVals));
    passRate = nPass / nTested;

    summaryRows(end+1, :) = { ...
        char(runTag), char(mode), double(sum(validMask)), double(nTested), double(nPass), double(passRate), ...
        double(median(finiteCorr)), double(min(finiteCorr)), double(prctile(finiteCorr, 10)), ...
        double(nRawNegHigh / nTested) ...
    }; %#ok<AGROW>

    if opt.Verbose
        fprintf('  Tested=%d | Pass=%d | PassRate=%.3f | CorrTCfix median=%.4f min=%.4f\n', ...
            nTested, nPass, passRate, median(finiteCorr), min(finiteCorr));
    end
end

QC.details = cell2table(detailRows, 'VariableNames', detailVar);
QC.summary = cell2table(summaryRows, 'VariableNames', summaryVar);
QC.qcDir = qcDir;

writetable(QC.summary, outSummary);
writetable(QC.details, outDetail);

fprintf('\nWrote:\n  %s\n  %s\n', outSummary, outDetail);

end

function [x, flipped, tref, vpre] = tc_sign_fix(x, mode)
x = double(x(:));
flipped = 0;
tref = NaN;
vpre = NaN;

mode = lower(char(mode));
if strcmp(mode, 'none')
    return;
end

[~, tref] = max(abs(x));
vpre = x(tref);
if isfinite(vpre) && vpre < 0
    x = -x;
    flipped = 1;
end
end

function r = fast_corr(a, b)
a = double(a(:));
b = double(b(:));
mask = isfinite(a) & isfinite(b);
a = a(mask);
b = b(mask);
if numel(a) < 3
    r = NaN;
    return;
end
a = a - mean(a);
b = b - mean(b);
den = sqrt(sum(a.^2) * sum(b.^2));
if den <= 0
    r = NaN;
else
    r = (a' * b) / den;
end
end
