function QC = r01_qc_v3_sign_convention_parcelpc(outDir, varargin)
%R01_QC_V3_SIGN_CONVENTION_PARCELPC (v3-compatible, timecourse-based sign fix)
% Randomly samples parcels per run and verifies exporter-consistent PC1 sign.
%
% IMPORTANT: v3 exporter applies sign convention on the *timecourse* (PC1),
% not on the eigenvector. This QC mirrors that exactly.
%
% Outputs in outDir\qc_v3_sign\:
%  - qc_sign_v3_summary.csv
%  - qc_sign_v3_details.csv

ip = inputParser;
ip.addRequired('outDir', @(x)ischar(x)||isstring(x));
ip.addParameter('QCSubdir', "qc_v3_sign", @(x)ischar(x)||isstring(x));
ip.addParameter('PreferGNORM', true, @(x)islogical(x)&&isscalar(x));
ip.addParameter('ParcelsPerRun', 25, @(x)isnumeric(x)&&isscalar(x)&&x>=1);
ip.addParameter('RandomSeed', 1, @(x)isnumeric(x)&&isscalar(x));
ip.addParameter('CorrThr', 0.99, @(x)isnumeric(x)&&isscalar(x));
ip.addParameter('Verbose', true, @(x)islogical(x)&&isscalar(x));
ip.addParameter('Overwrite', true, @(x)islogical(x)&&isscalar(x));
ip.parse(outDir, varargin{:});
opt = ip.Results;

outDir = char(outDir);
qcDir = fullfile(outDir, char(opt.QCSubdir));
if ~exist(qcDir,'dir'), mkdir(qcDir); end

outSummary = fullfile(qcDir, 'qc_sign_v3_summary.csv');
outDetail  = fullfile(qcDir, 'qc_sign_v3_details.csv');

if ~opt.Overwrite && exist(outSummary,'file') && exist(outDetail,'file')
    QC.summary = readtable(outSummary);
    QC.details = readtable(outDetail);
    QC.qcDir = qcDir;
    return;
end

% Choose mats
gnormFiles = dir(fullfile(outDir, '*_parcelPC_gnorm.mat'));
rawFiles   = dir(fullfile(outDir, '*_parcelPC_raw.mat'));
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

    % Required pointers
    if ~isfield(d,'resFile') || ~exist(d.resFile,'file')
        warning('Missing resFile for %s. Skipping.', runTag); continue;
    end
    if ~isfield(d,'scoutMat') || ~exist(d.scoutMat,'file')
        warning('Missing scoutMat for %s. Skipping.', runTag); continue;
    end
    if ~isfield(d,'eegSetFile') || ~exist(d.eegSetFile,'file')
        warning('Missing eegSetFile for %s. Skipping.', runTag); continue;
    end

    % Sign convention used by exporter
    signConvention = "maxabs";
    if isfield(d,'SignConvention') && ~isempty(d.SignConvention)
        signConvention = lower(string(d.SignConvention));
    end

    % Kernel
    R = load(d.resFile, 'ImagingKernel','GridLoc');
    K = double(R.ImagingKernel);                    % [nRowsKernel x nChan]
    [nRowsKernel, nChanK] = size(K);
    nVert = size(R.GridLoc,1);

    if mod(nRowsKernel, nVert) ~= 0
        warning('Kernel rows not divisible by nVert for %s. Skipping.', runTag); continue;
    end
    rows_per_vertex = nRowsKernel / nVert;

    % Scouts
    A = load(d.scoutMat, 'Scouts');
    Scouts = A.Scouts;
    nParc = numel(Scouts);

    % === matfile vars -> load into memory before indexing ===
    vmask = logical(M.valid_parcel_mask); vmask = vmask(:);
    pn = M.parcelNames; pn = pn(:);
    nv = double(M.n_vertices); nv = nv(:);
    nr = double(M.n_rows);     nr = nr(:);

    if numel(vmask) ~= nParc || numel(pn) ~= nParc
        warning('Mask/names mismatch for %s. Skipping.', runTag); continue;
    end

    validIdx = find(vmask);
    if isempty(validIdx)
        warning('No valid parcels for %s. Skipping.', runTag); continue;
    end

    nPick = min(opt.ParcelsPerRun, numel(validIdx));
    pick = validIdx(randperm(numel(validIdx), nPick));

    % EEG
    [eegPath, eegName, eegExt] = fileparts(char(d.eegSetFile));
    EEG = pop_loadset('filename', [eegName eegExt], 'filepath', eegPath);
    if size(EEG.data,1) ~= nChanK
        warning('Channel mismatch for %s. Skipping.', runTag); continue;
    end
    F = double(EEG.data);
    F = F - mean(F,2);
    nTime = size(F,2);

    % Sensor covariance
    Sxx = (F * F.') / max(1,(nTime-1));

    corrVals = [];
    nTested = 0;
    nPass = 0;
    nRawNegHigh = 0;

    for ii = 1:numel(pick)
        p = pick(ii);

        v = double(Scouts(p).Vertices(:));
        v = v(isfinite(v) & v>=1 & v<=nVert);
        v = unique(v);
        if numel(v) < 5, continue; end

        % vertices -> kernel rows
        dipIdx = zeros(numel(v)*rows_per_vertex, 1);
        kk = 1;
        for j = 1:numel(v)
            base = (v(j)-1)*rows_per_vertex;
            for o = 1:rows_per_vertex
                dipIdx(kk) = base + o;
                kk = kk + 1;
            end
        end
        dipIdx = dipIdx(dipIdx>=1 & dipIdx<=nRowsKernel);
        if isempty(dipIdx), continue; end

        Kp = K(dipIdx,:);

        Cp = Kp * Sxx * Kp.';
        Cp = 0.5*(Cp + Cp.');
        trCp = trace(Cp);
        if ~(isfinite(trCp) && trCp > 0), continue; end

        % top eigenvector
        try
            [V1,~] = eigs(Cp, 1, 'largestreal');
            w1 = real(V1(:,1));
        catch
            [Vfull,Dfull] = eig(Cp);
            lam = real(diag(Dfull));
            [~,ord] = sort(lam,'descend');
            w1 = real(Vfull(:,ord(1)));
        end

        % raw timecourse from w1
        pc_raw = ((w1.' * Kp) * F).';

        % apply exporter-style sign convention ON TIMECOURSE
        [pc_tcfix, tcflip, tref, vpre] = tc_sign_fix(pc_raw, signConvention);

        % stored PC1
        pc_stored = double(M.PC1(:,p));

        r_tc = fast_corr(pc_stored, pc_tcfix);
        r_raw = fast_corr(pc_stored, pc_raw);
        r_neg_raw = fast_corr(pc_stored, -pc_raw);

        pass = (r_tc >= opt.CorrThr);

        nTested = nTested + 1;
        nPass = nPass + double(pass);
        corrVals(end+1,1) = r_tc; %#ok<AGROW>

        if isfinite(r_neg_raw) && r_neg_raw >= opt.CorrThr
            nRawNegHigh = nRawNegHigh + 1;
        end

        detailRows(end+1,:) = { ...
            char(runTag), char(mode), double(p), string(pn{p}), true, ...
            nv(p), nr(p), ...
            double(r_tc), double(r_raw), double(r_neg_raw), logical(pass), ...
            double(tcflip), double(tref), double(vpre) ...
        };
    end

    if nTested == 0
        warning('No parcels tested for %s. Skipping summary row.', runTag);
        continue;
    end

    cf = corrVals(isfinite(corrVals));
    passRate = nPass / nTested;

    summaryRows(end+1,:) = { ...
        char(runTag), char(mode), double(sum(vmask)), double(nTested), double(nPass), double(passRate), ...
        double(median(cf)), double(min(cf)), double(prctile(cf,10)), ...
        double(nRawNegHigh / nTested) ...
    };

    if opt.Verbose
        fprintf('  Tested=%d | Pass=%d | PassRate=%.3f | CorrTCfix median=%.4f min=%.4f\n', ...
            nTested, nPass, passRate, median(cf), min(cf));
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
tref = NaN; vpre = NaN;

mode = lower(char(mode));
if strcmp(mode,'none')
    return;
end

[~,tref] = max(abs(x));
vpre = x(tref);
if isfinite(vpre) && vpre < 0
    x = -x;
    flipped = 1;
end
end

function r = fast_corr(a,b)
a = double(a(:)); b = double(b(:));
m = isfinite(a) & isfinite(b);
a = a(m); b = b(m);
if numel(a) < 3, r = NaN; return; end
a = a - mean(a);
b = b - mean(b);
den = sqrt(sum(a.^2) * sum(b.^2));
if den <= 0, r = NaN; else, r = (a'*b)/den; end
end
