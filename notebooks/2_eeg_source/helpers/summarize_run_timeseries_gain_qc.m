function T = summarize_run_timeseries_gain_qc(outDir, varargin)
%SUMMARIZE_RUN_TIMESERIES_GAIN_QC Build run-level parcel timeseries QC.
%
% What this helper does:
%   Loads the Stage-2 parcel export MAT files, computes per-run timeseries
%   summaries for PC1, and writes the gain/timeseries QC CSV used by the
%   public Stage-2 QC notebook.
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
%   - `qc_v3/qc_run_timeseries_gain_summary.csv`
%
% Important note:
%   This preserves the current preference for gain-normalized MAT files and
%   the current batch summary columns exactly.

ip = inputParser;
ip.addRequired('outDir', @(x) ischar(x) || isstring(x));
ip.addParameter('QCSubdir', "qc_v3", @(x) ischar(x) || isstring(x));
ip.addParameter('PreferGNORM', true, @(x) islogical(x) && isscalar(x));
ip.addParameter('ChunkRows', 20000, @(x) isnumeric(x) && isscalar(x) && x >= 1000);
ip.addParameter('Verbose', true, @(x) islogical(x) && isscalar(x));
ip.parse(outDir, varargin{:});
opt = ip.Results;

outDir = char(outDir);
qcDir = fullfile(outDir, char(opt.QCSubdir));
if ~exist(qcDir, 'dir')
    mkdir(qcDir);
end
outCsv = fullfile(qcDir, 'qc_run_timeseries_gain_summary.csv');

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

rows = {};
vars = { ...
    'runTag','mode','matFile', ...
    'nTime','nParc', ...
    'nan_frac_pc1', ...
    'pc1_std_median','pc1_std_p05','pc1_std_p95', ...
    'pc1_maxabs', ...
    'valid_parcels', ...
    'rows_per_vertex', ...
    'eeg_chStd_median','kVertNorm_median', ...
    'gnorm_scale_to_med_kgain','gnorm_ref_med_kgain','gnorm_basis' ...
};

for i = 1:numel(matFiles)
    matPath = fullfile(matFiles(i).folder, matFiles(i).name);
    M = matfile(matPath);

    try
        d = M.diagOut;
        runTag = string(d.runTag);
    catch
        runTag = erase(string(matFiles(i).name), "_parcelPC_" + mode + ".mat");
    end

    [nTime, nParc] = size(M, 'PC1');

    try
        validMask = logical(M.valid_parcel_mask(:));
        nValid = sum(validMask);
    catch
        validMask = true(nParc, 1);
        nValid = nParc;
    end

    chunk = double(opt.ChunkRows);
    n = zeros(1, nParc);
    mu = zeros(1, nParc);
    M2 = zeros(1, nParc);

    nanCount = 0;
    totalCount = 0;
    maxabs = 0;

    for r0 = 1:chunk:nTime
        r1 = min(nTime, r0 + chunk - 1);
        X = double(M.PC1(r0:r1, :));
        totalCount = totalCount + numel(X);
        nanCount = nanCount + sum(isnan(X(:)));

        finiteX = X(~isnan(X));
        if ~isempty(finiteX)
            maxabs = max(maxabs, max(abs(finiteX)));
        end

        for c = 1:nParc
            x = X(:, c);
            finiteMask = isfinite(x);
            x = x(finiteMask);
            if isempty(x)
                continue;
            end
            for t = 1:numel(x)
                n(c) = n(c) + 1;
                delta = x(t) - mu(c);
                mu(c) = mu(c) + delta / n(c);
                delta2 = x(t) - mu(c);
                M2(c) = M2(c) + delta * delta2;
            end
        end
    end

    varc = M2 ./ max(n - 1, 1);
    stdc = sqrt(varc);
    stdc(~isfinite(stdc)) = NaN;

    s = stdc;
    s(~validMask') = NaN;

    finiteS = s(isfinite(s));
    pc1_std_median = median(s, 'omitnan');
    pc1_std_p05 = prctile(finiteS, 5);
    pc1_std_p95 = prctile(finiteS, 95);

    nan_frac = nanCount / max(totalCount, 1);

    rows_per_vertex = NaN;
    eeg_chStd_median = NaN;
    kVertNorm_median = NaN;
    gnorm_scale = NaN;
    gnorm_ref = NaN;
    gnorm_basis = "";

    try
        d = M.diagOut;
        if isfield(d, 'rows_per_vertex'), rows_per_vertex = double(d.rows_per_vertex); end
        if isfield(d, 'eeg_chStd_median'), eeg_chStd_median = double(d.eeg_chStd_median); end
        if isfield(d, 'kVertNorm_median'), kVertNorm_median = double(d.kVertNorm_median); end
        if isfield(d, 'gnorm_scale_to_med_kgain'), gnorm_scale = double(d.gnorm_scale_to_med_kgain); end
        if isfield(d, 'gnorm_ref_med_kgain'), gnorm_ref = double(d.gnorm_ref_med_kgain); end
        if isfield(d, 'gnorm_basis'), gnorm_basis = string(d.gnorm_basis); end
    catch
    end

    if opt.Verbose
        fprintf('[%d/%d] %s | mode=%s | pc1_std_med=%.4g | nan=%.4g\n', ...
            i, numel(matFiles), runTag, mode, pc1_std_median, nan_frac);
    end

    rows(end+1, :) = { ...
        char(runTag), char(mode), matPath, ...
        nTime, nParc, ...
        nan_frac, ...
        pc1_std_median, pc1_std_p05, pc1_std_p95, ...
        maxabs, ...
        nValid, ...
        rows_per_vertex, ...
        eeg_chStd_median, kVertNorm_median, ...
        gnorm_scale, gnorm_ref, char(gnorm_basis) ...
    }; 
end

T = cell2table(rows, 'VariableNames', vars);
medStd = median(T.pc1_std_median, 'omitnan');
T.pc1_std_ratio_vs_run_median = T.pc1_std_median / max(medStd, eps);

writetable(T, outCsv);
fprintf('Wrote: %s\n', outCsv);

end
