function T = r01_qc_v3_run_timeseries_and_gain_summary(outDir, varargin)
%R01_QC_V3_RUN_TIMESERIES_AND_GAIN_SUMMARY
% Run-level QC for v3 exporter outputs:
% - Loads *_parcelPC_gnorm.mat if present else *_parcelPC_raw.mat
% - Computes chunked column-wise std (per parcel), NaN fraction, maxabs
% - Summarizes per-run: median/std quantiles across parcels
% - Checks gain normalization consistency across runs (using PC1 std medians)
%
% Outputs:
%   outDir\qc_v3\qc_run_timeseries_gain_summary.csv

ip = inputParser;
ip.addRequired('outDir', @(x)ischar(x)||isstring(x));
ip.addParameter('QCSubdir', "qc_v3", @(x)ischar(x)||isstring(x));
ip.addParameter('PreferGNORM', true, @(x)islogical(x)&&isscalar(x));
ip.addParameter('ChunkRows', 20000, @(x)isnumeric(x)&&isscalar(x)&&x>=1000);
ip.addParameter('Verbose', true, @(x)islogical(x)&&isscalar(x));
ip.parse(outDir, varargin{:});
opt = ip.Results;

outDir = char(outDir);
qcDir = fullfile(outDir, char(opt.QCSubdir));
if ~exist(qcDir,'dir'), mkdir(qcDir); end
outCsv = fullfile(qcDir, 'qc_run_timeseries_gain_summary.csv');

% Find mats
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

    % runTag from diagOut if possible
    try
        d = M.diagOut;
        runTag = string(d.runTag);
    catch
        runTag = erase(string(matFiles(i).name), "_parcelPC_" + mode + ".mat");
    end

    % Dimensions
    [nTime, nParc] = size(M, 'PC1');

    % Valid mask
    try
        vmask = logical(M.valid_parcel_mask(:));
        nValid = sum(vmask);
    catch
        vmask = true(nParc,1);
        nValid = nParc;
    end

    % Chunked stats for PC1
    chunk = double(opt.ChunkRows);

    % Welford per-column variance (ignore NaNs)
    n = zeros(1,nParc);
    mu = zeros(1,nParc);
    M2 = zeros(1,nParc);

    nanCount = 0;
    totalCount = 0;
    maxabs = 0;

    for r0 = 1:chunk:nTime
        r1 = min(nTime, r0 + chunk - 1);
        X = double(M.PC1(r0:r1, :));        % [chunk x nParc]
        totalCount = totalCount + numel(X);
        nanCount = nanCount + sum(isnan(X(:)));

        maxabs = max(maxabs, max(abs(X(~isnan(X))), [], 'omitnan'));

        % update per-column mean/var ignoring NaNs
        for c = 1:nParc
            x = X(:,c);
            m = isfinite(x);
            x = x(m);
            if isempty(x), continue; end
            for t = 1:numel(x)
                n(c) = n(c) + 1;
                delta = x(t) - mu(c);
                mu(c) = mu(c) + delta / n(c);
                delta2 = x(t) - mu(c);
                M2(c) = M2(c) + delta * delta2;
            end
        end
    end

    varc = M2 ./ max(n-1,1);
    stdc = sqrt(varc);
    stdc(~isfinite(stdc)) = NaN;

    % Summaries across parcels (prefer valid)
    s = stdc;
    s(~vmask') = NaN;

    pc1_std_median = median(s, 'omitnan');
    pc1_std_p05    = prctile(s(isfinite(s)), 5);
    pc1_std_p95    = prctile(s(isfinite(s)), 95);

    nan_frac = nanCount / max(totalCount,1);

    % Pull diagOut fields if present
    rows_per_vertex = NaN;
    eeg_chStd_median = NaN;
    kVertNorm_median = NaN;
    gnorm_scale = NaN;
    gnorm_ref   = NaN;
    gnorm_basis = "";

    try
        d = M.diagOut;
        if isfield(d,'rows_per_vertex'), rows_per_vertex = double(d.rows_per_vertex); end
        if isfield(d,'eeg_chStd_median'), eeg_chStd_median = double(d.eeg_chStd_median); end
        if isfield(d,'kVertNorm_median'), kVertNorm_median = double(d.kVertNorm_median); end
        if isfield(d,'gnorm_scale_to_med_kgain'), gnorm_scale = double(d.gnorm_scale_to_med_kgain); end
        if isfield(d,'gnorm_ref_med_kgain'), gnorm_ref = double(d.gnorm_ref_med_kgain); end
        if isfield(d,'gnorm_basis'), gnorm_basis = string(d.gnorm_basis); end
    catch
    end

    if opt.Verbose
        fprintf('[%d/%d] %s | mode=%s | pc1_std_med=%.4g | nan=%.4g\n', ...
            i, numel(matFiles), runTag, mode, pc1_std_median, nan_frac);
    end

    rows(end+1,:) = { ...
        char(runTag), char(mode), matPath, ...
        nTime, nParc, ...
        nan_frac, ...
        pc1_std_median, pc1_std_p05, pc1_std_p95, ...
        maxabs, ...
        nValid, ...
        rows_per_vertex, ...
        eeg_chStd_median, kVertNorm_median, ...
        gnorm_scale, gnorm_ref, char(gnorm_basis) ...
    }; %#ok<AGROW>
end

T = cell2table(rows, 'VariableNames', vars);

% Add normalization diagnostic columns (relative to median across runs)
medStd = median(T.pc1_std_median, 'omitnan');
T.pc1_std_ratio_vs_run_median = T.pc1_std_median / max(medStd, eps);

writetable(T, outCsv);
fprintf('Wrote: %s\n', outCsv);

end
