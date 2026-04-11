function [RUNSUM, PARCSUM] = r01_qc_v3_pve1_hist_and_lowparcels(outDir, varargin)
%R01_QC_V3_PVE1_HIST_AND_LOWPARCELS
% v3-compatible PVE1 QC that avoids matfile linear indexing issues.
%
% Reads v3 exporter mats (_gnorm preferred else _raw) and computes:
%  1) batch_pve1_run_quantiles_v3.csv
%  2) batch_pve1_histogram_v3.csv
%  3) batch_pve1_lowparcels_frequency_v3.csv
%  4) batch_pve1_lowparcels_frequency_named_v3.csv
%
% Notes:
% - Uses M.varname to pull into memory before (:), because MatFile objects
%   do not support linear indexing.
% - Assumes exporter saved PVE1 and valid_parcel_mask.
%
% Example:
%   [RUNSUM, PARCSUM] = r01_qc_v3_pve1_hist_and_lowparcels(outDir, ...
%       'PreferGNORM', true, 'BottomFrac', 0.05, 'HistBins', 40);

ip = inputParser;
ip.addRequired('outDir', @(x)ischar(x)||isstring(x));
ip.addParameter('PreferGNORM', true, @(x)islogical(x)&&isscalar(x));
ip.addParameter('BottomFrac', 0.05, @(x)isnumeric(x)&&isscalar(x)&&x>0&&x<1);
ip.addParameter('HistBins', 40, @(x)isnumeric(x)&&isscalar(x)&&x>=5);
ip.addParameter('Verbose', true, @(x)islogical(x)&&isscalar(x));
ip.parse(outDir, varargin{:});
opt = ip.Results;

outDir = char(outDir);

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

% Template from first file (load into memory before indexing)
M0 = matfile(fullfile(matFiles(1).folder, matFiles(1).name));
pn0 = M0.parcelNames;      % load to memory
pn0 = pn0(:);
parcelNames = string(pn0);
nParc = numel(parcelNames);
parcel_ids = (1:nParc)';

% Output filenames
outRun  = fullfile(outDir, 'batch_pve1_run_quantiles_v3.csv');
outHist = fullfile(outDir, 'batch_pve1_histogram_v3.csv');
outParc = fullfile(outDir, 'batch_pve1_lowparcels_frequency_v3.csv');
outParc2= fullfile(outDir, 'batch_pve1_lowparcels_frequency_named_v3.csv');

% pooled storage
all_pve1 = [];
all_pid  = [];
all_run  = strings(0,1);

% run-level rows
runRows = {};
runVars = { ...
    'runTag','mode','n_valid_parcels', ...
    'pve1_q10','pve1_q25','pve1_q50','pve1_q75','pve1_q90', ...
    'frac_pve1_lt10','frac_pve1_lt20','frac_pve1_lt30' ...
};

% parcel frequency of bottom frac across runs
parcelLowCount  = containers.Map('KeyType','double','ValueType','double');
parcelSeenCount = containers.Map('KeyType','double','ValueType','double');

for i = 1:numel(matFiles)
    matPath = fullfile(matFiles(i).folder, matFiles(i).name);
    M = matfile(matPath);

    % runTag
    try
        d = M.diagOut;
        runTag = string(d.runTag);
    catch
        runTag = erase(string(matFiles(i).name), "_parcelPC_" + mode + ".mat");
    end

    % Load PVE1 + valid mask into memory safely
    pve1 = double(M.PVE1); pve1 = pve1(:);
    valid = logical(M.valid_parcel_mask); valid = valid(:);

    if numel(pve1) ~= nParc
        warning('PVE1 length mismatch for %s (got %d, expected %d). Skipping.', runTag, numel(pve1), nParc);
        continue;
    end
    if numel(valid) ~= nParc
        warning('valid_parcel_mask length mismatch for %s (got %d, expected %d). Skipping.', runTag, numel(valid), nParc);
        continue;
    end

    % Apply validity and finite
    use = valid & isfinite(pve1);
    pid_u = parcel_ids(use);
    pve_u = pve1(use);

    if isempty(pve_u)
        warning('No valid PVE1 values for %s. Skipping.', runTag);
        continue;
    end

    % pooled vectors
    all_pve1 = [all_pve1; pve_u]; %#ok<AGROW>
    all_pid  = [all_pid;  pid_u]; %#ok<AGROW>
    all_run  = [all_run;  repmat(runTag, numel(pve_u), 1)]; %#ok<AGROW>

    % run quantiles
    q10 = quantile(pve_u, 0.10);
    q25 = quantile(pve_u, 0.25);
    q50 = quantile(pve_u, 0.50);
    q75 = quantile(pve_u, 0.75);
    q90 = quantile(pve_u, 0.90);

    frac_lt10 = mean(pve_u < 0.10);
    frac_lt20 = mean(pve_u < 0.20);
    frac_lt30 = mean(pve_u < 0.30);

    runRows(end+1,:) = { ...
        runTag, mode, numel(pve_u), ...
        q10,q25,q50,q75,q90, ...
        frac_lt10, frac_lt20, frac_lt30 ...
    }; %#ok<AGROW>

    % bottom-frac parcels for this run
    thr = quantile(pve_u, opt.BottomFrac);
    lowMask = (pve_u <= thr);

    for k = 1:numel(pid_u)
        key = double(pid_u(k));

        if ~isKey(parcelSeenCount, key), parcelSeenCount(key) = 0; end
        parcelSeenCount(key) = parcelSeenCount(key) + 1;

        if lowMask(k)
            if ~isKey(parcelLowCount, key), parcelLowCount(key) = 0; end
            parcelLowCount(key) = parcelLowCount(key) + 1;
        end
    end

    if opt.Verbose
        fprintf('[%d/%d] %s | mode=%s | n_valid=%d | q50=%.3f | frac<0.2=%.2f\n', ...
            i, numel(matFiles), runTag, mode, numel(pve_u), q50, frac_lt20);
    end
end

% RUNSUM table
RUNSUM = cell2table(runRows, 'VariableNames', runVars);
writetable(RUNSUM, outRun);

% Histogram pooled
edges = linspace(0,1,opt.HistBins+1);
counts = histcounts(all_pve1, edges);
centers = (edges(1:end-1)+edges(2:end))/2;
H = table(centers(:), counts(:), 'VariableNames', {'pve1_bin_center','count'});
writetable(H, outHist);

% PARCSUM table
keys = cell2mat(parcelSeenCount.keys);
keys = sort(keys(:));
n = numel(keys);

parcel_id = zeros(n,1);
seen = zeros(n,1);
low = zeros(n,1);

for j = 1:n
    parcel_id(j) = keys(j);
    seen(j) = parcelSeenCount(keys(j));
    if isKey(parcelLowCount, keys(j))
        low(j) = parcelLowCount(keys(j));
    else
        low(j) = 0;
    end
end

freq_bottom = low ./ max(seen,1);

PARCSUM = table(parcel_id, seen, low, freq_bottom, ...
    'VariableNames', {'parcel_id','n_runs_seen','n_runs_bottomFrac','freq_bottomFrac'});
PARCSUM = sortrows(PARCSUM, 'freq_bottomFrac', 'descend');
writetable(PARCSUM, outParc);

% Named version
parcel_name = strings(height(PARCSUM),1);
for j = 1:height(PARCSUM)
    pid = PARCSUM.parcel_id(j);
    if pid >= 1 && pid <= nParc
        parcel_name(j) = parcelNames(pid);
    else
        parcel_name(j) = "";
    end
end
PARCSUM_named = PARCSUM;
PARCSUM_named.parcel_name = parcel_name;
writetable(PARCSUM_named, outParc2);

if opt.Verbose
    fprintf('\nWrote:\n  %s\n  %s\n  %s\n  %s\n', outRun, outHist, outParc, outParc2);
    fprintf('PVE1 pooled: N=%d | median=%.3f | q10=%.3f | q90=%.3f\n', ...
        numel(all_pve1), median(all_pve1), quantile(all_pve1,0.10), quantile(all_pve1,0.90));
end

end
