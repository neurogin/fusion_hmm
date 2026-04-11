function [Tparcel, Tsens] = r01_qc_v3_pve1_per_parcel_summary(outDir, varargin)
%R01_QC_V3_PVE1_PER_PARCEL_SUMMARY
% Across-run summary of PVE1 per parcel (v3 exporter naming).
% MatFile-safe: loads variables into memory before (:).
%
% Outputs:
%  - batch_pve1_per_parcel_summary_v3.csv
%  - sensitivity_drop_parcels_by_pve1_v3.csv
%
% Example:
%   [Tparcel, Tsens] = r01_qc_v3_pve1_per_parcel_summary(outDir, ...
%       'PreferGNORM', true, 'NSens', 10);

ip = inputParser;
ip.addRequired('outDir', @(x)ischar(x) || isstring(x));
ip.addParameter('PreferGNORM', true, @(x)islogical(x)&&isscalar(x));
ip.addParameter('NSens', 10, @(x)isnumeric(x) && isscalar(x) && x>=1);
ip.addParameter('Quantiles', [0.10 0.25 0.50 0.75 0.90], @(x)isnumeric(x)&&isvector(x));
ip.addParameter('OutPrefix', 'batch', @(x)ischar(x) || isstring(x));
ip.addParameter('Verbose', true, @(x)islogical(x) && isscalar(x));
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

% Template from first file (matfile-safe)
M0 = matfile(fullfile(matFiles(1).folder, matFiles(1).name));
pn0 = M0.parcelNames;  % load to memory
pn0 = pn0(:);
parcelNames = string(pn0);
nP = numel(parcelNames);

parcel_ids = (1:nP)';

nR = numel(matFiles);

% PVE matrix: runs x parcels
PVE = NaN(nR, nP);
runTag = strings(nR,1);

for r = 1:nR
    matPath = fullfile(matFiles(r).folder, matFiles(r).name);
    M = matfile(matPath);

    % runTag
    try
        d = M.diagOut;
        runTag(r) = string(d.runTag);
    catch
        runTag(r) = erase(string(matFiles(r).name), "_parcelPC_" + mode + ".mat");
    end

    % Load into memory (no linear indexing on matfile)
    pve = double(M.PVE1); pve = pve(:);
    v   = logical(M.valid_parcel_mask); v = v(:);

    if numel(pve) ~= nP
        warning('PVE1 length mismatch for %s (got %d, expected %d). Skipping run.', runTag(r), numel(pve), nP);
        continue;
    end
    if numel(v) ~= nP
        warning('valid_parcel_mask mismatch for %s (got %d, expected %d). Skipping run.', runTag(r), numel(v), nP);
        continue;
    end

    pve(~v) = NaN;
    PVE(r,:) = pve(:).';
end

% Per-parcel summary
n_seen = sum(~isnan(PVE), 1)';   % number of runs contributing
pve_min  = min(PVE, [], 1, 'omitnan')';
pve_mean = mean(PVE, 1, 'omitnan')';
pve_std  = std(PVE, 0, 1, 'omitnan')';

qs = opt.Quantiles(:)';
Q = NaN(nP, numel(qs));

for j = 1:nP
    v = PVE(:,j);
    v = v(~isnan(v));
    if ~isempty(v)
        Q(j,:) = quantile(v, qs);
    end
end

Tparcel = table(parcel_ids, parcelNames, n_seen, pve_min, pve_mean, pve_std, ...
    'VariableNames', {'parcel_id','parcel_name','n_runs_seen','pve1_min','pve1_mean','pve1_std'});

for c = 1:numel(qs)
    colName = sprintf('pve1_q%02d', round(100*qs(c)));
    Tparcel.(colName) = Q(:,c);
end

out1 = fullfile(outDir, sprintf('%s_pve1_per_parcel_summary_v3.csv', string(opt.OutPrefix)));
writetable(Tparcel, out1);

% Sensitivity list: lowest q10 (then min, then median)
q10name = sprintf('pve1_q%02d', 10);
q50name = sprintf('pve1_q%02d', 50);

Tsort = sortrows(Tparcel, {q10name, 'pve1_min', q50name}, {'ascend','ascend','ascend'});
k = min(opt.NSens, height(Tsort));

Tsens = Tsort(1:k, {'parcel_id','parcel_name', q10name, q50name, 'pve1_min','pve1_mean','pve1_std','n_runs_seen'});
Tsens.Properties.VariableNames = {'parcel_id','parcel_name','q10_for_drop','med_for_drop','pve1_min','pve1_mean','pve1_std','n_runs_seen'};

out2 = fullfile(outDir, 'sensitivity_drop_parcels_by_pve1_v3.csv');
writetable(Tsens, out2);

if opt.Verbose
    fprintf('Mode=%s | Runs=%d | Parcels=%d\n', mode, nR, nP);
    fprintf('Wrote: %s\n', out1);
    fprintf('Wrote: %s\n', out2);
end

end
