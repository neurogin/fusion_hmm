function [outRawMatFile, diagOut] = export_parcel_pc1_one_run(runTag, eegSetFile, resFile, scoutMat, outDir, opts)
%EXPORT_PARCEL_PC1_ONE_RUN Export parcel PCs for one EEG source run.
%
% What this helper does:
%   Loads one cleaned EEGLAB run, one Brainstorm inverse-kernel file, and
%   one standardized scout MAT, then computes parcel PC1 and PC2 together
%   with the same metadata, support counts, sign-fix information, and PVE
%   summaries used in the preserved Stage-2 workflow.
%
% When it is used:
%   Called by `batch_export_eeg_parcel_pc_outputs.m`.
%
% Key inputs:
%   - `runTag`: standardized run identifier
%   - `eegSetFile`: cleaned Stage-1 EEGLAB `.set`
%   - `resFile`: Brainstorm inverse-kernel MAT
%   - `scoutMat`: standardized scout MAT from Step 22
%   - `outDir`: Stage-2 parcel-output directory
%   - `opts`: one-run export options
%
% Key outputs:
%   Writes `<runTag>_parcelPC_raw.mat` with:
%   - `PC1`, `PC2`
%   - `PVE1`, `PVE2`
%   - `valid_parcel_mask`
%   - `parcelNames`, `parcel_ids`
%   - `n_vertices`, `n_rows`
%   - `diagOut`
%
% Important note:
%   This preserves the current unconstrained-volume-kernel handling, the
%   minimum-vertex gate, the deterministic timecourse sign convention, and
%   all saved metadata used later in Stage 2 and Stage 4.

if nargin < 6 || isempty(opts)
    opts = struct();
end
if ~isfield(opts, 'ExpectedNScouts'), opts.ExpectedNScouts = 200; end
if ~isfield(opts, 'StrictTessMatch'), opts.StrictTessMatch = true; end
if ~isfield(opts, 'MinVertices'), opts.MinVertices = 40; end
if ~isfield(opts, 'NumPC'), opts.NumPC = 2; end
if ~isfield(opts, 'SavePC2'), opts.SavePC2 = (opts.NumPC >= 2); end
if ~isfield(opts, 'SignConvention'), opts.SignConvention = 'maxabs'; end
if ~isfield(opts, 'BlockSize'), opts.BlockSize = 0; end
if ~isfield(opts, 'Verbose'), opts.Verbose = true; end
if ~isfield(opts, 'Overwrite'), opts.Overwrite = true; end
if ~isfield(opts, 'EpsRank'), opts.EpsRank = 1e-12; end
if ~isfield(opts, 'UseEigs'), opts.UseEigs = true; end
if ~isfield(opts, 'EigsTol'), opts.EigsTol = 1e-3; end
if ~isfield(opts, 'EigsMaxIt'), opts.EigsMaxIt = 300; end
if ~isfield(opts, 'DenseFallbackMaxDim'), opts.DenseFallbackMaxDim = 1200; end

if ~exist(eegSetFile, 'file')
    error('EEGLAB .set not found: %s', eegSetFile);
end
if ~exist(resFile, 'file')
    error('Kernel file not found: %s', resFile);
end
if ~exist(scoutMat, 'file')
    error('Scout MAT not found: %s', scoutMat);
end
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

outRawMatFile = fullfile(outDir, sprintf('%s_parcelPC_raw.mat', runTag));
if exist(outRawMatFile, 'file') && ~opts.Overwrite
    tmp = load(outRawMatFile, 'diagOut');
    if isfield(tmp, 'diagOut')
        diagOut = tmp.diagOut;
        if opts.Verbose
            fprintf('[%s] Exists (overwrite=false): %s\n', runTag, outRawMatFile);
        end
        return;
    end
end

tStart = tic;

if opts.Verbose
    fprintf('[%s] Loading kernel: %s\n', runTag, resFile);
end
kernelData = load(resFile, 'ImagingKernel', 'GridLoc');
if ~isfield(kernelData, 'ImagingKernel')
    error('Kernel file missing ImagingKernel: %s', resFile);
end
K = double(kernelData.ImagingKernel);
[nRowsKernel, nChanK] = size(K);

if ~isfield(kernelData, 'GridLoc') || isempty(kernelData.GridLoc)
    error('[%s] Kernel file missing GridLoc (needed for tess/grid match).', runTag);
end
nVert = size(kernelData.GridLoc, 1);

if mod(nRowsKernel, nVert) ~= 0
    error('[%s] Kernel rows (%d) not divisible by GridLoc vertices (%d).', runTag, nRowsKernel, nVert);
end
rows_per_vertex = nRowsKernel / nVert;
if ~(rows_per_vertex == 1 || rows_per_vertex == 3)
    warning('[%s] rows_per_vertex=%g (unusual). Proceeding anyway.', runTag, rows_per_vertex);
end

if opts.Verbose
    fprintf('[%s] Loading scouts: %s\n', runTag, scoutMat);
end
atlasData = load(scoutMat);

Scouts = [];
if isfield(atlasData, 'Scouts')
    Scouts = atlasData.Scouts;
elseif isfield(atlasData, 'atlas') && isfield(atlasData.atlas, 'Scouts')
    Scouts = atlasData.atlas.Scouts;
end
if isempty(Scouts)
    error('Scout MAT has no Scouts / atlas.Scouts: %s', scoutMat);
end

nScouts = numel(Scouts);
if ~isempty(opts.ExpectedNScouts) && ~isnan(opts.ExpectedNScouts) && nScouts ~= opts.ExpectedNScouts
    error('[%s] Expected %d scouts but found %d in %s', runTag, opts.ExpectedNScouts, nScouts, scoutMat);
end

tessN = NaN;
if isfield(atlasData, 'TessNbVertices')
    tessN = double(atlasData.TessNbVertices);
end
if opts.StrictTessMatch
    if ~isfinite(tessN)
        error('[%s] StrictTessMatch=true but TessNbVertices missing in %s', runTag, scoutMat);
    end
    if tessN ~= nVert
        error('[%s] Tess mismatch: TessNbVertices=%d vs GridLoc vertices=%d. Scout=%s', ...
            runTag, tessN, nVert, scoutMat);
    end
end

if opts.Verbose
    fprintf('[%s] Loading EEG: %s\n', runTag, eegSetFile);
end
[eegPath, eegName, eegExt] = fileparts(char(eegSetFile));
EEG = pop_loadset('filename', [eegName eegExt], 'filepath', eegPath);

F = double(EEG.data);
F = F - mean(F, 2);
[nChan, nTime] = size(F);

if nChan ~= nChanK
    error('[%s] Channel mismatch: EEG chans=%d, kernel expects=%d', runTag, nChan, nChanK);
end

diagOut = struct();
diagOut.runTag = char(runTag);
diagOut.eegSetFile = char(eegSetFile);
diagOut.resFile = char(resFile);
diagOut.scoutMat = char(scoutMat);

diagOut.nVert_gridloc = nVert;
diagOut.nRowsKernel = nRowsKernel;
diagOut.rows_per_vertex = rows_per_vertex;
diagOut.TessNbVertices = tessN;

diagOut.nChan = nChan;
diagOut.nTime = nTime;
diagOut.srate = EEG.srate;

diagOut.nScouts = nScouts;
diagOut.MinVertices = double(opts.MinVertices);
diagOut.NumPC = double(opts.NumPC);
diagOut.SavePC2 = logical(opts.SavePC2);
diagOut.SignConvention = char(opts.SignConvention);

channelStd = std(F, 0, 2);
diagOut.eeg_chStd_median = median(channelStd);
diagOut.eeg_chStd_p05 = prctile(channelStd, 5);
diagOut.eeg_chStd_p95 = prctile(channelStd, 95);
diagOut.eeg_maxabs = max(abs(F(:)));

kernelRowNorm = sqrt(sum(K.^2, 2));
diagOut.kRowNorm_median = median(kernelRowNorm);
diagOut.kRowNorm_p05 = prctile(kernelRowNorm, 5);
diagOut.kRowNorm_p95 = prctile(kernelRowNorm, 95);

kernelVertNorm = zeros(nVert, 1);
for v = 1:nVert
    r0 = (v - 1) * rows_per_vertex + 1;
    r1 = v * rows_per_vertex;
    rowNorms = kernelRowNorm(r0:r1);
    kernelVertNorm(v) = sqrt(sum(rowNorms.^2));
end
diagOut.kVertNorm_median = median(kernelVertNorm);
diagOut.kVertNorm_p05 = prctile(kernelVertNorm, 5);
diagOut.kVertNorm_p95 = prctile(kernelVertNorm, 95);

parcel_ids = int32((1:nScouts).');
parcelNames = cell(nScouts, 1);

vertexIdxByParcel = cell(nScouts, 1);
n_vertices = zeros(nScouts, 1, 'int32');
n_rows = zeros(nScouts, 1, 'int32');

assign_count = zeros(nVert, 1);
scoutVertexMin = inf;
scoutVertexMax = 0;

for p = 1:nScouts
    if isfield(Scouts(p), 'Label') && ~isempty(Scouts(p).Label)
        parcelNames{p} = Scouts(p).Label;
    elseif isfield(Scouts(p), 'Name') && ~isempty(Scouts(p).Name)
        parcelNames{p} = Scouts(p).Name;
    else
        parcelNames{p} = sprintf('parcel_%03d', p);
    end

    vertices = double(Scouts(p).Vertices(:));
    vertices = vertices(isfinite(vertices));
    if ~isempty(vertices)
        scoutVertexMin = min(scoutVertexMin, min(vertices));
        scoutVertexMax = max(scoutVertexMax, max(vertices));
    end

    vertices = vertices(vertices >= 1 & vertices <= nVert);
    vertices = unique(vertices);

    vertexIdxByParcel{p} = vertices;
    n_vertices(p) = int32(numel(vertices));
    n_rows(p) = int32(numel(vertices) * rows_per_vertex);

    if ~isempty(vertices)
        assign_count(vertices) = assign_count(vertices) + 1;
    end
end

if isinf(scoutVertexMin)
    scoutVertexMin = NaN;
end
diagOut.scout_vertex_min = scoutVertexMin;
diagOut.scout_vertex_max = scoutVertexMax;

diagOut.n_assigned_vertices = sum(assign_count > 0);
diagOut.overlap_vertex_count = sum(assign_count > 1);

valid_parcel_mask = double(n_vertices) >= double(opts.MinVertices);
diagOut.n_valid_parcels = sum(valid_parcel_mask);

if opts.Verbose
    fprintf('[%s] Computing S (nChan=%d, nTime=%d)...\n', runTag, nChan, nTime);
end

if opts.BlockSize <= 0
    S = (F * F.') / max(1, (nTime - 1));
else
    blockSize = double(opts.BlockSize);
    Sacc = zeros(nChan, nChan, 'double');
    t0 = 1;
    while t0 <= nTime
        t1 = min(nTime, t0 + blockSize - 1);
        Fb = F(:, t0:t1);
        Sacc = Sacc + (Fb * Fb.');
        t0 = t1 + 1;
    end
    S = Sacc / max(1, (nTime - 1));
end

eigsOpts = struct('issym', true, 'isreal', true, 'tol', opts.EigsTol, 'maxit', opts.EigsMaxIt, 'disp', 0);

PC1 = NaN(nTime, nScouts, 'single');
PC2 = NaN(nTime, nScouts, 'single');
PVE1 = NaN(nScouts, 1, 'single');
PVE2 = NaN(nScouts, 1, 'single');

m1_norm = NaN(nScouts, 1);
m2_norm = NaN(nScouts, 1);
pc1_std = NaN(nScouts, 1);
pc2_std = NaN(nScouts, 1);

pc1_flipped = zeros(nScouts, 1, 'int8');
pc2_flipped = zeros(nScouts, 1, 'int8');
pc1_ref_t = NaN(nScouts, 1);
pc2_ref_t = NaN(nScouts, 1);
pc1_ref_val_pre = NaN(nScouts, 1);
pc2_ref_val_pre = NaN(nScouts, 1);

if opts.Verbose
    fprintf('[%s] Parcel loop (%d parcels)... rows_per_vertex=%g\n', runTag, nScouts, rows_per_vertex);
end

for p = 1:nScouts
    if ~valid_parcel_mask(p)
        continue;
    end
    vertices = vertexIdxByParcel{p};
    if isempty(vertices)
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

    Kp = K(dipIdx, :);
    nDip = size(Kp, 1);
    if nDip < 5
        continue;
    end

    M = Kp * S;
    trCp = sum(M(:) .* Kp(:));
    if ~(isfinite(trCp) && trCp > 0)
        continue;
    end

    Afun = @(x) Kp * (S * (Kp.' * x));

    nComponents = min(opts.NumPC, nDip);
    if nComponents < 1
        continue;
    end

    V = [];
    lam = [];

    if opts.UseEigs
        try
            eigsOpts.v0 = ones(nDip, 1);
            [Vraw, Draw] = eigs(Afun, nDip, nComponents, 'largestreal', eigsOpts);
            lamRaw = real(diag(Draw));
            [lam, order] = sort(lamRaw, 'descend');
            V = real(Vraw(:, order));
        catch
            V = [];
            lam = [];
        end
    end

    if isempty(V) || isempty(lam)
        if nDip <= opts.DenseFallbackMaxDim
            Cp = Kp * S * Kp.';
            Cp = 0.5 * (Cp + Cp.');
            [Vfull, Dfull] = eig(Cp);
            lamAll = real(diag(Dfull));
            [lamAll, order] = sort(lamAll, 'descend');
            order = order(1:nComponents);
            V = real(Vfull(:, order));
            lam = lamAll(1:nComponents);
        else
            continue;
        end
    end

    if isempty(lam) || any(~isfinite(lam)) || lam(1) < opts.EpsRank
        continue;
    end

    w1 = V(:, 1);
    m1 = (w1.' * Kp);
    x1 = (m1 * F).';

    [x1, flipped1, tref1, vpre1] = tc_sign_fix(x1, opts.SignConvention);
    PC1(:, p) = single(x1);
    m1_norm(p) = norm(m1);
    pc1_std(p) = std(x1);

    pc1_flipped(p) = int8(flipped1);
    pc1_ref_t(p) = tref1;
    pc1_ref_val_pre(p) = vpre1;

    PVE1(p) = single(lam(1) / trCp);

    if opts.SavePC2 && nComponents >= 2 && isfinite(lam(2)) && lam(2) >= opts.EpsRank
        w2 = V(:, 2);
        m2 = (w2.' * Kp);
        x2 = (m2 * F).';

        [x2, flipped2, tref2, vpre2] = tc_sign_fix(x2, opts.SignConvention);
        PC2(:, p) = single(x2);
        m2_norm(p) = norm(m2);
        pc2_std(p) = std(x2);

        pc2_flipped(p) = int8(flipped2);
        pc2_ref_t(p) = tref2;
        pc2_ref_val_pre(p) = vpre2;

        PVE2(p) = single(lam(2) / trCp);
    end
end

validMask = logical(valid_parcel_mask(:));
diagOut.m1_norm_median = median(m1_norm(validMask & isfinite(m1_norm)));
diagOut.pc1_std_median = median(pc1_std(validMask & isfinite(pc1_std)));
diagOut.pve1_median = median(double(PVE1(validMask & isfinite(PVE1))));
diagOut.pve1_q10 = quantile(double(PVE1(validMask & isfinite(PVE1))), 0.10);
diagOut.pve1_q90 = quantile(double(PVE1(validMask & isfinite(PVE1))), 0.90);

diagOut.n_flipped_pc1 = sum(pc1_flipped ~= 0);
diagOut.n_flipped_pc2 = sum(pc2_flipped ~= 0);

diagOut.elapsed_sec = toc(tStart);

diagOut.n_vertices = n_vertices;
diagOut.n_rows = n_rows;
diagOut.m1_norm = m1_norm(:);
diagOut.m2_norm = m2_norm(:);
diagOut.pc1_std = pc1_std(:);
diagOut.pc2_std = pc2_std(:);

diagOut.pc1_flipped = pc1_flipped(:);
diagOut.pc2_flipped = pc2_flipped(:);
diagOut.pc1_ref_t = pc1_ref_t(:);
diagOut.pc2_ref_t = pc2_ref_t(:);
diagOut.pc1_ref_val_pre = pc1_ref_val_pre(:);
diagOut.pc2_ref_val_pre = pc2_ref_val_pre(:);

if opts.Verbose
    fprintf(['[%s] RAW OK | rows/vert=%g | EEGchStd=%.4g | ' ...
             'kVertNorm=%.4g | PC1std(med)=%.4g | flipsPC1=%d\n'], ...
        runTag, rows_per_vertex, diagOut.eeg_chStd_median, ...
        diagOut.kVertNorm_median, diagOut.pc1_std_median, diagOut.n_flipped_pc1);
end

save(outRawMatFile, ...
    'PC1', 'PC2', 'PVE1', 'PVE2', ...
    'valid_parcel_mask', 'parcelNames', 'parcel_ids', 'n_vertices', 'n_rows', ...
    'diagOut', '-v7.3');

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
