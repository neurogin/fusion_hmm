function [outRawMatFile, diagOut] = r01_export_parcel_pc1_one_run_v3(runTag, eegSetFile, resFile, scoutMat, outDir, opts)
%R01_EXPORT_PARCEL_PC1_ONE_RUN_V3 (fixed for 3-component volume kernels)
%
% Exports parcel PC1/PC2 for one run using volume-grid scouts from tess.
% Handles Brainstorm volume kernels where ImagingKernel has:
%   - 1 row per vertex (constrained orientation), OR
%   - 3 rows per vertex (unconstrained orientation: x/y/z), i.e. nRows = nVert*3.
%
% Outputs a RAW MAT (v7.3):
%   <runTag>_parcelPC_raw.mat
% containing:
%   PC1, PC2 (time x parcels)   [single]
%   PVE1, PVE2                  [single]  (per parcel, lambda_k / trace(Cp))
%   valid_parcel_mask           [logical]
%   parcelNames                 [cellstr]
%   parcel_ids                  [int32]
%   n_vertices                  [int32]   (per parcel)
%   n_rows                      [int32]   (per parcel = n_vertices * rows_per_vertex)
%   diagOut                     [struct]  (gain + sign + safety + summaries)

% --------------------------
% Defaults
% --------------------------
if nargin < 6 || isempty(opts), opts = struct(); end
if ~isfield(opts,'ExpectedNScouts'), opts.ExpectedNScouts = 200; end
if ~isfield(opts,'StrictTessMatch'), opts.StrictTessMatch = true; end
if ~isfield(opts,'MinVertices'),     opts.MinVertices = 40; end   % threshold on vertices (NOT kernel rows)
if ~isfield(opts,'NumPC'),           opts.NumPC = 2; end
if ~isfield(opts,'SavePC2'),         opts.SavePC2 = (opts.NumPC >= 2); end
if ~isfield(opts,'SignConvention'),  opts.SignConvention = 'maxabs'; end % maxabs|none
if ~isfield(opts,'BlockSize'),       opts.BlockSize = 0; end            % 0=full covariance
if ~isfield(opts,'Verbose'),         opts.Verbose = true; end
if ~isfield(opts,'Overwrite'),       opts.Overwrite = true; end
if ~isfield(opts,'EpsRank'),         opts.EpsRank = 1e-12; end
if ~isfield(opts,'UseEigs'),         opts.UseEigs = true; end
if ~isfield(opts,'EigsTol'),         opts.EigsTol = 1e-3; end
if ~isfield(opts,'EigsMaxIt'),       opts.EigsMaxIt = 300; end
if ~isfield(opts,'DenseFallbackMaxDim'), opts.DenseFallbackMaxDim = 1200; end % if n_rows <= this, allow dense fallback

% --------------------------
% Validate inputs
% --------------------------
if ~exist(eegSetFile,'file'), error('EEGLAB .set not found: %s', eegSetFile); end
if ~exist(resFile,'file'),    error('Kernel file not found: %s', resFile); end
if ~exist(scoutMat,'file'),   error('Scout MAT not found: %s', scoutMat); end
if ~exist(outDir,'dir'), mkdir(outDir); end

outRawMatFile = fullfile(outDir, sprintf('%s_parcelPC_raw.mat', runTag));
if exist(outRawMatFile,'file') && ~opts.Overwrite
    tmp = load(outRawMatFile, 'diagOut');
    if isfield(tmp,'diagOut')
        diagOut = tmp.diagOut;
        if opts.Verbose
            fprintf('[%s] Exists (overwrite=false): %s\n', runTag, outRawMatFile);
        end
        return;
    end
end

tStart = tic;

% --------------------------
% Load kernel + determine rows/vertex
% --------------------------
if opts.Verbose
    fprintf('[%s] Loading kernel: %s\n', runTag, resFile);
end
R = load(resFile, 'ImagingKernel','GridLoc');
if ~isfield(R,'ImagingKernel')
    error('Kernel file missing ImagingKernel: %s', resFile);
end
K = double(R.ImagingKernel);           % [nRowsKernel x nChan]
[nRowsKernel, nChanK] = size(K);

if ~isfield(R,'GridLoc') || isempty(R.GridLoc)
    error('[%s] Kernel file missing GridLoc (needed for tess/grid match).', runTag);
end
nVert = size(R.GridLoc,1);

% rows per vertex: 1 or 3 typically
if mod(nRowsKernel, nVert) ~= 0
    error('[%s] Kernel rows (%d) not divisible by GridLoc vertices (%d).', runTag, nRowsKernel, nVert);
end
rows_per_vertex = nRowsKernel / nVert;
if ~(rows_per_vertex==1 || rows_per_vertex==3)
    warning('[%s] rows_per_vertex=%g (unusual). Proceeding anyway.', runTag, rows_per_vertex);
end

% --------------------------
% Load scouts + tess checks
% --------------------------
if opts.Verbose
    fprintf('[%s] Loading scouts: %s\n', runTag, scoutMat);
end
A = load(scoutMat);

Scouts = [];
if isfield(A,'Scouts')
    Scouts = A.Scouts;
elseif isfield(A,'atlas') && isfield(A.atlas,'Scouts')
    Scouts = A.atlas.Scouts;
end
if isempty(Scouts)
    error('Scout MAT has no Scouts / atlas.Scouts: %s', scoutMat);
end

nScouts = numel(Scouts);
if ~isempty(opts.ExpectedNScouts) && ~isnan(opts.ExpectedNScouts) && nScouts ~= opts.ExpectedNScouts
    error('[%s] Expected %d scouts but found %d in %s', runTag, opts.ExpectedNScouts, nScouts, scoutMat);
end

tessN = NaN;
if isfield(A,'TessNbVertices'), tessN = double(A.TessNbVertices); end
if opts.StrictTessMatch
    if ~isfinite(tessN)
        error('[%s] StrictTessMatch=true but TessNbVertices missing in %s', runTag, scoutMat);
    end
    if tessN ~= nVert
        error('[%s] Tess mismatch: TessNbVertices=%d vs GridLoc vertices=%d. Scout=%s', ...
            runTag, tessN, nVert, scoutMat);
    end
end

% --------------------------
% Load EEG
% --------------------------
if opts.Verbose
    fprintf('[%s] Loading EEG: %s\n', runTag, eegSetFile);
end
[eegPath,eegName,eegExt] = fileparts(char(eegSetFile));
EEG = pop_loadset('filename', [eegName eegExt], 'filepath', eegPath);

F = double(EEG.data);                 % [nChan x nTime]
F = F - mean(F,2);
[nChan, nTime] = size(F);

if nChan ~= nChanK
    error('[%s] Channel mismatch: EEG chans=%d, kernel expects=%d', runTag, nChan, nChanK);
end

% --------------------------
% diagOut core
% --------------------------
diagOut = struct();
diagOut.runTag     = char(runTag);
diagOut.eegSetFile = char(eegSetFile);
diagOut.resFile    = char(resFile);
diagOut.scoutMat   = char(scoutMat);

diagOut.nVert_gridloc     = nVert;
diagOut.nRowsKernel       = nRowsKernel;
diagOut.rows_per_vertex   = rows_per_vertex;
diagOut.TessNbVertices    = tessN;

diagOut.nChan = nChan;
diagOut.nTime = nTime;
diagOut.srate = EEG.srate;

diagOut.nScouts = nScouts;
diagOut.MinVertices = double(opts.MinVertices);
diagOut.NumPC = double(opts.NumPC);
diagOut.SavePC2 = logical(opts.SavePC2);
diagOut.SignConvention = char(opts.SignConvention);

% EEG scale diagnostics
ch_std = std(F,0,2);
diagOut.eeg_chStd_median = median(ch_std);
diagOut.eeg_chStd_p05    = prctile(ch_std,5);
diagOut.eeg_chStd_p95    = prctile(ch_std,95);
diagOut.eeg_maxabs       = max(abs(F(:)));

% Kernel gain diagnostics
k_row_norm = sqrt(sum(K.^2, 2));               % per kernel row
diagOut.kRowNorm_median = median(k_row_norm);
diagOut.kRowNorm_p05    = prctile(k_row_norm,5);
diagOut.kRowNorm_p95    = prctile(k_row_norm,95);

% Vertex-level norm (stable for 3-comp kernels): Frobenius norm of [rows_per_vertex x nChan] block
k_vert_norm = zeros(nVert,1);
for v = 1:nVert
    r0 = (v-1)*rows_per_vertex + 1;
    r1 = v*rows_per_vertex;
    rn = k_row_norm(r0:r1);
    k_vert_norm(v) = sqrt(sum(rn.^2));
end
diagOut.kVertNorm_median = median(k_vert_norm);
diagOut.kVertNorm_p05    = prctile(k_vert_norm,5);
diagOut.kVertNorm_p95    = prctile(k_vert_norm,95);

% --------------------------
% Build parcel vertex indices
% --------------------------
parcel_ids = int32((1:nScouts).');
parcelNames = cell(nScouts,1);

vertexIdxByParcel = cell(nScouts,1);
n_vertices = zeros(nScouts,1,'int32');
n_rows     = zeros(nScouts,1,'int32');

assign_count = zeros(nVert,1); % how many parcels each vertex belongs to
minV = inf; maxV = 0;

for p = 1:nScouts
    if isfield(Scouts(p),'Label') && ~isempty(Scouts(p).Label)
        parcelNames{p} = Scouts(p).Label;
    elseif isfield(Scouts(p),'Name') && ~isempty(Scouts(p).Name)
        parcelNames{p} = Scouts(p).Name;
    else
        parcelNames{p} = sprintf('parcel_%03d', p);
    end

    v = double(Scouts(p).Vertices(:));
    v = v(isfinite(v));
    if ~isempty(v)
        minV = min(minV, min(v));
        maxV = max(maxV, max(v));
    end

    v = v(v >= 1 & v <= nVert);
    v = unique(v);

    vertexIdxByParcel{p} = v;
    n_vertices(p) = int32(numel(v));
    n_rows(p)     = int32(numel(v) * rows_per_vertex);

    if ~isempty(v)
        assign_count(v) = assign_count(v) + 1;
    end
end

if isinf(minV), minV = NaN; end
diagOut.scout_vertex_min = minV;
diagOut.scout_vertex_max = maxV;

diagOut.n_assigned_vertices = sum(assign_count > 0);
diagOut.overlap_vertex_count = sum(assign_count > 1);

valid_parcel_mask = (double(n_vertices) >= double(opts.MinVertices));
diagOut.n_valid_parcels = sum(valid_parcel_mask);

% --------------------------
% Sensor covariance S
% --------------------------
if opts.Verbose
    fprintf('[%s] Computing S (nChan=%d, nTime=%d)...\n', runTag, nChan, nTime);
end

if opts.BlockSize <= 0
    S = (F * F.') / max(1,(nTime - 1));
else
    B = double(opts.BlockSize);
    Sacc = zeros(nChan, nChan, 'double');
    t0 = 1;
    while t0 <= nTime
        t1 = min(nTime, t0 + B - 1);
        Fb = F(:, t0:t1);
        Sacc = Sacc + (Fb * Fb.');
        t0 = t1 + 1;
    end
    S = Sacc / max(1,(nTime - 1));
end

% eigs options
eigsOpts = struct('issym',true,'isreal',true,'tol',opts.EigsTol,'maxit',opts.EigsMaxIt,'disp',0);

% --------------------------
% Allocate outputs (store as single)
% --------------------------
PC1  = NaN(nTime, nScouts, 'single');
PC2  = NaN(nTime, nScouts, 'single');
PVE1 = NaN(nScouts, 1, 'single');
PVE2 = NaN(nScouts, 1, 'single');

m1_norm = NaN(nScouts,1);
m2_norm = NaN(nScouts,1);
pc1_std = NaN(nScouts,1);
pc2_std = NaN(nScouts,1);

pc1_flipped = zeros(nScouts,1,'int8');
pc2_flipped = zeros(nScouts,1,'int8');
pc1_ref_t = NaN(nScouts,1);
pc2_ref_t = NaN(nScouts,1);
pc1_ref_val_pre = NaN(nScouts,1);
pc2_ref_val_pre = NaN(nScouts,1);

% --------------------------
% Parcel loop: implicit Cp operator (no huge Cp)
% --------------------------
if opts.Verbose
    fprintf('[%s] Parcel loop (%d parcels)... rows_per_vertex=%g\n', runTag, nScouts, rows_per_vertex);
end

for p = 1:nScouts
    if ~valid_parcel_mask(p), continue; end
    v = vertexIdxByParcel{p};
    if isempty(v), continue; end

    % expand vertices -> kernel row indices
    % rows = (v-1)*rows_per_vertex + (1:rows_per_vertex)
    dipIdx = zeros(numel(v)*rows_per_vertex, 1);
    kk = 1;
    for j = 1:numel(v)
        base = (v(j)-1)*rows_per_vertex;
        for o = 1:rows_per_vertex
            dipIdx(kk) = base + o;
            kk = kk + 1;
        end
    end

    Kp = K(dipIdx, :);                   % [n_rows x nChan]
    nDip = size(Kp,1);
    if nDip < 5, continue; end

    % trace(Cp) without forming Cp:
    % Cp = Kp*S*Kp' => trace = sum(sum((Kp*S).*Kp))
    M = Kp * S;
    trCp = sum(M(:) .* Kp(:));
    if ~(isfinite(trCp) && trCp > 0), continue; end

    % Operator for Cp*x
    Afun = @(x) Kp * (S * (Kp.' * x));

    % Compute top eigenpairs
    k = min(opts.NumPC, nDip);
    if k < 1, continue; end

    V = [];
    lam = [];

    if opts.UseEigs
        try
            eigsOpts.v0 = ones(nDip,1); % deterministic-ish seed
            [Vraw,Draw] = eigs(Afun, nDip, k, 'largestreal', eigsOpts);
            lam_raw = real(diag(Draw));
            [lam,ord] = sort(lam_raw, 'descend');
            V = real(Vraw(:,ord));
        catch
            V = [];
            lam = [];
        end
    end

    % Dense fallback if eigs failed and size is manageable
    if isempty(V) || isempty(lam)
        if nDip <= opts.DenseFallbackMaxDim
            Cp = Kp * S * Kp.';
            Cp = 0.5*(Cp + Cp.');
            [Vfull,Dfull] = eig(Cp);
            lam_all = real(diag(Dfull));
            [lam_all,ord] = sort(lam_all, 'descend');
            ord = ord(1:k);
            V = real(Vfull(:,ord));
            lam = lam_all(1:k);
        else
            continue; % skip pathological parcel
        end
    end

    if isempty(lam) || any(~isfinite(lam)) || lam(1) < opts.EpsRank
        continue;
    end

    % PC1
    w1 = V(:,1);
    m1 = (w1.' * Kp);            % [1 x nChan]
    x1 = (m1 * F).';             % [nTime x 1] double

    [x1, flipped1, tref1, vpre1] = tc_sign_fix(x1, opts.SignConvention);
    PC1(:,p) = single(x1);
    m1_norm(p) = norm(m1);
    pc1_std(p) = std(x1);

    pc1_flipped(p) = int8(flipped1);
    pc1_ref_t(p) = tref1;
    pc1_ref_val_pre(p) = vpre1;

    PVE1(p) = single(lam(1) / trCp);

    % PC2
    if opts.SavePC2 && k >= 2 && isfinite(lam(2)) && lam(2) >= opts.EpsRank
        w2 = V(:,2);
        m2 = (w2.' * Kp);
        x2 = (m2 * F).';

        [x2, flipped2, tref2, vpre2] = tc_sign_fix(x2, opts.SignConvention);
        PC2(:,p) = single(x2);
        m2_norm(p) = norm(m2);
        pc2_std(p) = std(x2);

        pc2_flipped(p) = int8(flipped2);
        pc2_ref_t(p) = tref2;
        pc2_ref_val_pre(p) = vpre2;

        PVE2(p) = single(lam(2) / trCp);
    end
end

% --------------------------
% Summaries
% --------------------------
vp = logical(valid_parcel_mask(:));
diagOut.m1_norm_median = median(m1_norm(vp & isfinite(m1_norm)));
diagOut.pc1_std_median = median(pc1_std(vp & isfinite(pc1_std)));
diagOut.pve1_median    = median(double(PVE1(vp & isfinite(PVE1))));
diagOut.pve1_q10       = quantile(double(PVE1(vp & isfinite(PVE1))), 0.10);
diagOut.pve1_q90       = quantile(double(PVE1(vp & isfinite(PVE1))), 0.90);

diagOut.n_flipped_pc1 = sum(pc1_flipped ~= 0);
diagOut.n_flipped_pc2 = sum(pc2_flipped ~= 0);

diagOut.elapsed_sec = toc(tStart);

% vectors for forensic/QC
diagOut.n_vertices = n_vertices;
diagOut.n_rows     = n_rows;
diagOut.m1_norm    = m1_norm(:);
diagOut.m2_norm    = m2_norm(:);
diagOut.pc1_std    = pc1_std(:);
diagOut.pc2_std    = pc2_std(:);

diagOut.pc1_flipped = pc1_flipped(:);
diagOut.pc2_flipped = pc2_flipped(:);
diagOut.pc1_ref_t = pc1_ref_t(:);
diagOut.pc2_ref_t = pc2_ref_t(:);
diagOut.pc1_ref_val_pre = pc1_ref_val_pre(:);
diagOut.pc2_ref_val_pre = pc2_ref_val_pre(:);

if opts.Verbose
    fprintf('[%s] RAW OK | rows/vert=%g | EEGchStd=%.4g | kVertNorm=%.4g | PC1std(med)=%.4g | flipsPC1=%d\n', ...
        runTag, rows_per_vertex, diagOut.eeg_chStd_median, diagOut.kVertNorm_median, diagOut.pc1_std_median, diagOut.n_flipped_pc1);
end

save(outRawMatFile, ...
    'PC1','PC2','PVE1','PVE2', ...
    'valid_parcel_mask','parcelNames','parcel_ids','n_vertices','n_rows', ...
    'diagOut', '-v7.3');

end % main

% --------------------------
function [x, flipped, tref, vpre] = tc_sign_fix(x, mode)
% Deterministic sign convention on the timecourse.
x = double(x(:));
flipped = 0;
tref = NaN;
vpre = NaN;

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
