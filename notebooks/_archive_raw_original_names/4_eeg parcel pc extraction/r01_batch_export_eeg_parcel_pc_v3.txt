function batch = r01_batch_export_eeg_parcel_pc_v3(protocolRoot, outDir, varargin)
%R01_BATCH_EXPORT_EEG_PARCEL_PC_V3 (fixed for 3-component volume kernels)
% 1) Export RAW MATs for all runs (PC1/PC2 + PVE + metadata)
% 2) Compute global median kernel gain (kVertNorm_median)
% 3) Create GNORM MATs by scaling PC1/PC2 in-place in chunks
% 4) Write NPY outputs from GNORM MATs
% 5) Write batch summaries + manifest CSVs

ip = inputParser;
ip.addRequired('protocolRoot', @(x)ischar(x)||isstring(x));
ip.addRequired('outDir', @(x)ischar(x)||isstring(x));

% Discovery / naming
ip.addParameter('KernelPattern', "results_MN_EEG_KERNEL_*.mat", @(x)ischar(x)||isstring(x));
ip.addParameter('EEGCleanDir', "", @(x)ischar(x)||isstring(x));
ip.addParameter('DescTag', "desc-ICRej70", @(x)ischar(x)||isstring(x));  % without _clean
ip.addParameter('ScoutFilename', "scout_Schaefer2018_200_7N_dilated_MNI.mat", @(x)ischar(x)||isstring(x));

% Safety locks
ip.addParameter('ExpectedNScouts', 200, @(x)isnumeric(x)&&isscalar(x)&&x>=1);
ip.addParameter('StrictTessMatch', true, @(x)islogical(x)&&isscalar(x));

% Export options
ip.addParameter('MinVertices', 40, @(x)isnumeric(x)&&isscalar(x)&&x>=0);
ip.addParameter('NumPC', 2, @(x)isnumeric(x)&&isscalar(x)&&x>=1);
ip.addParameter('SavePC2', true, @(x)islogical(x)&&isscalar(x));
ip.addParameter('SignConvention', "maxabs", @(x)ischar(x)||isstring(x));
ip.addParameter('BlockSize', 0, @(x)isnumeric(x)&&isscalar(x)&&x>=0);
ip.addParameter('Overwrite', true, @(x)islogical(x)&&isscalar(x));
ip.addParameter('Verbose', true, @(x)islogical(x)&&isscalar(x));
ip.addParameter('EpsRank', 1e-12, @(x)isnumeric(x)&&isscalar(x)&&x>0);
ip.addParameter('UseEigs', true, @(x)islogical(x)&&isscalar(x));
ip.addParameter('EigsTol', 1e-3, @(x)isnumeric(x)&&isscalar(x));
ip.addParameter('EigsMaxIt', 300, @(x)isnumeric(x)&&isscalar(x));
ip.addParameter('DenseFallbackMaxDim', 1200, @(x)isnumeric(x)&&isscalar(x));

% Gain normalization
ip.addParameter('GainNorm', true, @(x)islogical(x)&&isscalar(x));
ip.addParameter('GainCap', [0, Inf], @(x)isnumeric(x)&&numel(x)==2); % cap applied scale

% NPY export
ip.addParameter('WriteNPY', true, @(x)islogical(x)&&isscalar(x));
ip.addParameter('NPYDir', "", @(x)ischar(x)||isstring(x));

% Chunking for scaling large PC matrices
ip.addParameter('ScaleChunkRows', 20000, @(x)isnumeric(x)&&isscalar(x)&&x>=1000);

% Control
ip.addParameter('ContinueOnFail', false, @(x)islogical(x)&&isscalar(x));

ip.parse(protocolRoot, outDir, varargin{:});
opt = ip.Results;

protocolRoot = char(protocolRoot);
outDir = char(outDir);
if ~exist(outDir,'dir'), mkdir(outDir); end

descTag = string(opt.DescTag);
if ~startsWith(descTag,"desc-")
    error('DescTag must start with "desc-". Got: %s', descTag);
end

% EEG clean dir default
eegCleanDir = string(opt.EEGCleanDir);
if strlength(eegCleanDir)==0
    eegCleanDir = string(fullfile(fileparts(outDir), 'ic_pruned','clean'));
end
eegCleanDir = char(eegCleanDir);

% NPY dir default
npyDir = string(opt.NPYDir);
if strlength(npyDir)==0
    npyDir = string(fullfile(outDir, 'npy'));
end
npyDir = char(npyDir);
if opt.WriteNPY && ~exist(npyDir,'dir'), mkdir(npyDir); end

% Find kernels
kernelFiles = dir(fullfile(protocolRoot, 'data', '**', char(opt.KernelPattern)));
kernelFiles = kernelFiles(~[kernelFiles.isdir]);
if isempty(kernelFiles)
    error('No kernel files found under %s with pattern %s', protocolRoot, char(opt.KernelPattern));
end

fprintf('=== r01_batch_export_eeg_parcel_pc_v3 ===\n');
fprintf('protocolRoot : %s\n', protocolRoot);
fprintf('outDir       : %s\n', outDir);
fprintf('eegCleanDir  : %s\n', eegCleanDir);
fprintf('KernelPattern: %s\n', char(opt.KernelPattern));
fprintf('ScoutFilename: %s\n', char(opt.ScoutFilename));
fprintf('DescTag      : %s\n', char(descTag));
fprintf('Found %d kernel files.\n', numel(kernelFiles));
fprintf('WriteNPY     : %d | npyDir: %s\n', opt.WriteNPY, npyDir);

% one-run opts
xopts = struct();
xopts.ExpectedNScouts = double(opt.ExpectedNScouts);
xopts.StrictTessMatch = logical(opt.StrictTessMatch);
xopts.MinVertices     = double(opt.MinVertices);
xopts.NumPC           = double(opt.NumPC);
xopts.SavePC2         = logical(opt.SavePC2);
xopts.SignConvention  = char(opt.SignConvention);
xopts.BlockSize       = double(opt.BlockSize);
xopts.Verbose         = logical(opt.Verbose);
xopts.Overwrite       = logical(opt.Overwrite);
xopts.EpsRank         = double(opt.EpsRank);
xopts.UseEigs         = logical(opt.UseEigs);
xopts.EigsTol         = double(opt.EigsTol);
xopts.EigsMaxIt       = double(opt.EigsMaxIt);
xopts.DenseFallbackMaxDim = double(opt.DenseFallbackMaxDim);

gainRows = {};
covRows  = {};
maniRows = {};

rawMats = strings(0,1);
runTags = strings(0,1);
kgainMed = []; % kVertNorm_median per run

parcelNamesSaved = false;

for i = 1:numel(kernelFiles)
    resFile = fullfile(kernelFiles(i).folder, kernelFiles(i).name);

    sub = regexp(resFile, '(sub-[A-Za-z0-9]+)', 'match', 'once');
    ses = regexp(resFile, '(ses-[A-Za-z0-9]+)', 'match', 'once');

    if isempty(sub) || isempty(ses)
        fprintf('[%d/%d] SKIP (no sub/ses): %s\n', i, numel(kernelFiles), resFile);
        continue;
    end

    runTag = sprintf('%s_%s_%s_clean', sub, ses, char(descTag));

    eegSetFile = fullfile(eegCleanDir, [runTag '.set']);
    if ~exist(eegSetFile,'file')
        msg = sprintf('[%d/%d] Missing EEG .set: %s', i, numel(kernelFiles), eegSetFile);
        if opt.ContinueOnFail, fprintf(2,'%s\n',msg); continue; else, error('%s', msg); end
    end

    anatDir = fullfile(protocolRoot, 'anat', sprintf('%s_%s', sub, ses));
    scoutFile = fullfile(anatDir, char(opt.ScoutFilename));
    if ~exist(scoutFile,'file')
        msg = sprintf('[%d/%d] Missing scout MAT: %s', i, numel(kernelFiles), scoutFile);
        if opt.ContinueOnFail, fprintf(2,'%s\n',msg); continue; else, error('%s', msg); end
    end

    try
        [outRawMat, d] = r01_export_parcel_pc1_one_run_v3(runTag, eegSetFile, resFile, scoutFile, outDir, xopts);

        % Save parcelNames once
        if ~parcelNamesSaved
            tmp = load(outRawMat, 'parcelNames');
            pn = tmp.parcelNames(:);
            Tpn = table((1:numel(pn))', string(pn), 'VariableNames', {'parcel_id','parcel_name'});
            writetable(Tpn, fullfile(outDir, 'parcelNames_200.csv'));
            parcelNamesSaved = true;
        end

        rawMats(end+1,1) = string(outRawMat); %#ok<AGROW>
        runTags(end+1,1) = string(runTag); %#ok<AGROW>
        kgainMed(end+1,1) = double(d.kVertNorm_median); %#ok<AGROW>

        fprintf('[%d/%d] RAW OK %s | rows/vert=%g | EEGchStd=%.4g | kVertNorm=%.4g | PC1std=%.4g\n', ...
            i, numel(kernelFiles), runTag, d.rows_per_vertex, d.eeg_chStd_median, d.kVertNorm_median, d.pc1_std_median);

        % Coverage summary
        covRows(end+1,:) = { ...
            runTag, sub, ses, ...
            d.nScouts, sum(d.n_vertices>0), d.nScouts-sum(d.n_vertices>0), sum(d.n_vertices>0)/max(1,d.nScouts), ...
            d.MinVertices, d.n_valid_parcels, d.n_valid_parcels/max(1,d.nScouts), ...
            d.TessNbVertices, d.nVert_gridloc, d.scout_vertex_min, d.scout_vertex_max, ...
            d.n_assigned_vertices, d.overlap_vertex_count, d.rows_per_vertex ...
        }; %#ok<AGROW>

        % Gain summary
        gainRows(end+1,:) = { ...
            runTag, sub, ses, ...
            d.eeg_chStd_median, d.eeg_maxabs, ...
            d.kRowNorm_median, d.kVertNorm_median, ...
            d.m1_norm_median, d.pc1_std_median, ...
            d.pve1_median, d.pve1_q10, d.pve1_q90, ...
            d.n_flipped_pc1, d.elapsed_sec, outRawMat ...
        }; %#ok<AGROW>

    catch ME
        fprintf(2,'[%d/%d] FAIL %s | %s\n', i, numel(kernelFiles), runTag, ME.message);
        if opt.ContinueOnFail, continue; else, rethrow(ME); end
    end
end

if isempty(kgainMed)
    error('No runs exported. Nothing to normalize.');
end

% -------- Gain normalization basis: kVertNorm_median --------
kgain_med_global = median(kgainMed);
fprintf('\nGlobal median kVertNorm_median = %.6g\n', kgain_med_global);

% Write gnorm mats + NPY
for i = 1:numel(rawMats)
    runTag = char(runTags(i));
    rawMat = char(rawMats(i));

    scale = 1.0;
    if opt.GainNorm
        scale = kgain_med_global / kgainMed(i);
        scale = max(opt.GainCap(1), min(opt.GainCap(2), scale));
    end

    gnormMat = fullfile(outDir, sprintf('%s_parcelPC_gnorm.mat', runTag));

    % copy and scale in-place (chunked) to avoid huge RAM
    copyfile(rawMat, gnormMat);

    M = matfile(gnormMat, 'Writable', true);
    [nT, nP] = size(M, 'PC1');

    chunk = double(opt.ScaleChunkRows);
    for r0 = 1:chunk:nT
        r1 = min(nT, r0 + chunk - 1);
        tmp = M.PC1(r0:r1, :);
        M.PC1(r0:r1, :) = tmp * single(scale);

        if opt.SavePC2
            tmp2 = M.PC2(r0:r1, :);
            M.PC2(r0:r1, :) = tmp2 * single(scale);
        end
    end

    % update diagOut + add gnorm datasets
    d = M.diagOut;
    d.gnorm_scale_to_med_kgain = double(scale);
    d.gnorm_ref_med_kgain      = double(kgain_med_global);
    d.gnorm_basis              = 'kVertNorm_median';
    M.diagOut = d;

    M.gnorm_scale_to_med_kgain = double(scale);
    M.gnorm_ref_med_kgain      = double(kgain_med_global);
    M.gnorm_basis              = 'kVertNorm_median';

    % NPY export from GNORM (load once per run)
    pc1n = fullfile(npyDir, sprintf('%s_PC1_gnorm.npy', runTag));
    pc2n = fullfile(npyDir, sprintf('%s_PC2_gnorm.npy', runTag));
    pve1n = fullfile(npyDir, sprintf('%s_PVE1.npy', runTag));
    pve2n = fullfile(npyDir, sprintf('%s_PVE2.npy', runTag));
    vmn  = fullfile(npyDir, sprintf('%s_valid_parcel_mask.npy', runTag));
    nv_n = fullfile(npyDir, sprintf('%s_n_vertices.npy', runTag));
    nr_n = fullfile(npyDir, sprintf('%s_n_rows.npy', runTag));
    pidn = fullfile(npyDir, sprintf('%s_parcel_ids.npy', runTag));

    if opt.WriteNPY
        if exist('writeNPY','file') ~= 2
            warning('writeNPY not found on MATLAB path; skipping NPY export.');
        else
            PC1 = M.PC1; %#ok<NASGU>
            PC2 = []; %#ok<NASGU>
            if opt.SavePC2, PC2 = M.PC2; end %#ok<NASGU>
            PVE1 = M.PVE1; PVE2 = M.PVE2;
            valid_parcel_mask = M.valid_parcel_mask;
            n_vertices = M.n_vertices;
            n_rows = M.n_rows;
            parcel_ids = M.parcel_ids;

            writeNPY(single(PC1), pc1n);
            if opt.SavePC2
                writeNPY(single(PC2), pc2n);
            end
            writeNPY(single(PVE1), pve1n);
            writeNPY(single(PVE2), pve2n);
            writeNPY(uint8(valid_parcel_mask(:)), vmn);
            writeNPY(int32(n_vertices(:)), nv_n);
            writeNPY(int32(n_rows(:)), nr_n);
            writeNPY(int32(parcel_ids(:)), pidn);
        end
    end

    maniRows(end+1,:) = {runTag, rawMat, gnormMat, scale, kgain_med_global, pc1n, pc2n}; %#ok<AGROW>

    if opt.Verbose
        fprintf('[GNORM] %s scale=%.6g -> %s\n', runTag, scale, gnormMat);
    end
end

% -------- Write tables --------
gainT = cell2table(gainRows, 'VariableNames', { ...
    'runTag','sub','ses', ...
    'eeg_chStd_median','eeg_maxabs', ...
    'kRowNorm_median','kVertNorm_median', ...
    'm1_norm_median','pc1_std_median', ...
    'pve1_median','pve1_q10','pve1_q90', ...
    'n_flipped_pc1','elapsed_sec','rawMatFile'});

covT = cell2table(covRows, 'VariableNames', { ...
    'runTag','sub','ses', ...
    'n_scouts','n_found','n_missing','coverage_frac', ...
    'minVertices','n_valid','coverage_valid', ...
    'TessNbVertices','nVert_gridloc','scout_vmin','scout_vmax', ...
    'n_assigned_vertices','overlap_vertices','rows_per_vertex'});

maniT = cell2table(maniRows, 'VariableNames', { ...
    'runTag','rawMatFile','gnormMatFile','gnorm_scale_to_med_kgain','gnorm_ref_med_kgain','pc1_gnorm_npy','pc2_gnorm_npy'});

gainCsv = fullfile(outDir, 'batch_parcel_gain_summary_v3.csv');
covCsv  = fullfile(outDir, 'batch_parcel_coverage_summary_v3.csv');
maniCsv = fullfile(outDir, 'batch_parcel_manifest_v3.csv');

writetable(gainT, gainCsv);
writetable(covT,  covCsv);
writetable(maniT, maniCsv);

fprintf('\nWrote:\n  %s\n  %s\n  %s\n', gainCsv, covCsv, maniCsv);

batch = struct();
batch.gainTable = gainT;
batch.coverageTable = covT;
batch.manifestTable = maniT;
batch.gainCsv = gainCsv;
batch.coverageCsv = covCsv;
batch.manifestCsv = maniCsv;
batch.kgain_med_global = kgain_med_global;

end
