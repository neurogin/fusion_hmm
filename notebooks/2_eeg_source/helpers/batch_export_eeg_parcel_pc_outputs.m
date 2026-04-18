function batch = batch_export_eeg_parcel_pc_outputs(protocolRoot, outDir, varargin)
%BATCH_EXPORT_EEG_PARCEL_PC_OUTPUTS Export Stage-2 parcel outputs in batch.
%
% What this helper does:
%   Finds all matching Brainstorm kernel files for a protocol, loads the
%   matching Stage-1 cleaned EEG files and standardized scout MAT files,
%   writes run-wise raw parcel-PC exports, applies the preserved gain
%   normalization, writes the NPY sidecars, and builds the batch summary
%   tables used by the later Stage-2 and Stage-4 steps.
%
% When it is used:
%   Called by `export_eeg_parcel_pc1_and_gain_normalize_23.m`.
%
% Key inputs:
%   - Brainstorm protocol root
%   - parcel-output directory
%   - the same name-value options preserved from the original batch export
%
% Key outputs:
%   Writes the same MAT, NPY, and CSV outputs as the original helper and
%   returns a batch struct pointing to the main summary tables.
%
% Important note:
%   This is now the active descriptive implementation for the Stage-2
%   parcel export. It preserves the current gain-normalization basis, the
%   restored `*_time_sec.npy` rule, the sign convention, and the current
%   summary-file schema exactly.

this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
if ~isempty(this_dir)
    addpath(this_dir);
end

assert_dependency_exists(fullfile(this_dir, 'export_parcel_pc1_one_run.m'), ...
    ['Missing Stage-2 one-run parcel exporter:' newline ...
     '  notebooks/2_eeg_source/helpers/export_parcel_pc1_one_run.m']);

ip = inputParser;
ip.addRequired('protocolRoot', @(x) ischar(x) || isstring(x));
ip.addRequired('outDir', @(x) ischar(x) || isstring(x));

ip.addParameter('KernelPattern', "results_MN_EEG_KERNEL_*.mat", @(x) ischar(x) || isstring(x));
ip.addParameter('EEGCleanDir', "", @(x) ischar(x) || isstring(x));
ip.addParameter('DescTag', "desc-ICRej70", @(x) ischar(x) || isstring(x));
ip.addParameter('ScoutFilename', "scout_Schaefer2018_200_7N_dilated_MNI.mat", @(x) ischar(x) || isstring(x));

ip.addParameter('ExpectedNScouts', 200, @(x) isnumeric(x) && isscalar(x) && x >= 1);
ip.addParameter('StrictTessMatch', true, @(x) islogical(x) && isscalar(x));

ip.addParameter('MinVertices', 40, @(x) isnumeric(x) && isscalar(x) && x >= 0);
ip.addParameter('NumPC', 2, @(x) isnumeric(x) && isscalar(x) && x >= 1);
ip.addParameter('SavePC2', true, @(x) islogical(x) && isscalar(x));
ip.addParameter('SignConvention', "maxabs", @(x) ischar(x) || isstring(x));
ip.addParameter('BlockSize', 0, @(x) isnumeric(x) && isscalar(x) && x >= 0);
ip.addParameter('Overwrite', true, @(x) islogical(x) && isscalar(x));
ip.addParameter('Verbose', true, @(x) islogical(x) && isscalar(x));
ip.addParameter('EpsRank', 1e-12, @(x) isnumeric(x) && isscalar(x) && x > 0);
ip.addParameter('UseEigs', true, @(x) islogical(x) && isscalar(x));
ip.addParameter('EigsTol', 1e-3, @(x) isnumeric(x) && isscalar(x));
ip.addParameter('EigsMaxIt', 300, @(x) isnumeric(x) && isscalar(x));
ip.addParameter('DenseFallbackMaxDim', 1200, @(x) isnumeric(x) && isscalar(x));

ip.addParameter('GainNorm', true, @(x) islogical(x) && isscalar(x));
ip.addParameter('GainCap', [0, Inf], @(x) isnumeric(x) && numel(x) == 2);

ip.addParameter('WriteNPY', true, @(x) islogical(x) && isscalar(x));
ip.addParameter('NPYDir', "", @(x) ischar(x) || isstring(x));

ip.addParameter('ScaleChunkRows', 20000, @(x) isnumeric(x) && isscalar(x) && x >= 1000);
ip.addParameter('ContinueOnFail', false, @(x) islogical(x) && isscalar(x));

ip.parse(protocolRoot, outDir, varargin{:});
opt = ip.Results;

protocolRoot = char(protocolRoot);
outDir = char(outDir);
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

descTag = string(opt.DescTag);
if ~startsWith(descTag, "desc-")
    error('DescTag must start with "desc-". Got: %s', descTag);
end

eegCleanDir = string(opt.EEGCleanDir);
if strlength(eegCleanDir) == 0
    eegCleanDir = string(fullfile(fileparts(outDir), 'ic_pruned', 'clean'));
end
eegCleanDir = char(eegCleanDir);

npyDir = string(opt.NPYDir);
if strlength(npyDir) == 0
    npyDir = string(fullfile(outDir, 'npy'));
end
npyDir = char(npyDir);
if opt.WriteNPY && ~exist(npyDir, 'dir')
    mkdir(npyDir);
end

kernelFiles = dir(fullfile(protocolRoot, 'data', '**', char(opt.KernelPattern)));
kernelFiles = kernelFiles(~[kernelFiles.isdir]);
if isempty(kernelFiles)
    error('No kernel files found under %s with pattern %s', protocolRoot, char(opt.KernelPattern));
end

fprintf('=== batch_export_eeg_parcel_pc_outputs ===\n');
fprintf('protocolRoot : %s\n', protocolRoot);
fprintf('outDir       : %s\n', outDir);
fprintf('eegCleanDir  : %s\n', eegCleanDir);
fprintf('KernelPattern: %s\n', char(opt.KernelPattern));
fprintf('ScoutFilename: %s\n', char(opt.ScoutFilename));
fprintf('DescTag      : %s\n', char(descTag));
fprintf('Found %d kernel files.\n', numel(kernelFiles));
fprintf('WriteNPY     : %d | npyDir: %s\n', opt.WriteNPY, npyDir);

xopts = struct();
xopts.ExpectedNScouts = double(opt.ExpectedNScouts);
xopts.StrictTessMatch = logical(opt.StrictTessMatch);
xopts.MinVertices = double(opt.MinVertices);
xopts.NumPC = double(opt.NumPC);
xopts.SavePC2 = logical(opt.SavePC2);
xopts.SignConvention = char(opt.SignConvention);
xopts.BlockSize = double(opt.BlockSize);
xopts.Verbose = logical(opt.Verbose);
xopts.Overwrite = logical(opt.Overwrite);
xopts.EpsRank = double(opt.EpsRank);
xopts.UseEigs = logical(opt.UseEigs);
xopts.EigsTol = double(opt.EigsTol);
xopts.EigsMaxIt = double(opt.EigsMaxIt);
xopts.DenseFallbackMaxDim = double(opt.DenseFallbackMaxDim);

gainRows = {};
coverageRows = {};
manifestRows = {};

rawMats = strings(0, 1);
runTags = strings(0, 1);
kgainMed = [];

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
    if ~exist(eegSetFile, 'file')
        msg = sprintf('[%d/%d] Missing EEG .set: %s', i, numel(kernelFiles), eegSetFile);
        if opt.ContinueOnFail
            fprintf(2, '%s\n', msg);
            continue;
        else
            error('%s', msg);
        end
    end

    anatDir = fullfile(protocolRoot, 'anat', sprintf('%s_%s', sub, ses));
    scoutFile = fullfile(anatDir, char(opt.ScoutFilename));
    if ~exist(scoutFile, 'file')
        msg = sprintf('[%d/%d] Missing scout MAT: %s', i, numel(kernelFiles), scoutFile);
        if opt.ContinueOnFail
            fprintf(2, '%s\n', msg);
            continue;
        else
            error('%s', msg);
        end
    end

    try
        [outRawMat, diagOut] = export_parcel_pc1_one_run(runTag, eegSetFile, resFile, scoutFile, outDir, xopts);

        if ~parcelNamesSaved
            tmp = load(outRawMat, 'parcelNames');
            pn = tmp.parcelNames(:);
            Tpn = table((1:numel(pn)).', string(pn), 'VariableNames', {'parcel_id','parcel_name'});
            writetable(Tpn, fullfile(outDir, 'parcelNames_200.csv'));
            parcelNamesSaved = true;
        end

        rawMats(end+1, 1) = string(outRawMat); %#ok<AGROW>
        runTags(end+1, 1) = string(runTag); %#ok<AGROW>
        kgainMed(end+1, 1) = double(diagOut.kVertNorm_median); %#ok<AGROW>

        fprintf(['[%d/%d] RAW OK %s | rows/vert=%g | EEGchStd=%.4g | ' ...
                 'kVertNorm=%.4g | PC1std=%.4g\n'], ...
            i, numel(kernelFiles), runTag, diagOut.rows_per_vertex, ...
            diagOut.eeg_chStd_median, diagOut.kVertNorm_median, diagOut.pc1_std_median);

        coverageRows(end+1, :) = { ...
            runTag, sub, ses, ...
            diagOut.nScouts, sum(diagOut.n_vertices > 0), diagOut.nScouts - sum(diagOut.n_vertices > 0), ...
            sum(diagOut.n_vertices > 0) / max(1, diagOut.nScouts), ...
            diagOut.MinVertices, diagOut.n_valid_parcels, diagOut.n_valid_parcels / max(1, diagOut.nScouts), ...
            diagOut.TessNbVertices, diagOut.nVert_gridloc, diagOut.scout_vertex_min, diagOut.scout_vertex_max, ...
            diagOut.n_assigned_vertices, diagOut.overlap_vertex_count, diagOut.rows_per_vertex ...
        }; %#ok<AGROW>

        gainRows(end+1, :) = { ...
            runTag, sub, ses, ...
            diagOut.eeg_chStd_median, diagOut.eeg_maxabs, ...
            diagOut.kRowNorm_median, diagOut.kVertNorm_median, ...
            diagOut.m1_norm_median, diagOut.pc1_std_median, ...
            diagOut.pve1_median, diagOut.pve1_q10, diagOut.pve1_q90, ...
            diagOut.n_flipped_pc1, diagOut.elapsed_sec, outRawMat ...
        }; %#ok<AGROW>

    catch ME
        fprintf(2, '[%d/%d] FAIL %s | %s\n', i, numel(kernelFiles), runTag, ME.message);
        if opt.ContinueOnFail
            continue;
        else
            rethrow(ME);
        end
    end
end

if isempty(kgainMed)
    error('No runs exported. Nothing to normalize.');
end

kgainMedGlobal = median(kgainMed);
fprintf('\nGlobal median kVertNorm_median = %.6g\n', kgainMedGlobal);

for i = 1:numel(rawMats)
    runTag = char(runTags(i));
    rawMat = char(rawMats(i));

    scale = 1.0;
    if opt.GainNorm
        scale = kgainMedGlobal / kgainMed(i);
        scale = max(opt.GainCap(1), min(opt.GainCap(2), scale));
    end

    gnormMat = fullfile(outDir, sprintf('%s_parcelPC_gnorm.mat', runTag));
    copyfile(rawMat, gnormMat);

    M = matfile(gnormMat, 'Writable', true);
    [nT, ~] = size(M, 'PC1');

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

    d = M.diagOut;
    d.gnorm_scale_to_med_kgain = double(scale);
    d.gnorm_ref_med_kgain = double(kgainMedGlobal);
    d.gnorm_basis = 'kVertNorm_median';
    M.diagOut = d;

    M.gnorm_scale_to_med_kgain = double(scale);
    M.gnorm_ref_med_kgain = double(kgainMedGlobal);
    M.gnorm_basis = 'kVertNorm_median';

    pc1n = fullfile(npyDir, sprintf('%s_PC1_gnorm.npy', runTag));
    pc2n = fullfile(npyDir, sprintf('%s_PC2_gnorm.npy', runTag));
    pve1n = fullfile(npyDir, sprintf('%s_PVE1.npy', runTag));
    pve2n = fullfile(npyDir, sprintf('%s_PVE2.npy', runTag));
    tsn = fullfile(npyDir, sprintf('%s_time_sec.npy', runTag));
    vmn = fullfile(npyDir, sprintf('%s_valid_parcel_mask.npy', runTag));
    nvn = fullfile(npyDir, sprintf('%s_n_vertices.npy', runTag));
    nrn = fullfile(npyDir, sprintf('%s_n_rows.npy', runTag));
    pidn = fullfile(npyDir, sprintf('%s_parcel_ids.npy', runTag));

    if opt.WriteNPY
        if exist('writeNPY', 'file') ~= 2
            warning('writeNPY not found on MATLAB path; skipping NPY export.');
        else
            PC1 = M.PC1; 
            PC2 = []; 
            if opt.SavePC2
                PC2 = M.PC2; 
            end
            PVE1 = M.PVE1;
            PVE2 = M.PVE2;
            valid_parcel_mask = M.valid_parcel_mask;
            n_vertices = M.n_vertices;
            n_rows = M.n_rows;
            parcel_ids = M.parcel_ids;
            time_sec = single((0:nT-1)' / double(d.srate));

            writeNPY(single(PC1), pc1n);
            if opt.SavePC2
                writeNPY(single(PC2), pc2n);
            end
            writeNPY(single(PVE1), pve1n);
            writeNPY(single(PVE2), pve2n);
            writeNPY(time_sec, tsn);
            writeNPY(uint8(valid_parcel_mask(:)), vmn);
            writeNPY(int32(n_vertices(:)), nvn);
            writeNPY(int32(n_rows(:)), nrn);
            writeNPY(int32(parcel_ids(:)), pidn);
        end
    end

    manifestRows(end+1, :) = { ...
        runTag, rawMat, gnormMat, scale, kgainMedGlobal, pc1n, pc2n ...
    }; %#ok<AGROW>

    if opt.Verbose
        fprintf('[GNORM] %s scale=%.6g -> %s\n', runTag, scale, gnormMat);
    end
end

gainT = cell2table(gainRows, 'VariableNames', { ...
    'runTag','sub','ses', ...
    'eeg_chStd_median','eeg_maxabs', ...
    'kRowNorm_median','kVertNorm_median', ...
    'm1_norm_median','pc1_std_median', ...
    'pve1_median','pve1_q10','pve1_q90', ...
    'n_flipped_pc1','elapsed_sec','rawMatFile'});

coverageT = cell2table(coverageRows, 'VariableNames', { ...
    'runTag','sub','ses', ...
    'n_scouts','n_found','n_missing','coverage_frac', ...
    'minVertices','n_valid','coverage_valid', ...
    'TessNbVertices','nVert_gridloc','scout_vmin','scout_vmax', ...
    'n_assigned_vertices','overlap_vertices','rows_per_vertex'});

manifestT = cell2table(manifestRows, 'VariableNames', { ...
    'runTag','rawMatFile','gnormMatFile','gnorm_scale_to_med_kgain', ...
    'gnorm_ref_med_kgain','pc1_gnorm_npy','pc2_gnorm_npy'});

gainCsv = fullfile(outDir, 'batch_parcel_gain_summary_v3.csv');
coverageCsv = fullfile(outDir, 'batch_parcel_coverage_summary_v3.csv');
manifestCsv = fullfile(outDir, 'batch_parcel_manifest_v3.csv');

writetable(gainT, gainCsv);
writetable(coverageT, coverageCsv);
writetable(manifestT, manifestCsv);

fprintf('\nWrote:\n  %s\n  %s\n  %s\n', gainCsv, coverageCsv, manifestCsv);

batch = struct();
batch.gainTable = gainT;
batch.coverageTable = coverageT;
batch.manifestTable = manifestT;
batch.gainCsv = gainCsv;
batch.coverageCsv = coverageCsv;
batch.manifestCsv = manifestCsv;
batch.kgain_med_global = kgainMedGlobal;

end

function assert_dependency_exists(path_to_file, message_text)
if exist(path_to_file, 'file') ~= 2
    error('%s', message_text);
end
end
