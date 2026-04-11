function T = r01_batch_make_volgrid_scouts_from_tess(protocolRoot, scoutFilename, atlasNameContains, kernelPattern)
% Create per-subject/session vol-grid scout files from tess files, using kernel GridLoc size.

if nargin < 2 || isempty(scoutFilename)
    scoutFilename = 'scout_Schaefer2018_200_7N_dilated_MNI.mat';
end
if nargin < 3 || isempty(atlasNameContains)
    atlasNameContains = 'atlas-Schaefer2018_desc-200Parcels7Networks';
end
if nargin < 4 || isempty(kernelPattern)
    kernelPattern = 'results_MN_EEG_KERNEL_*.mat';
end

protocolRoot = char(protocolRoot);

% find all kernel files (one per run)
K = dir(fullfile(protocolRoot, 'data', '**', kernelPattern));
K = K(~[K.isdir]);

rows = {};
for i = 1:numel(K)
    resFile = fullfile(K(i).folder, K(i).name);

    % infer sub/ses from path
    sub = regexp(resFile, '(sub-[A-Za-z0-9]+)', 'tokens', 'once');
    ses = regexp(resFile, '(ses-[A-Za-z0-9]+)', 'tokens', 'once');
    if isempty(sub) || isempty(ses)
        continue;
    end
    sub = string(sub{1}); ses = string(ses{1});

    anatDir = fullfile(protocolRoot, 'anat', char(sub + "_" + ses));
    if exist(anatDir,'dir') ~= 7
        continue;
    end

    % pick tess file (prefer *_fix if present)
    tFix = dir(fullfile(anatDir, 'tess_cortex_pial_low*_fix.mat'));
    tAny = dir(fullfile(anatDir, 'tess_cortex_pial_low*.mat'));
    if ~isempty(tFix)
        tessFile = fullfile(tFix(1).folder, tFix(1).name);
    elseif ~isempty(tAny)
        tessFile = fullfile(tAny(1).folder, tAny(1).name);
    else
        warning('No tess file found in %s', anatDir);
        continue;
    end

    % get nVertExpected from kernel GridLoc
    R = load(resFile, 'GridLoc');
    if ~isfield(R,'GridLoc')
        warning('No GridLoc in %s', resFile);
        continue;
    end
    nVertExpected = size(R.GridLoc,1);

    outScoutFile = fullfile(anatDir, scoutFilename);

    d = r01_make_volgrid_scout_from_tess(tessFile, outScoutFile, atlasNameContains, nVertExpected);

    rows(end+1,:) = { ...
        char(sub), char(ses), ...
        tessFile, outScoutFile, ...
        char(d.atlasName), ...
        double(d.nScouts), double(d.nEmptyScouts), ...
        double(d.vertexIndexMin), double(d.vertexIndexMax), double(d.nVertExpected) ...
    };
end

T = cell2table(rows, 'VariableNames', { ...
    'sub','ses','tessFile','outScoutFile','atlasName', ...
    'nScouts','nEmptyScouts','vertexMin','vertexMax','nVertExpected' ...
});
end
