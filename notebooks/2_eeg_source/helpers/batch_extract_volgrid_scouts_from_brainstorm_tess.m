function T = batch_extract_volgrid_scouts_from_brainstorm_tess(protocolRoot, scoutFilename, atlasNameContains, kernelPattern)
%BATCH_EXTRACT_VOLGRID_SCOUTS_FROM_BRAINSTORM_TESS Build scouts for all runs.
%
% What this helper does:
%   Scans a Brainstorm protocol for run-level EEG inverse-kernel files,
%   finds the matching tess files that already contain the imported atlas,
%   and writes one standardized scout MAT per subject/session.
%
% When it is used:
%   Called by `step22_extract_volgrid_scouts_from_brainstorm_tess.m`.
%
% Key inputs:
%   - Brainstorm protocol root
%   - scout output filename
%   - atlas-name filter string
%   - kernel-file glob pattern
%
% Key outputs:
%   Writes the same standardized scout MAT files as the original helper and
%   returns a build-summary table with one row per subject/session.
%
% Important note:
%   This is the active descriptive implementation for the Stage-2
%   scout-building step used by the public workflow.

this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
if ~isempty(this_dir)
    addpath(this_dir);
end

assert_dependency_exists(fullfile(this_dir, 'make_volgrid_scout_from_brainstorm_tess.m'), ...
    ['Missing Stage-2 one-run scout builder:' newline ...
     '  notebooks/2_eeg_source/helpers/make_volgrid_scout_from_brainstorm_tess.m']);

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

kernelFiles = dir(fullfile(protocolRoot, 'data', '**', kernelPattern));
kernelFiles = kernelFiles(~[kernelFiles.isdir]);

rows = {};
for i = 1:numel(kernelFiles)
    resFile = fullfile(kernelFiles(i).folder, kernelFiles(i).name);

    sub = regexp(resFile, '(sub-[A-Za-z0-9]+)', 'tokens', 'once');
    ses = regexp(resFile, '(ses-[A-Za-z0-9]+)', 'tokens', 'once');
    if isempty(sub) || isempty(ses)
        continue;
    end
    sub = string(sub{1});
    ses = string(ses{1});

    anatDir = fullfile(protocolRoot, 'anat', char(sub + "_" + ses));
    if exist(anatDir, 'dir') ~= 7
        continue;
    end

    tessPreferred = dir(fullfile(anatDir, 'tess_cortex_pial_low*_fix.mat'));
    tessFallback = dir(fullfile(anatDir, 'tess_cortex_pial_low*.mat'));
    if ~isempty(tessPreferred)
        tessFile = fullfile(tessPreferred(1).folder, tessPreferred(1).name);
    elseif ~isempty(tessFallback)
        tessFile = fullfile(tessFallback(1).folder, tessFallback(1).name);
    else
        warning('No tess file found in %s', anatDir);
        continue;
    end

    kernelData = load(resFile, 'GridLoc');
    if ~isfield(kernelData, 'GridLoc')
        warning('No GridLoc in %s', resFile);
        continue;
    end
    nVertExpected = size(kernelData.GridLoc, 1);

    outScoutFile = fullfile(anatDir, scoutFilename);

    diag = make_volgrid_scout_from_brainstorm_tess( ...
        tessFile, ...
        outScoutFile, ...
        atlasNameContains, ...
        nVertExpected);

    rows(end+1, :) = { ...
        char(sub), char(ses), ...
        tessFile, outScoutFile, ...
        char(diag.atlasName), ...
        double(diag.nScouts), double(diag.nEmptyScouts), ...
        double(diag.vertexIndexMin), double(diag.vertexIndexMax), double(diag.nVertExpected) ...
    }; 
end

T = cell2table(rows, 'VariableNames', { ...
    'sub','ses','tessFile','outScoutFile','atlasName', ...
    'nScouts','nEmptyScouts','vertexMin','vertexMax','nVertExpected' ...
});

end

function assert_dependency_exists(path_to_file, message_text)
if exist(path_to_file, 'file') ~= 2
    error('%s', message_text);
end
end
