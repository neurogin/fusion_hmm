function diag = r01_make_volgrid_scout_from_tess(tessFile, outScoutFile, atlasNameContains, nVertExpected)
% Extract the "Volume ####: <atlas>" scouts from a Brainstorm tess_*.mat file
% and save a compact scout file compatible with volumetric kernels (GridLoc indexing).
%
% Inputs:
%   tessFile          : ...\anat\sub-XX_ses-YY\tess_cortex_pial_low*.mat
%   outScoutFile      : ...\anat\sub-XX_ses-YY\scout_Schaefer2018_200_7N_dilated_MNI.mat
%   atlasNameContains : string to identify atlas entry (e.g. 'atlas-Schaefer2018_desc-200Parcels7Networks')
%   nVertExpected     : kernel GridLoc vertex count (e.g. 43722)
%
% Output:
%   diag struct with summary checks

S = load(tessFile);

if ~isfield(S,'Atlas') || isempty(S.Atlas)
    error('No Atlas field found in tess file: %s', tessFile);
end

names = arrayfun(@(a) string(a.Name), S.Atlas, 'UniformOutput', true);
hit = find(contains(names, "Volume") & contains(names, atlasNameContains), 1, 'first');

if isempty(hit)
    % fall back: any "Volume" atlas
    hit = find(contains(names, "Volume"), 1, 'first');
end

if isempty(hit)
    error('No Volume atlas entry found in tess file: %s', tessFile);
end

VolAtlas = S.Atlas(hit);

if ~isfield(VolAtlas,'Scouts') || isempty(VolAtlas.Scouts)
    error('Selected atlas has no Scouts: %s (%s)', tessFile, VolAtlas.Name);
end

Scouts = VolAtlas.Scouts;
nScouts = numel(Scouts);

% Validate vertex indices range
Vmax = 0; Vmin = inf; nEmpty = 0;
for k = 1:nScouts
    vv = double(Scouts(k).Vertices(:));
    vv = vv(~isnan(vv) & vv > 0);
    if isempty(vv)
        nEmpty = nEmpty + 1;
        continue;
    end
    Vmax = max(Vmax, max(vv));
    Vmin = min(Vmin, min(vv));
end

if isfinite(nVertExpected) && nVertExpected > 0
    if Vmax > nVertExpected
        error('Atlas scout vertex index exceeds expected GridLoc size: max=%d > nVertExpected=%d', Vmax, nVertExpected);
    end
end

Name = char(VolAtlas.Name);
TessNbVertices = double(nVertExpected);

% Save in your standardized compact format
if exist(fileparts(outScoutFile), 'dir') ~= 7
    mkdir(fileparts(outScoutFile));
end
save(outScoutFile, 'Name', 'Scouts', 'TessNbVertices', '-v7.3');

diag = struct();
diag.tessFile = tessFile;
diag.outScoutFile = outScoutFile;
diag.atlasName = string(VolAtlas.Name);
diag.nScouts = nScouts;
diag.nEmptyScouts = nEmpty;
diag.vertexIndexMin = Vmin;
diag.vertexIndexMax = Vmax;
diag.nVertExpected = nVertExpected;

fprintf('Wrote vol-grid scout: %s\n', outScoutFile);
fprintf('  Atlas: %s\n', VolAtlas.Name);
fprintf('  nScouts=%d, empty=%d, vertexRange=[%d,%d], nVertExpected=%d\n', ...
    nScouts, nEmpty, Vmin, Vmax, nVertExpected);
end
