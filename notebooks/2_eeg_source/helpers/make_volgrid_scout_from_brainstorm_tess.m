function diag = make_volgrid_scout_from_brainstorm_tess(tessFile, outScoutFile, atlasNameContains, nVertExpected)
%MAKE_VOLGRID_SCOUT_FROM_BRAINSTORM_TESS Build one standardized scout MAT.
%
% What this helper does:
%   Reads one Brainstorm tess file that already contains a volume-atlas
%   entry, extracts the atlas scouts, checks that their vertex indices fit
%   the expected inverse-kernel grid, and writes the compact scout MAT used
%   downstream by the Stage-2 parcel exporter.
%
% When it is used:
%   Called by `batch_extract_volgrid_scouts_from_brainstorm_tess.m`.
%
% Key inputs:
%   - `tessFile`: Brainstorm tess file in `anat/sub-XX_ses-YY/`
%   - `outScoutFile`: standardized scout MAT to write
%   - `atlasNameContains`: atlas-name filter string
%   - `nVertExpected`: expected GridLoc vertex count from the kernel file
%
% Key outputs:
%   Writes a scout MAT containing:
%   - `Name`
%   - `Scouts`
%   - `TessNbVertices`
%   Returns a diagnostic struct with summary counts and vertex-range checks.

S = load(tessFile);

if ~isfield(S, 'Atlas') || isempty(S.Atlas)
    error('No Atlas field found in tess file: %s', tessFile);
end

names = arrayfun(@(a) string(a.Name), S.Atlas, 'UniformOutput', true);
hit = find(contains(names, "Volume") & contains(names, atlasNameContains), 1, 'first');

if isempty(hit)
    hit = find(contains(names, "Volume"), 1, 'first');
end

if isempty(hit)
    error('No Volume atlas entry found in tess file: %s', tessFile);
end

volumeAtlas = S.Atlas(hit);

if ~isfield(volumeAtlas, 'Scouts') || isempty(volumeAtlas.Scouts)
    error('Selected atlas has no Scouts: %s (%s)', tessFile, volumeAtlas.Name);
end

Scouts = volumeAtlas.Scouts;
nScouts = numel(Scouts);

vertexIndexMax = 0;
vertexIndexMin = inf;
nEmptyScouts = 0;

for k = 1:nScouts
    vertices = double(Scouts(k).Vertices(:));
    vertices = vertices(~isnan(vertices) & vertices > 0);
    if isempty(vertices)
        nEmptyScouts = nEmptyScouts + 1;
        continue;
    end
    vertexIndexMax = max(vertexIndexMax, max(vertices));
    vertexIndexMin = min(vertexIndexMin, min(vertices));
end

if isfinite(nVertExpected) && nVertExpected > 0 && vertexIndexMax > nVertExpected
    error(['Atlas scout vertex index exceeds the expected GridLoc size: ' ...
           'max=%d > nVertExpected=%d'], vertexIndexMax, nVertExpected);
end

if ~isfinite(vertexIndexMin)
    vertexIndexMin = NaN;
end

Name = char(volumeAtlas.Name);
TessNbVertices = double(nVertExpected);

outDir = fileparts(outScoutFile);
if ~isempty(outDir) && exist(outDir, 'dir') ~= 7
    mkdir(outDir);
end
save(outScoutFile, 'Name', 'Scouts', 'TessNbVertices', '-v7.3');

diag = struct();
diag.tessFile = tessFile;
diag.outScoutFile = outScoutFile;
diag.atlasName = string(volumeAtlas.Name);
diag.nScouts = nScouts;
diag.nEmptyScouts = nEmptyScouts;
diag.vertexIndexMin = vertexIndexMin;
diag.vertexIndexMax = vertexIndexMax;
diag.nVertExpected = nVertExpected;

fprintf('Wrote vol-grid scout: %s\n', outScoutFile);
fprintf('  Atlas: %s\n', volumeAtlas.Name);
fprintf('  nScouts=%d, empty=%d, vertexRange=[%d,%d], nVertExpected=%d\n', ...
    nScouts, nEmptyScouts, vertexIndexMin, vertexIndexMax, nVertExpected);

end
