function T = batch_extract_volgrid_scouts_from_brainstorm_tess(protocolRoot, scoutFilename, atlasNameContains, kernelPattern)
%BATCH_EXTRACT_VOLGRID_SCOUTS_FROM_BRAINSTORM_TESS Public Stage-2 wrapper.
%
% What this helper does:
%   Scans a Brainstorm protocol for run-level EEG inverse-kernel files,
%   finds the matching tess files that already contain the imported atlas,
%   and writes one standardized scout MAT per subject/session.
%
% When it is used:
%   Called by `22_extract_volgrid_scouts_from_brainstorm_tess.m`.
%
% Key inputs:
%   - Brainstorm protocol root
%   - scout output filename
%   - atlas-name filter string
%   - kernel-file glob pattern
%
% Key outputs:
%   Returns the same summary table as the preserved helper and writes the
%   same scout MAT files on disk.
%
% Important note:
%   This is the descriptive public-facing wrapper. The preserved low-level
%   implementation remains in `r01_batch_make_volgrid_scouts_from_tess.m`.

T = r01_batch_make_volgrid_scouts_from_tess( ...
    protocolRoot, ...
    scoutFilename, ...
    atlasNameContains, ...
    kernelPattern);

end
