function T = batch_extract_volgrid_scouts_from_brainstorm_tess(protocolRoot, scoutFilename, atlasNameContains, kernelPattern)
%BATCH_EXTRACT_VOLGRID_SCOUTS_FROM_BRAINSTORM_TESS Public Stage-2 wrapper.
%
% What this helper does:
%   Scans a Brainstorm protocol for run-level EEG inverse-kernel files,
%   finds the matching tess files that already contain the imported atlas,
%   and writes one standardized scout MAT per subject/session.
%
% When it is used:
%   Called by `extract_volgrid_scouts_from_brainstorm_tess_22.m`.
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

this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
stage2_dir = fileparts(this_dir);

if ~isempty(stage2_dir)
    addpath(stage2_dir);
end
if ~isempty(this_dir)
    addpath(this_dir);
end

assert_dependency_exists(fullfile(stage2_dir, 'r01_batch_make_volgrid_scouts_from_tess.m'), ...
    ['Missing preserved Stage-2 batch scout extractor:' newline ...
     '  notebooks/2_eeg_source/r01_batch_make_volgrid_scouts_from_tess.m']);
assert_dependency_exists(fullfile(stage2_dir, 'r01_make_volgrid_scout_from_tess.m'), ...
    ['Missing preserved Stage-2 one-run scout extractor:' newline ...
     '  notebooks/2_eeg_source/r01_make_volgrid_scout_from_tess.m']);

T = r01_batch_make_volgrid_scouts_from_tess( ...
    protocolRoot, ...
    scoutFilename, ...
    atlasNameContains, ...
    kernelPattern);

end

function assert_dependency_exists(path_to_file, message_text)
if exist(path_to_file, 'file') ~= 2
    error('%s', message_text);
end
end
