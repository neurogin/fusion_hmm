function T = r01_batch_make_volgrid_scouts_from_tess(protocolRoot, scoutFilename, atlasNameContains, kernelPattern)
%R01_BATCH_MAKE_VOLGRID_SCOUTS_FROM_TESS Legacy compatibility wrapper.
%
% The active descriptive Stage-2 scout builder now lives in:
%   notebooks/2_eeg_source/helpers/batch_extract_volgrid_scouts_from_brainstorm_tess.m
%
% This preserved `r01_*` file remains only so older provenance code can
% still resolve the same behavior without changing scientific outputs.

this_dir = fileparts(mfilename('fullpath'));
helper_dir = fullfile(this_dir, 'helpers');
if exist(helper_dir, 'dir') == 7
    addpath(helper_dir);
end

T = batch_extract_volgrid_scouts_from_brainstorm_tess( ...
    protocolRoot, ...
    scoutFilename, ...
    atlasNameContains, ...
    kernelPattern);

end
