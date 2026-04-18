function diag = r01_make_volgrid_scout_from_tess(tessFile, outScoutFile, atlasNameContains, nVertExpected)
%R01_MAKE_VOLGRID_SCOUT_FROM_TESS Legacy compatibility wrapper.
%
% The active descriptive Stage-2 one-run scout builder now lives in:
%   notebooks/2_eeg_source/helpers/make_volgrid_scout_from_brainstorm_tess.m

this_dir = fileparts(mfilename('fullpath'));
helper_dir = fullfile(this_dir, 'helpers');
if exist(helper_dir, 'dir') == 7
    addpath(helper_dir);
end

diag = make_volgrid_scout_from_brainstorm_tess( ...
    tessFile, ...
    outScoutFile, ...
    atlasNameContains, ...
    nVertExpected);

end
