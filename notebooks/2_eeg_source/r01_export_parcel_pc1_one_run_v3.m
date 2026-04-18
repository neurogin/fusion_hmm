function [outRawMatFile, diagOut] = r01_export_parcel_pc1_one_run_v3(runTag, eegSetFile, resFile, scoutMat, outDir, opts)
%R01_EXPORT_PARCEL_PC1_ONE_RUN_V3 Legacy compatibility wrapper.
%
% The active descriptive Stage-2 one-run parcel exporter now lives in:
%   notebooks/2_eeg_source/helpers/export_parcel_pc1_one_run.m

this_dir = fileparts(mfilename('fullpath'));
helper_dir = fullfile(this_dir, 'helpers');
if exist(helper_dir, 'dir') == 7
    addpath(helper_dir);
end

[outRawMatFile, diagOut] = export_parcel_pc1_one_run( ...
    runTag, eegSetFile, resFile, scoutMat, outDir, opts);

end
