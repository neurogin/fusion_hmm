function T = r01_qc_v3_run_timeseries_and_gain_summary(outDir, varargin)
%R01_QC_V3_RUN_TIMESERIES_AND_GAIN_SUMMARY Legacy compatibility wrapper.
%
% The active descriptive Stage-2 QC helper now lives in:
%   notebooks/2_eeg_source/helpers/summarize_run_timeseries_gain_qc.m

this_dir = fileparts(mfilename('fullpath'));
helper_dir = fullfile(this_dir, 'helpers');
if exist(helper_dir, 'dir') == 7
    addpath(helper_dir);
end

T = summarize_run_timeseries_gain_qc(outDir, varargin{:});

end
