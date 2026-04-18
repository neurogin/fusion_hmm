function [RUNSUM, PARCSUM] = r01_qc_v3_pve1_hist_and_lowparcels(outDir, varargin)
%R01_QC_V3_PVE1_HIST_AND_LOWPARCELS Legacy compatibility wrapper.
%
% The active descriptive Stage-2 QC helper now lives in:
%   notebooks/2_eeg_source/helpers/summarize_pve1_histogram_and_lowparcel_qc.m

this_dir = fileparts(mfilename('fullpath'));
helper_dir = fullfile(this_dir, 'helpers');
if exist(helper_dir, 'dir') == 7
    addpath(helper_dir);
end

[RUNSUM, PARCSUM] = summarize_pve1_histogram_and_lowparcel_qc(outDir, varargin{:});

end
