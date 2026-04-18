function QC = r01_qc_v3_sign_convention_parcelpc(outDir, varargin)
%R01_QC_V3_SIGN_CONVENTION_PARCELPC Legacy compatibility wrapper.
%
% The active descriptive Stage-2 QC helper now lives in:
%   notebooks/2_eeg_source/helpers/summarize_sign_convention_qc.m

this_dir = fileparts(mfilename('fullpath'));
helper_dir = fullfile(this_dir, 'helpers');
if exist(helper_dir, 'dir') == 7
    addpath(helper_dir);
end

QC = summarize_sign_convention_qc(outDir, varargin{:});

end
