function r01_qc_excl_union_folder(union_dir, out_dir, varargin)
%R01_QC_EXCL_UNION_FOLDER Legacy compatibility wrapper.
%
% The active Stage-1 scientific implementation now lives in:
%   notebooks/1_eeg_sensor/helpers/summarize_exclusion_union_folder_qc.m
%
% This preserved file remains only so older provenance callers still run.

summarize_exclusion_union_folder_qc(union_dir, out_dir, varargin{:});

end
