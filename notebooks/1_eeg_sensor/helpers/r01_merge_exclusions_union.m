function out_tsv = r01_merge_exclusions_union(in_tsv, out_tsv, varargin)
%R01_MERGE_EXCLUSIONS_UNION Legacy compatibility wrapper.
%
% The active Stage-1 scientific implementation now lives in:
%   notebooks/1_eeg_sensor/helpers/merge_exclusion_union_masks.m
%
% This preserved file remains only so older provenance callers still run.

out_tsv = merge_exclusion_union_masks(in_tsv, out_tsv, varargin{:});

end
