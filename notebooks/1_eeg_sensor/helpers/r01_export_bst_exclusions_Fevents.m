function r01_export_bst_exclusions_Fevents(rawlink_mat, out_tsv)
%R01_EXPORT_BST_EXCLUSIONS_FEVENTS Legacy compatibility wrapper.
%
% The active Stage-1 scientific implementation now lives in:
%   notebooks/1_eeg_sensor/helpers/export_brainstorm_exclusion_events.m
%
% This preserved file remains only so older provenance callers still run.

export_brainstorm_exclusion_events(rawlink_mat, out_tsv);

end
