function P = r01_params()
%R01_PARAMS  Single source of truth for R01 rerun pipeline settings.

% ------------------------
% Identity / policy
% ------------------------
P.project_name = "R01_rerun";

% IC retention policy
P.ic_policy = "reject_artifacts";   % {"brain_threshold","reject_artifacts"}

% Reject-artifacts policy parameters
P.ic_reject_threshold = 0.70;
P.ic_reject_classes = ["Eye","Muscle","Heart","LineNoise","ChannelNoise","Other"]; % as requested

% Tag used in filenames and filtering
P.iclabel_tag = sprintf("ICRej%02d", round(100 * P.ic_reject_threshold)); % e.g., ICRej70
P.file_filter = P.iclabel_tag + "_clean";  % e.g., ICRej70_clean

% (Optional) if you ever want the brain-threshold policy again:
P.iclabel_brain_threshold = 0.60; % unused unless P.ic_policy="brain_threshold"

% ------------------------
% Root paths (Windows)
% ------------------------
P.root = "C:\EEGFMRI\hmm\R01_rerun";

% Raw inputs
P.raw.eeg_eeglab = fullfile(P.root, "01_raw", "eeg_eeglab");

% Derivatives
P.deriv.eeg_ic_pruned = fullfile(P.root, "02_derivatives", "eeg_source", "ic_pruned");
P.deriv.masks_bst_exports = fullfile(P.root, "02_derivatives", "masks", "bst_exports");

% QC / logs
P.qc.tables = fullfile(P.root, "04_qc", "tables");
P.qc.exclusions = fullfile(P.root, "04_qc", "exclusions");

% ------------------------
% Brainstorm database root (NEW PROTOCOL)
% ------------------------
P.brainstorm.db_root = "C:\brainstorm_db\eegfmri_R01_ICRej70";

% ------------------------
% Masking / QC parameters
% ------------------------
P.mask.merge.adjacency_tol_sec = 0.0;
P.mask.merge.min_dur_sec = 0.0;

P.qc.excl.max_excl_frac_warn = 0.20;
P.qc.excl.max_interval_sec_warn = 30.0;
P.qc.excl.min_interval_sec_warn = 0.05;

% ------------------------
% Behavior
% ------------------------
P.overwrite = true;

end
