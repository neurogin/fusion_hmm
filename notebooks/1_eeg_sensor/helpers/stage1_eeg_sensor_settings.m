function P = stage1_eeg_sensor_settings()
%STAGE1_EEG_SENSOR_SETTINGS Public configuration helper for Stage 1.
%
% What this helper does:
%   Stores the editable path roots and the manuscript-default settings for
%   Stage 1 EEG sensor preprocessing, Brainstorm exclusion export, and
%   run-level QC.
%
% When it is used:
%   The cleaned Stage-1 public entry files call this helper near the top so
%   that all user-edited paths and frozen scientific defaults live in one
%   predictable place.
%
% Key inputs:
%   Edit the placeholder roots below before running:
%     - P.paths.project_root
%     - P.paths.brainstorm_db_root
%
% Key outputs:
%   Returns one struct `P` containing:
%     - manuscript-default ICLabel policy settings
%     - derived Stage-1 input/output folders
%     - exclusion-merge settings
%     - run-level EEG QC thresholds
%
% Important note:
%   This public helper is now the main editable config file for Stage 1.
%   The legacy compatibility helper `r01_stage1_params.m` remains in place
%   and forwards to this file so older provenance code can still run.

% -------------------------------------------------------------------------
% Manuscript-default ICLabel pruning policy
% -------------------------------------------------------------------------
P.ic_policy = "reject_artifacts";
P.ic_reject_threshold = 0.70;
P.ic_reject_classes = ["Eye","Muscle","Heart","LineNoise","ChannelNoise","Other"];
P.iclabel_tag = sprintf("ICRej%02d", round(100 * P.ic_reject_threshold));

% Preserved for provenance only. This is not the final paper path.
P.historical.iclabel_brain_threshold = 0.60;

% -------------------------------------------------------------------------
% User-edited path roots
% -------------------------------------------------------------------------
P.paths.project_root = "<SET_PROJECT_ROOT>";
P.paths.brainstorm_db_root = "<SET_BRAINSTORM_DB_ROOT>";

% Legacy compatibility alias preserved for older provenance code.
P.paths.r01_rerun_root = P.paths.project_root;

% -------------------------------------------------------------------------
% Derived Stage-1 paths
%
% These are the outsider-facing default locations used by the cleaned
% public workflow. Legacy aliases are kept below where helpful so older
% provenance helpers can still resolve the same folders.
% -------------------------------------------------------------------------
P.paths.raw_eeglab_dir = fullfile(P.paths.project_root, "01_raw", "eeg_eeglab");

P.paths.stage1_derivatives_root = fullfile(P.paths.project_root, "02_derivatives", "stage1_eeg_sensor");
P.paths.stage1_qc_root = fullfile(P.paths.project_root, "04_qc", "stage1_eeg_sensor");

P.paths.ic_pruned_dir = fullfile(P.paths.stage1_derivatives_root, "ic_pruned");
P.paths.with_ica_dir = fullfile(P.paths.ic_pruned_dir, "with_ica");
P.paths.clean_sets_dir = fullfile(P.paths.ic_pruned_dir, "clean_sets");

P.paths.exclusions_root = fullfile(P.paths.stage1_derivatives_root, "exclusions");
P.paths.brainstorm_export_dir = fullfile(P.paths.exclusions_root, "brainstorm_exports");
P.paths.union_mask_dir = fullfile(P.paths.exclusions_root, "union_masks");

P.paths.qc_tables_dir = fullfile(P.paths.stage1_qc_root, "tables");
P.paths.qc_exclusions_dir = fullfile(P.paths.stage1_qc_root, "exclusions");

% Legacy compatibility aliases
P.paths.withICA_dir = P.paths.with_ica_dir;
P.paths.clean_dir = P.paths.clean_sets_dir;
P.paths.bst_export_dir = P.paths.brainstorm_export_dir;

% Brainstorm protocol metadata
P.brainstorm.protocol_name = "eegfmri_R01_ICRej70";

% Only export Brainstorm raw-links corresponding to the manuscript-default
% cleaned EEG files.
P.file_filter = P.iclabel_tag + "_clean";

% -------------------------------------------------------------------------
% Exclusion merge settings
% -------------------------------------------------------------------------
P.mask.merge.adjacency_tol_sec = 0.0;
P.mask.merge.min_dur_sec = 0.0;

% -------------------------------------------------------------------------
% Folder-level exclusion QC warning settings
% -------------------------------------------------------------------------
P.qc.excl.max_excl_frac_warn = 0.20;
P.qc.excl.max_interval_sec_warn = 30.0;
P.qc.excl.min_interval_sec_warn = 0.05;

% -------------------------------------------------------------------------
% Run-level EEG QC gate settings
% -------------------------------------------------------------------------
P.qc.run.min_usable_frac = 0.70;
P.qc.run.max_emg_db = 3.0;
P.qc.run.max_badchan_abs = 10;
P.qc.run.max_badchan_frac = 0.10;
P.qc.run.allow_unknown_usable = false;
P.qc.run.load_raw_duration = false;
P.qc.run.hf_band = [30 80];
P.qc.run.lf_band = [8 13];

end
