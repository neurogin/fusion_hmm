function P = stage1_eeg_sensor_settings()
%STAGE1_EEG_SENSOR_SETTINGS Public configuration helper for Stage 1.
%
% Before you run any Stage-1 script:
%   1. Set `P.paths.project_root` below.
%   2. Set `P.paths.brainstorm_db_root` below.
%   3. Save this file.
%
% What those two required paths mean:
%   - `P.paths.project_root`
%       Your main project folder for this repo run. The scripts expect the
%       raw EEGLAB `.set` inputs under:
%         <project_root>\01_raw\eeg_eeglab\
%       and they will write Stage-1 derivatives and QC outputs under:
%         <project_root>\02_derivatives\stage1_eeg_sensor\
%         <project_root>\04_qc\stage1_eeg_sensor\
%
%   - `P.paths.brainstorm_db_root`
%       The Brainstorm protocol folder for this study. This folder must
%       contain Brainstorm's `data\` subfolder underneath it, because the
%       Stage-1 export script scans:
%         <brainstorm_db_root>\data\...
%       for the saved `data_0raw_*.mat` raw-link files.
%
% What this helper returns:
%   One struct `P` containing:
%   - the two user-edited root paths above
%   - all derived Stage-1 input/output folders
%   - the manuscript-default pruning and QC settings
%
% Important note:
%   This is the main user-editable Stage-1 settings file. Most users only
%   need to edit the two placeholder paths below. The legacy compatibility
%   helper `r01_stage1_params.m` remains in place so older provenance code
%   can still run.

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
%
% These are the only placeholders most users need to change.
% -------------------------------------------------------------------------
P.paths.project_root = "<SET_PROJECT_ROOT>";          % Main project folder containing 01_raw, 02_derivatives, and 04_qc
P.paths.brainstorm_db_root = "<SET_BRAINSTORM_DB_ROOT>"; % Brainstorm protocol folder that contains the data\ subfolder

% Legacy compatibility alias preserved for older provenance code.
P.paths.r01_rerun_root = P.paths.project_root;

% -------------------------------------------------------------------------
% Derived Stage-1 paths
%
% You usually do not need to edit anything below this line.
%
% These are the outsider-facing default locations used by the cleaned
% public workflow. They are derived automatically from the two root paths
% above so the Stage-1 scripts all write to consistent locations.
% -------------------------------------------------------------------------
P.paths.raw_eeglab_dir = fullfile(P.paths.project_root, "01_raw", "eeg_eeglab"); % Raw EEGLAB .set files read by step 10

P.paths.stage1_derivatives_root = fullfile(P.paths.project_root, "02_derivatives", "stage1_eeg_sensor");
P.paths.stage1_qc_root = fullfile(P.paths.project_root, "04_qc", "stage1_eeg_sensor");

P.paths.ic_pruned_dir = fullfile(P.paths.stage1_derivatives_root, "ic_pruned"); % Parent folder for Stage-1 pruned EEG outputs
P.paths.with_ica_dir = fullfile(P.paths.ic_pruned_dir, "with_ica");              % Auditable outputs that keep ICA metadata
P.paths.clean_sets_dir = fullfile(P.paths.ic_pruned_dir, "clean_sets");          % Brainstorm-facing clean .set files

P.paths.exclusions_root = fullfile(P.paths.stage1_derivatives_root, "exclusions");           % Parent folder for exported exclusion TSV files
P.paths.brainstorm_export_dir = fullfile(P.paths.exclusions_root, "brainstorm_exports");     % Raw Brainstorm event exports
P.paths.union_mask_dir = fullfile(P.paths.exclusions_root, "union_masks");                    % Merged exclusion windows used downstream

P.paths.qc_tables_dir = fullfile(P.paths.stage1_qc_root, "tables");           % Run-level QC tables, manifests, and Table-S1 support files
P.paths.qc_exclusions_dir = fullfile(P.paths.stage1_qc_root, "exclusions");   % Exclusion-summary QC outputs

% Legacy compatibility aliases
P.paths.withICA_dir = P.paths.with_ica_dir;
P.paths.clean_dir = P.paths.clean_sets_dir;
P.paths.bst_export_dir = P.paths.brainstorm_export_dir;

% Brainstorm protocol metadata
%
% This is used as stage metadata. The actual files are still located by the
% `brainstorm_db_root` path above.
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
