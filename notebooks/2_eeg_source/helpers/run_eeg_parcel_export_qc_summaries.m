function qc = run_eeg_parcel_export_qc_summaries(outDir)
%RUN_EEG_PARCEL_EXPORT_QC_SUMMARIES Public helper for Stage-2 QC sidecars.
%
% What this helper does:
%   Runs the preserved Stage-2 exporter QC summaries that feed the cleaned
%   public notebook `25_qc_eeg_parcel_exports_table_s3_and_figures_s2_s4`.
%
% When it is used:
%   Run this after `export_eeg_parcel_pc1_and_gain_normalize_23.m` and
%   before the Stage-2 QC notebook if the required CSV sidecars do not yet
%   exist.
%
% Key inputs:
%   - `outDir`: the Stage-2 parcel-output directory written by step 23
%
% Key outputs:
%   Writes the same QC CSV outputs as the preserved helper trio, including:
%     - `qc_v3/qc_run_timeseries_gain_summary.csv`
%     - `qc_v3_sign/qc_sign_v3_summary.csv`
%     - `batch_pve1_run_quantiles_v3.csv`
%     - `batch_pve1_histogram_v3.csv`
%     - `batch_pve1_lowparcels_frequency_named_v3.csv`
%
% Important note:
%   This wrapper preserves the current v3 QC behavior and default options.
%   The underlying implementations remain in the preserved legacy helpers.

this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
stage2_dir = fileparts(this_dir);

if ~isempty(stage2_dir)
    addpath(stage2_dir);
end
if ~isempty(this_dir)
    addpath(this_dir);
end

assert_dependency_exists(fullfile(stage2_dir, 'r01_qc_v3_run_timeseries_and_gain_summary.m'), ...
    ['Missing preserved Stage-2 QC helper:' newline ...
     '  notebooks/2_eeg_source/r01_qc_v3_run_timeseries_and_gain_summary.m']);
assert_dependency_exists(fullfile(stage2_dir, 'r01_qc_v3_sign_convention_parcelpc.m'), ...
    ['Missing preserved Stage-2 QC helper:' newline ...
     '  notebooks/2_eeg_source/r01_qc_v3_sign_convention_parcelpc.m']);
assert_dependency_exists(fullfile(stage2_dir, 'r01_qc_v3_pve1_hist_and_lowparcels.m'), ...
    ['Missing preserved Stage-2 QC helper:' newline ...
     '  notebooks/2_eeg_source/r01_qc_v3_pve1_hist_and_lowparcels.m']);

qc = struct();

qc.timeseries_gain = r01_qc_v3_run_timeseries_and_gain_summary( ...
    outDir, ...
    'PreferGNORM', true, ...
    'ChunkRows', 20000);

qc.sign = r01_qc_v3_sign_convention_parcelpc( ...
    outDir, ...
    'PreferGNORM', true, ...
    'ParcelsPerRun', 25, ...
    'CorrThr', 0.99);

[qc.pve_run_summary, qc.pve_parcel_summary] = r01_qc_v3_pve1_hist_and_lowparcels( ...
    outDir, ...
    'PreferGNORM', true, ...
    'BottomFrac', 0.05);

end

function assert_dependency_exists(path_to_file, message_text)
if exist(path_to_file, 'file') ~= 2
    error('%s', message_text);
end
end
