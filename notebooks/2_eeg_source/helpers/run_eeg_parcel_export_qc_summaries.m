function qc = run_eeg_parcel_export_qc_summaries(outDir)
%RUN_EEG_PARCEL_EXPORT_QC_SUMMARIES Public helper for Stage-2 QC sidecars.
%
% What this helper does:
%   Runs the active descriptive Stage-2 QC helpers that feed the cleaned
%   public notebook `step26_qc_eeg_parcel_exports_table_s3_and_figures_s2_s4`.
%
% When it is used:
%   The public Stage-2 workflow calls this helper from
%   `step25_generate_eeg_parcel_export_qc_sidecars.m` after Step 23 and
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
%   This wrapper preserves the current v3 QC behavior and default options
%   while exposing that behavior through the active public helper layer.

this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
if ~isempty(this_dir)
    addpath(this_dir);
end

assert_dependency_exists(fullfile(this_dir, 'summarize_run_timeseries_gain_qc.m'), ...
    ['Missing Stage-2 run-timeseries QC helper:' newline ...
     '  notebooks/2_eeg_source/helpers/summarize_run_timeseries_gain_qc.m']);
assert_dependency_exists(fullfile(this_dir, 'summarize_sign_convention_qc.m'), ...
    ['Missing Stage-2 sign-convention QC helper:' newline ...
     '  notebooks/2_eeg_source/helpers/summarize_sign_convention_qc.m']);
assert_dependency_exists(fullfile(this_dir, 'summarize_pve1_histogram_and_lowparcel_qc.m'), ...
    ['Missing Stage-2 PVE1 QC helper:' newline ...
     '  notebooks/2_eeg_source/helpers/summarize_pve1_histogram_and_lowparcel_qc.m']);

qc = struct();

qc.timeseries_gain = summarize_run_timeseries_gain_qc( ...
    outDir, ...
    'PreferGNORM', true, ...
    'ChunkRows', 20000);

qc.sign = summarize_sign_convention_qc( ...
    outDir, ...
    'PreferGNORM', true, ...
    'ParcelsPerRun', 25, ...
    'CorrThr', 0.99);

[qc.pve_run_summary, qc.pve_parcel_summary] = summarize_pve1_histogram_and_lowparcel_qc( ...
    outDir, ...
    'PreferGNORM', true, ...
    'BottomFrac', 0.05);

end

function assert_dependency_exists(path_to_file, message_text)
if exist(path_to_file, 'file') ~= 2
    error('%s', message_text);
end
end
