function qc = run_eeg_parcel_export_qc_summaries(outDir)
%RUN_EEG_PARCEL_EXPORT_QC_SUMMARIES Public helper for Stage-2 QC sidecars.
%
% What this helper does:
%   Runs the preserved Stage-2 exporter QC summaries that feed the cleaned
%   public notebook `25_qc_eeg_parcel_exports_table_s3_and_figures_s2_s4`.
%
% When it is used:
%   Run this after `23_export_eeg_parcel_pc1_and_gain_normalize.m` and
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
