outDir = 'C:\EEGFMRI\hmm\R01_rerun\02_derivatives\eeg_source\parcel_pc1';

Tts = r01_qc_v3_run_timeseries_and_gain_summary(outDir, 'PreferGNORM', true, 'ChunkRows', 20000);

QCsign = r01_qc_v3_sign_convention_parcelpc(outDir, 'PreferGNORM', true, 'ParcelsPerRun', 25, 'CorrThr', 0.99);

[RUNSUM, PARCSUM] = r01_qc_v3_pve1_hist_and_lowparcels(outDir, 'PreferGNORM', true, 'BottomFrac', 0.05);

[Tparcel, Tsens] = r01_qc_v3_pve1_per_parcel_summary(outDir, 'PreferGNORM', true, 'NSens', 10);
