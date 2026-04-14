function P = r01_stage1_params()
%R01_STAGE1_PARAMS Legacy compatibility wrapper for Stage-1 settings.
%
% The cleaned public Stage-1 workflow now uses
% `stage1_eeg_sensor_settings.m` as the descriptive editable config file.
% This legacy helper remains in place so older provenance code can still
% resolve the same settings without changing behavior.

P = stage1_eeg_sensor_settings();

end
