% qc_eeg_source_alignment_table_s2_24
%
% What this file does:
%   Read the Stage-2 parcel coverage summary and write a manuscript-facing
%   Table-S2 support CSV describing atlas coverage on the EEG volumetric
%   source grid.
%
% When to run it:
%   Run this after step 23 has written the coverage summary CSV.
%
% Manuscript linkage:
%   - Main Methods 2.2.2
%   - Supplementary Methods 1.2
%   - Supplementary Results 2.2
%   - Supplementary Table S2
%   - Supplementary Fig. S1A,B support
%
% Inputs expected:
%   - batch_parcel_coverage_summary_v3.csv from step 23
%   - optional scout-build summary CSV from step 22
%
% Outputs written:
%   - table_s2_eeg_atlas_alignment_summary.csv
%
% Manual dependency:
%   The upstream Brainstorm source localization and atlas import remain
%   hybrid/manual. This script summarizes the downstream scripted outputs
%   from that workflow; it does not replace Brainstorm screenshots or final
%   manual figure assembly for Figure S1A,B.

% -------------------------------------------------------------------------
% Step 0. Locate this stage folder and add it to the MATLAB path
% -------------------------------------------------------------------------
this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
if isempty(this_dir)
    error('Could not resolve the stage-2 script location. Run this file from disk.');
end
addpath(this_dir);

% -------------------------------------------------------------------------
% Step 1. User-editable roots and inputs
% -------------------------------------------------------------------------
project_root = '<SET_PROJECT_ROOT>';

stage2_derivatives_root = fullfile(project_root, '02_derivatives', 'stage2_eeg_source');
parcel_output_dir = fullfile(stage2_derivatives_root, 'parcel_exports');
stage2_qc_root = fullfile(project_root, '04_qc', 'stage2_eeg_source', 'tables');

coverage_summary_csv = fullfile(parcel_output_dir, 'batch_parcel_coverage_summary_v3.csv');
scout_build_summary_csv = fullfile(stage2_qc_root, 'batch_volgrid_scout_build.csv');
table_s2_csv = fullfile(stage2_qc_root, 'table_s2_eeg_atlas_alignment_summary.csv');

% -------------------------------------------------------------------------
% Step 2. Resolve directories
% -------------------------------------------------------------------------
assert_configured_input_dir(project_root, 'project_root');
assert_configured_input_dir(parcel_output_dir, 'parcel_output_dir');
ensure_parent_dir(table_s2_csv);

% -------------------------------------------------------------------------
% Step 3. Print a short run summary for the user
% -------------------------------------------------------------------------
fprintf('\nStage 2 / Step 24: Build Table-S2 EEG source-alignment support file\n');
fprintf('  Project root:          %s\n', project_root);
fprintf('  Parcel output dir:     %s\n', parcel_output_dir);
fprintf('  Coverage summary CSV:  %s\n', coverage_summary_csv);
fprintf('  Scout-build summary:   %s\n', scout_build_summary_csv);
fprintf('  Table-S2 output CSV:   %s\n\n', table_s2_csv);

% -------------------------------------------------------------------------
% Step 4. Read the coverage summary written by the preserved exporter
% -------------------------------------------------------------------------
if ~exist(coverage_summary_csv, 'file')
    error('Coverage summary CSV not found: %s', coverage_summary_csv);
end

coverageT = readtable(coverage_summary_csv);
required_cols = { ...
    'runTag','sub','ses', ...
    'n_scouts','n_found','n_missing','coverage_frac', ...
    'minVertices','n_valid', ...
    'TessNbVertices','nVert_gridloc', ...
    'n_assigned_vertices','overlap_vertices'};
assert_has_columns(coverageT, required_cols, coverage_summary_csv);

% -------------------------------------------------------------------------
% Step 5. Optionally cross-check the scout-build summary
% -------------------------------------------------------------------------
if ~isempty(scout_build_summary_csv) && exist(scout_build_summary_csv, 'file')
    scoutT = readtable(scout_build_summary_csv);
    scout_required_cols = {'sub','ses','nScouts','nVertExpected'};
    assert_has_columns(scoutT, scout_required_cols, scout_build_summary_csv);

    joined = outerjoin( ...
        coverageT(:, {'sub','ses','n_scouts','TessNbVertices','nVert_gridloc'}), ...
        scoutT(:, {'sub','ses','nScouts','nVertExpected'}), ...
        'Keys', {'sub','ses'}, ...
        'MergeKeys', true, ...
        'Type', 'left');

    scout_count_mismatch = joined.n_scouts ~= joined.nScouts;
    grid_size_mismatch = joined.TessNbVertices ~= joined.nVertExpected | joined.nVert_gridloc ~= joined.nVertExpected;

    if any(scout_count_mismatch, 'omitnan')
        warning('Scout-count mismatch detected between coverage summary and scout-build summary.');
    end
    if any(grid_size_mismatch, 'omitnan')
        warning('Grid-size mismatch detected between coverage summary and scout-build summary.');
    end
else
    fprintf('Scout-build summary not found. Continuing without the optional cross-check.\n');
end

% -------------------------------------------------------------------------
% Step 6. Build the manuscript-facing Table-S2 support output
% -------------------------------------------------------------------------
overlap_rate = coverageT.overlap_vertices ./ max(coverageT.n_assigned_vertices, 1);

tableS2 = table( ...
    string(coverageT.sub), ...
    string(coverageT.ses), ...
    string(coverageT.runTag), ...
    coverageT.n_scouts, ...
    coverageT.n_found, ...
    coverageT.n_missing, ...
    coverageT.coverage_frac, ...
    coverageT.minVertices, ...
    coverageT.n_valid, ...
    coverageT.TessNbVertices, ...
    coverageT.nVert_gridloc, ...
    coverageT.n_assigned_vertices, ...
    coverageT.overlap_vertices, ...
    overlap_rate, ...
    'VariableNames', { ...
    'Subject','Session','Run','Scouts','Parcels_found','Parcels_missing', ...
    'Coverage_fraction','MinDipoles','Parcels_valid_ge_MinDipoles', ...
    'Scout_grid_size','Kernel_grid_size','Assigned_vertices', ...
    'Overlap_vertices','Overlap_rate'});

% -------------------------------------------------------------------------
% Step 7. Write the Table-S2 support CSV
% -------------------------------------------------------------------------
writetable(tableS2, table_s2_csv);

% -------------------------------------------------------------------------
% Step 8. Point the user to the manual/hybrid figure note
% -------------------------------------------------------------------------
fprintf('\nStage 2 / Step 24 complete.\n');
fprintf('Wrote Table-S2 support CSV: %s\n', table_s2_csv);
fprintf(['Figure S1A,B still requires manual/hybrid Brainstorm screenshots or later ' ...
    'assembly outside this script.\n']);

function assert_configured_input_dir(path_value, label)
path_char = char(path_value);
if isempty(strtrim(path_char)) || contains(path_char, '<SET_')
    error('%s is still using a placeholder path. Edit this script before running.', label);
end
if ~exist(path_char, 'dir')
    error('%s does not exist: %s', label, path_char);
end
end

function assert_has_columns(T, required_cols, label)
missing = required_cols(~ismember(required_cols, T.Properties.VariableNames));
if ~isempty(missing)
    error('Missing required columns in %s: %s', label, strjoin(missing, ', '));
end
end

function ensure_parent_dir(file_path)
parent_dir = fileparts(file_path);
if ~isempty(parent_dir) && ~exist(parent_dir, 'dir')
    mkdir(parent_dir);
end
end
