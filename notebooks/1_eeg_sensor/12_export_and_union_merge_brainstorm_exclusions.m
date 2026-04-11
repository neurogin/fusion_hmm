% 12_export_and_union_merge_brainstorm_exclusions
%
% Public-facing stage-1 entry point for the scripted handoff from manual
% Brainstorm exclusion marking to exported and merged exclusion intervals.
%
% Manuscript linkage:
%   - Main Methods 2.2.1
%   - Supplementary Methods 1.1
%
% Inputs:
%   - Brainstorm raw-link MAT files under <brainstorm_db_root>\data\**\data_0raw_*.mat
%   - Manual Brainstorm marking already completed:
%       * boundary
%       * BAD
%
% Outputs:
%   - <bst_export_dir>\*_bst_exclusions.tsv
%   - <bst_export_dir>\bst_exclusions_batch_summary.csv
%   - <bst_export_dir>\*_excl_union.tsv
%   - <bst_export_dir>\*_excl_union_qc.csv
%
% Notes:
%   - This stage remains hybrid because the Brainstorm marking step is
%     manual and documented separately in 11_brainstorm_exclusion_marking_manual.md.
%   - The recovered exporter helper is preserved as-is in helpers/ and is
%     intentionally not "improved" in this first public refactor pass.

stage1_dir = fileparts(mfilename('fullpath'));
if isempty(stage1_dir)
    error('Could not resolve the stage-1 script location. Run this file from disk.');
end

helper_dir = fullfile(stage1_dir, 'helpers');
addpath(stage1_dir);
addpath(helper_dir);

P = r01_stage1_params();

% -------------------------------------------------------------------------
% User-configurable parameters
% -------------------------------------------------------------------------
overwrite_existing_outputs = true;
recursive_bst_scan = true;

assert_configured_input_dir(P.paths.brainstorm_db_root, 'P.paths.brainstorm_db_root');

fprintf('\nStage 1 / Step 12: Export and union-merge Brainstorm exclusions\n');
fprintf('  Brainstorm DB root: %s\n', char(P.paths.brainstorm_db_root));
fprintf('  Export directory:   %s\n', char(P.paths.bst_export_dir));
fprintf('  File filter:        %s\n', char(P.file_filter));
fprintf('  Labels kept:        BAD, boundary, bad_boundary\n');
fprintf('  Merge tolerance:    %.3f sec\n', P.mask.merge.adjacency_tol_sec);
fprintf('  Min union duration: %.3f sec\n\n', P.mask.merge.min_dur_sec);

% Step 12A. Export Brainstorm BAD / boundary intervals to TSV.
r01_batch_export_bst_exclusions_Fevents( ...
    char(P.paths.brainstorm_db_root), ...
    char(P.paths.bst_export_dir), ...
    'overwrite', overwrite_existing_outputs, ...
    'file_filter', char(P.file_filter), ...
    'recursive', recursive_bst_scan);

% Step 12B. Merge overlapping or touching exclusions into per-run unions.
r01_batch_merge_exclusions_union( ...
    char(P.paths.bst_export_dir), ...
    'adjacency_tol_sec', P.mask.merge.adjacency_tol_sec, ...
    'min_dur_sec', P.mask.merge.min_dur_sec, ...
    'labels', ["BAD","boundary","bad_boundary"], ...
    'overwrite', overwrite_existing_outputs);

fprintf('\nStage 1 / Step 12 complete.\n');
fprintf('Next scripted step: run 13_eeg_run_qc_and_table_s1.m.\n');

function assert_configured_input_dir(path_value, label)
path_char = char(path_value);
if contains(path_char, '<SET_')
    error('%s is still using a placeholder path. Edit helpers/r01_stage1_params.m first.', label);
end
if ~exist(path_char, 'dir')
    error('%s does not exist: %s', label, path_char);
end
end
