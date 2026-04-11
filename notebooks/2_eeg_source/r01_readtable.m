T = readtable(fullfile(P.qc.tables, sprintf('include_manifest_%s.csv', tag)));

% Sort by continuity risk
disp('Lowest kept_median_sec (most chopped typical segments):');
disp(sortrows(T(:,{'stem','usable_fraction','kept_n_segments','kept_median_sec','kept_max_sec','kept_largest_frac','effective_transition_pairs'}), 'kept_median_sec', 'ascend'));

disp('Highest kept_n_segments (most islands):');
disp(sortrows(T(:,{'stem','usable_fraction','kept_n_segments','kept_median_sec','kept_max_sec','kept_largest_frac','effective_transition_pairs'}), 'kept_n_segments', 'descend'));

disp('Lowest kept_largest_frac (retained time spread out):');
disp(sortrows(T(:,{'stem','usable_fraction','kept_n_segments','kept_median_sec','kept_max_sec','kept_largest_frac','effective_transition_pairs'}), 'kept_largest_frac', 'ascend'));
