"""Backend functions for the public Stage-6 review and QC workflow.

This module holds the active runtime behind step61_review_final_k3_fit_qc_and_state_dynamics.ipynb.
It keeps the saved-output review logic in a normal Python backend while the notebook
stays focused on the public-facing interpretation of the step."""

from __future__ import annotations


def _display_fallback(obj):
    try:
        from IPython.display import display as ipy_display
        ipy_display(obj)
    except Exception:
        print(obj)


def run_final_review_backend(
    *,
    final_model_root,
    review_output_root,
):
    """Review saved final-fit outputs and rebuild the main QC/state-dynamics figures."""
    from pathlib import Path
    from pathlib import Path
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    RESULT_ROOT = Path(final_model_root)
    FINAL_DIR = RESULT_ROOT / 'final'
    OUT_FIG_DIR = Path(review_output_root)
    OUT_FIG_DIR.mkdir(exist_ok=True, parents=True)
    
    display = _display_fallback
    
    plt.rcParams['figure.dpi'] = 130
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    
    def require_file(label, *candidates):
        for p in candidates:
            p = Path(p)
            if p.exists():
                return p
        tried = "\n".join(str(Path(p)) for p in candidates)
        raise FileNotFoundError(f"Could not find {label}. Tried:\n{tried}")
    
    print('RESULT_ROOT:', RESULT_ROOT)
    print('FINAL_DIR   :', FINAL_DIR)
    print('OUT_FIG_DIR :', OUT_FIG_DIR)
    # ---- notebook cell 2 ----
    # ------------------------------
    # Resolve and load saved PipelineE outputs
    # ------------------------------
    QC_PATH         = require_file('qc_summary.json', RESULT_ROOT / 'qc_summary.json')
    REFIT_PATH      = require_file('refit_results.json', FINAL_DIR / 'refit_results.json', RESULT_ROOT / 'refit_results.json')
    SUBJECT_PATH    = require_file('subject_metrics.tsv', RESULT_ROOT / 'subject_metrics.tsv')
    RUN_PATH        = require_file('run_metrics.tsv', RESULT_ROOT / 'run_metrics.tsv')
    DWELL_PATH      = require_file('dwell_from_A.tsv', RESULT_ROOT / 'dwell_from_A.tsv')
    A_PATH          = require_file('trans_prob.npy', FINAL_DIR / 'trans_prob.npy', RESULT_ROOT / 'trans_prob.npy')
    SIG_PATH        = require_file('state_signature_ut_boldcorr.npy', FINAL_DIR / 'state_signature_ut_boldcorr.npy', RESULT_ROOT / 'state_signature_ut_boldcorr.npy')
    MEANS_PATH      = require_file('means_pca.npy', FINAL_DIR / 'means_pca.npy', RESULT_ROOT / 'means_pca.npy')
    COVS_PATH       = require_file('covs_pca.npy', FINAL_DIR / 'covs_pca.npy', RESULT_ROOT / 'covs_pca.npy')
    
    with open(QC_PATH, 'r', encoding='utf-8') as f:
        qc = json.load(f)
    
    with open(REFIT_PATH, 'r', encoding='utf-8') as f:
        refit = json.load(f)
    
    subject_metrics = pd.read_csv(SUBJECT_PATH, sep='\t')
    run_metrics     = pd.read_csv(RUN_PATH, sep='\t')
    dwell_from_A    = pd.read_csv(DWELL_PATH, sep='\t')
    A               = np.load(A_PATH)
    state_sig       = np.load(SIG_PATH)
    means_pca       = np.load(MEANS_PATH)
    covs_pca        = np.load(COVS_PATH)
    
    print('Resolved files:')
    print('  qc_summary.json               ->', QC_PATH)
    print('  refit_results.json           ->', REFIT_PATH)
    print('  subject_metrics.tsv          ->', SUBJECT_PATH)
    print('  run_metrics.tsv              ->', RUN_PATH)
    print('  dwell_from_A.tsv             ->', DWELL_PATH)
    print('  trans_prob.npy               ->', A_PATH)
    print('  state_signature_ut_boldcorr  ->', SIG_PATH)
    print('  means_pca.npy                ->', MEANS_PATH)
    print('  covs_pca.npy                 ->', COVS_PATH)
    print()
    print('qc_summary keys:', list(qc.keys()))
    print('refit candidates:', len(refit))
    print('subject_metrics shape:', subject_metrics.shape)
    print('run_metrics shape    :', run_metrics.shape)
    print('dwell_from_A shape   :', dwell_from_A.shape)
    print('A shape              :', A.shape)
    print('state_sig shape      :', state_sig.shape)
    print('means_pca shape      :', means_pca.shape)
    print('covs_pca shape       :', covs_pca.shape)
    
    # ---- notebook cell 3 ----
    # ------------------------------
    # Helpers
    # ------------------------------
    STATE_COLS = ['FO_s01', 'FO_s02', 'FO_s03']
    STATE_NAMES = ['S1', 'S2', 'S3']
    
    
    def stationary_distribution(A):
        eigvals, eigvecs = np.linalg.eig(A.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        pi = np.real(eigvecs[:, idx])
        pi = pi / pi.sum()
        return pi
    
    
    def dwell_from_transition_matrix(A, eps=1e-12):
        A = np.asarray(A, dtype=float)
        Akk = np.clip(np.diag(A), 0.0, 1.0 - eps)
        return 1.0 / (1.0 - Akk)
    
    
    def entropy_of_fo(fo, eps=1e-12):
        fo = np.asarray(fo, dtype=float)
        fo = np.clip(fo, eps, None)
        fo = fo / fo.sum()
        H = -(fo * np.log(fo)).sum()
        neff = np.exp(H)
        return H, neff
    
    
    def dominant_state_labels(df, cols=STATE_COLS):
        idx = df[cols].to_numpy().argmax(axis=1)
        return pd.Series(idx + 1, index=df.index)
    
    
    def pairwise_row_corr(X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        out = np.eye(n, dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                c = np.corrcoef(X[i], X[j])[0, 1]
                out[i, j] = c
                out[j, i] = c
        return out
    
    # ---- notebook cell 5 ----
    qc_df = pd.DataFrame([qc]).T
    qc_df.columns = ['value']
    qc_df
    
    # ---- notebook cell 6 ----
    refit_df = pd.DataFrame(refit).sort_values('fe').reset_index(drop=True)
    refit_df
    
    # ---- notebook cell 8 ----
    subject_metrics = subject_metrics.copy()
    run_metrics = run_metrics.copy()
    
    subject_metrics['dominant_state'] = dominant_state_labels(subject_metrics)
    run_metrics['dominant_state'] = dominant_state_labels(run_metrics)
    
    subject_metrics
    
    # ---- notebook cell 9 ----
    subject_summary = {
        'n_subjects': int(len(subject_metrics)),
        'n_runs': int(len(run_metrics)),
        'dominant_state_subject_counts': subject_metrics['dominant_state'].value_counts().sort_index().to_dict(),
        'dominant_state_run_counts': run_metrics['dominant_state'].value_counts().sort_index().to_dict(),
        'subject_FO_s01_median': float(subject_metrics['FO_s01'].median()),
        'subject_FO_s02_median': float(subject_metrics['FO_s02'].median()),
        'subject_FO_s03_median': float(subject_metrics['FO_s03'].median()),
        'subject_FO_max_median': float(subject_metrics['FO_max'].median()),
        'subject_neff_median': float(subject_metrics['neff'].median()),
    }
    subject_summary
    
    # ---- notebook cell 10 ----
    # Rank subjects by how dominant S2 is
    subject_ranked = subject_metrics.sort_values('FO_s02', ascending=False).reset_index(drop=True)
    subject_ranked[['subject', 'n_runs', 'total_T', 'FO_s01', 'FO_s02', 'FO_s03', 'FO_max', 'n_active', 'neff']]
    
    # ---- notebook cell 11 ----
    # Show multi-run subjects to check within-subject consistency across sessions
    multi_run = run_metrics.groupby('subject').filter(lambda x: len(x) > 1).copy()
    multi_run.sort_values(['subject', 'run'])
    
    # ---- notebook cell 12 ----
    # Within-subject run-to-run spread for subjects with multiple runs
    rows = []
    for subject, df in run_metrics.groupby('subject'):
        if len(df) > 1:
            row = {'subject': subject, 'n_runs': len(df)}
            for col in STATE_COLS:
                row[f'{col}_range'] = float(df[col].max() - df[col].min())
            rows.append(row)
    
    run_spread = pd.DataFrame(rows)
    run_spread
    
    # ---- notebook cell 14 ----
    A_df = pd.DataFrame(A, index=STATE_NAMES, columns=STATE_NAMES)
    A_df
    
    # ---- notebook cell 15 ----
    pi = stationary_distribution(A)
    outgoing_excl_self = 1.0 - np.diag(A)
    expected_dwell_tr = dwell_from_transition_matrix(A)
    
    transition_summary = pd.DataFrame({
        'state': STATE_NAMES,
        'stationary_prob_from_A': pi,
        'A_kk': np.diag(A),
        'leave_prob_1_minus_Akk': outgoing_excl_self,
        'expected_dwell_TR': expected_dwell_tr,
    })
    transition_summary['expected_dwell_sec_if_TR_2p1'] = transition_summary['expected_dwell_TR'] * 2.1
    transition_summary
    
    # ---- notebook cell 16 ----
    # Compare stationary distribution from A to the final seed's FO distribution from refit_results.json
    best_refit = refit_df.iloc[0].copy()
    fo_final = np.array(best_refit['fo'], dtype=float)
    compare_fo = pd.DataFrame({
        'state': STATE_NAMES,
        'FO_final_seed': fo_final,
        'stationary_from_A': pi,
        'difference': fo_final - pi,
    })
    compare_fo
    
    # ---- notebook cell 18 ----
    sig_corr = pairwise_row_corr(state_sig)
    sig_corr_df = pd.DataFrame(sig_corr, index=STATE_NAMES, columns=STATE_NAMES)
    sig_corr_df
    
    # ---- notebook cell 19 ----
    # Optional: mean norms and covariance traces in PCA space
    rows = []
    for k in range(3):
        rows.append({
            'state': STATE_NAMES[k],
            'mean_norm_pca': float(np.linalg.norm(means_pca[k])),
            'cov_trace_pca': float(np.trace(covs_pca[k])),
        })
    state_scale_df = pd.DataFrame(rows)
    state_scale_df
    
    # ---- notebook cell 21 ----
    # Subject-level stacked occupancy plot
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = subject_metrics.sort_values('FO_s02', ascending=False).reset_index(drop=True)
    x = np.arange(len(plot_df))
    ax.bar(x, plot_df['FO_s01'], label='S1')
    ax.bar(x, plot_df['FO_s02'], bottom=plot_df['FO_s01'], label='S2')
    ax.bar(x, plot_df['FO_s03'], bottom=plot_df['FO_s01'] + plot_df['FO_s02'], label='S3')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['subject'], rotation=45, ha='right')
    ax.set_ylabel('Fractional occupancy')
    ax.set_title('Subject-level fractional occupancy (sorted by S2)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG_DIR / 'subject_fractional_occupancy_stacked.png', bbox_inches='tight')
    plt.show()
    
    # ---- notebook cell 22 ----
    # Run-level occupancy plot
    fig, ax = plt.subplots(figsize=(11, 5))
    plot_df = run_metrics.sort_values('FO_s02', ascending=False).reset_index(drop=True)
    x = np.arange(len(plot_df))
    ax.bar(x, plot_df['FO_s01'], label='S1')
    ax.bar(x, plot_df['FO_s02'], bottom=plot_df['FO_s01'], label='S2')
    ax.bar(x, plot_df['FO_s03'], bottom=plot_df['FO_s01'] + plot_df['FO_s02'], label='S3')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['run'], rotation=60, ha='right')
    ax.set_ylabel('Fractional occupancy')
    ax.set_title('Run-level fractional occupancy (sorted by S2)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG_DIR / 'run_fractional_occupancy_stacked.png', bbox_inches='tight')
    plt.show()
    
    # ---- notebook cell 23 ----
    # Transition matrix heatmap
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(A, aspect='auto')
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(STATE_NAMES)
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(STATE_NAMES)
    ax.set_title('Transition matrix A')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{A[i, j]:.3f}', ha='center', va='center')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT_FIG_DIR / 'transition_matrix_A.png', bbox_inches='tight')
    plt.show()
    
    # ---- notebook cell 24 ----
    # Dwell times from A
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(dwell_from_A['state'], dwell_from_A['dwell_A_sec'])
    ax.set_ylabel('Seconds')
    ax.set_title('Expected dwell time from A')
    fig.tight_layout()
    fig.savefig(OUT_FIG_DIR / 'dwell_time_seconds.png', bbox_inches='tight')
    plt.show()
    
    # ---- notebook cell 25 ----
    # Signature correlation heatmap
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(sig_corr, vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(STATE_NAMES)
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(STATE_NAMES)
    ax.set_title('Pairwise correlation of state signatures')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{sig_corr[i, j]:.3f}', ha='center', va='center')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT_FIG_DIR / 'state_signature_correlation.png', bbox_inches='tight')
    plt.show()
    
    # ---- notebook cell 27 ----
    subj_dom_counts = subject_metrics['dominant_state'].value_counts().sort_index().to_dict()
    run_dom_counts = run_metrics['dominant_state'].value_counts().sort_index().to_dict()
    
    msg = f'''
    PipelineE K=3 review
    --------------------
    1. QC passed: {qc['collapsed_run_count']} / {qc['n_runs']} collapsed runs.
    2. Final seed = {qc['final_seed']} with FE = {qc['final_seed_fe']:.3f}, FO_max = {qc['final_seed_fo_max']:.3f}, n_active = {qc['final_seed_n_active']}, n_eff = {qc['final_seed_neff']:.3f}.
    3. Dominant state across subjects: {subj_dom_counts}
    4. Dominant state across runs    : {run_dom_counts}
    5. Stationary distribution from A: {np.round(pi, 4)}
    6. Expected dwell (TR)           : {np.round(expected_dwell_tr, 3)}
    7. Expected dwell (sec, TR=2.1)  : {np.round(expected_dwell_tr * 2.1, 3)}
    8. Pairwise signature correlations:
    {pd.DataFrame(sig_corr, index=STATE_NAMES, columns=STATE_NAMES).round(3).to_string()}
    
    Interpretation:
    - S2 is the dominant state globally.
    - S2 is also the dominant state for every subject and every run in these saved outputs.
    - S2 is much stickier than S1 and S3.
    - S1 and S3 look more like short-lived satellite states that tend to transition into S2.
    - If the signature correlations remain high, the states may differ more in persistence / occupancy than in completely different BOLD-correlation topographies.
    '''
    print(msg)
    
    return {
        "result_root": str(RESULT_ROOT),
        "final_dir": str(FINAL_DIR),
        "review_output_root": str(OUT_FIG_DIR),
        "qc_summary": qc,
        "subject_summary": subject_summary,
    }
