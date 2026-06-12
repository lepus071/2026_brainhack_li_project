#!/usr/bin/env python3
# ==============================================================================
# check_e_global_signal_corr.py
#
# Sanity check for Phase 6 Limbic-network warning on Unique_e / Shared_e_xf /
# Shared_e_xs: quantify how correlated the HRF-convolved error signal (e) is
# with fMRIPrep's global_signal confound, per subject and at the group level.
# ==============================================================================

import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_1samp

PROJ_DIR = '/home/ser/2026_brainheck_li_project'
SUBJECT_LIST = f'{PROJ_DIR}/codes/subjectlist.txt'
AFNI_DIR = f'{PROJ_DIR}/data/derivatives/afni_fmriprep'
FMRIPREP_DIR = f'{PROJ_DIR}/data/derivatives/fmriprep'

TR = 2.0
WINDOW_SIZE = 30
TASK_RUN_GLOB = '*task-rightlearning*desc-confounds_timeseries.tsv'  # matches r03


def spm_hrf(dt=0.1):
    t = np.arange(0, 30.0, dt)
    alpha1, beta1 = 6.0, 1.0
    alpha2, beta2 = 16.0, 1.0
    c = 1.0 / 6.0
    from scipy.special import gammaln

    def gamma_pdf(x, alpha, beta):
        x = np.maximum(x, 1e-10)
        return np.exp(alpha * np.log(beta) + (alpha - 1) * np.log(x) - beta * x - gammaln(alpha))

    g1 = gamma_pdf(t, alpha1, beta1)
    g2 = gamma_pdf(t, alpha2, beta2)
    hrf = g1 - c * g2
    return hrf / np.sum(hrf)


def main():
    with open(SUBJECT_LIST, 'r') as f:
        subjects = [line.strip() for line in f if line.strip()]

    dt = 0.1
    hrf_kernel = spm_hrf(dt=dt)
    results = []

    for subj in subjects:
        behav_csv = f"{PROJ_DIR}/data/derivatives/behavioral_features/{subj}/CSV_Data/{subj}_Triarchic_Fitted.csv"
        if not os.path.exists(behav_csv):
            continue
        subj_behav = pd.read_csv(behav_csv)
        if subj_behav.empty:
            continue

        # --- Load global_signal from the task-rightlearning confounds file ---
        pattern = f"{FMRIPREP_DIR}/{subj}/ses-*/func/{subj}_{TASK_RUN_GLOB}"
        matches = glob.glob(pattern)
        if not matches:
            print(f"  [{subj}] confounds file not found, skipping.")
            continue
        confounds = pd.read_csv(matches[0], sep='\t')
        global_signal = confounds['global_signal'].fillna(0).values
        n_volumes = len(global_signal)

        # --- Build the same HRF-convolved error (e) time course as Phase 4 ---
        onsets = 16.5 + (subj_behav['Original_TrialNo'].values - 1) * 6.0
        total_time = n_volumes * TR
        t_grid = np.arange(0, total_time, dt)

        neural_e = np.zeros_like(t_grid)
        for j in range(len(subj_behav)):
            onset = onsets[j]
            duration = 4.5
            idx = np.where((t_grid >= onset) & (t_grid <= onset + duration))[0]
            if len(idx) > 0:
                neural_e[idx] = np.abs(subj_behav['Metacognitive_Engagement'].iloc[j])

        hemo_e = np.convolve(neural_e, hrf_kernel)[:len(t_grid)]

        # --- Sliding-window mean (Size=30, Step=1) for both signals ---
        n_windows = n_volumes - WINDOW_SIZE + 1
        if n_windows <= 0:
            continue

        y_err = np.zeros(n_windows)
        y_gs = np.zeros(n_windows)
        for i in range(n_windows):
            t_start = i * TR
            t_end = i * TR + WINDOW_SIZE * TR
            idx_win = np.where((t_grid >= t_start) & (t_grid <= t_end))[0]
            y_err[i] = np.mean(hemo_e[idx_win]) if len(idx_win) > 0 else 0.0
            y_gs[i] = np.mean(global_signal[i:i + WINDOW_SIZE])

        r, p = pearsonr(y_err, y_gs)
        results.append({'subject': subj, 'r': r, 'p': p, 'n_windows': n_windows})
        print(f"  [{subj}] corr(e, global_signal) = {r:+.3f} (p={p:.4f}, n={n_windows})")

    if not results:
        print("No subjects processed.")
        return

    df = pd.DataFrame(results)
    print("\n========================================================")
    print("Group-level summary: corr(e, global_signal)")
    print("========================================================")
    print(f"  mean r = {df['r'].mean():+.3f}")
    print(f"  std  r = {df['r'].std():.3f}")
    t, p = ttest_1samp(df['r'], 0.0)
    print(f"  one-sample t-test vs 0: t={t:.3f}, p={p:.4f}")
    print(f"  |r| > 0.3 in {np.sum(np.abs(df['r']) > 0.3)}/{len(df)} subjects")

    out_csv = f"{PROJ_DIR}/data/derivatives/riemannian_decoding/ml_results/e_global_signal_corr.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved per-subject results to: {out_csv}")


if __name__ == '__main__':
    main()
