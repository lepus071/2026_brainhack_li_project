#!/usr/bin/env python3
# ==============================================================================
# check_e_global_signal_corr_detrend.py
#
# Follow-up to check_e_global_signal_corr.py: tests whether the strong
# corr(e, global_signal) (often < -0.7) is driven by shared low-frequency
# temporal drift (e decreases over the run as a learning curve; global_signal
# may show an unrelated scanner/physiological drift in the opposite direction)
# rather than a genuine trial-by-trial relationship.
#
# For each subject, computes corr(e, global_signal):
#   (a) raw (as before)
#   (b) after linear detrend of both signals
#   (c) after removing a low-order polynomial (cubic) trend from both signals
#
# If |r| collapses after detrending, the raw correlation is mostly a shared
# drift artifact, not a real trial-level relationship.
# ==============================================================================

import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_1samp
from scipy.signal import detrend

PROJ_DIR = '/home/ser/2026_brainheck_li_project'
SUBJECT_LIST = f'{PROJ_DIR}/codes/subjectlist.txt'
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


def poly_detrend(y, order=3):
    """Remove a polynomial trend of given order (returns residuals)."""
    x = np.arange(len(y))
    coefs = np.polyfit(x, y, order)
    trend = np.polyval(coefs, x)
    return y - trend


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

        pattern = f"{FMRIPREP_DIR}/{subj}/ses-*/func/{subj}_{TASK_RUN_GLOB}"
        matches = glob.glob(pattern)
        if not matches:
            print(f"  [{subj}] confounds file not found, skipping.")
            continue
        confounds = pd.read_csv(matches[0], sep='\t')
        global_signal = confounds['global_signal'].fillna(0).values
        n_volumes = len(global_signal)

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

        # (a) raw
        r_raw, p_raw = pearsonr(y_err, y_gs)

        # (b) linear detrend
        y_err_lin = detrend(y_err, type='linear')
        y_gs_lin = detrend(y_gs, type='linear')
        r_lin, p_lin = pearsonr(y_err_lin, y_gs_lin)

        # (c) cubic polynomial detrend
        y_err_cub = poly_detrend(y_err, order=3)
        y_gs_cub = poly_detrend(y_gs, order=3)
        r_cub, p_cub = pearsonr(y_err_cub, y_gs_cub)

        results.append({
            'subject': subj,
            'r_raw': r_raw, 'p_raw': p_raw,
            'r_linear_detrend': r_lin, 'p_linear_detrend': p_lin,
            'r_cubic_detrend': r_cub, 'p_cubic_detrend': p_cub,
            'n_windows': n_windows,
        })
        print(f"  [{subj}] raw r={r_raw:+.3f} | linear-detrend r={r_lin:+.3f} | cubic-detrend r={r_cub:+.3f}")

    if not results:
        print("No subjects processed.")
        return

    df = pd.DataFrame(results)
    print("\n========================================================")
    print("Group-level summary")
    print("========================================================")
    for col in ['r_raw', 'r_linear_detrend', 'r_cubic_detrend']:
        t, p = ttest_1samp(df[col], 0.0)
        print(f"  {col}: mean={df[col].mean():+.3f}  std={df[col].std():.3f}  "
              f"t-test vs 0: t={t:.3f}, p={p:.4f}  |r|>0.3 in {np.sum(np.abs(df[col]) > 0.3)}/{len(df)}")

    out_csv = f"{PROJ_DIR}/data/derivatives/riemannian_decoding/ml_results/e_global_signal_corr_detrend.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved per-subject results to: {out_csv}")


if __name__ == '__main__':
    main()
