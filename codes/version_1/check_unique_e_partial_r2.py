#!/usr/bin/env python3
# ==============================================================================
# check_unique_e_partial_r2.py
#
# Partial-R^2 sensitivity analysis for Unique_e.
#
# Adds the (window-averaged) fMRIPrep global_signal as a 4th predictor
# alongside (e, x_f, x_s), then computes:
#
#   Unique_e_partial = R2(e, x_f, x_s, gs)  -  R2(x_f, x_s, gs)
#
# i.e. the variance e explains ON TOP OF global signal (and x_f, x_s).
# If Unique_e_partial collapses relative to the original Unique_e (which did
# not control for gs), the original thalamus/Limbic finding was largely a
# global-signal artifact. If it survives, e carries genuine information beyond
# global signal.
#
# Both inner (GroupKFold) and outer (LeaveOneGroupOut) CV are subject-grouped,
# matching the fix already applied to run_hybrid_group_decoding.py.
# ==============================================================================

import os
import sys
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.metrics import r2_score
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_hybrid_group_decoding import spm_hrf, PROJ_DIR, FEATURE_DIR, TR, WINDOW_SIZE

SUBJECT_LIST = f'{PROJ_DIR}/codes/subjectlist.txt'
ATLAS_DIR = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/atlases'
FMRIPREP_DIR = f'{PROJ_DIR}/data/derivatives/fmriprep'
TASK_RUN_GLOB = '*task-rightlearning*desc-confounds_timeseries.tsv'  # matches r03


def load_and_align_data_with_gs(subjects):
    X_behav_all, X_brain_all, groups_all = [], [], []
    dt = 0.1
    hrf_kernel = spm_hrf(dt=dt)

    for subj in subjects:
        feat_path = f"{FEATURE_DIR}/Riemannian_Features_{subj}.npy"
        behav_csv = f"{PROJ_DIR}/data/derivatives/behavioral_features/{subj}/CSV_Data/{subj}_Triarchic_Fitted.csv"
        gs_files = glob.glob(f"{FMRIPREP_DIR}/{subj}/ses-*/func/{subj}_{TASK_RUN_GLOB}")

        if not os.path.exists(feat_path) or not os.path.exists(behav_csv) or not gs_files:
            print(f"  [{subj}] missing data, skipping.")
            continue

        features = np.load(feat_path)  # (N_windows, 464)
        n_windows = features.shape[0]

        subj_behav = pd.read_csv(behav_csv)
        if subj_behav.empty:
            continue

        confounds = pd.read_csv(gs_files[0], sep='\t')
        global_signal = confounds['global_signal'].fillna(0).values

        onsets = 16.5 + (subj_behav['Original_TrialNo'].values - 1) * 6.0

        total_time = (n_windows + WINDOW_SIZE) * TR
        t_grid = np.arange(0, total_time, dt)

        neural_e = np.zeros_like(t_grid)
        neural_xf = np.zeros_like(t_grid)
        neural_xs = np.zeros_like(t_grid)

        for j in range(len(subj_behav)):
            onset = onsets[j]
            duration = 4.5
            idx = np.where((t_grid >= onset) & (t_grid <= onset + duration))[0]
            if len(idx) > 0:
                neural_e[idx] = np.abs(subj_behav['Metacognitive_Engagement'].iloc[j])
                neural_xf[idx] = subj_behav['CCN_Engagement'].iloc[j]
                neural_xs[idx] = subj_behav['Representation_Engagement'].iloc[j]

        hemo_e = np.convolve(neural_e, hrf_kernel)[:len(t_grid)]
        hemo_xf = np.convolve(neural_xf, hrf_kernel)[:len(t_grid)]
        hemo_xs = np.convolve(neural_xs, hrf_kernel)[:len(t_grid)]

        y_err = np.zeros(n_windows)
        y_xf = np.zeros(n_windows)
        y_xs = np.zeros(n_windows)
        y_gs = np.zeros(n_windows)
        valid_windows = np.zeros(n_windows, dtype=bool)

        for i in range(n_windows):
            t_start = i * TR
            t_end = i * TR + WINDOW_SIZE * TR

            if (t_start <= onsets[-1] + 4.5) and (t_end >= 16.5):
                idx_win = np.where((t_grid >= t_start) & (t_grid <= t_end))[0]
                if len(idx_win) > 0:
                    y_err[i] = np.mean(hemo_e[idx_win])
                    y_xf[i] = np.mean(hemo_xf[idx_win])
                    y_xs[i] = np.mean(hemo_xs[idx_win])
                    valid_windows[i] = True

            # global_signal is at TR resolution (1 sample per volume), not the
            # fine dt grid -> window-average directly over volumes i..i+WINDOW_SIZE-1
            gs_end = min(i + WINDOW_SIZE, len(global_signal))
            if i < len(global_signal):
                y_gs[i] = np.mean(global_signal[i:gs_end])

        X_brain_all.append(features[valid_windows])
        X_behav_all.append(np.vstack([
            y_err[valid_windows], y_xf[valid_windows], y_xs[valid_windows], y_gs[valid_windows]
        ]).T)
        groups_all.append(np.full(np.sum(valid_windows), int(subj.split('-')[1])))

    if not X_brain_all:
        return None, None, None

    return np.vstack(X_behav_all), np.vstack(X_brain_all), np.concatenate(groups_all)


def run_loso_r2(X_behav_raw, Y_brain_raw, cv_groups, cols):
    """LOSO + inner GroupKFold RidgeCV on X_behav_raw[:, cols] -> Y_brain_raw."""
    Y_pred = np.zeros_like(Y_brain_raw)
    logo = LeaveOneGroupOut()
    alphas = np.logspace(-3, 3, 10)

    for train_idx, test_idx in logo.split(X_behav_raw, Y_brain_raw, cv_groups):
        X_tr_raw, X_te_raw = X_behav_raw[train_idx][:, cols], X_behav_raw[test_idx][:, cols]
        Y_tr_raw, Y_te_raw = Y_brain_raw[train_idx], Y_brain_raw[test_idx]

        scaler_X = StandardScaler().fit(X_tr_raw)
        scaler_Y = StandardScaler().fit(Y_tr_raw)
        X_tr = scaler_X.transform(X_tr_raw)
        X_te = scaler_X.transform(X_te_raw)
        Y_tr = scaler_Y.transform(Y_tr_raw)

        groups_tr = cv_groups[train_idx]
        n_inner = len(np.unique(groups_tr))
        cv_inner = GroupKFold(n_splits=n_inner).split(X_tr, Y_tr, groups=groups_tr)

        clf = RidgeCV(alphas=alphas, cv=cv_inner).fit(X_tr, Y_tr)
        Y_pred[test_idx] = scaler_Y.inverse_transform(clf.predict(X_te))

    return np.maximum(0, r2_score(Y_brain_raw, Y_pred, multioutput='raw_values'))


def main():
    with open(SUBJECT_LIST, 'r') as f:
        subjects = [line.strip() for line in f if line.strip()]

    print("Loading data with global_signal as 4th predictor (e, x_f, x_s, gs)...")
    X_behav_raw, Y_brain_raw, cv_groups = load_and_align_data_with_gs(subjects)
    print(f"  > Total samples: {X_behav_raw.shape[0]} windows, subjects: {len(np.unique(cv_groups))}")

    print("\nFitting Full model (e, x_f, x_s, gs)...")
    r2_full = run_loso_r2(X_behav_raw, Y_brain_raw, cv_groups, cols=[0, 1, 2, 3])

    print("Fitting No-e model (x_f, x_s, gs)...")
    r2_no_e = run_loso_r2(X_behav_raw, Y_brain_raw, cv_groups, cols=[1, 2, 3])

    unique_e_partial = np.maximum(0, r2_full - r2_no_e)

    OUTPUT_DIR = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/ml_results'
    np.save(f"{OUTPUT_DIR}/Unique_e_partial_R2.npy", unique_e_partial)

    # Compare to original Unique_e
    orig = np.load(f"{OUTPUT_DIR}/Unique_e_R2.npy")

    labels_df = pd.read_csv(f'{ATLAS_DIR}/Combined_Atlas_Labels.csv').set_index('Index')['Label']

    print("\n========================================================")
    print("Comparison: Unique_e (original) vs Unique_e_partial (controlling for global signal)")
    print("========================================================")
    print(f"  mean R2: original={orig.mean():.4f}  partial={unique_e_partial.mean():.4f}")
    print(f"  max  R2: original={orig.max():.4f}  partial={unique_e_partial.max():.4f}")
    print(f"  nonzero ROIs: original={np.sum(orig>0)}/{len(orig)}  partial={np.sum(unique_e_partial>0)}/{len(unique_e_partial)}")

    print("\nTop-5 ROIs (Unique_e_partial):")
    order = np.argsort(unique_e_partial)[::-1][:5]
    for rank, i in enumerate(order, 1):
        roi_id = i + 2
        label = labels_df.get(roi_id, "UNKNOWN")
        print(f"  {rank}. ROI#{roi_id:3d}  partial_R2={unique_e_partial[i]:.4f}  (orig={orig[i]:.4f})  {label}")

    # Thalamus-specific check
    print("\nThalamus ROIs (orig vs partial):")
    for roi_id in [450, 438]:
        i = roi_id - 2
        label = labels_df.get(roi_id, "UNKNOWN")
        print(f"  ROI#{roi_id} {label}: orig={orig[i]:.4f}  partial={unique_e_partial[i]:.4f}")

    # Limbic network mean
    limbic_idx = [i for i, lbl in labels_df.items() if 'Limbic' in str(lbl)]
    limbic_orig = [orig[r-2] for r in limbic_idx if 0 <= r-2 < len(orig)]
    limbic_partial = [unique_e_partial[r-2] for r in limbic_idx if 0 <= r-2 < len(unique_e_partial)]
    print(f"\nLimbic network mean Unique_e R2: original={np.mean(limbic_orig):.4f}  partial={np.mean(limbic_partial):.4f}")


if __name__ == '__main__':
    main()
