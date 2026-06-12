#!/usr/bin/env python3
# ==============================================================================
# run_hybrid_group_decoding.py  (version_2: Smith et al. 2006 dual-rate, 2-var)
#
# Hybrid Dynamic Neural Decoding Pipeline - Phase 4 (2-var VPA version):
# Joint Encoding & Variance Partitioning using ONLY x_f, x_s (no error signal e).
#
# This script executes:
# 1. Read Riemannian projected feature matrix (N_windows, 464) and behavioral data.
# 2. Use "behavioral parameters (x_f, x_s)" as X to predict "brain features (464 dimensions)" as Y.
# 3. Train 3 combined Ridge models via Leave-One-Subject-Out (LOSO).
# 4. Calculate 3 variance partitions (Unique_xf, Unique_xs, Shared_xf_xs) + Full_xf_xs.
# 5. Output R^2 numpy arrays (group + per-subject) for Phase 5 inverse mapping.
# ==============================================================================

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.metrics import r2_score
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

PROJ_DIR = '/home/ser/2026_brainheck_li_project'
SUBJECT_LIST = f'{PROJ_DIR}/codes/subjectlist.txt'
FEATURE_DIR  = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/features'
OUTPUT_DIR   = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/ml_results_v2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TR = 2.0
WINDOW_SIZE = 30


def spm_hrf(dt=0.1):
    """Generates SPM's canonical HRF kernel on a fine time grid (dt = 0.1s by default)"""
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
    return hrf / np.sum(hrf)  # Normalize area to 1


def load_and_align_data(subjects):
    """Read feature and behavioral data, and perform HRF convolution and window alignment (x_f, x_s only)."""
    X_brain_all, X_behav_all, groups_all = [], [], []
    dt = 0.1
    hrf_kernel = spm_hrf(dt=dt)

    for subj in subjects:
        feat_path = f"{FEATURE_DIR}/Riemannian_Features_{subj}.npy"
        behav_csv = f"{PROJ_DIR}/data/derivatives/behavioral_features/{subj}/CSV_Data/{subj}_Triarchic_Fitted.csv"

        if not os.path.exists(feat_path) or not os.path.exists(behav_csv):
            continue

        features = np.load(feat_path)  # Shape: (N_windows, 464)
        n_windows = features.shape[0]

        subj_behav = pd.read_csv(behav_csv)
        if subj_behav.empty:
            continue

        onsets = 16.5 + (subj_behav['Original_TrialNo'].values - 1) * 6.0

        total_time = (n_windows + WINDOW_SIZE) * TR
        t_grid = np.arange(0, total_time, dt)

        neural_xf = np.zeros_like(t_grid)
        neural_xs = np.zeros_like(t_grid)

        for j in range(len(subj_behav)):
            onset = onsets[j]
            duration = 4.5  # Trial duration / motor execution phase
            idx = np.where((t_grid >= onset) & (t_grid <= onset + duration))[0]
            if len(idx) > 0:
                neural_xf[idx] = subj_behav['CCN_Engagement'].iloc[j]
                neural_xs[idx] = subj_behav['Representation_Engagement'].iloc[j]

        hemo_xf = np.convolve(neural_xf, hrf_kernel)[:len(t_grid)]
        hemo_xs = np.convolve(neural_xs, hrf_kernel)[:len(t_grid)]

        y_xf = np.zeros(n_windows)
        y_xs = np.zeros(n_windows)
        valid_windows = np.zeros(n_windows, dtype=bool)

        for i in range(n_windows):
            t_start = i * TR
            t_end = i * TR + WINDOW_SIZE * TR

            if (t_start <= onsets[-1] + 4.5) and (t_end >= 16.5):
                idx_win = np.where((t_grid >= t_start) & (t_grid <= t_end))[0]
                if len(idx_win) > 0:
                    y_xf[i] = np.mean(hemo_xf[idx_win])
                    y_xs[i] = np.mean(hemo_xs[idx_win])
                    valid_windows[i] = True

        X_brain_all.append(features[valid_windows])
        X_behav_all.append(np.vstack([y_xf[valid_windows], y_xs[valid_windows]]).T)
        groups_all.append(np.full(np.sum(valid_windows), int(subj.split('-')[1])))

    if not X_brain_all:
        return None, None, None

    return np.vstack(X_behav_all), np.vstack(X_brain_all), np.concatenate(groups_all)


def main():
    if not os.path.exists(SUBJECT_LIST):
        return
    with open(SUBJECT_LIST, 'r') as f:
        subjects = [line.strip() for line in f if line.strip()]

    print("\nAligning HRF and compiling dataset (Encoding Model, x_f/x_s only)...")
    X_behav_raw, Y_brain_raw, cv_groups = load_and_align_data(subjects)
    if X_behav_raw is None:
        print("Cannot load data.")
        return

    print(f"  > Total samples: {X_behav_raw.shape[0]} windows")
    print(f"  > Predictor features X (Behavior): 2 dimensions (x_f, x_s)")
    print(f"  > Target Y (Brain): {Y_brain_raw.shape[1]} dimensions (Riemannian space connections)")

    # 3 prediction placeholders (in RAW scale): full, xf-only, xs-only
    Y_pred_full = np.zeros_like(Y_brain_raw)
    Y_pred_xf = np.zeros_like(Y_brain_raw)
    Y_pred_xs = np.zeros_like(Y_brain_raw)

    logo = LeaveOneGroupOut()
    alphas = np.logspace(-3, 3, 10)

    print("\nExecuting Leave-One-Subject-Out cross-validation (LOSO) and 3 Ridge modeling sets...")
    fold = 1
    for train_idx, test_idx in logo.split(X_behav_raw, Y_brain_raw, cv_groups):
        test_subj = cv_groups[test_idx[0]]
        print(f"  > Training Fold {fold} (Left out subject sub-{test_subj:02d})...", end="\r")

        X_tr_raw, X_te_raw = X_behav_raw[train_idx], X_behav_raw[test_idx]
        Y_tr_raw, Y_te_raw = Y_brain_raw[train_idx], Y_brain_raw[test_idx]

        # STRICT DATA LEAKAGE PREVENTION: Fit scaler ONLY on training data!
        scaler_X = StandardScaler().fit(X_tr_raw)
        scaler_Y = StandardScaler().fit(Y_tr_raw)

        X_tr = scaler_X.transform(X_tr_raw)
        X_te = scaler_X.transform(X_te_raw)

        Y_tr = scaler_Y.transform(Y_tr_raw)

        # GROUP-AWARE INNER CV: alpha selection must respect subject grouping,
        # otherwise RidgeCV's default leave-one-sample-out GCV leaks across
        # autocorrelated sliding-window samples from the same subject.
        groups_tr = cv_groups[train_idx]
        n_inner_splits = len(np.unique(groups_tr))

        def make_cv():
            return GroupKFold(n_splits=n_inner_splits).split(X_tr, Y_tr, groups=groups_tr)

        # 1. Full Model (xf, xs)
        clf = RidgeCV(alphas=alphas, cv=make_cv()).fit(X_tr, Y_tr)
        Y_pred_full[test_idx] = scaler_Y.inverse_transform(clf.predict(X_te))

        # 2. xf only - col 0
        clf = RidgeCV(alphas=alphas, cv=make_cv()).fit(X_tr[:, [0]], Y_tr)
        Y_pred_xf[test_idx] = scaler_Y.inverse_transform(clf.predict(X_te[:, [0]]))

        # 3. xs only - col 1
        clf = RidgeCV(alphas=alphas, cv=make_cv()).fit(X_tr[:, [1]], Y_tr)
        Y_pred_xs[test_idx] = scaler_Y.inverse_transform(clf.predict(X_te[:, [1]]))

        fold += 1

    print(f"\nLOSO cross-validation complete! Calculating Variance Partitioning Analysis (VPA)...")

    def calc_r2(pred, Y_true):
        return np.maximum(0, r2_score(Y_true, pred, multioutput='raw_values'))

    r2_full = calc_r2(Y_pred_full, Y_brain_raw)
    r2_xf = calc_r2(Y_pred_xf, Y_brain_raw)
    r2_xs = calc_r2(Y_pred_xs, Y_brain_raw)

    # --- Venn Diagram Math (2-var) ---
    unique_xf = np.maximum(0, r2_full - r2_xs)
    unique_xs = np.maximum(0, r2_full - r2_xf)
    shared_xf_xs = np.maximum(0, r2_xf + r2_xs - r2_full)

    print("\nSaving Group-level 2-var VPA neural axes (R^2 arrays)...")
    np.save(f"{OUTPUT_DIR}/Unique_xf_R2.npy", unique_xf)
    np.save(f"{OUTPUT_DIR}/Unique_xs_R2.npy", unique_xs)
    np.save(f"{OUTPUT_DIR}/Shared_xf_xs_R2.npy", shared_xf_xs)
    np.save(f"{OUTPUT_DIR}/Full_xf_xs_R2.npy", r2_full)

    print("Calculating and saving Individual-level VPA neural axes...")
    unique_subjs = np.unique(cv_groups)
    for subj_id in unique_subjs:
        idx = (cv_groups == subj_id)
        Y_sub = Y_brain_raw[idx]
        if len(Y_sub) < 10:
            continue

        sr2_full = calc_r2(Y_pred_full[idx], Y_sub)
        sr2_xf = calc_r2(Y_pred_xf[idx], Y_sub)
        sr2_xs = calc_r2(Y_pred_xs[idx], Y_sub)

        sunique_xf = np.maximum(0, sr2_full - sr2_xs)
        sunique_xs = np.maximum(0, sr2_full - sr2_xf)
        sshared_xf_xs = np.maximum(0, sr2_xf + sr2_xs - sr2_full)

        subj_str = f"sub-{int(subj_id):02d}"
        np.save(f"{OUTPUT_DIR}/Unique_xf_R2_{subj_str}.npy", sunique_xf)
        np.save(f"{OUTPUT_DIR}/Unique_xs_R2_{subj_str}.npy", sunique_xs)
        np.save(f"{OUTPUT_DIR}/Shared_xf_xs_R2_{subj_str}.npy", sshared_xf_xs)
        np.save(f"{OUTPUT_DIR}/Full_xf_xs_R2_{subj_str}.npy", sr2_full)

    print("All 2-var VPA matrices (including group and individual) have been successfully output to ml_results_v2/! Ready for Phase 5 inverse mapping.")


if __name__ == '__main__':
    main()
